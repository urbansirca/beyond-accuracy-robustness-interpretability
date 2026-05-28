from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from .adapters import EXTERNAL_RESIDUAL_MODELS, get_forward_fn
from .features import make_block_hook, n_passes_per_forward


def _metrics(y_true, y_pred):
    return {
        "bacc":        balanced_accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
    }


def _build_probe(in_dim, n_classes, n_epochs, batches_per_epoch, device):
    """Construct one linear probe + Adam + CosineAnnealingLR."""
    clf = nn.Linear(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs * batches_per_epoch, eta_min=1e-5)
    return clf, opt, sched


def _results_row(probe_dim, train_m, test_m, best_epoch):
    """Assemble the per-probe results dict written to the CSV."""
    return {
        "probe_dim":         probe_dim,
        "bacc":              test_m["bacc"],
        "f1_weighted":       test_m["f1_weighted"],
        "f1_macro":          test_m["f1_macro"],
        "train_bacc":        train_m["bacc"],
        "train_f1_weighted": train_m["f1_weighted"],
        "train_f1_macro":    train_m["f1_macro"],
        "best_epoch":        best_epoch,
    }


def run_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    device: torch.device,
    n_epochs: int = 20, batch_size: int = 1000, patience: int = 5,
) -> dict:
    """GPU linear probe with cosine LR + early stopping. 15 % of training
    as a validation split for the early-stop."""
    n_classes = int(np.unique(y_train).size)

    rng   = np.random.default_rng(0)
    idx   = rng.permutation(len(X_train))
    n_val = max(1, int(0.15 * len(idx)))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    X_fit    = torch.from_numpy(X_train[fit_idx].astype(np.float32)).to(device)
    y_fit    = torch.from_numpy(y_train[fit_idx]).long().to(device)
    X_val    = torch.from_numpy(X_train[val_idx].astype(np.float32)).to(device)
    y_val_np = y_train[val_idx]

    batches_per_epoch = (len(fit_idx) + batch_size - 1) // batch_size
    clf, opt, sched = _build_probe(X_fit.size(1), n_classes, n_epochs,
                                   batches_per_epoch, device)

    best_val_bacc, best_epoch_num, best_state, patience_count = -1.0, 0, None, 0
    
    for epoch in range(n_epochs):
        clf.train()
        perm = torch.randperm(len(X_fit), device=device)
        for start in range(0, len(X_fit), batch_size):
            idx_b = perm[start : start + batch_size]
            opt.zero_grad()
            F.cross_entropy(clf(X_fit[idx_b]), y_fit[idx_b]).backward()
            opt.step()
            sched.step()

        clf.eval()
        with torch.no_grad():
            val_preds = clf(X_val).argmax(dim=1).cpu().numpy()
        bacc = balanced_accuracy_score(y_val_np, val_preds)
        if bacc > best_val_bacc:
            best_val_bacc  = bacc
            best_epoch_num = epoch + 1
            best_state     = {k: v.clone() for k, v in clf.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    clf.load_state_dict(best_state)
    clf.eval()

    X_tr_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_te_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        p_train = clf(X_tr_t).argmax(dim=1).cpu().numpy()
        p_test  = clf(X_te_t).argmax(dim=1).cpu().numpy()

    test_m  = _metrics(y_test,  p_test)
    train_m = _metrics(y_train, p_train)
    return _results_row(X_fit.size(1), train_m, test_m, best_epoch_num)

def run_probe_concat(
    model_name: str, nn_model: nn.Module, blocks,
    X_np: np.ndarray, y_np: np.ndarray,
    train_mask: np.ndarray, test_mask: np.ndarray,
    ch_names, batch_size: int, n_epochs: int = 20, patience: int = 5,
) -> dict:
    """Fit one nn.Linear per block by streaming (B, concat_dim) features."""
    device       = next(nn_model.parameters()).device
    n_passes     = n_passes_per_forward(model_name)
    add_residual = model_name in EXTERNAL_RESIDUAL_MODELS
    n_classes    = int(np.unique(y_np).size)
    forward_fn = get_forward_fn(model_name, nn_model, ch_names, X_np.shape[2])

    linears:    dict[str, nn.Linear]                            = {}
    optimizers: dict[str, torch.optim.Optimizer]                = {}
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}

    X_tensor = torch.from_numpy(np.array(X_np, dtype=np.float32))

    # 15 % of training indices for val/early-stop
    train_idx = np.where(train_mask)[0]
    np.random.default_rng(0).shuffle(train_idx)
    n_val    = max(1, int(0.15 * len(train_idx)))
    val_idx  = train_idx[:n_val]
    fit_idx  = train_idx[n_val:]
    fit_mask = np.zeros(len(y_np), dtype=bool); fit_mask[fit_idx] = True
    val_mask = np.zeros(len(y_np), dtype=bool); val_mask[val_idx] = True

    n_train           = int(fit_mask.sum())
    batches_per_epoch = (n_train + batch_size - 1) // batch_size
    total_batches     = (n_epochs + 2) * batches_per_epoch
    pbar = tqdm(total=total_batches, desc="  concat probe", unit="batch", leave=False)

    fwd_error_printed = [False]

    def _run_pass(mask: np.ndarray, fit: bool, epoch: int = 0):
        idx = np.where(mask)[0]
        if fit:
            idx = np.random.default_rng(42 + epoch).permutation(idx)

        all_preds = {name: [] for name, _ in blocks}

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            x_batch   = X_tensor[batch_idx]
            y_batch   = torch.from_numpy(y_np[batch_idx]).long().to(device)
            batch_acts: dict[str, torch.Tensor | None] = {name: None for name, _ in blocks}

            def on_first(name, out):
                flat = out.reshape(out.size(0), -1)
                print(f"    [concat] {name}: {tuple(out.shape[1:])} → {flat.size(1)} dims")

            def sink(name, out):
                flat = out.reshape(out.size(0), -1).detach().float()
                batch_acts[name] = F.rms_norm(flat, (flat.size(-1),), eps=1e-6)

            handles = [
                mod.register_forward_hook(make_block_hook(
                    name, n_passes=n_passes, sink=sink,
                    on_first_call=on_first if name not in linears else None,
                    add_residual=add_residual,
                ))
                for name, mod in blocks
            ]
            nn_model.eval()
            with torch.no_grad():
                try:
                    forward_fn(x_batch)
                except Exception as e:
                    if not fwd_error_printed[0]:
                        print(f"    [forward error] {type(e).__name__}: {e}")
                        fwd_error_printed[0] = True
            for h in handles:
                h.remove()

            for name, _ in blocks:
                feat = batch_acts[name]
                if feat is None:
                    continue
                if name not in linears:
                    linears[name], optimizers[name], schedulers[name] = _build_probe(
                        feat.size(1), n_classes, n_epochs, batches_per_epoch, device,
                    )
                if fit:
                    linears[name].train()
                    optimizers[name].zero_grad()
                    F.cross_entropy(linears[name](feat), y_batch).backward()
                    optimizers[name].step()
                    schedulers[name].step()
                else:
                    linears[name].eval()
                    with torch.no_grad():
                        all_preds[name].append(
                            linears[name](feat).argmax(dim=1).cpu().numpy()
                        )

            pbar.update(1)
        return all_preds

    best_val_bacc  = {name: -1.0  for name, _ in blocks}
    best_epoch     = {name: 0     for name, _ in blocks}
    best_weights:   dict[str, dict] = {}
    patience_count = {name: 0     for name, _ in blocks}
    stopped        = {name: False for name, _ in blocks}

    for epoch in range(n_epochs):
        if all(stopped.values()):
            break
        pbar.set_description(f"  concat probe [epoch {epoch+1}/{n_epochs}]")
        _run_pass(fit_mask, fit=True, epoch=epoch)

        val_preds = _run_pass(val_mask, fit=False)
        for name, _ in blocks:
            if stopped[name] or not val_preds[name]:
                continue
            bacc = balanced_accuracy_score(y_np[val_mask], np.concatenate(val_preds[name]))
            if bacc > best_val_bacc[name]:
                best_val_bacc[name]  = bacc
                best_epoch[name]     = epoch + 1
                best_weights[name]   = {k: v.clone() for k, v in linears[name].state_dict().items()}
                patience_count[name] = 0
            else:
                patience_count[name] += 1
                if patience_count[name] >= patience:
                    stopped[name] = True

    print("  [concat probe] best epochs:")
    for i_block, (name, _) in enumerate(blocks):
        print(f"    block {i_block:2d}: epoch {best_epoch[name]:3d}  val_bacc={best_val_bacc[name]:.4f}")

    pbar.set_description("  concat probe [eval train]")
    for name, _ in blocks:
        if name in best_weights:
            linears[name].load_state_dict(best_weights[name])
    train_preds = _run_pass(train_mask, fit=False)
    pbar.set_description("  concat probe [eval test]")
    test_preds  = _run_pass(test_mask,  fit=False)
    pbar.close()

    results = {}
    y_train = y_np[train_mask]
    y_test  = y_np[test_mask]
    for name, _ in blocks:
        if not train_preds[name] or not test_preds[name]:
            continue
        p_tr = np.concatenate(train_preds[name])
        p_te = np.concatenate(test_preds[name])
        test_m  = _metrics(y_test,  p_te)
        train_m = _metrics(y_train, p_tr)
        results[name] = _results_row(
            linears[name].in_features, train_m, test_m, best_epoch[name],
        )
    return results
