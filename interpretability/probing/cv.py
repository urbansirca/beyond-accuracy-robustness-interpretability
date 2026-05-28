from __future__ import annotations

import os

import numpy as np
import torch
from sklearn import model_selection

from data.loaders import load_benchmark, get_subdir
from models.wrappers import get_model

from .adapters import get_blocks, get_callable_model
from .classifier import run_probe, run_probe_concat
from .features import extract_block_features
from .io import build_row, write_rows

from data.loaders import slug_for


def _run_probes_from_cache(blocks, cached_features, labels, train_mask, test_mask,
                           device, n_epochs, batch_size, patience) -> dict:
    """Run nn.Linear probes on pre-extracted features (mean/cls path)."""
    out = {}
    for name, _ in blocks:
        if name not in cached_features:
            continue
        feats = cached_features[name]
        out[name] = run_probe(
            feats[train_mask], labels[train_mask],
            feats[test_mask],  labels[test_mask],
            device, n_epochs=n_epochs, batch_size=batch_size, patience=patience,
        )
    return out


def probe_model(
    model_name: str,
    benchmark_name: str,
    ckpt_type: str,
    device: torch.device,
    writer,
    csv_file,
    cfg: dict,
    completed_keys: set,
    pooling: str,
    large_head: bool,
) -> None:
    """Run k-fold CV probing for one combination."""
    subdir = get_subdir(model_name)
    sfreq  = 256 if model_name == "BrainOmni" else 200

    bench = load_benchmark(benchmark_name, cfg["data_root"], subdir=subdir)
    eeg, subject_ids, labels, chnames = bench.get_data()
    n_times   = eeg.shape[2]
    n_outputs = len(np.unique(labels))

    kf = model_selection.KFold(n_splits=cfg["n_folds"], shuffle=True,
                               random_state=cfg["cv_seed"])
    subject_ids_unique = np.unique(subject_ids)

    batch_size     = cfg.get("model_batch_size", {}).get(model_name, cfg["batch_size"])
    n_epochs       = cfg["n_epochs"]
    patience       = cfg["patience"]
    overwrite      = cfg["overwrite"]

    # ── Resource setup ──────────────────────────────────────────────────────
    # Pretrained: build wrapper once. Finetuned: built per-fold below.
    cached_features: dict | None = None
    pretrained_wrapper = pretrained_nn_model = pretrained_blocks = None

    if ckpt_type == "pretrained":
        pretrained_wrapper  = get_model(
            model_name, n_chans=len(chnames), ch_names=chnames, sfreq=sfreq,
            n_times=n_times, n_outputs=n_outputs, sbj_ids=subject_ids,
            encoder_only=False, ckpt_path=None, train_head_only=False,
            large_head=large_head,
        )
        pretrained_nn_model = get_callable_model(pretrained_wrapper).to(device)
        pretrained_blocks   = get_blocks(model_name, pretrained_nn_model)
        print(f"found {pretrained_blocks} blocks in pretrained {model_name}")
        if not pretrained_blocks:
            raise ValueError(f"  [skip] No transformer blocks found in {model_name}")
        print(f"  Found {len(pretrained_blocks)} blocks")

        # mean/cls: extract features once, reuse across folds.
        if pooling != "concat":
            cached_features = extract_block_features(
                model_name, pretrained_nn_model, pretrained_blocks,
                eeg, chnames, batch_size, pooling
            )

    # ── Fold loop ───────────────────────────────────────────────────────────
    for i_fold, (sbj_idx_train, sbj_idx_test) in enumerate(kf.split(subject_ids_unique)):
        fold_key = (model_name, benchmark_name, ckpt_type, pooling, large_head, i_fold)
        if not overwrite and fold_key in completed_keys:
            print(f"  [skip fold {i_fold}] Already in results (overwrite=False)")
            continue

        train_subjects = subject_ids_unique[sbj_idx_train].tolist()
        test_subjects  = subject_ids_unique[sbj_idx_test].tolist()
        bench.check_and_save_split(i_fold, train_subjects, test_subjects)
        train_mask = np.isin(subject_ids, subject_ids_unique[sbj_idx_train])
        test_mask  = np.isin(subject_ids, subject_ids_unique[sbj_idx_test])

        # Pick the model and blocks for this fold.
        if ckpt_type == "pretrained":
            nn_model, blocks = pretrained_nn_model, pretrained_blocks
        else:
            prefix    = "full_large_head" if large_head else "full"
            ckpt_path = os.path.join(
                cfg["ckpt_dir"], model_name,
                slug_for(benchmark_name),
                f"{prefix}_train-clean_fold{i_fold}_best.pt",
            )
            if not os.path.exists(ckpt_path):
                print(f"  [skip fold {i_fold}] Checkpoint not found: {ckpt_path}")
                continue
            wrapper = get_model(
                model_name, n_chans=len(chnames), ch_names=chnames, sfreq=sfreq,
                n_times=n_times, n_outputs=n_outputs, sbj_ids=subject_ids,
                encoder_only=False, ckpt_path=None, train_head_only=False,
                large_head=large_head,
            )
            wrapper.load_model(ckpt_path)
            nn_model = get_callable_model(wrapper).to(device)
            blocks   = get_blocks(model_name, nn_model)
            if not blocks:
                print(f"  [skip] No transformer blocks found in {model_name}")
                raise ValueError(f"  [skip] No transformer blocks found in {model_name}")
            if i_fold == 0:
                print(f"  Found {len(blocks)} blocks")

        # Compute per-block metrics for this fold.
        if pooling == "concat":
            block_metrics = run_probe_concat(
                model_name, nn_model, blocks, eeg, labels,
                train_mask, test_mask, chnames, batch_size,
                n_epochs=n_epochs, patience=patience,
            )
        elif cached_features is not None:
            block_metrics = _run_probes_from_cache(
                blocks, cached_features, labels, train_mask, test_mask,
                device, n_epochs, batch_size, patience,
            )
        else:
            # Finetuned + (mean|cls): re-extract for this fold's model.
            features = extract_block_features(
                model_name, nn_model, blocks, eeg, chnames,
                batch_size, pooling,
            )
            block_metrics = _run_probes_from_cache(
                blocks, features, labels, train_mask, test_mask,
                device, n_epochs, batch_size, patience,
            )

        # Stream rows for this fold.
        fold_rows = []
        for i_block, (name, _) in enumerate(blocks):
            if name not in block_metrics:
                continue
            m = block_metrics[name]
            print(f"    Fold {i_fold} Block {i_block:2d}: "
                  f"bacc={m['bacc']:.4f}  best_epoch={m.get('best_epoch', '')}")
            fold_rows.append(build_row(
                model=model_name, benchmark=benchmark_name, ckpt_type=ckpt_type,
                pooling=pooling, large_head=large_head, fold=i_fold,
                block_idx=i_block, block_name=name, n_blocks=len(blocks),
                metrics=m,
            ))
        write_rows(writer, csv_file, fold_rows)
