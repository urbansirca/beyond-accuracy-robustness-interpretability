import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from captum.attr import InputXGradient

from interpretability.common.numeric import normalize_p99
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output
from models.BrainOmni._repo import ensure_repo_on_path


def _grad_friendly_forward(m, X):
    """Inline BrainOmniModule.forward without @torch.no_grad() barriers."""
    X = X - X.mean(dim=1, keepdim=True)
    std = X.std(dim=(1, 2), keepdim=True) + 1e-5
    X = X / std
    B = X.shape[0]
    pos = m._pos.unsqueeze(0).expand(B, -1, -1).float()
    st  = m._sensor_type.unsqueeze(0).expand(B, -1)

    tokenizer     = m.backbone.tokenizer
    overlap_ratio = m.backbone.overlap_ratio

    t = tokenizer.unfold(X, overlap_ratio=overlap_ratio)
    sensor_emb = tokenizer.sensor_embed(pos, st)
    # cuDNN LSTM backward requires training mode
    tokenizer.encoder.train()
    t = tokenizer.encoder(t, sensor_emb)
    tokenizer.encoder.eval()

    t_normed = F.normalize(t, p=2.0, dim=-1, eps=1e-12)
    if getattr(m, "skip_tokenizer", False):
        # No codebook lookup: gradients flow through F.normalize directly,
        # so no straight-through estimator is needed.
        t_q = t_normed
    else:
        t_q, _, _ = tokenizer.quantizer.rvq(t_normed)
        # Straight-through estimator (VQ.forward only applies it in training mode).
        t_q = t_normed + (t_q - t_normed).detach()
    feature = rearrange(t_q, "B C N T D -> B C (N T) D")

    backbone = m.backbone
    _, C_, W_, _ = feature.shape
    neuro = tokenizer.encoder.neuros.type_as(feature).detach().view(1, C_, 1, -1)
    t = feature + neuro
    t = backbone.projection(t)
    for block in backbone.blocks[:-1]:
        t = block(t)
    t = F.normalize(t, p=2.0, dim=-1, eps=1e-6)

    if t.ndim == 4:
        t = t.mean(dim=2)
    t = t.contiguous().view(t.shape[0], -1)
    return m.downstream_model.class_head(t)


def run(model, X_test, batch_size, ch_names, channels, *, y_test=None, **_):
    """Compute Gradient × Input attribution for BrainOmni.  ``channels`` ignored.

    Output is averaged within BrainOmni's 512-sample patches.  Returns (N, C, A).
    """
    ensure_repo_on_path()
    m = model.model if hasattr(model, 'model') else model
    m.eval()

    patch_size = 512
    ixg = InputXGradient(lambda x: _grad_friendly_forward(m, x))

    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()

        with torch.no_grad():
            out = unwrap_output(model(x.detach()))
            preds_batch, conf_batch = extract_preds_and_confidence(out)
            predictions.append(preds_batch)
            confidences.append(conf_batch)

        x_in = x.detach().float().requires_grad_(True)

        if y_test is not None:
            target = torch.tensor(
                y_test[start : start + batch_size], dtype=torch.long, device=x.device
            )
        else:
            with torch.no_grad():
                target = _grad_friendly_forward(m, x_in).argmax(dim=1)

        rel = ixg.attribute(x_in, target=target).detach()
        rel_np = normalize_p99(rel).cpu().numpy()

        # rel: (B, C, T) → (B, C, A)
        B, C, T = rel_np.shape
        print(f"  batch {start} - {start + len(x)}: relevance shape {rel_np.shape}")
        n_patches = T // patch_size
        if n_patches == 0:
            rel_3d = np.abs(rel_np).mean(axis=-1, keepdims=True)
        else:
            rel_3d = np.abs(rel_np[:, :, :n_patches * patch_size]).reshape(
                B, C, n_patches, patch_size
            ).mean(axis=-1)
            
        print(f"    → patched relevance shape {rel_3d.shape}")
        relevances.append(rel_3d)

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
