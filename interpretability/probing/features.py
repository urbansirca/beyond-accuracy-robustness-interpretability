from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .adapters import EXTERNAL_RESIDUAL_MODELS, get_forward_fn


def n_passes_per_forward(model_name: str) -> int:
    return 4 if model_name == "NeuroRVQ" else 1


def make_block_hook(
    name: str,
    *,
    n_passes: int,
    sink,
    on_first_call=None,
    add_residual: bool = False,
):
    """
    Build a forward hook that calls `sink(name, output_tensor)` once per
    input. For `n_passes > 1`, buffers each branch's block output and emits a
    single tensor formed by concatenating the branches on the last (embedding)
    axis.
    """
    counter = [0]
    first = [True]
    branch_buf: list = [None] * n_passes

    def hook(_module, _inp, output):
        counter[0] += 1
        branch_idx = (counter[0] - 1) % n_passes
        out = output[0] if isinstance(output, tuple) else output
        if add_residual:
            out = out + _inp[0]
        branch_buf[branch_idx] = out
        if branch_idx != n_passes - 1:
            return
        full = out if n_passes == 1 else torch.cat(branch_buf, dim=-1)
        if first[0] and on_first_call is not None:
            on_first_call(name, full)
            first[0] = False
        sink(name, full)
        for k in range(n_passes):
            branch_buf[k] = None

    return hook


def _pool(out: torch.Tensor, pooling: str) -> torch.Tensor:
    """Reduce a block output to (B, d)"""
    if out.dim() == 3:
        if pooling == "cls":
            return out[:, 0, :]
        if pooling == "concat":
            return out.reshape(out.size(0), -1)
        return out.mean(dim=1)
    if out.dim() == 4:
        if pooling == "concat":
            return out.reshape(out.size(0), -1)
        return out.mean(dim=(1, 2))
    return out.view(out.size(0), -1)


def extract_block_features(
    model_name: str,
    nn_model: nn.Module,
    blocks,
    X_np: np.ndarray,
    ch_names,
    batch_size: int,
    pooling: str = "mean",
) -> dict:
    """Forward pass through `nn_model`; pool each block's output per `pooling`.
    Returns {block_name: np.ndarray (N, feature_dim)}."""
    activations: dict[str, list[torch.Tensor]] = {name: [] for name, _ in blocks}
    n_passes = n_passes_per_forward(model_name)
    add_residual = model_name in EXTERNAL_RESIDUAL_MODELS

    def on_first(name, out):
        pooled = _pool(out, pooling)
        print(f"    [{pooling}] {name}: {tuple(out.shape[1:])} → {pooled.size(1)} dims")

    def sink(name, out):
        feat = _pool(out, pooling).detach().cpu().float()
        activations[name].append(feat)

    handles = [
        mod.register_forward_hook(
            make_block_hook(name, n_passes=n_passes, sink=sink,
                            on_first_call=on_first, add_residual=add_residual)
        )
        for name, mod in blocks
    ]
    forward_fn = get_forward_fn(model_name, nn_model, ch_names, X_np.shape[2])

    nn_model.eval()
    X_f32    = np.array(X_np, dtype=np.float32)
    X_tensor = torch.from_numpy(X_f32)
    device   = next(nn_model.parameters()).device
    n_batches = (len(X_f32) + batch_size - 1) // batch_size
    print(f"    extracting: {len(X_f32)} trials, {n_batches} batches "
          f"of {batch_size} on {device}")
    fwd_error_printed = False
    with torch.no_grad():
        for i in tqdm(range(0, len(X_f32), batch_size), total=n_batches,
                      desc=f"    {model_name} forward", leave=False):
            x = X_tensor[i : i + batch_size]
            try:
                forward_fn(x)
            except Exception as e:
                if not fwd_error_printed:
                    print(f"    [forward error] {type(e).__name__}: {e}")
                    fwd_error_printed = True
                break
    print(f"    extraction done for {model_name}")

    for h in handles:
        h.remove()

    return {
        name: torch.cat(acts, dim=0).numpy()
        for name, acts in activations.items()
        if acts
    }
