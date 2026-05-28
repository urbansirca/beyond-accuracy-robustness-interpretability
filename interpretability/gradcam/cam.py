import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ─────────────────────────────────────────────────────────────────────────────
# Token → channel projection (used by the manual NeuroRVQ CAM below)
# ─────────────────────────────────────────────────────────────────────────────

def _token_scores_to_channels(
    scores: torch.Tensor,
    model_name: str,
    n_channels: int,
    return_time_patches: bool = False,
) -> np.ndarray:
    """Map per-token CAM scores (B, N) → (B, C) or (B, C, A) by reshaping.

    Token layout:
      LaBraM   — CLS at index 0 followed by C*A patch tokens (CLS dropped).
      NeuroRVQ — CLS at index 0 followed by C*A patch tokens (CLS dropped).
      REVE     — no CLS, N = C*A patch tokens.

    Result is L∞-normalised per sample.
    """
    tokens = scores[:, 1:] if model_name in ("LaBraM", "NeuroRVQ") else scores
    B, N = tokens.shape
    n_time = N // n_channels
    out = tokens[:, : n_channels * n_time].reshape(B, n_channels, n_time)  # (B, C, A)

    if not return_time_patches:
        out = out.mean(dim=-1)  # (B, C)

    dims = list(range(1, out.dim()))
    max_val = out.abs().amax(dim=dims, keepdim=True).clamp(min=1e-9)
    return (out / max_val).cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Target layer discovery
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_layer(model_name: str, model: nn.Module, layer_idx: int = -1) -> nn.Module:
    """Return the module to hook for GradCAM at the requested depth."""
    if model_name in ("LaBraM", "NeuroRVQ"):
        blocks = [
            m for m in model.modules()
            if m.__class__.__name__ == "Block"
            and hasattr(m, "attn") and hasattr(m, "norm1")
        ]
    elif model_name == "REVE":
        backbone = next(
            m for m in model.modules()
            if m.__class__.__name__ == "TransformerBackbone"
        )
        blocks = [pair[1] for pair in backbone.layers]   # FeedForward modules
    elif model_name == "CBraMod":
        blocks = list(model.backbone.encoder.layers)
    else:
        raise ValueError(f"GradCAM not implemented for '{model_name}'.")

    if not blocks:
        raise RuntimeError(f"No hookable blocks found in '{model_name}'.")
    return blocks[layer_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Reshape transform for pytorch-grad-cam
# ─────────────────────────────────────────────────────────────────────────────

def _make_reshape_transform(model_name: str, n_channels: int, n_time: int):
    """
    Convert block output to (B, D, n_channels, n_time) for pytorch-grad-cam.

    pytorch-grad-cam treats D as the "channel" axis and (n_channels, n_time) as
    the 2-D spatial axes, so the final CAM output is (B, n_channels, n_time).

    LaBraM/REVE/NeuroRVQ : block output is (B, N_flat, D); reshape then permute.
      LaBraM also prepends a CLS token that is dropped before reshaping.
    CBraMod              : block output is already (B, C, A, D); just permute.
    """
    if model_name == "CBraMod":
        def reshape_transform(tensor):
            return tensor.permute(0, 3, 1, 2).contiguous()
        return reshape_transform

    has_cls = (model_name in ["LaBraM", "NeuroRVQ"])

    def reshape_transform(tensor):
        # tensor: (B, N_total, D)
        t = tensor[:, 1:, :] if has_cls else tensor   # drop CLS → (B, C*A, D)
        B, _, D = t.shape
        return t.reshape(B, n_channels, n_time, D).permute(0, 3, 1, 2).contiguous()
        # → (B, D, n_channels, n_time)

    return reshape_transform


# ─────────────────────────────────────────────────────────────────────────────
# NeuroRVQ manual GradCAM (shared blocks fire 4× per forward)
# ─────────────────────────────────────────────────────────────────────────────

def _gradcam_neurorvq(
    model: nn.Module,
    x: torch.Tensor,
    n_channels: int,
    target_class: torch.Tensor,
    target_layer: nn.Module,
    return_time_patches: bool,
    forward_kwargs: dict,
) -> np.ndarray:
    """
    GradCAM for NeuroRVQ — averages activations/gradients across the 4 branch
    firings before applying the GradCAM formula.
    """
    act_store, grad_store = [], []

    def fwd_hook(module, inp, out):
        act_store.append(out)

    def bwd_hook(module, grad_in, grad_out):
        grad_store.append(grad_out[0])

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        with torch.enable_grad():
            out = model(x, **forward_kwargs)
            if isinstance(out, dict):
                out = out.get("logits", next(iter(out.values())))
            elif isinstance(out, (list, tuple)):
                out = out[0]
            # Binary sigmoid model → (B,1): expand to (B,2) so indexing by class works.
            if out.shape[-1] == 1:
                out = torch.cat([-out, out], dim=-1)

            seed = torch.zeros_like(out)
            seed[torch.arange(out.shape[0]), target_class] = 1.0
            model.zero_grad()
            out.backward(gradient=seed)
    finally:
        fh.remove()
        bh.remove()

    grads = list(reversed(grad_store))

    act  = torch.stack(act_store).mean(0).detach()
    grad = torch.stack(grads).mean(0).detach()
    alpha = grad.mean(dim=1, keepdim=True)          # (B, 1, D)
    cam   = (alpha * act).sum(dim=-1).clamp(min=0)  # (B, N)
    return _token_scores_to_channels(cam, "NeuroRVQ", n_channels, return_time_patches)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(
    model_name: str,
    model: nn.Module,
    x: torch.Tensor,
    n_channels: int,
    target_class=None,
    target_layer: int = -1,
    return_time_patches: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compute GradCAM for an EEG transformer model.

    Parameters
    ----------
    model_name          : "LaBraM", "REVE", "NeuroRVQ", or "CBraMod"
    model               : fine-tuned model in eval mode on the correct device
    x                   : input batch, float32
    n_channels          : number of EEG channels (for token → channel mapping)
    target_class        : int | (B,) LongTensor | None
                          Class to attribute to.  None → model argmax.
    target_layer        : transformer block index to hook (default -1 = last)
    return_time_patches : if True return (B, C, A), else (B, C)
    **kwargs            : model-specific forward args
                          (input_chans=… for LaBraM,
                           temporal_embedding_ix=… for NeuroRVQ)

    Returns
    -------
    np.ndarray, L∞-normalised per sample.
    Shape (B, n_channels) or (B, n_channels, n_time_patches).
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)

    forward_kwargs = kwargs

    # Infer number of time patches from input shape
    if model_name in ("LaBraM", "NeuroRVQ"):
        n_time = x.shape[2]          # (B, C, A, T) — A is already the patch count
    else:  # REVE, CBraMod: (B, C, T_total), patch_size = 200
        n_time = x.shape[-1] // 200

    # Resolve target class if not given
    if target_class is None:
        with torch.no_grad():
            out = model(x.detach(), **forward_kwargs)
            if isinstance(out, dict):
                out = out.get("logits", next(iter(out.values())))
            elif isinstance(out, (list, tuple)):
                out = out[0]
            target_class = out.argmax(dim=-1)
    elif isinstance(target_class, int):
        target_class = torch.full(
            (x.shape[0],), target_class, dtype=torch.long, device=device
        )

    # ── NeuroRVQ: manual implementation
    if model_name == "NeuroRVQ":
        layer_module = _get_target_layer(model_name, model, target_layer)
        return _gradcam_neurorvq(
            model, x, n_channels, target_class,
            layer_module, return_time_patches, forward_kwargs,
        )

    # ── LaBraM / REVE / CBraMod: pytorch-grad-cam library ───────────────────

    # Thin wrapper so the library can call model(input_tensor) without extra kwargs
    class _ModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self._m = model

        def forward(self, inp):
            out = self._m(inp, **forward_kwargs)
            if isinstance(out, dict):
                out = out.get("logits", next(iter(out.values())))
            elif isinstance(out, (list, tuple)):
                out = out[0]
            # Binary model outputs (B,) or (B,1): convert to (B,2) so
            # ClassifierOutputTarget can index class 0 or 1
            if out.dim() == 1:
                out = torch.stack([-out, out], dim=-1)
            elif out.shape[-1] == 1:
                out = torch.cat([-out, out], dim=-1)
            return out

    wrapped = _ModelWrapper()
    layer_module = _get_target_layer(model_name, model, target_layer)
    reshape_fn = _make_reshape_transform(model_name, n_channels, n_time)

    targets = [ClassifierOutputTarget(int(c)) for c in target_class.cpu()]

    _nc, _nt = n_channels, n_time

    class _EEGCam(GradCAM):
        def get_target_width_height(self, _input_tensor):
            # cv2.resize(img, (width, height)) → output shape (height, width)
            # We want output (n_channels, n_time), so pass (n_time, n_channels).
            return _nt, _nc

    with _EEGCam(
        model=wrapped,
        target_layers=[layer_module],
        reshape_transform=reshape_fn,
    ) as cam:
        result = cam(input_tensor=x, targets=targets)   # (B, n_channels, n_time) numpy

    if not return_time_patches:
        result = result.mean(axis=-1)   # (B, n_channels)

    # Per-sample L∞ normalisation
    dims = tuple(range(1, result.ndim))
    max_val = np.abs(result).max(axis=dims, keepdims=True).clip(min=1e-9)
    return result / max_val
