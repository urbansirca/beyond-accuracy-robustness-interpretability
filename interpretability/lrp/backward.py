import numpy as np
import torch

from interpretability.common.numeric import normalize_p99
from interpretability.common.predict import unwrap_output

# Set to True to print relevance range and overflow diagnostics per batch.
LRP_DEBUG = False


def run_backward(model, x, target_class, **forward_kwargs):
    """
    Seed relevance at the target logit and run one backward pass in float64.
    Returns the LRP relevance tensor (same shape as x, dtype float32).

    This is model-agnostic — all model-specific logic is in the patches.
    """
    model.eval()
    model.double()
    x = x.detach().double().requires_grad_(True)

    # Also cast any tensor forward_kwargs to float64
    fwd_kwargs_cast = {}
    for k, v in forward_kwargs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            fwd_kwargs_cast[k] = v.double()
        else:
            fwd_kwargs_cast[k] = v

    try:
        with torch.enable_grad():
            out = unwrap_output(model(x, **fwd_kwargs_cast))

            if target_class is None:
                target_class = out.argmax(dim=1)

            if out.shape[1] == 1:
                # Binary model: single logit, column index 0 is the only valid one.
                # Always seed at column 0 regardless of target_class.
                seed = torch.ones_like(out)
            else:
                seed = torch.zeros_like(out)
                seed[torch.arange(out.shape[0]), target_class] = 1.0
            out.backward(gradient=seed)

        # x.grad IS the LRP relevance
        relevance = x.grad.detach()

        if LRP_DEBUG:
            mn, mx = relevance.min().item(), relevance.max().item()
            has_nan = torch.isnan(relevance).any().item()
            has_inf = torch.isinf(relevance).any().item()
            print(f"  [LRP_DEBUG] raw (float64):  "
                  f"min={mn:.3e}  max={mx:.3e}  "
                  f"nan={has_nan}  inf={has_inf}")

        # Per-sample normalisation in float64 before float32 conversion.
        relevance = normalize_p99(relevance)

        if LRP_DEBUG:
            mn2, mx2 = relevance.min().item(), relevance.max().item()
            print(f"  [LRP_DEBUG] normalised:     min={mn2:.3e}  max={mx2:.3e}")

    finally:
        model.float()
        x.grad = None

    return relevance.float()


def compute_lrp(model, x, patch, unpatch, *,
                target_class=None, conservative=False, **forward_kwargs):
    """Patch the model, run one LRP backward pass, unpatch, return numpy.

    Parameters
    ----------
    model        : nn.Module — fine-tuned, in eval mode, on the right device
    x            : (B, ...) float32 input tensor
    patch        : callable ``patch(model, conservative=...)`` — installs the
                   per-model LRP forwards
    unpatch      : callable ``unpatch(model)`` — restores original forwards
    target_class : int, (B,) tensor, or None (None → predicted class)
    conservative : bool — CP-LRP
    **forward_kwargs : model-specific forward args

    Returns
    -------
    relevance : np.ndarray — same shape as x
    """
    patch(model, conservative=conservative)
    try:
        rel = run_backward(model, x, target_class, **forward_kwargs)
    finally:
        unpatch(model)
    return rel.cpu().numpy()
