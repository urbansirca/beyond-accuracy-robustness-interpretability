import numpy as np
import torch

from captum.attr import InputXGradient

from interpretability.common.numeric import normalize_p99, to_patches_3d
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output


def compute_grad_x_input(model, x, target_class=None, **forward_kwargs):
    """Run Gradient × Input for a batch of inputs. Returns a numpy array of the same shape as x."""
    model.eval()
    x_in = x.detach().float().requires_grad_(True)

    kw_keys = list(forward_kwargs.keys())
    kw_vals = tuple(forward_kwargs[k] for k in kw_keys)

    def forward_fn(inp, *extra):
        out = unwrap_output(model(inp, **dict(zip(kw_keys, extra))))
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if out.shape[1] == 1:
            out = torch.cat([out, out], dim=1)
        return out

    if target_class is None:
        with torch.no_grad():
            target_class = forward_fn(x_in, *kw_vals).argmax(dim=1)

    ixg = InputXGradient(forward_fn)
    relevance = ixg.attribute(
        x_in,
        target=target_class,
        additional_forward_args=kw_vals if kw_vals else None,
    ).detach()

    return normalize_p99(relevance).cpu().numpy()


def run_ixg_generic(model, X_test, batch_size, y_test=None, *,
                    preprocess=None, forward_kwargs_fn=None,
                    preds_fn=None, postprocess=None):
    """Batch loop for Gradient × Input.  Returns (rel, preds, conf).

    Hooks for per-model variation:
      - ``preprocess(x_np) -> x_tensor``   : numpy → CUDA tensor (defaults to
        ``torch.from_numpy(x).float().cuda()``).  Use for channel masking,
        scaling, or shape rearrangement (LaBraM does ``[:, ch_mask, :] / 100``
        then rearrange).
      - ``forward_kwargs_fn(x) -> dict``   : extra kwargs for ``model(x, **kw)``
        and for ``compute_grad_x_input`` (e.g. ``input_chans`` for LaBraM).
      - ``preds_fn(out) -> (preds, conf)`` : defaults to
        ``extract_preds_and_confidence``.  Override only for log-prob heads
        (EEGNet's LogSoftmax).
      - ``postprocess(rel_np) -> rel_3d``  : defaults to ``to_patches_3d``.
        Pass a callable for adapters whose IxG output isn't (B, C, T).
    """
    if preprocess is None:
        preprocess = lambda x_np: torch.from_numpy(x_np).float().cuda()
    if preds_fn is None:
        preds_fn = extract_preds_and_confidence
    if postprocess is None:
        postprocess = to_patches_3d

    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = preprocess(X_test[start : start + batch_size])
        fwd_kw = forward_kwargs_fn(x) if forward_kwargs_fn is not None else {}

        with torch.no_grad():
            preds_b, conf_b = preds_fn(unwrap_output(model(x.detach(), **fwd_kw)))
            predictions.append(preds_b)
            confidences.append(conf_b)

        target = (torch.tensor(y_test[start : start + batch_size],
                               dtype=torch.long, device=x.device)
                  if y_test is not None else None)

        rel = compute_grad_x_input(model, x, target_class=target, **fwd_kw)
        relevances.append(postprocess(rel))

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
