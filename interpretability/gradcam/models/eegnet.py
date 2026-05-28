import numpy as np
import torch

from captum.attr import LayerGradCam

from interpretability.common.numeric import to_patches_3d
from interpretability.common.predict import unwrap_output


def _eegnet_preds_conf(out):
    """EEGNet uses LogSoftmax → log-probs; binary head is single-logit."""
    if out.dim() == 1 or out.shape[-1] == 1:
        prob = torch.sigmoid(out.float().squeeze(-1) if out.dim() > 1 else out.float())
        preds = (prob >= 0.5).long().cpu().numpy()
        conf  = torch.where(prob >= 0.5, prob, 1 - prob).cpu().numpy()
    else:
        probs = out.float().exp()
        conf, preds = probs.max(dim=-1)
        preds = preds.cpu().numpy()
        conf  = conf.cpu().numpy()
    return preds, conf


def _find_conv1(model):
    """Locate EEGNet's ``conv_1`` (1×1 pointwise Conv2d) regardless of wrapping."""
    for name, module in model.named_modules():
        if name.split(".")[-1] == "conv_1":
            return module
    raise RuntimeError("conv_1 not found on EEGNet model.")


def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, target_layer=-1, **_):
    """GradCAM for EEGNet at conv_1, via captum ``LayerGradCam``.

    ``conv_1`` is a 1×1 pointwise Conv2d whose *input-channel* axis is the
    electrode axis — every later activation has collapsed electrodes into learned
    "virtual channels", so conv_1's input is the only place a per-electrode map
    exists.  We attribute to that input (``attribute_to_layer_input=True``) and
    keep the electrode axis (``attr_dim_summation=False`` — the default would sum
    it away), giving the canonical GradCAM ``ReLU(αₖ · Xₖ(t))`` with αₖ the
    time-averaged gradient per electrode.  ``channels``/``target_layer`` ignored.
    Returns (N, C, A).
    """
    model.eval()

    def forward_func(inp):
        out = unwrap_output(model(inp))
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if out.shape[-1] == 1:                    # binary single-logit → 2 cols
            out = torch.cat([-out, out], dim=-1)
        return out

    grad_cam = LayerGradCam(forward_func, _find_conv1(model))

    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()

        with torch.no_grad():
            preds_b, conf_b = _eegnet_preds_conf(unwrap_output(model(x.detach())))
            predictions.append(preds_b)
            confidences.append(conf_b)

        target = torch.tensor(
            y_test[start : start + batch_size] if y_test is not None else preds_b,
            dtype=torch.long, device=x.device,
        )

        cam = grad_cam.attribute(
            x, target=target,
            attribute_to_layer_input=True,   # hook conv_1's input (electrodes live here)
            attr_dim_summation=False,        # keep the electrode axis, don't sum it
            relu_attributions=True,          # the GradCAM ReLU
        )
        if isinstance(cam, tuple):
            cam = cam[0]
        cam = cam.squeeze(-1).detach().cpu().numpy()   # (B, C, T, 1) → (B, C, T)

        rel = to_patches_3d(cam)             # (B, C, A)  (abs is a no-op here)

        # Per-sample L∞ (matches compute_gradcam for the other models)
        max_val = np.abs(rel).max(axis=(1, 2), keepdims=True).clip(min=1e-9)
        relevances.append(rel / max_val)

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
