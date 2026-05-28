"""EEGNet Gradient × Input (LogSoftmax → log-prob head)."""

import torch

from interpretability.ixg.captum import run_ixg_generic


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


def run(model, X_test, batch_size, ch_names, channels, *, y_test=None, **_):
    """Gradient × Input for EEGNet. Returns (N, C, A)."""
    return run_ixg_generic(model, X_test, batch_size, y_test,
                           preds_fn=_eegnet_preds_conf)
