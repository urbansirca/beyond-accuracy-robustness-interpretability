import numpy as np
import torch
import torch.nn as nn

from interpretability.lrp.backward import compute_lrp
from interpretability.lrp.patching import unpatch_all, walk_and_patch
from interpretability.lrp.rules import (
    elu_identity_forward,
    conv2d_epsilon_forward,
)
from interpretability.common.numeric import to_patches_3d
from interpretability.common.predict import unwrap_output


# ─────────────────────────────────────────────────────────────────────────────
# BatchNorm2d identity rule
# ─────────────────────────────────────────────────────────────────────────────

def batchnorm2d_identity_forward(self, x):
    """
    Identity rule for BatchNorm2d (eval mode).
    """
    mean = self.running_mean.view(1, -1, 1, 1)
    var  = self.running_var.view(1, -1, 1, 1)
    std  = (var + self.eps).sqrt()
    y    = (x - mean.detach()) / std.detach()
    if self.weight is not None:
        y = y * self.weight.view(1, -1, 1, 1)
    if self.bias is not None:
        y = y + self.bias.view(1, -1, 1, 1)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# LogSoftmax identity rule  (keep logit values; avoid log-space distortion)
# ─────────────────────────────────────────────────────────────────────────────

def logsoftmax_identity_forward(self, x):
    """Pass logits through unchanged so LRP seeds at the raw score."""
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_eegnet(model, conservative=False):
    """Apply LRP patches to an EEGNetv1 instance."""
    from braindecode.models.modules import Expression

    is_elu_expr = lambda m: (
        isinstance(m, Expression)
        and getattr(m.expression_fn, "__name__", "") == "elu"
    )
    is_elu_expr.__name__ = "Expression(elu)"

    walk_and_patch(model, [
        (nn.Conv2d,      conv2d_epsilon_forward),
        (nn.BatchNorm2d, batchnorm2d_identity_forward),
        (nn.LogSoftmax,  logsoftmax_identity_forward),
        (is_elu_expr,    elu_identity_forward),
    ], debug_label="patch_eegnet")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

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


def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, conservative=False, **_):
    """
    Compute LRP relevances for EEGNet via the ε-rule CNN backend.
    Returns (N, C, A) — |relevance| averaged per 1-s time patch.
    """
    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()

        with torch.no_grad():
            preds_b, conf_b = _eegnet_preds_conf(unwrap_output(model(x.detach())))
            predictions.append(preds_b)
            confidences.append(conf_b)

        target = (torch.tensor(y_test[start : start + batch_size],
                               dtype=torch.long, device=x.device)
                  if y_test is not None else None)

        rel = compute_lrp(model, x, patch_eegnet, unpatch_all,
                          target_class=target, conservative=False)
        model.float()
        relevances.append(to_patches_3d(rel))

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
