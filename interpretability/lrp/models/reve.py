import numpy as np
import torch
import torch.nn as nn

import lxt.explicit.functional as lf

from interpretability.lrp.backward import compute_lrp
from interpretability.lrp.patching import (
    unpatch_all, by_classname, walk_and_patch,
)
from interpretability.lrp.rules import (
    gelu_identity_forward, geglu_identity_forward,
    conv2d_epsilon_forward,
    linear_lrp_forward, layernorm_lrp_forward, rmsnorm_lrp_forward,
)
from interpretability.common.numeric import to_patches_3d
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output

_CONSERVATIVE = False


# ─────────────────────────────────────────────────────────────────────────────
# REVE ClassicalAttention forward (decompose SDPA)
# ─────────────────────────────────────────────────────────────────────────────

def _classical_attention_lrp_forward(self, qkv):
    """
    Replaces F.scaled_dot_product_attention with explicit LRP-aware ops:
      (1) lf.matmul for Q·K^T
      (2) lf.softmax for attention weights
      (3) lf.matmul for A·V
    """
    from einops import rearrange

    q, k, v = qkv.chunk(3, dim=-1)
    q, k, v = (
        rearrange(t, "batch seq (heads dim) -> batch heads seq dim", heads=self.heads)
        for t in (q, k, v)
    )

    scale = q.shape[-1] ** -0.5
    q = q * scale

    # CP-LRP: detach Q,K so all relevance flows through V only
    if _CONSERVATIVE:
        q = q.detach()
        k = k.detach()

    dots = lf.matmul(q, k.transpose(-2, -1))
    attn = lf.softmax(dots, dim=-1)
    out = lf.matmul(attn, v)

    out = rearrange(out, "batch heads seq dim -> batch seq (heads dim)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REVE TransformerBackbone forward (lf.add2 residuals)
# ─────────────────────────────────────────────────────────────────────────────

def _transformer_backbone_lrp_forward(self, x, return_out_layers=False, exit_block=None):
    """Replaces residual additions with LXT sign-preserving add2 ε-rule.

    Accepts ``exit_block`` to match the upstream signature, but LRP requires
    the full backbone pass so a non-None value is rejected.
    """
    if exit_block is not None:
        raise ValueError("exit_block must be None for LRP (full-network attribution only).")
    out_layers = [x] if return_out_layers else []
    for attn, ff in self.layers:
        x = lf.add2(x, attn(x))
        x = lf.add2(x, ff(x))
        if return_out_layers:
            out_layers.append(x)
    return out_layers if return_out_layers else x


# ─────────────────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_reve(model, conservative=False):
    """Apply all AttnLRP patches to a REVE model."""
    global _CONSERVATIVE
    _CONSERVATIVE = conservative

    walk_and_patch(model, [
        (by_classname("ClassicalAttention"),   _classical_attention_lrp_forward),
        (by_classname("TransformerBackbone"),  _transformer_backbone_lrp_forward),
        (by_classname("RMSNorm"),              rmsnorm_lrp_forward),
        (by_classname("GEGLU"),                geglu_identity_forward),
        (nn.LayerNorm,                         layernorm_lrp_forward),
        (nn.GELU,                              gelu_identity_forward),
        (nn.Linear,                            linear_lrp_forward),
        (nn.Conv2d,                            conv2d_epsilon_forward),
    ], debug_label="patch_reve")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, conservative=False, **_):
    """Compute LRP relevances for REVE. Returns (N, C, A).  ``channels`` ignored."""
    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()

        with torch.no_grad():
            out = unwrap_output(model(x.detach()))
            preds_batch, conf_batch = extract_preds_and_confidence(out)
            predictions.append(preds_batch)
            confidences.append(conf_batch)

        target = (torch.tensor(y_test[start : start + batch_size],
                               dtype=torch.long, device=x.device)
                  if y_test is not None else None)

        rel = compute_lrp(model, x, patch_reve, unpatch_all,
                          target_class=target, conservative=conservative)
        model.float()
        relevances.append(to_patches_3d(rel))

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
