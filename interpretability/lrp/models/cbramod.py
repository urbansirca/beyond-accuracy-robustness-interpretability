import numpy as np
import torch
import torch.nn as nn

import lxt.explicit.functional as lf

from interpretability.lrp.backward import compute_lrp
from interpretability.lrp.patching import (
    save_and_replace, unpatch_all, walk_and_patch,
)
from interpretability.lrp.rules import (
    groupnorm_lrp_forward,
    GELUIdentityFn,
    gelu_identity_forward,
    elu_identity_forward,
    conv2d_epsilon_forward,
    linear_lrp_forward, layernorm_lrp_forward,
)
from interpretability.common.numeric import to_patches_3d
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output


# Module-level flag: when True, use CP-LRP (detach Q,K → only V gets relevance)
_CONSERVATIVE = False


# ─────────────────────────────────────────────────────────────────────────────
# nn.MultiheadAttention LRP forward
# ─────────────────────────────────────────────────────────────────────────────

def _mha_lrp_forward(self, query, key, value,
                     attn_mask=None, key_padding_mask=None,
                     need_weights=False, **kwargs):
    """
    Drop-in for nn.MultiheadAttention.forward with AttnLRP rules.

    Decomposes the fused kernel into lf.linear_epsilon (QKV), lf.matmul (Q·K^T),
    lf.softmax, lf.matmul (A·V); out_proj is patched by the module walk.
    Returns (output, None).  Input is batch_first=True: (B, N, D).
    """
    B, N, D = query.shape

    # (1) Combined QKV projection with LXT ε-rule
    qkv = lf.linear_epsilon(query, self.in_proj_weight, self.in_proj_bias)    # (B, N, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)                   # each (B, N, D)

    # Reshape to multi-head format: (B, H, N, head_dim)
    head_dim = D // self.num_heads
    q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)
    k = k.view(B, N, self.num_heads, head_dim).transpose(1, 2)
    v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)

    scale = head_dim ** -0.5
    q = q * scale

    if _CONSERVATIVE:
        q = q.detach()
        k = k.detach()

    # (2-4) LXT AttnLRP attention
    dots = lf.matmul(q, k.transpose(-2, -1))
    attn = lf.softmax(dots, dim=-1)
    out  = lf.matmul(attn, v)

    out = out.transpose(1, 2).reshape(B, N, D)
    return self.out_proj(out), None   # out_proj already patched


# ─────────────────────────────────────────────────────────────────────────────
# TransformerEncoderLayer LRP forward (lf.add2 residuals + inline GELU rule)
# ─────────────────────────────────────────────────────────────────────────────

def _encoder_layer_lrp_forward(self, src, src_mask=None,
                                src_key_padding_mask=None, is_causal=False):
    """TransformerEncoderLayer.forward."""
    x = src
    # Residual 1: criss-cross attention (_sa_block calls already-patched MHAs)
    x = lf.add2(x, self._sa_block(self.norm1(x), src_mask, src_key_padding_mask,
                                is_causal=is_causal))
    # Residual 2: FFN with GELU identity rule inlined
    normed = self.norm2(x)
    ffn    = self.linear2(self.dropout(GELUIdentityFn.apply(self.linear1(normed))))
    x      = lf.add2(x, self.dropout2(ffn))
    return x


def _patch_embedding_lrp_forward(self, x, mask=None):
    import torch
    bz, ch_num, patch_num, patch_size = x.shape

    if mask is None:
        mask_x = x
    else:
        mask_x = x.clone()
        mask_x[mask == 1] = self.mask_encoding

    # ── Time-domain branch (Conv2d × 3) — main LRP signal path ───────────
    flat = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
    patch_emb = self.proj_in(flat)                                          # patched Conv2d+GroupNorm+GELU
    patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(
        bz, ch_num, patch_num, self.d_model
    )

    # ── Spectral branch - has no rule ──────────
    with torch.no_grad():
        flat2    = flat.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(flat2, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)                         # Linear+Dropout
    patch_emb = patch_emb + spectral_emb.detach()

    # ── Positional encoding — detached to act as fixed bias ──────────────
    pos_emb   = self.positional_encoding(patch_emb.detach().permute(0, 3, 1, 2))
    pos_emb   = pos_emb.permute(0, 2, 3, 1)
    patch_emb = patch_emb + pos_emb

    return patch_emb


# ─────────────────────────────────────────────────────────────────────────────
# Model (top-level wrapper) LRP forward
# ─────────────────────────────────────────────────────────────────────────────

def _model_lrp_forward(self, x):
    """
    Drop-in for Model.forward.

    The binary (.view(bz)) reshape is removed, keeping output shape (B, 1) so
    run_backward's out.shape[1] check works for binary benchmarks.
    """
    bz, ch_num, t = x.shape
    x    = x.reshape(bz, ch_num, self.len_seconds, -1)
    feat = self.backbone(x)
    feat = feat.contiguous().view(bz, ch_num * self.len_seconds * 200)
    out  = self.classifier(feat)                                            # (B, num_classes) or (B, 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_cbramod(model, conservative=False):
    """Apply all AttnLRP patches to a CBraMod Model instance."""
    global _CONSERVATIVE
    _CONSERVATIVE = conservative

    from models.CBraMod.criss_cross_transformer import TransformerEncoderLayer
    from models.CBraMod.cbramod import PatchEmbedding

    walk_and_patch(model, [
        (TransformerEncoderLayer, _encoder_layer_lrp_forward),
        (PatchEmbedding,          _patch_embedding_lrp_forward),
        (nn.MultiheadAttention,   _mha_lrp_forward),
        (nn.LayerNorm,            layernorm_lrp_forward),
        (nn.GroupNorm,            groupnorm_lrp_forward),
        (nn.GELU,                 gelu_identity_forward),
        (nn.ELU,                  elu_identity_forward),
        (nn.Linear,               linear_lrp_forward),
        (nn.Conv2d,               conv2d_epsilon_forward),
    ], debug_label="patch_cbramod")

    save_and_replace(model, _model_lrp_forward)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, conservative=False, **_):
    """Compute LRP relevances for CBraMod. Returns (N, C, A).  ``channels`` ignored."""
    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()

        with torch.no_grad():
            preds_b, conf_b = extract_preds_and_confidence(unwrap_output(model(x.detach())))
            predictions.append(preds_b)
            confidences.append(conf_b)

        target = (torch.tensor(y_test[start : start + batch_size],
                               dtype=torch.long, device=x.device)
                  if y_test is not None else None)

        rel = compute_lrp(model, x, patch_cbramod, unpatch_all,
                          target_class=target, conservative=conservative)
        model.float()
        relevances.append(to_patches_3d(rel))

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
