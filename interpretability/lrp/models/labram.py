import numpy as np
import torch
import torch.nn as nn

import lxt.explicit.functional as lf
from einops import rearrange

from interpretability.lrp.backward import compute_lrp
from interpretability.lrp.patching import unpatch_all, walk_and_patch
from interpretability.lrp.rules import (
    scale_epsilon,
    groupnorm_lrp_forward, gelu_identity_forward,
    conv2d_epsilon_forward,
    linear_lrp_forward, layernorm_lrp_forward,
)
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output

_CONSERVATIVE = False

# ─────────────────────────────────────────────────────────────────────────────
# LaBraM Attention forward
# ─────────────────────────────────────────────────────────────────────────────

def _attention_lrp_forward(
    self, x, rel_pos_bias=None, return_attention=False, return_qkv=False
):
    """Drop-in for LaBraM Attention.forward with AttnLRP rules."""
    B, N, C = x.shape

    if self.q_bias is not None:
        qkv_bias = torch.cat(
            (
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            )
        )
        qkv = lf.linear_epsilon(x, self.qkv.weight, qkv_bias)
    else:
        qkv = lf.linear_epsilon(x, self.qkv.weight)

    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if self.q_norm is not None:
        q = self.q_norm(q).type_as(v)
    if self.k_norm is not None:
        k = self.k_norm(k).type_as(v)

    q = q * self.scale

    if _CONSERVATIVE:
        q = q.detach()
        k = k.detach()

    # (1) LXT bilinear matmul for Q·K^T
    attn = lf.matmul(q, k.transpose(-2, -1))

    if self.relative_position_bias_table is not None:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

    if rel_pos_bias is not None:
        attn = attn + rel_pos_bias

    # (2) LXT softmax rule
    attn = lf.softmax(attn, dim=-1)
    attn = self.attn_drop(attn)

    if return_attention:
        return attn

    # (3) LXT bilinear matmul for A·V
    x = lf.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)

    # Projection layers — patched to _linear_lrp_forward
    x = self.proj(x)
    x = self.proj_drop(x)

    if return_qkv:
        return x, qkv
    return x


# ─────────────────────────────────────────────────────────────────────────────
# LaBraM Block forward
# ─────────────────────────────────────────────────────────────────────────────

def _block_lrp_forward(
    self, x, rel_pos_bias=None, return_attention=False, return_qkv=False
):
    """Drop-in for LaBraM Block.forward with LXT add2 residuals."""
    if return_attention:
        return self.attn(
            self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True
        )

    attn_out = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
    if self.gamma_1 is not None:
        attn_out = scale_epsilon(attn_out, self.gamma_1)

    if return_qkv:
        attn_out, qkv = self.attn(
            self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=True
        )
        if self.gamma_1 is not None:
            attn_out = scale_epsilon(attn_out, self.gamma_1)
        x = lf.add2(x, self.drop_path1(attn_out))
        mlp_out = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            mlp_out = scale_epsilon(mlp_out, self.gamma_2)
        x = lf.add2(x, self.drop_path2(mlp_out))
        return x, qkv

    x = lf.add2(x, self.drop_path1(attn_out))

    mlp_out = self.mlp(self.norm2(x))
    if self.gamma_2 is not None:
        mlp_out = scale_epsilon(mlp_out, self.gamma_2)
    x = lf.add2(x, self.drop_path2(mlp_out))

    return x


# ─────────────────────────────────────────────────────────────────────────────
# NeuralTransformer top-level forward (finetuning path)
# ─────────────────────────────────────────────────────────────────────────────

def _neural_transformer_lrp_forward(self, x, input_chans=None,
                                     return_patch_tokens=False,
                                     return_all_tokens=False, **kwargs):
    """
    Drop-in for NeuralTransformer.forward (finetuning path).

    Changes vs original:
      - pos_embed and time_embed detached
      - x[:, 1:, :] += time_embed replaced with non-in-place addition
      - t.mean(1) → lf.mean(t, dim=1)  (LXT ε-rule)
    """
    batch_size, n, a, t = x.shape
    input_time_window = a if t == self.patch_size else t
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
    if self.pos_embed is not None:
        pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(
            batch_size, -1, input_time_window, -1).flatten(1, 2)
        pos_embed = torch.cat(
            (pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1
        ).detach()
        x = x + pos_embed

    if self.time_embed is not None:
        nc = n if t == self.patch_size else a
        time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
            batch_size, nc, -1, -1).flatten(1, 2).detach()   # fixed temporal bias
        time_pad = torch.zeros(batch_size, 1, time_embed.shape[-1],
                               dtype=time_embed.dtype, device=time_embed.device)
        x = x + torch.cat([time_pad, time_embed], dim=1)

    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x, rel_pos_bias=None)

    x = self.norm(x)
    if self.fc_norm is not None:
        t_tokens = x[:, 1:, :]
        x = self.fc_norm(lf.mean(t_tokens, dim=1, keep_dim=False))   # LXT ε-rule mean
    else:
        x = x[:, 0]   # cls-token classification path

    return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_labram(model, conservative=False):
    """Apply all AttnLRP patches to a LaBraM NeuralTransformer."""
    global _CONSERVATIVE
    _CONSERVATIVE = conservative

    from models.LaBraM.modeling_finetune import Attention, Block, NeuralTransformer

    walk_and_patch(model, [
        (NeuralTransformer, _neural_transformer_lrp_forward),
        (Attention,         _attention_lrp_forward),
        (Block,             _block_lrp_forward),
        (nn.LayerNorm,      layernorm_lrp_forward),
        (nn.GroupNorm,      groupnorm_lrp_forward),
        (nn.GELU,           gelu_identity_forward),
        (nn.Linear,         linear_lrp_forward),
        (nn.Conv2d,         conv2d_epsilon_forward),
    ], debug_label="patch_labram")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, conservative=False, **_):
    """Compute LRP relevances for LaBraM.

    Pulls ``input_chans`` and ``ch_mask`` from ``channels``.

    Returns
    -------
    relevances  : np.ndarray (N, C_masked, A) — |relevance| averaged per time patch
    predictions : np.ndarray (N,)
    confidences : np.ndarray (N,) — max softmax/sigmoid probability per sample
    """
    input_chans = channels['input_chans']
    ch_mask     = channels['ch_mask']
    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()
        x = x[:, ch_mask, :]  # keep only channels LaBraM knows
        x = x / 100  # same scaling as engine_for_finetuning
        x = rearrange(x, "B N (A T) -> B N A T", T=200)  # split into patches
        n_time_patches = x.shape[2]  # A

        with torch.no_grad():
            out = unwrap_output(model(x.detach(), input_chans=input_chans))
            preds_batch, conf_batch = extract_preds_and_confidence(out)
            predictions.append(preds_batch)
            confidences.append(conf_batch)

        if y_test is not None:
            target = torch.tensor(
                y_test[start : start + batch_size], dtype=torch.long, device=x.device
            )
        else:
            target = None

        rel = compute_lrp(model, x, patch_labram, unpatch_all,
                          target_class=target, conservative=conservative,
                          input_chans=input_chans)
        model.float()
        rel = rel.reshape(rel.shape[0], rel.shape[1], -1)  # (B, C_masked, T_total)
        rel_3d = np.abs(rel).reshape(rel.shape[0], rel.shape[1], n_time_patches, -1).mean(axis=-1)
        relevances.append(rel_3d)

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
