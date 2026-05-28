from __future__ import annotations

import numpy as np
import torch.nn as nn

from models.LaBraM.utils import get_input_chans


# Model capability flags consulted by the CLI before kicking off a run.
CLS_POOLING_MODELS = {"LaBraM", "NeuroRVQ"}            # have a meaningful CLS token
LARGE_HEAD_MODELS  = {"LaBraM", "NeuroRVQ", "REVE"}    # support large_head variant

# Models whose residual add happens outside the hooked submodule
EXTERNAL_RESIDUAL_MODELS = {"REVE"}


def get_callable_model(wrapper) -> nn.Module:
    """Unwrap a benchmarking wrapper down to the nn.Module we can call directly."""
    m = wrapper.model
    if isinstance(m, nn.Module):
        return m
    if hasattr(m, "model") and isinstance(m.model, nn.Module):
        return m.model
    raise ValueError(f"Cannot extract nn.Module from {type(m)}")


def get_forward_fn(model_name: str, nn_model: nn.Module, ch_names, n_times: int | None = None):
    """Return a function that takes input and calls the model's forward pass."""
    if model_name == "LaBraM":
        input_chans, ch_mask = get_input_chans(ch_names)
        patch_size = nn_model.patch_size  # 200

        def labram_forward(x):
            x = x[:, ch_mask, :]
            x = x / 100.0
            B, N, total_t = x.shape
            x = x.reshape(B, N, total_t // patch_size, patch_size)
            x = x.to(next(nn_model.parameters()).device)
            nn_model(x, input_chans=input_chans)

        return labram_forward

    if model_name == "NeuroRVQ":
        from models.NeuroRVQm.modules import (
            create_embedding_ix, ch_names_global,
            patch_size as neuro_patch_size, n_patches as neuro_n_patches,
        )
        ch_names_enc = np.array([c.lower().encode() for c in ch_names])
        ch_mask      = np.isin(ch_names_enc, ch_names_global)
        ch_names_msk = ch_names_enc[ch_mask]
        n_time       = n_times // neuro_patch_size
        temp_ix, spat_ix = create_embedding_ix(n_time, neuro_n_patches, ch_names_msk, ch_names_global)

        def neurorvq_forward(x):
            device = next(nn_model.parameters()).device
            x = x[:, ch_mask, :]
            B, C, T = x.shape
            x = x.reshape(B, C, n_time, neuro_patch_size)
            x = x.to(device)
            t_ix = temp_ix.expand(B, -1).to(device)
            s_ix = spat_ix.expand(B, -1).to(device)
            nn_model(x, t_ix, s_ix)

        return neurorvq_forward

    def default_forward(x):
        x = x.to(next(nn_model.parameters()).device)
        nn_model(x)

    return default_forward


def _find_transformer_blocks(model: nn.Module):
    """Walk named_modules and return (name, module)."""
    CONTAINER_CLASSES = {"TransformerEncoder", "TransformerDecoder"}
    BLOCK_PATTERNS = ("TransformerEncoderLayer", "TransformerDecoderLayer",
                      "TransformerBlock", "EncoderLayer", "Block")

    blocks = []
    seen_ids = set()
    for name, mod in model.named_modules():
        if id(mod) in seen_ids:
            continue
        cls = type(mod).__name__
        if cls in CONTAINER_CLASSES:
            continue
        if any(kw in cls for kw in BLOCK_PATTERNS):
            blocks.append((name, mod))
            seen_ids.add(id(mod))
    return blocks


def get_blocks(model_name: str, nn_model: nn.Module):
    """Return (name, module) pairs for the transformer blocks to probe."""
    if model_name == "BrainOmni":
        return [(f"backbone.blocks.{i}", m)
                for i, m in enumerate(nn_model.backbone.blocks)]
    if model_name == "REVE":
        
        return [(f"backbone.transformer.layers.{i}.ff", layer[1])
                for i, layer in enumerate(nn_model.backbone.transformer.layers)]
    blocks = _find_transformer_blocks(nn_model)
    return blocks