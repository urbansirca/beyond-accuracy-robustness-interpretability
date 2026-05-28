"""Numeric helpers shared by the attribution methods (LRP / IxG / GradCAM)."""

import numpy as np
import torch


def normalize_p99(relevance):
    """Per-sample L∞-style normalisation using the 99th percentile of |rel|.

    Used by both the LRP backward pass and Gradient × Input.  
    Percentile (vs. absolute max) avoids outlier positions.
    """
    abs_rel = relevance.abs()
    flat = abs_rel.flatten(1)                       # (B, -1)
    p99 = torch.quantile(flat, 0.99, dim=1)         # (B,)
    for _ in range(1, relevance.dim()):
        p99 = p99.unsqueeze(-1)
    p99 = p99.clamp(min=1e-30)
    relevance = (relevance / p99).clamp(-1.0, 1.0)
    return relevance


def to_patches_3d(rel, patch_size=200):
    """(B, C, T) → (B, C, A) by averaging |rel| within `patch_size`-sample patches.

    Truncates trailing samples that don't fill a full patch.  Falls back to a
    single-patch mean if T < patch_size.
    """
    B, C, T = rel.shape
    n_patches = max(1, T // patch_size)
    end = n_patches * patch_size if T >= patch_size else T
    return np.abs(rel[:, :, :end]).reshape(B, C, n_patches, -1).mean(axis=-1)
