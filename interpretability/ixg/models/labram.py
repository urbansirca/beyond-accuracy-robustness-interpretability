"""LaBraM Gradient × Input (channel-mask + scale + patch rearrange)."""

import numpy as np
import torch
from einops import rearrange

from interpretability.ixg.captum import run_ixg_generic


def run(model, X_test, batch_size, ch_names, channels, *, y_test=None, **_):
    """Gradient × Input for LaBraM. Input reshaped to (B, N, A, T=200).

    Returns (N, C_masked, A) attribution averaged per time patch.
    """
    input_chans = channels['input_chans']
    ch_mask     = channels['ch_mask']

    def preprocess(x_np):
        x = torch.from_numpy(x_np).float().cuda()
        x = x[:, ch_mask, :] / 100  # same scaling as engine_for_finetuning
        return rearrange(x, "B N (A T) -> B N A T", T=200)

    def postprocess(rel):
        # rel: (B, N, A, 200) → (B, N, A) by mean over patch_size of |rel|
        return np.abs(rel).mean(axis=-1)

    return run_ixg_generic(
        model, X_test, batch_size, y_test,
        preprocess=preprocess,
        forward_kwargs_fn=lambda x: {"input_chans": input_chans},
        postprocess=postprocess,
    )
