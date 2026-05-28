"""NeuroRVQ Gradient × Input (4-branch tokeniser, masked channels)."""

import numpy as np
import torch

from interpretability.ixg.captum import compute_grad_x_input
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output


def _build_neurorvq_inputs(X_test, batch_size, ch_mask, ch_names_masked, n_time):
    """Shared batch prep for NeuroRVQ. Yields (x_4d, t_ix, s_ix, start_idx)."""
    from models.NeuroRVQm.modules import (
        patch_size, n_patches, create_embedding_ix, ch_names_global,
    )

    temp_embed_ix, spat_embed_ix = create_embedding_ix(
        n_time, n_patches, ch_names_masked, ch_names_global
    )
    temp_embed_ix = temp_embed_ix.cuda()
    spat_embed_ix = spat_embed_ix.cuda()

    for start in range(0, len(X_test), batch_size):
        x_raw = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()
        x_raw = x_raw[:, ch_mask, :]
        b, c, _ = x_raw.shape
        x = x_raw.reshape(b, c, n_time, patch_size)
        t_ix = temp_embed_ix.expand(b, -1)
        s_ix = spat_embed_ix.expand(b, -1)
        yield x, t_ix, s_ix, start


def run(model, X_test, batch_size, ch_names, channels, *, y_test=None, **_):
    """Gradient × Input attribution for NeuroRVQ. Returns (N, C_masked, A)."""
    ch_mask         = channels['ch_mask']
    ch_names_masked = channels['ch_names_masked']
    n_time          = X_test.shape[2] // 200
    relevances, predictions, confidences = [], [], []
    for x, t_ix, s_ix, start in _build_neurorvq_inputs(
        X_test, batch_size, ch_mask, ch_names_masked, n_time
    ):
        with torch.no_grad():
            out = unwrap_output(model(x.detach(), t_ix, s_ix))
            preds_b, conf_b = extract_preds_and_confidence(out)
            predictions.append(preds_b)
            confidences.append(conf_b)

        target = (torch.tensor(y_test[start : start + batch_size],
                               dtype=torch.long, device=x.device)
                  if y_test is not None else None)

        rel = compute_grad_x_input(
            model, x, target_class=target,
            temporal_embedding_ix=t_ix, spatial_embedding_ix=s_ix,
        )
        relevances.append(np.abs(rel).mean(axis=-1))

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
