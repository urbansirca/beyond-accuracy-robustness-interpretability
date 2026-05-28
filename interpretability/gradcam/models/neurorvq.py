"""NeuroRVQ GradCAM (branch-averaged; manual hook in cam.py)."""

import numpy as np
import torch

from interpretability.gradcam.cam import compute_gradcam
from interpretability.common.predict import extract_preds_and_confidence, unwrap_output



def run(model, X_test, batch_size, ch_names, channels,
        *, y_test=None, target_layer=-1, **_):
    """GradCAM for NeuroRVQ (branch-averaged)."""
    from models.NeuroRVQm.modules import (
        patch_size, n_patches, create_embedding_ix, ch_names_global,
    )

    ch_mask         = channels['ch_mask']
    ch_names_masked = channels['ch_names_masked']
    n_time          = X_test.shape[2] // 200
    n_channels      = int(ch_mask.sum())

    temp_embed_ix, spat_embed_ix = create_embedding_ix(
        n_time, n_patches, ch_names_masked, ch_names_global
    )
    temp_embed_ix = temp_embed_ix.cuda()
    spat_embed_ix = spat_embed_ix.cuda()

    relevances, predictions, confidences = [], [], []
    for start in range(0, len(X_test), batch_size):
        x_raw = torch.from_numpy(X_test[start : start + batch_size]).float().cuda()
        x_raw = x_raw[:, ch_mask, :]
        n, c, t = x_raw.shape
        x = x_raw.reshape(n, c, n_time, patch_size)

        b = x.shape[0]
        t_ix = temp_embed_ix.expand(b, -1)
        s_ix = spat_embed_ix.expand(b, -1)

        with torch.no_grad():
            out = unwrap_output(model(x.detach(), t_ix, s_ix))
            preds_batch, conf_batch = extract_preds_and_confidence(out)
            predictions.append(preds_batch)
            confidences.append(conf_batch)

        target = (
            torch.tensor(y_test[start : start + batch_size], dtype=torch.long, device=x.device)
            if y_test is not None else None
        )

        rel = compute_gradcam(
            "NeuroRVQ", model, x,
            n_channels=n_channels,
            target_class=target,
            target_layer=target_layer,
            return_time_patches=True,
            temporal_embedding_ix=t_ix,
            spatial_embedding_ix=s_ix,
        )
        relevances.append(rel)

    return np.concatenate(relevances), np.concatenate(predictions), np.concatenate(confidences)
