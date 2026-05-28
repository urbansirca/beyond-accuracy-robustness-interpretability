import torch


def run(model, X_batch, ch_names, channels):
    """One forward batch under an active `attention_context`."""
    from models.NeuroRVQm.modules import (
        patch_size, n_patches, create_embedding_ix, ch_names_global,
    )
    ch_mask         = channels['ch_mask']
    ch_names_masked = channels['ch_names_masked']
    n_time          = X_batch.shape[2] // 200

    temp_embed_ix, spat_embed_ix = create_embedding_ix(
        n_time, n_patches, ch_names_masked, ch_names_global
    )
    temp_embed_ix = temp_embed_ix.cuda()
    spat_embed_ix = spat_embed_ix.cuda()

    x_raw = torch.from_numpy(X_batch).float().cuda()[:, ch_mask, :]
    b, c, _ = x_raw.shape
    x = x_raw.reshape(b, c, n_time, patch_size)
    with torch.no_grad():
        model(x, temp_embed_ix.expand(b, -1), spat_embed_ix.expand(b, -1))
