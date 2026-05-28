import torch
from einops import rearrange


def run(model, X_batch, ch_names, channels):
    """One forward batch under an active `attention_context`."""
    input_chans = channels['input_chans']
    ch_mask     = channels['ch_mask']
    x = torch.from_numpy(X_batch[:, ch_mask, :]).float().cuda()
    x = rearrange(x, "B N (A T) -> B N A T", T=200)
    with torch.no_grad():
        model(x, input_chans=input_chans)
