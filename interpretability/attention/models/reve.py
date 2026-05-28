import torch


def run(model, X_batch, ch_names, channels):
    """One forward batch under an active `attention_context`.  ``channels`` ignored."""
    x = torch.from_numpy(X_batch).float().cuda()
    with torch.no_grad():
        model(x)
