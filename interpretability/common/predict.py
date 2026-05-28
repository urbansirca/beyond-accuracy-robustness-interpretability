"""Prediction / output unwrapping helpers shared across methods."""

import torch


def extract_preds_and_confidence(out):
    """Return (predictions, confidences) from raw logits.

    Binary (single-logit or 1-D) models: sigmoid threshold at 0.5 for class
    label, confidence is the probability of the predicted class.
    Multi-class: argmax for label, max softmax probability for confidence.
    """
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.shape[-1] == 1:
        prob = torch.sigmoid(out.float().squeeze(-1))          # (B,)
        preds = (prob >= 0.5).long().cpu().numpy()             # 0 or 1
        conf = torch.where(prob >= 0.5, prob, 1 - prob).cpu().numpy()
    else:
        probs = torch.softmax(out.float(), dim=-1)             # (B, C)
        conf, preds = probs.max(dim=-1)
        preds = preds.cpu().numpy()
        conf = conf.cpu().numpy()
    return preds, conf


def unwrap_output(out):
    """Unwrap dict/tuple model outputs to raw logit tensor."""
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    elif isinstance(out, (list, tuple)):
        out = out[0]
    return out
