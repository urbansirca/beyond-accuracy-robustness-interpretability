"""Shared, model-agnostic helpers used by the method packages.
  ``data``        — benchmark loading, channel resolution, fold splitting.
  ``checkpoints`` — checkpoint paths, batch sizes, fold-model loading, unwrap.
  ``predict``     — logits → (preds, confidence); output unwrapping.
  ``numeric``     — relevance normalisation + time-patch reduction.
  ``csv``         — attribution + attention CSV export.
  ``plots``       — the topomap cascade shared by the attribution methods.
"""
