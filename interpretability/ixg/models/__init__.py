"""Per-model IxG code: input prep + batch loop, one file per model.

Each module exposes ``run(model, X_test, batch_size, ch_names, channels, *,
y_test=None, **_) -> (rel, preds, conf)``.
"""
