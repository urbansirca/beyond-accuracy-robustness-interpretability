"""Per-model GradCAM code: batch loop, one file per model.

Each module exposes ``run(model, X_test, batch_size, ch_names, channels, *,
y_test=None, target_layer=-1, **_) -> (rel, preds, conf)``.
"""
