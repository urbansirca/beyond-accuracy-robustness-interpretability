"""CBraMod Gradient × Input."""

from interpretability.ixg.captum import run_ixg_generic


def run(model, X_test, batch_size, ch_names, channels, *, y_test=None, **_):
    """Gradient × Input for CBraMod. Returns (N, C, A).  ``channels`` ignored."""
    return run_ixg_generic(model, X_test, batch_size, y_test)
