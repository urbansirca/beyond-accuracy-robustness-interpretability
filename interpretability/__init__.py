"""
Interpretability for EEG foundation models.

Structure
---------
    common/     model-agnostic helpers (data, checkpoints, predict, numeric,
                csv, plots) — composed by the method orchestrators.
    plotting/   matplotlib topomap / attention figure functions.

    lrp/        run_analysis + backward + patching + rules/ + models/
    ixg/        run_analysis + captum + models/
    gradcam/    run_analysis + cam + models/
    attention/  run_analysis + capture + models/
    probing/    linear-probing workflow (separate CLI: cli/probe.py)

Each method is a self-contained top-level package: a ``run_analysis``
orchestrator, the shared compute for that method, and one ``models/<name>.py``
per supported model. ``lrp/rules/`` holds the LRP autograd primitives (only LRP uses them).

Run a method:
    from interpretability.lrp import run_analysis
    run_analysis(data_root, "LaBraM", "KU MI", fold=-1)

Add a model to a method:
    create ``interpretability/<method>/models/<name>.py`` exposing ``run(...)``
    and add the model name to that method's ``_SUPPORTED`` set.
"""
