"""
    python -m cli.interpret --config configs/interpret/lrp.yaml
"""

import argparse
import os
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cli._config import load_config


DATA_ROOT          = "/projects/prjs1721/benchmarks"
DEFAULT_MODELS     = ["LaBraM", "REVE", "NeuroRVQ", "CBraMod", "BrainOmni", "EEGNet"]
DEFAULT_BENCHMARKS = ["Physionet Eyes", "KU MI", "High Gamma", "KU ERP"]

# Models that use the flatten/concat head variant by default.  Override per-run
# with `--large_head` or `--no-large-head`.
_LARGE_HEAD_DEFAULTS = {"REVE"}

# Per-method hardcoded defaults — applied first, then the YAML config (if any),
_COMMON_DEFAULTS = {
    "data_root": DATA_ROOT,
    "models": DEFAULT_MODELS,
    "benchmarks": DEFAULT_BENCHMARKS,
    "fold": -1,
    "large_head": None,
    "overwrite": False,
    "out_root": "results/attribution",
}

_METHOD_DEFAULTS = {
    "lrp": {
        **_COMMON_DEFAULTS,
        "augmentations": [None],
        "confidence_quantile": 0.75,
        "target": "gt",
        "conservative": False,
    },
    "ixg": {
        **_COMMON_DEFAULTS,
        "augmentations": [None],
        "confidence_quantile": 0.75,
        "target": "gt",
        "skip_brainomni_tokenizer": False,
    },
    "gradcam": {
        **_COMMON_DEFAULTS,
        "confidence_quantile": 0.75,
        "target": "gt",
        "target_layer": -1,
    },
    "attention": {
        **_COMMON_DEFAULTS,
        "augmentations": [None],
        "train_head_only": False,
    },
}


def _resolve_large_head(model_name: str, override) -> bool:
    """Override is None (use default), True, or False."""
    if override is None:
        return model_name in _LARGE_HEAD_DEFAULTS
    return override


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    """Single flat parser. All defaults are ``argparse.SUPPRESS`` so unparsed
    args don't land in the namespace — ``_resolve_cfg`` then treats anything
    present on ``args`` as an explicit override that wins over the YAML.
    """
    S = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        prog="interpret",
        description="Run an attribution method across models × benchmarks × folds.",
    )
    p.add_argument("--config", default=None,
                   help="Path to a YAML config (e.g. configs/interpret/lrp.yaml). "
                        "Its `method:` field selects the analysis. CLI flags override config values.")
    p.add_argument("--method", default=S, choices=list(_METHOD_DEFAULTS),
                   help="Override the method named in the YAML (or supply it when no --config is given).")

    # Common
    p.add_argument("--data_root", default=S)
    p.add_argument("--models",     nargs="+", default=S)
    p.add_argument("--benchmarks", nargs="+", default=S)
    p.add_argument("--fold", type=int, default=S,
                   help="Fold index, or -1 to iterate all 10 folds (default).")
    p.add_argument("--large_head", action=argparse.BooleanOptionalAction, default=S,
                   help="Force the flatten/concat large-head variant on/off. "
                        "Default: on for REVE, off for everyone else.")
    p.add_argument("--overwrite", action="store_true", default=S,
                   help="Recompute and overwrite existing CSV files.")
    p.add_argument("--out_root", default=S,
                   help="Root directory for attribution CSV outputs "
                        "(default: results/attribution). The method subdir "
                        "(e.g. input_x_gradient/) is appended automatically.")

    # Method-specific (flat, ignored when the chosen method doesn't use them)
    p.add_argument("--augmentations", nargs="+", default=S,
                   help="One or more augmentation names (e.g. white_noise_0db). "
                        "Used by lrp/ixg/attention.")
    p.add_argument("--confidence_quantile", type=float, default=S,
                   help="High-confidence threshold quantile. Used by lrp/ixg/gradcam.")
    p.add_argument("--target", default=S, choices=["gt", "predicted"],
                   help="Target class for the backward seed. Used by lrp/ixg/gradcam.")
    p.add_argument("--conservative", action="store_true", default=S,
                   help="CP-LRP: detach Q,K in attention for sharper maps. Used by lrp.")
    p.add_argument("--target_layer", type=int, default=S,
                   help="Transformer block index to hook. Used by gradcam.")
    p.add_argument("--train_head_only", action="store_true", default=S,
                   help="Load checkpoints trained with head-only fine-tuning. Used by attention.")
    p.add_argument("--skip_brainomni_tokenizer", action=argparse.BooleanOptionalAction, default=S,
                   help="BrainOmni only: bypass the VQ codebook lookup in the IxG forward "
                        "so gradients flow through F.normalize directly (no straight-through).")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Config resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_cfg(args) -> SimpleNamespace:
    """Merge hardcoded per-method defaults < YAML config < explicit CLI args."""
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}

    yaml_cfg = {}
    cfg_path = getattr(args, "config", None)
    if cfg_path is not None:
        yaml_cfg = load_config(cfg_path)

    cli_method = cli_overrides.pop("method", None)
    yaml_method = yaml_cfg.pop("method", None)
    method = cli_method or yaml_method
    if method is None:
        sys.exit("error: no method specified — pass --method or set `method:` "
                 "in the YAML config.")
    if method not in _METHOD_DEFAULTS:
        sys.exit(f"error: unknown method {method!r}. "
                 f"Choices: {list(_METHOD_DEFAULTS)}.")

    cfg = dict(_METHOD_DEFAULTS[method])
    cfg.update(yaml_cfg)
    cfg.update(cli_overrides)

    # YAML stores `null` augmentations as Python None; argparse may produce the
    # string "None" if the user typed it. Normalise both to None.
    if "augmentations" in cfg and cfg["augmentations"] is not None:
        cfg["augmentations"] = [None if a in (None, "null", "None") else a
                                for a in cfg["augmentations"]]

    return SimpleNamespace(method=method, **cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _iter_runs(args):
    """Yield (model_name, benchmark_name, augmentation) tuples."""
    augmentations = getattr(args, "augmentations", [None])
    for model_name in args.models:
        for benchmark_name in args.benchmarks:
            for augmentation in augmentations:
                yield model_name, benchmark_name, augmentation


def _run_lrp(args):
    from interpretability.lrp import run_analysis
    for model_name, benchmark_name, augmentation in _iter_runs(args):
        _banner("LRP", model_name, benchmark_name, augmentation)
        try:
            run_analysis(
                args.data_root, model_name, benchmark_name, args.fold,
                lrp_target=args.target, conservative=args.conservative,
                confidence_quantile=args.confidence_quantile,
                large_head=_resolve_large_head(model_name, args.large_head), overwrite=args.overwrite,
                augmentation=augmentation,
                out_root=args.out_root,
            )
        except Exception:
            traceback.print_exc()


def _run_ixg(args):
    from interpretability.ixg import run_analysis
    for model_name, benchmark_name, augmentation in _iter_runs(args):
        _banner("IxG", model_name, benchmark_name, augmentation)
        try:
            run_analysis(
                args.data_root, model_name, benchmark_name, args.fold,
                target=args.target, confidence_quantile=args.confidence_quantile,
                large_head=_resolve_large_head(model_name, args.large_head), overwrite=args.overwrite,
                augmentation=augmentation,
                skip_brainomni_tokenizer=getattr(args, "skip_brainomni_tokenizer", False),
                out_root=args.out_root,
            )
        except Exception:
            traceback.print_exc()


def _run_gradcam(args):
    from interpretability.gradcam import run_analysis
    # gradcam doesn't take augmentation today; iterate without it
    for model_name in args.models:
        for benchmark_name in args.benchmarks:
            _banner("GradCAM", model_name, benchmark_name, None)
            try:
                run_analysis(
                    args.data_root, model_name, benchmark_name, args.fold,
                    gradcam_target=args.target, target_layer=args.target_layer,
                    confidence_quantile=args.confidence_quantile,
                    large_head=_resolve_large_head(model_name, args.large_head), overwrite=args.overwrite,
                    out_root=args.out_root,
                )
            except Exception:
                traceback.print_exc()


def _run_attention(args):
    from interpretability.attention import run_analysis
    for model_name, benchmark_name, augmentation in _iter_runs(args):
        _banner("Attention", model_name, benchmark_name, augmentation)
        try:
            run_analysis(args.out_root,
                args.data_root, model_name, benchmark_name, args.fold,
                large_head=_resolve_large_head(model_name, args.large_head),
                train_head_only=args.train_head_only,
                overwrite=args.overwrite,
                augmentation=augmentation,
            )
        except Exception:
            traceback.print_exc()


def _banner(method_name, model_name, benchmark_name, augmentation):
    aug = f" ({augmentation})" if augmentation else ""
    print(f"\n{'='*60}")
    print(f"{method_name}: {model_name} on {benchmark_name}{aug}")
    print(f"{'='*60}")


_DISPATCH = {
    "lrp":       _run_lrp,
    "ixg":       _run_ixg,
    "gradcam":   _run_gradcam,
    "attention": _run_attention,
}


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = _resolve_cfg(args)
    _DISPATCH[cfg.method](cfg)


if __name__ == "__main__":
    main()
