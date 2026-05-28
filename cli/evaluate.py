import argparse
import sys

from cli._config import load_config
from experiments.runner import run_experiments


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate EEG-FM checkpoints under perturbations.")
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (configs/eval/*.yaml).")
    p.add_argument("--models", nargs="+",
                   help="Override config's models list.")
    p.add_argument("--benchmarks", nargs="+",
                   help="Override config's benchmarks list.")
    p.add_argument("--output-dir",
                   help="Override config's output_dir.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute fold results even if a CSV row already exists.")
    p.add_argument("--evaluate-on", choices=["best", "last"],
                   help="Which per-fold checkpoint to load.")
    p.add_argument("--finetune", choices=["head_only", "full"],
                   help="Restrict to one finetune mode (overrides finetune_modes).")
    p.add_argument("--fold-filter", nargs="+", type=int,
                   help="Only evaluate these fold indices (0-based).")
    p.add_argument("--test-aug",
                   help="Override config's augmentations with a single clean-trained "
                        "(train=null) pair using this test augmentation. Pass "
                        "'clean'/'null'/'none' for the clean test set.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.get("phase") != "evaluate":
        sys.exit(f"Expected phase=evaluate in {args.config}, got phase={cfg.get('phase')!r}")

    if args.models:       cfg["models"] = args.models
    if args.benchmarks:   cfg["benchmarks"] = args.benchmarks
    if args.output_dir:   cfg["output_dir"] = args.output_dir
    if args.overwrite:    cfg["overwrite"] = True
    if args.evaluate_on:  cfg["evaluate_on"] = args.evaluate_on
    if args.finetune:     cfg["finetune_modes"] = [args.finetune]
    if args.fold_filter:  cfg["fold_filter"] = args.fold_filter
    if args.test_aug:
        test = None if args.test_aug.lower() in {"clean", "null", "none"} else args.test_aug
        cfg["augmentations"] = [{"train": None, "test": test}]

    run_experiments(cfg)


if __name__ == "__main__":
    main()
