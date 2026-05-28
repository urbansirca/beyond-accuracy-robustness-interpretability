import argparse
import sys

from cli._config import load_config
from experiments.runner import run_experiments


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train EEG-FM models on benchmarks (cross-validated).")
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (configs/train/*.yaml).")
    p.add_argument("--models", nargs="+",
                   help="Override config's models list.")
    p.add_argument("--benchmarks", nargs="+",
                   help="Override config's benchmarks list.")
    p.add_argument("--output-dir",
                   help="Override config's output_dir.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing checkpoints.")
    p.add_argument("--finetune", choices=["head_only", "full"],
                   help="Restrict to one finetune mode (overrides finetune_modes).")
    p.add_argument("--fold-filter", nargs="+", type=int,
                   help="Only run these fold indices (0-based).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.get("phase") != "train":
        sys.exit(f"Expected phase=train in {args.config}, got phase={cfg.get('phase')!r}")

    if args.models:     cfg["models"] = args.models
    if args.benchmarks: cfg["benchmarks"] = args.benchmarks
    if args.output_dir: cfg["output_dir"] = args.output_dir
    if args.overwrite:  cfg["overwrite"] = True
    if args.finetune:   cfg["finetune_modes"] = [args.finetune]
    if args.fold_filter: cfg["fold_filter"] = args.fold_filter

    run_experiments(cfg)


if __name__ == "__main__":
    main()
