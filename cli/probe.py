"""Linear probing CLI. Takes a YAML config and runs the
cross-product (model × benchmark × ckpt_type) for one pooling strategy.

python -m cli.probe --config configs/probing/mean.yaml
python -m cli.probe --config configs/probing/concat.yaml --models CBraMod
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from cli._config import load_config
from interpretability.probing import load_completed_keys, open_writer, probe_model
from interpretability.probing.adapters import CLS_POOLING_MODELS, LARGE_HEAD_MODELS


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Linear probing of EEG foundation models.")
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (configs/probing/*.yaml).")
    p.add_argument("--models",      nargs="+", help="Override config's models list.")
    p.add_argument("--benchmarks",  nargs="+", help="Override config's benchmarks list.")
    p.add_argument("--ckpt-types",  nargs="+", choices=["pretrained", "finetuned"],
                   help="Override config's ckpt_types list.")
    p.add_argument("--output-csv",  help="Override config's output_csv.")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-run and overwrite existing fold results.")
    p.add_argument("--large-head",  action="store_true",
                   help="Use large_head variant (LaBraM/NeuroRVQ/REVE only).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if "pooling" not in cfg:
        sys.exit(f"Config {args.config} must set `pooling` (mean|cls|concat).")

    if args.models:       cfg["models"]      = args.models
    if args.benchmarks:   cfg["benchmarks"]  = args.benchmarks
    if args.ckpt_types:   cfg["ckpt_types"]  = args.ckpt_types
    if args.output_csv:   cfg["output_csv"]  = args.output_csv
    if args.overwrite:    cfg["overwrite"]   = True
    if args.large_head:   cfg["large_head"]  = True

    pooling    = cfg["pooling"]
    large_head = cfg["large_head"]
    
    output_dir = cfg["output_dir"]
    output_csv = os.path.join(output_dir, cfg["output_csv"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # keys already present anywhere under `output_dir`.
    completed_keys = load_completed_keys(cfg["output_dir"]) if not cfg["overwrite"] else set()
    writer, csv_file = open_writer(output_csv)

    try:
        for model_name in cfg["models"]:
            for benchmark_name in cfg["benchmarks"]:
                for ckpt_type in cfg["ckpt_types"]:
                    if pooling == "cls" and model_name not in CLS_POOLING_MODELS:
                        print(f"\n[skip] {model_name} has no CLS token — skipping cls pooling")
                        continue
                    if large_head and model_name not in LARGE_HEAD_MODELS:
                        print(f"\n[skip] {model_name} does not support large_head — skipping")
                        continue
                    print(f"\n=== {model_name} | {benchmark_name} | {ckpt_type} "
                          f"| pooling={pooling} | large_head={large_head} ===")
                    try:
                        probe_model(
                            model_name, benchmark_name, ckpt_type, device,
                            writer, csv_file, cfg,
                            completed_keys=completed_keys,
                            pooling=pooling, large_head=large_head,
                        )
                    except Exception as e:
                        print(f"  [error] {type(e).__name__}: {e}")
    finally:
        csv_file.close()

    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
