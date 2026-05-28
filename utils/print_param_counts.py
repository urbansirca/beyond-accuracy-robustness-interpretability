"""
Print parameter counts for each model × dataset × finetune_type combination.

For each combination, shows:
  - Total parameters (all)
  - Trainable parameters (requires_grad=True)
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loaders import load_benchmark, BENCHMARK_DIRS, get_subdir
from models.wrappers import get_model


DATA_ROOT = "/projects/prjs1721/benchmarks"
FOUNDATION_MODELS = ["REVE", "NeuroRVQ", "LaBraM", "CBraMod", "BIOT", "BrainOmni"]
BASELINE_MODEL = "EEGNet"
ALL_MODELS = FOUNDATION_MODELS + [BASELINE_MODEL]
LARGE_HEAD_MODELS = ["NeuroRVQ", "REVE", "LaBraM"]
FINETUNE_CONFIGS = [
    # (train_head_only, large_head, display_name)
    (True,  False, "Head"),
    (True,  True,  "Head (LH)"),
    (False, False, "Full FT"),
    (False, True,  "Full FT (LH)"),
]


HEAD_PATHS = {
    "CBraMod":   ["classifier"],
    "BIOT":      ["classifier"],
    "LaBraM":    ["head"],
    "NeuroRVQ":  ["head", "fc_norm"],
    "REVE":      ["backbone.final_layer"],
    "BrainOmni": ["downstream_model.class_head"],
    "EEGNet":    []
}

ALL_BENCHMARKS = [
    "Physionet Eyes",
    # "Physionet MI",
    # "Physionet ME",
    # "High Gamma",
    # "KU MI",
    # "Pavlov memory",
    # "Sleep EDF",
    # "KU ERP",
]


def valid_configs(model):
    """Return valid (train_head_only, large_head, display) configs for a model."""
    configs = []
    for head_only, large_head, display in FINETUNE_CONFIGS:
        if large_head and model not in LARGE_HEAD_MODELS:
            continue
        if model == BASELINE_MODEL and head_only:
            continue
        configs.append((head_only, large_head, display))
    return configs


def _get_nn_module(wrapper):
    """Navigate wrapper → Module → nn.Module to find the actual torch module."""
    obj = wrapper.model
    if hasattr(obj, 'model') and isinstance(obj.model, torch.nn.Module):
        return obj.model
    if isinstance(obj, torch.nn.Module):
        return obj
    if hasattr(obj, 'model'):
        return obj.model
    return obj


def _apply_deferred_freezing(wrapper, head_only):
    """Simulate the freezing that some models defer to fit()/prepare() time."""
    if not head_only:
        return
    module = _get_nn_module(wrapper)
    model_cls = type(wrapper.model).__name__

    if model_cls == "LaBraMModule":
        for name, param in module.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    elif model_cls == "NeuroRVQModule":
        for name, param in module.named_parameters():
            if 'head.' in name or 'fc_norm.' in name:
                continue
            param.requires_grad = False

    elif model_cls == "CBraModModule":
        if hasattr(module, 'backbone'):
            for param in module.backbone.parameters():
                param.requires_grad = False


def _get_submodule(module, dotted_path):
    """Walk dotted attribute path on a module."""
    obj = module
    for part in dotted_path.split('.'):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _head_param_ids(model_name, module):
    ids = set()
    for path in HEAD_PATHS.get(model_name, []):
        sub = _get_submodule(module, path)
        if sub is None or not isinstance(sub, torch.nn.Module):
            continue
        for p in sub.parameters():
            ids.add(id(p))
    return ids


def count_params(model_wrapper, model_name, head_only=False):
    """Return dict with total / trainable / backbone / head / head_trainable counts."""
    _apply_deferred_freezing(model_wrapper, head_only)
    module = _get_nn_module(model_wrapper)
    head_ids = _head_param_ids(model_name, module)

    total = trainable = head_total = head_trainable = 0
    seen = set()
    for p in module.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if id(p) in head_ids:
            head_total += n
            if p.requires_grad:
                head_trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "head": head_total,
        "head_trainable": head_trainable,
        "backbone": total - head_total,
    }


def get_trainable_layers(model_wrapper, head_only=False):
    """Return list of (name, shape, numel) for all trainable parameters."""
    _apply_deferred_freezing(model_wrapper, head_only)
    module = _get_nn_module(model_wrapper)
    layers = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            layers.append((name, tuple(param.shape), param.numel()))
    return layers


def fmt_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


def print_table(title, headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = []
        for i in range(len(headers)):
            w = len(headers[i])
            for row in rows:
                if i < len(row):
                    w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            parts.append(str(cell).center(col_widths[i]))
        return "|" + "|".join(parts) + "|"

    print(f"\n{'=' * len(sep)}")
    print(f" {title}")
    print(f"{'=' * len(sep)}")
    print(sep)
    print(fmt_row(headers))
    print(sep.replace("-", "="))
    for row in rows:
        print(fmt_row(row))
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Print parameter counts per model × dataset × finetune type.")
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"Models to include. Default: all. Options: {ALL_MODELS}")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"Datasets to include. Default: all. Options: {ALL_BENCHMARKS}")
    parser.add_argument("--device", default="cpu", help="Device for model instantiation (default: cpu)")
    args = parser.parse_args()

    models = args.models or ALL_MODELS
    datasets = args.datasets or ALL_BENCHMARKS

    bench_cache = {}

    for dataset_name in datasets:
        print(f"\n{'#' * 80}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#' * 80}")

        headers = ["Model", "FT Type", "Total", "Backbone", "Head", "Trainable", "% Tr", "Flag"]
        rows = []
        head_details = []

        for model_name in models:
            subdir = get_subdir(model_name)
            cache_key = (dataset_name, subdir)

            # Load benchmark to get shapes (cached)
            if cache_key not in bench_cache:
                try:
                    bench = load_benchmark(dataset_name, args.data_root, subdir=subdir, apply_car=True)
                    X, sbj_id, y, ch_names = bench.get_data()
                    n_samples, n_chans, n_times = X.shape
                    n_outputs = len(np.unique(y))
                    sfreq = 200  # all datasets use 200 Hz
                    bench_cache[cache_key] = (n_chans, ch_names, sfreq, n_times, n_outputs, sbj_id)
                    del X  # free memory
                except Exception as e:
                    print(f"  WARNING: Could not load {dataset_name} (subdir={subdir}): {e}")
                    bench_cache[cache_key] = None
                    continue

            meta = bench_cache[cache_key]
            if meta is None:
                continue
            n_chans, ch_names, sfreq, n_times, n_outputs, sbj_id = meta

            for head_only, large_head, display in valid_configs(model_name):
                try:
                    wrapper = get_model(
                        model_name,
                        n_chans=n_chans,
                        ch_names=ch_names,
                        sfreq=sfreq,
                        n_times=n_times,
                        n_outputs=n_outputs,
                        sbj_ids=sbj_id,
                        encoder_only=False,
                        ckpt_path=None,
                        train_head_only=head_only,
                        large_head=large_head,
                    )
                    stats = count_params(wrapper, model_name, head_only=head_only)
                    trainable_layers = get_trainable_layers(wrapper, head_only=head_only)
                    pct = f"{100 * stats['trainable'] / stats['total']:.1f}%" if stats["total"] > 0 else "—"


                    flag = ""
                    if HEAD_PATHS.get(model_name) is not None and HEAD_PATHS.get(model_name):
                        if head_only and stats["trainable"] != stats["head"]:
                            flag = f"MISMATCH (head={fmt_params(stats['head'])}, tr={fmt_params(stats['trainable'])})"
                        elif (not head_only) and stats["trainable"] != stats["total"]:
                            frozen = stats["total"] - stats["trainable"]
                            flag = f"MISMATCH (frozen={fmt_params(frozen)})"

                    rows.append([
                        model_name, display,
                        fmt_params(stats["total"]),
                        fmt_params(stats["backbone"]),
                        fmt_params(stats["head"]),
                        fmt_params(stats["trainable"]),
                        pct,
                        flag,
                    ])
                    if head_only:
                        head_details.append((model_name, display, trainable_layers))

                    del wrapper
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    rows.append([model_name, display, "ERROR", "—", "—", str(e)[:40], "—", ""])

        print_table(f"Parameter counts — {dataset_name}", headers, rows)

        for model_name, display, layers in head_details:
            if not layers:
                continue
            print(f"\n  {model_name} [{display}] — trainable layers ({len(layers)}):")
            for name, shape, numel in layers:
                print(f"    {name:<60s} {str(shape):<30s} {fmt_params(numel):>10s}")


if __name__ == "__main__":
    main()
