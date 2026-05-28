"""
python interpretability/plotting/plotting_probing.py --pooling mean
python interpretability/plotting/plotting_probing.py --pooling cls 
python interpretability/plotting/plotting_probing.py --pooling concat
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data.loaders import slug_for

METRIC_LABELS = {
    "bacc":        "Balanced Accuracy",
    "f1_weighted": "F1 (Weighted)",
    "f1_macro":    "F1 (Macro)",
}

MODELS = ["LaBraM", "CBraMod", "BrainOmni", "NeuroRVQ", "REVE"]
BENCHMARKS = ["Physionet Eyes", "High Gamma", "KU ERP", "KU MI"]

CHANCE_LEVELS = {
    "Physionet Eyes": 0.5,
    "High Gamma":     0.25,
    "KU ERP":         0.5,
    "KU MI":          0.5,
}

COLORS = {"pretrained": "#2196F3", "finetuned": "#F44336"}

N_FOLDS_EXPECTED = 10


def model_display_name(model: str, large_head: bool) -> str:
    return f"{model} (large head)" if large_head else model

def load_results(results_dir: str) -> pd.DataFrame:
    """Concatenate every `*.csv` under `results_dir` (one per model)."""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")
    csvs = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSVs under {results_dir}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df["large_head"] = df["large_head"].astype(bool)
    return df


def check_fold_counts(df: pd.DataFrame, pooling: str):
    """Warn if any (display_model, benchmark, ckpt_type) has fewer than N_FOLDS_EXPECTED folds."""
    df = df.copy()
    df["display_model"] = [model_display_name(m, lh) for m, lh in zip(df["model"], df["large_head"])]

    groups = df.groupby(["display_model", "benchmark", "ckpt_type"])["fold"].nunique()
    for (display_model, benchmark, ckpt_type), n_folds in groups.items():
        if n_folds < N_FOLDS_EXPECTED:
            print(
                f"  WARNING: {display_model} | {benchmark} | {ckpt_type} | pooling={pooling}"
                f" — only {n_folds}/{N_FOLDS_EXPECTED} folds found"
            )


def plot_benchmark(df: pd.DataFrame, benchmark: str, metric: str, out_dir: str):
    sub = df[df["benchmark"] == benchmark].copy()
    if sub.empty:
        return

    sub["display_model"] = [model_display_name(m, lh) for m, lh in zip(sub["model"], sub["large_head"])]

    # Preserve canonical order, inserting large_head variants right after base model
    ordered = []
    for m in MODELS:
        for lh in [False, True]:
            name = model_display_name(m, lh)
            if name in sub["display_model"].unique():
                ordered.append(name)
    models_present = ordered
    n_models = len(models_present)
    if n_models == 0:
        return

    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"{benchmark} — {METRIC_LABELS.get(metric, metric)}", fontsize=14, y=1.01)

    for idx, display_model in enumerate(models_present):
        ax = axes[idx // ncols][idx % ncols]
        msub = sub[sub["display_model"] == display_model]

        for ckpt_type, color in COLORS.items():
            csub = msub[msub["ckpt_type"] == ckpt_type]
            if csub.empty:
                continue
            stats = csub.groupby("block_idx")[metric].agg(["mean", "std"]).reset_index()
            x = stats["block_idx"].values
            y = stats["mean"].values
            yerr = stats["std"].values
            ax.plot(x, y, color=color, label=ckpt_type, marker="o", markersize=3, linewidth=1.5)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15)

        ax.axhline(CHANCE_LEVELS[benchmark], color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.set_title(display_model)
        ax.set_xlabel("Block index")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{slug_for(benchmark)}_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling",     required=True, choices=["concat", "cls", "mean"])
    parser.add_argument("--results_dir", default=None,
                        help="Defaults to results/probing/<pooling>.")
    parser.add_argument("--out_dir",     default=None,
                        help="Defaults to plots/probing/<pooling>.")
    parser.add_argument("--metric",      default="bacc", choices=list(METRIC_LABELS.keys()))
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join("results", "probing", args.pooling)
    out_dir     = args.out_dir     or os.path.join("plots",   "probing", args.pooling)

    df = load_results(results_dir)
    df = df[df["pooling"] == args.pooling]
    print(f"Loaded {len(df)} rows (pooling={args.pooling}) from {results_dir}")
    print(f"  Models:     {sorted(df['model'].unique())}")
    print(f"  Benchmarks: {sorted(df['benchmark'].unique())}")
    print(f"  Ckpt types: {sorted(df['ckpt_type'].unique())}")
    print(f"Checking fold counts...")
    check_fold_counts(df, args.pooling)

    benchmarks_present = [b for b in BENCHMARKS if b in df["benchmark"].unique()]
    for benchmark in benchmarks_present:
        plot_benchmark(df, benchmark, args.metric, out_dir)


if __name__ == "__main__":
    main()
