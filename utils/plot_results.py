"""
Plot robustness evaluation results from results_eval/ CSV files.

Produces one figure per condition group (5 figures):
  - random_dropout
  - region_dropout
  - white_noise
  - pink_noise
  - region_noise

Figure layout:
  - Rows:    top = full_finetuned,  bottom = head_only
  - Columns: one per dataset
  - X-axis:  Clean baseline + conditions that actually exist for that dataset
  - Lines:   coloured by model; ±1 std shading over folds
  - EEGNet:  shown in both rows using its full_finetuned results (it has no head_only)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR    = "results/results_eval"
OUTPUT_DIR     = "plots/robustness/15db"
DEFAULT_METRIC = "bacc_test_best"

FOUNDATION_MODELS = ["REVE", "NeuroRVQ", "BrainOmni" , "BIOT", "CBraMod", "LaBraM"]
BASELINE_MODEL    = "EEGNet"   # shown in both rows; only has full_finetuned
ALL_MODELS        = FOUNDATION_MODELS + [BASELINE_MODEL]

SKIP_CUT_DATASETS = True

# ---------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------
PLOT_LARGE_HEAD  = False  # also plot large_head variants of LaBraM/NeuroRVQ
INCLUDE_OLD_MODELS = False  # include REVE-old and BrainOmni-old

LARGE_HEAD_MODELS = ["NeuroRVQ", "REVE"]  # models that have large_head variants
OLD_MODELS        = ["REVE-old", "BrainOmni-old"]

MODEL_COLORS = {
    "BIOT":          "#1f77b4",
    "CBraMod":       "#ff7f0e",
    "EEGNet":        "#2ca02c",
    "LaBraM":        "#d62728",
    "NeuroRVQ":      "#9467bd",
    "BrainOmni":     "#f50093",
    "REVE":          "#54b3c9",
    # old variants — same hue, lighter shade
    "REVE-old":      "#a8dce8",
    "BrainOmni-old": "#faa8d8",
}

# line/marker used when EEGNet is plotted in the head_only row
EEGNET_BASELINE_LS  = "-"
EEGNET_BASELINE_MK  = "D"

FINETUNE_LINESTYLES = {"full_finetuned": "-",  "head_only": "--"}
FINETUNE_MARKERS    = {"full_finetuned": "o",  "head_only": "s"}

# style for large_head variants (overlaid on normal rows)
LARGE_HEAD_LS = ":"
LARGE_HEAD_MK = "^"

# ---------------------------------------------------------------------------
# Condition group definitions
# each entry: (condition_id, x_label)
# ---------------------------------------------------------------------------

CONDITION_GROUPS = {
    "random_dropout": {
        "title":   "Random channel dropout",
        "xlabel":  "Dropout probability",
        "conditions": [
            ("dropout_random_p10", "p=0.10"),
            ("dropout_random_p30", "p=0.30"),
            ("dropout_random_p50", "p=0.50"),
            ("dropout_random_p90", "p=0.90"),

        ],
    },
    "region_dropout": {
        "title":   "Region dropout",
        "xlabel":  "Region",
        "conditions": [
            ("dropout_region_control",   "Control"),
            # ("dropout_region_secondary", "Secondary"),
            ("dropout_region_primary",   "Primary"),
            
        ],
    },
    "white_noise": {
        "title":   "White noise",
        "xlabel":  "SNR (dB)",
        "conditions": [
            ("white_noise_10db",  "10"),
            ("white_noise_5db",    "5"),
            ("white_noise_0db",    "0"),
            ("white_noise_-3db",  "-3"),
            ("white_noise_-5db",  "-5"),
            ("white_noise_-15db", "-15"),
        ],
    },
    "pink_noise": {
        "title":   "Pink noise",
        "xlabel":  "SNR (dB)",
        "conditions": [
            ("pink_noise_10db",  "10"),
            ("pink_noise_5db",    "5"),
            ("pink_noise_0db",    "0"),
            ("pink_noise_-3db",  "-3"),
            ("pink_noise_-5db",  "-5"),
        ],
    },
    "region_noise": {
        "title":   "Region noise (white)",
        "xlabel":  "Region  /  SNR",
        "conditions": [
             ("region_noise_white_5db_control",    "Control\n5 dB"),
            ("region_noise_white_-3db_control",   "Control\n-3 dB"),
            # ("region_noise_white_5db_secondary",  "Secondary\n5 dB"),
            # ("region_noise_white_-3db_secondary", "Secondary\n-3 dB"),
            ("region_noise_white_5db_primary",    "Primary\n5 dB"),
            ("region_noise_white_-3db_primary",   "Primary\n-3 dB"),
           
        ],
    },
}

# Condition groups to suppress for specific datasets (results are invalid)
DATASET_SKIP_GROUPS = {
    "Sleep_EDF": {"random_dropout"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def csv_path(results_dir, model, dataset, finetune_type, condition):
    fname = f"train-clean_test-{condition}_folds.csv"
    return os.path.join(results_dir, model, dataset, finetune_type, fname)


def load_metric(results_dir, model, dataset, finetune_type, condition, metric):
    """Return array of per-fold metric values, or None if unavailable."""
    path = csv_path(results_dir, model, dataset, finetune_type, condition)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if metric not in df.columns:
            return None
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
        return vals if len(vals) > 0 else None
    except Exception:
        return None


def mean_std(vals):
    if vals is None or len(vals) == 0:
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def discover_datasets(results_dir):
    datasets = set()
    for model in ALL_MODELS:
        d = os.path.join(results_dir, model)
        if os.path.isdir(d):
            for ds in os.listdir(d):
                if os.path.isdir(os.path.join(d, ds)):
                    if ds.endswith("_cut") and SKIP_CUT_DATASETS:
                        continue
                    datasets.add(ds)
    return sorted(datasets)


# ---------------------------------------------------------------------------
# Core: draw one subplot
# ---------------------------------------------------------------------------

def draw_subplot(ax, results_dir, dataset, finetune_type, group_key,
                 metric, models_to_plot, relative=False):
    """
    Draw one subplot for (dataset, finetune_type, condition_group).

    models_to_plot: list of (model, actual_finetune_type, linestyle, marker)
    relative: if True, subtract each model's clean baseline from all values.
    """
    group = CONDITION_GROUPS[group_key]

    # Skip this subplot if the group is invalid for this dataset
    if group_key in DATASET_SKIP_GROUPS.get(dataset, set()):
        ax.set_visible(False)
        return

    all_conds = [("clean", "Clean")] + group["conditions"]

    # Determine which x positions actually have data (any model has the file)
    valid_indices = []
    for i, (cond, _) in enumerate(all_conds):
        for model, ft, _, _ in models_to_plot:
            if load_metric(results_dir, model, dataset, ft, cond, metric) is not None:
                valid_indices.append(i)
                break

    if not valid_indices:
        ax.set_visible(False)
        return

    x_pos    = np.arange(len(valid_indices))
    x_labels = [all_conds[i][1] for i in valid_indices]
    x_conds  = [all_conds[i][0] for i in valid_indices]

    for model, ft, ls, mk in models_to_plot:
        color  = MODEL_COLORS[model]
        means, stds = [], []
        for cond in x_conds:
            vals = load_metric(results_dir, model, dataset, ft, cond, metric)
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s)

        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)

        if relative:
            clean_vals = load_metric(results_dir, model, dataset, ft, "clean", metric)
            clean_mean, _ = mean_std(clean_vals)
            means = means - clean_mean  # clean becomes 0; corrupted conditions are drops

        valid = ~np.isnan(means)

        if valid.sum() == 0:
            continue

        ax.plot(x_pos[valid], means[valid],
                color=color, linestyle=ls, marker=mk,
                linewidth=1.8, markersize=5, zorder=3)
        ax.fill_between(x_pos[valid],
                        (means - stds)[valid], (means + stds)[valid],
                        color=color, alpha=0.12, zorder=2)

    if relative:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4, zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.25)

    # vertical separator after "Clean"
    if len(valid_indices) > 1 and valid_indices[0] == 0:
        ax.axvline(0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)


# ---------------------------------------------------------------------------
# Build one figure per condition group
# ---------------------------------------------------------------------------

def plot_group(group_key, datasets, results_dir, metric, output_dir,
               include_eegnet=True, plot_large_head=False, include_old_models=False,
               relative=False):
    group     = CONDITION_GROUPS[group_key]
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        2, n_datasets,
        figsize=(3.2 * n_datasets, 7),
        sharey="row",
    )
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Full fine-tuned", "Head only"]

    old_models_active = INCLUDE_OLD_MODELS and include_old_models

    for col, dataset in enumerate(datasets):
        for row, finetune_type in enumerate(["full_finetuned", "head_only"]):
            ax = axes[row, col]

            full_ft_models = ALL_MODELS if include_eegnet else FOUNDATION_MODELS
            if finetune_type == "full_finetuned":
                models_to_plot = [
                    (m, "full_finetuned",
                     FINETUNE_LINESTYLES["full_finetuned"],
                     FINETUNE_MARKERS["full_finetuned"])
                    for m in full_ft_models
                ]
                if plot_large_head and PLOT_LARGE_HEAD:
                    models_to_plot += [
                        (m, "full_finetuned_large_head", LARGE_HEAD_LS, LARGE_HEAD_MK)
                        for m in LARGE_HEAD_MODELS
                    ]
                if old_models_active:
                    models_to_plot += [
                        (m, "full_finetuned", FINETUNE_LINESTYLES["full_finetuned"], FINETUNE_MARKERS["full_finetuned"])
                        for m in OLD_MODELS
                    ]
            else:  # head_only row
                models_to_plot = [
                    (m, "head_only",
                     FINETUNE_LINESTYLES["head_only"],
                     FINETUNE_MARKERS["head_only"])
                    for m in FOUNDATION_MODELS
                ]
                if plot_large_head and PLOT_LARGE_HEAD:
                    models_to_plot += [
                        (m, "head_only_large_head", LARGE_HEAD_LS, LARGE_HEAD_MK)
                        for m in LARGE_HEAD_MODELS
                    ]
                if old_models_active:
                    models_to_plot += [
                        (m, "head_only", FINETUNE_LINESTYLES["head_only"], FINETUNE_MARKERS["head_only"])
                        for m in OLD_MODELS
                    ]
                if include_eegnet:
                    models_to_plot += [
                        # EEGNet baseline (full_finetuned) shown with distinct style
                        (BASELINE_MODEL, "full_finetuned",
                         EEGNET_BASELINE_LS, EEGNET_BASELINE_MK)
                    ]

            draw_subplot(ax, results_dir, dataset, finetune_type,
                         group_key, metric, models_to_plot, relative=relative)

            # Column header (dataset name) on top row only
            if row == 0:
                ax.set_title(dataset.replace("_", " "), fontsize=9, pad=4)
            # Row label on leftmost column only
            if col == 0:
                ylabel = f"{row_labels[row]}\nΔ{metric} (vs clean)" if relative else f"{row_labels[row]}\n{metric}"
                ax.set_ylabel(ylabel, fontsize=8)
            # X-axis label on bottom row only
            if row == 1:
                ax.set_xlabel(group["xlabel"], fontsize=8)

    fig.suptitle(group["title"], fontsize=13, y=1.01)

    # ---- Legend -------------------------------------------------------------
    # Models
    legend_models = ALL_MODELS if include_eegnet else FOUNDATION_MODELS
    if old_models_active:
        legend_models = legend_models + OLD_MODELS
    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=m)
        for m in legend_models
    ]
    # Finetune types
    ft_handles = [
        mlines.Line2D([], [], color="grey",
                      linestyle=FINETUNE_LINESTYLES[ft],
                      marker=FINETUNE_MARKERS[ft],
                      markersize=5, label=ft.replace("_", " "))
        for ft in ["full_finetuned", "head_only"]
    ]
    all_handles = model_handles + ft_handles
    if plot_large_head and PLOT_LARGE_HEAD:
        all_handles += [mlines.Line2D(
            [], [], color="grey",
            linestyle=LARGE_HEAD_LS, marker=LARGE_HEAD_MK,
            markersize=5, label="large head"
        )]
    if include_eegnet:
        all_handles += [mlines.Line2D(
            [], [], color=MODEL_COLORS[BASELINE_MODEL],
            linestyle=EEGNET_BASELINE_LS, marker=EEGNET_BASELINE_MK,
            markersize=5, label="EEGNet (full ft, baseline)"
        )]

    fig.legend(handles=all_handles,
               loc="lower center", ncol=len(all_handles),
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_relative" if relative else ""
    out_path = os.path.join(output_dir, f"{group_key}_{metric}{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot EEG robustness evaluation results.")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR)
    parser.add_argument("--metric",      default=DEFAULT_METRIC,
                        help="Column to plot. Options: accuracy_test_best, bacc_test_best, "
                             "kappa_test_best, f1_weighted_test_best, "
                             "f1_macro_test_best, roc_auc_test_best")
    parser.add_argument("--datasets",    nargs="*", default=None)
    parser.add_argument("--groups",      nargs="*", default=None,
                        help="Condition groups to plot (default: all). "
                             "Options: " + ", ".join(CONDITION_GROUPS))
    parser.add_argument("--no-eegnet", dest="no_eegnet", action="store_true",
                        help="Exclude EEGNet from all plots")
    parser.add_argument("--large-head", dest="large_head", action="store_true",
                        default=PLOT_LARGE_HEAD,
                        help="Also plot large_head variants of LaBraM/NeuroRVQ")
    parser.add_argument("--old-models", dest="old_models", action="store_true",
                        default=INCLUDE_OLD_MODELS,
                        help="Include REVE-old and BrainOmni-old")
    parser.add_argument("--relative", action="store_true", default=False,
                        help="Plot drops relative to each model's clean baseline "
                             "(clean = 0, corrupted conditions show absolute change)")
    args = parser.parse_args()

    datasets = args.datasets or discover_datasets(args.results_dir)
    groups   = args.groups   or list(CONDITION_GROUPS)

    print(f"Metric:      {args.metric}")
    print(f"Datasets:    {datasets}")
    print(f"Groups:      {groups}")
    print(f"Large head:  {args.large_head}")
    print(f"Old models:  {args.old_models}")
    print(f"Relative:    {args.relative}")

    include_eegnet = not args.no_eegnet

    for g in groups:
        print(f"\nPlotting: {g}")
        plot_group(g, datasets, args.results_dir, args.metric, args.output_dir,
                   include_eegnet=include_eegnet,
                   plot_large_head=args.large_head,
                   include_old_models=args.old_models,
                   relative=args.relative)

    print(f"\nDone. Plots saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()