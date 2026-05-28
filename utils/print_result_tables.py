import os
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config (mirrored from plot_results.py)
# ---------------------------------------------------------------------------

RESULTS_EVAL_DIR     = "results/results_eval"
RESULTS_TRAIN_DIR    = "results/results_train"
RESULTS_DROPPED_DIR  = "results/dropped-channels-eval"
DEFAULT_METRIC       = "bacc"

FOUNDATION_MODELS = ["REVE", "NeuroRVQ", "LaBraM", "CBraMod", "BIOT", "BrainOmni"]
BASELINE_MODEL    = "EEGNet"
ALL_MODELS        = FOUNDATION_MODELS + [BASELINE_MODEL]

LARGE_HEAD_MODELS = ["NeuroRVQ", "REVE", "LaBraM"]

FINETUNE_TYPES = [
    "head_only",
    "head_only_large_head",
    "full_finetuned",
    "full_finetuned_large_head",
]

FINETUNE_DISPLAY = {
    "head_only":                  "Head",
    "head_only_large_head":       "Head (LH)",
    "full_finetuned":             "Full FT",
    "full_finetuned_large_head":  "Full FT (LH)",
}

SKIP_CUT_DATASETS = True

CONDITION_GROUPS = {
    "clean": {
        "title": "Clean (train-clean / test-clean)",
        "conditions": [("clean", "Clean")],
    },
    "random_dropout": {
        "title": "Random channel dropout",
        "conditions": [
            ("dropout_random_p10", "p=0.10"),
            ("dropout_random_p30", "p=0.30"),
            ("dropout_random_p50", "p=0.50"),
            ("dropout_random_p90", "p=0.90"),
        ],
    },
    "region_dropout": {
        "title": "Region dropout",
        "conditions": [
            ("dropout_region_control",   "Control"),
            ("dropout_region_secondary", "Secondary"),
            ("dropout_region_primary",   "Primary"),
        ],
    },
    "true_random_dropout": {
        "title": "Random channel dropout (TRUE drop — channels removed from input)",
        "source": "dropped",
        "conditions": [
            ("dropout_random_p10", "p=0.10"),
            ("dropout_random_p30", "p=0.30"),
            ("dropout_random_p50", "p=0.50"),
            ("dropout_random_p90", "p=0.90"),
        ],
    },
    "true_region_dropout": {
        "title": "Region dropout (TRUE drop — channels removed from input)",
        "source": "dropped",
        "conditions": [
            ("dropout_region_control",   "Control"),
            ("dropout_region_secondary", "Secondary"),
            ("dropout_region_primary",   "Primary"),
        ],
    },
    "white_noise": {
        "title": "White noise",
        "conditions": [
            ("white_noise_10db", "10 dB"),
            ("white_noise_5db",   "5 dB"),
            ("white_noise_0db",   "0 dB"),
            ("white_noise_-3db", "-3 dB"),
            ("white_noise_-5db", "-5 dB"),
            ("white_noise_-15db", "-15 dB"),
        ],
    },
    "pink_noise": {
        "title": "Pink noise",
        "conditions": [
            ("pink_noise_10db", "10 dB"),
            ("pink_noise_5db",   "5 dB"),
            ("pink_noise_0db",   "0 dB"),
            ("pink_noise_-3db", "-3 dB"),
            ("pink_noise_-5db", "-5 dB"),
        ],
    },
    "region_noise": {
        "title": "Region noise (white)",
        "conditions": [
            ("region_noise_white_5db_control",    "Ctrl 5dB"),
            ("region_noise_white_-3db_control",   "Ctrl -3dB"),
            ("region_noise_white_5db_secondary",  "Sec 5dB"),
            ("region_noise_white_-3db_secondary", "Sec -3dB"),
            ("region_noise_white_5db_primary",    "Pri 5dB"),
            ("region_noise_white_-3db_primary",   "Pri -3dB"),
        ],
    },
}

DATASET_SKIP_GROUPS = {
    "sleep_edf": {"random_dropout"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_sleep(ds):
    """Case-insensitive match for the sleep dataset (dir is `sleep_edf`)."""
    return ds.strip().lower() == "sleep_edf"

def _update_paths(eval_dir, train_dir, dropped_dir=None):
    global RESULTS_EVAL_DIR, RESULTS_TRAIN_DIR, RESULTS_DROPPED_DIR
    RESULTS_EVAL_DIR  = eval_dir
    RESULTS_TRAIN_DIR = train_dir
    if dropped_dir is not None:
        RESULTS_DROPPED_DIR = dropped_dir


def csv_path_eval(model, dataset, finetune_type, condition, source="eval"):
    """source: 'eval' (default, results_eval) or 'dropped' (dropped-channels-eval)."""
    base = RESULTS_DROPPED_DIR if source == "dropped" else RESULTS_EVAL_DIR
    fname = f"train-clean_test-{condition}_folds.csv"
    return os.path.join(base, model, dataset, finetune_type, fname)


def csv_path_train(model, dataset, finetune_type):
    fname = "train-clean_test-clean_folds.csv"
    return os.path.join(RESULTS_TRAIN_DIR, model, dataset, finetune_type, fname)


def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None


def get_metric_vals(df, col):
    if df is None or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    return vals if len(vals) > 0 else None


def fmt_mean_std(vals):
    if vals is None or len(vals) == 0:
        return "   —   "
    m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
    return f"{m:.3f}±{s:.3f}"


def fmt_mean_std_short(vals):
    if vals is None or len(vals) == 0:
        return "  —  "
    m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
    return f"{m:.1f}±{s:.1f}"


def fmt_avg(per_ds_means):
    """Format average across datasets (mean of per-dataset means ± std across datasets)."""
    valid = [m for m in per_ds_means if not np.isnan(m)]
    if not valid:
        return "   —   "
    m = np.mean(valid)
    s = np.std(valid)
    return f"{m:.3f}±{s:.3f}"


def fmt_drop(clean_vals, corrupt_vals):
    """Format relative % drop from clean, computed per-fold.
    """
    if clean_vals is None or corrupt_vals is None:
        return "   —   "
    n = min(len(clean_vals), len(corrupt_vals))
    if n == 0:
        return "   —   "
    drops = corrupt_vals[:n] - clean_vals[:n]
    # Convert to percentage points (already in 0-1 scale for bacc etc.)
    drops_pct = drops * 100
    m, s = float(np.nanmean(drops_pct)), float(np.nanstd(drops_pct))
    sign = "+" if m > 0 else ""
    return f"{sign}{m:.1f}±{s:.1f}"


def fmt_drop_avg(per_ds_drops):
    """Format average drop across datasets."""
    valid = [d for d in per_ds_drops if not np.isnan(d)]
    if not valid:
        return "   —   "
    m = np.mean(valid)
    s = np.std(valid)
    sign = "+" if m > 0 else ""
    return f"{sign}{m:.1f}±{s:.1f}"


def fmt_rel(clean_vals, corrupt_vals):
    """Relative %-loss vs clean, per-fold: (clean - corrupted) / clean × 100."""
    if clean_vals is None or corrupt_vals is None:
        return "   —   "
    n = min(len(clean_vals), len(corrupt_vals))
    if n == 0:
        return "   —   "
    cl = np.asarray(clean_vals[:n], dtype=float)
    co = np.asarray(corrupt_vals[:n], dtype=float)
    rel = np.where(cl > 0, (cl - co) / cl * 100.0, np.nan)
    rel = rel[~np.isnan(rel)]
    if len(rel) == 0:
        return "   —   "
    m, s = float(np.mean(rel)), float(np.std(rel))
    return f"{m:.1f}±{s:.1f}"


def fmt_rel_avg(per_ds_rels):
    """Mean ± std of per-dataset."""
    valid = [r for r in per_ds_rels if not np.isnan(r)]
    if not valid:
        return "   —   "
    return f"{np.mean(valid):.1f}±{np.std(valid):.1f}"


RANK_COL_ALIASES = {
    "average": "avg",
    "avg excl sleep": "avg (excl. sleep)",
    "avg-excl-sleep": "avg (excl. sleep)",
    "hg": "high gamma",
    "eyes": "physionet eyes",
    "mi": "physionet mi",
    "me": "physionet me",
    "memory": "pavlov memory",
    "ku-mi": "ku mi",
    "ku-erp": "ku erp",
    "sleep": "sleep edf",
}


def _resolve_rank_target(rank_col, headers):
    """Return the index in `headers` matching `rank_col` (with alias support), or None."""
    if rank_col is None:
        return None
    norm = lambda s: s.strip().lower().replace("_", " ")
    target = norm(rank_col)
    target = RANK_COL_ALIASES.get(target, target)
    for i, h in enumerate(headers):
        if norm(h) == target:
            return i
    return None


def discover_datasets():
    datasets = set()
    for results_dir in [RESULTS_EVAL_DIR, RESULTS_TRAIN_DIR]:
        for model in ALL_MODELS:
            d = os.path.join(results_dir, model)
            if os.path.isdir(d):
                for ds in os.listdir(d):
                    if os.path.isdir(os.path.join(d, ds)):
                        if ds.endswith("_cut") and SKIP_CUT_DATASETS:
                            continue
                        datasets.add(ds)
    return sorted(datasets)


def valid_ft_types(model, include_lh=True, ft_mode="both"):
    """Return finetune types applicable to this model.

    Args:
        include_lh: Controls large_head variants. Either a bool (True = every
            LH-capable model gets LH variants, False = none do) or a collection
            of model names (only those models get LH variants, the rest are
            head-only).
        ft_mode: "head_only", "finetuned", or "both".
    """
    if isinstance(include_lh, (set, list, tuple)):
        model_has_lh = model in include_lh
    else:
        model_has_lh = bool(include_lh)
    fts = []
    for ft in FINETUNE_TYPES:
        if "large_head" in ft and (model not in LARGE_HEAD_MODELS or not model_has_lh):
            continue
        if model == BASELINE_MODEL and ft != "full_finetuned":
            continue
        if ft_mode == "head_only" and ft.startswith("full_finetuned"):
            continue
        if ft_mode == "finetuned" and ft.startswith("head_only"):
            continue
        fts.append(ft)
    return fts


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _sort_key(cell):
    """Extract mean from a 'mean±std' cell for sorting."""
    s = str(cell).strip()
    if "±" in s:
        try:
            return float(s.split("±")[0])
        except ValueError:
            pass
    return float("-inf")


def print_table(title, headers, rows, col_widths=None, rank_col=None):
    """Print a nicely formatted table. If rank_col is set, sort rows descending by that column."""
    if rank_col is not None:
        # Match rank_col to header (case-insensitive, spaces/underscores normalized)
        norm = lambda s: s.strip().lower().replace("_", " ")
        col_idx = None
        for i, h in enumerate(headers):
            if norm(h) == norm(rank_col):
                col_idx = i
                break
        if col_idx is not None:
            rows = sorted(rows, key=lambda r: _sort_key(r[col_idx]), reverse=True)
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

    # Build average summary row
    avg_row = ["", "Avg"]
    for col_i in range(2, len(headers)):
        col_means = []
        for row in rows:
            if col_i < len(row):
                val = _sort_key(row[col_i])
                if val != float("-inf"):
                    col_means.append(val)
        if col_means:
            m = np.mean(col_means)
            s = np.std(col_means)
            avg_row.append(f"{m:.3f}±{s:.3f}")
        else:
            avg_row.append("   —   ")
    # Include avg_row in col_widths calculation
    for i in range(len(headers)):
        if i < len(avg_row):
            col_widths[i] = max(col_widths[i], len(str(avg_row[i])) + 2)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    print(f"\n{'=' * len(sep)}")
    print(f" {title}")
    print(f"{'=' * len(sep)}")
    print(sep)
    print(fmt_row(headers))
    print(sep.replace("-", "="))
    for row in rows:
        print(fmt_row(row))
    print(sep.replace("-", "="))
    print(fmt_row(avg_row))
    print(sep)


def print_clean_table(datasets, metric, rank_col=None, skip_train=False, include_lh=True, ft_mode="both"):
    """Table for clean train/test performance + training epochs."""
    print("\n" + "#" * 80)
    print("# CLEAN BASELINE (train-clean / test-clean)")
    print("#" * 80)

    train_col = f"{metric}_train_best"
    test_col  = f"{metric}_test_best"

    headers = ["Model", "FT Type"] + [ds.replace("_", " ") for ds in datasets] + ["Avg", "Avg (excl. Sleep)", "Avg Epochs"]

    # --- Test metric table ---
    rows = []
    for model in ALL_MODELS:
        for ft in valid_ft_types(model, include_lh=include_lh, ft_mode=ft_mode):
            row = [model, FINETUNE_DISPLAY[ft]]
            all_epochs = []
            ds_means = []
            ds_means_excl_sleep = []
            for ds in datasets:
                df = load_csv(csv_path_eval(model, ds, ft, "clean"))
                if df is None:
                    df = load_csv(csv_path_train(model, ds, ft))
                vals = get_metric_vals(df, test_col)
                row.append(fmt_mean_std(vals))
                m = float(np.nanmean(vals)) if vals is not None else np.nan
                ds_means.append(m)
                if not _is_sleep(ds):
                    ds_means_excl_sleep.append(m)
                df_train = load_csv(csv_path_train(model, ds, ft))
                ep = get_metric_vals(df_train, "n_epochs")
                if ep is not None:
                    all_epochs.extend(ep)
            row.append(fmt_avg(ds_means))
            row.append(fmt_avg(ds_means_excl_sleep))
            if all_epochs:
                row.append(fmt_mean_std_short(np.array(all_epochs)))
            else:
                row.append("—")
            rows.append(row)

    print_table(f"Test {metric} (mean±std across folds)", headers, rows, rank_col=rank_col)


    if skip_train:
        return

    # --- Train metric table ---
    rows = []
    for model in ALL_MODELS:
        for ft in valid_ft_types(model, include_lh=include_lh, ft_mode=ft_mode):
            row = [model, FINETUNE_DISPLAY[ft]]
            all_epochs = []
            ds_means = []
            ds_means_excl_sleep = []
            for ds in datasets:
                df = load_csv(csv_path_train(model, ds, ft))
                vals = get_metric_vals(df, train_col)
                row.append(fmt_mean_std(vals))
                m = float(np.nanmean(vals)) if vals is not None else np.nan
                ds_means.append(m)
                if not _is_sleep(ds):
                    ds_means_excl_sleep.append(m)
                ep = get_metric_vals(df, "n_epochs")
                if ep is not None:
                    all_epochs.extend(ep)
            row.append(fmt_avg(ds_means))
            row.append(fmt_avg(ds_means_excl_sleep))
            if all_epochs:
                row.append(fmt_mean_std_short(np.array(all_epochs)))
            else:
                row.append("—")
            rows.append(row)

    print_table(f"Train {metric} (mean±std across folds)", headers, rows, rank_col=rank_col)


def print_corruption_table(group_key, datasets, metric, rank_col=None, include_lh=True,
                           ft_mode="both", relative=False, rank_by_relative=False):
    """Table for one corruption scheme.

    If `rank_col` is supplied, an extra "Rel" column (relative %-loss vs clean,
    (clean - corrupted) / clean × 100) is inserted next to the chosen column.
    `rank_by_relative=True` sorts rows by Rel ascending (smallest relative loss
    first); otherwise rows are sorted by the chosen column's existing value
    (descending — largest = smallest absolute loss for Δ% mode).
    """
    group = CONDITION_GROUPS[group_key]
    conditions = group["conditions"]
    source = group.get("source", "eval")

    print("\n" + "#" * 80)
    print(f"# {group['title'].upper()}")
    print("#" * 80)

    test_col  = f"{metric}_test_best"
    train_col = f"{metric}_train_best"

    for cond_id, cond_label in conditions:
        # --- Test ---
        ds_headers = [ds.replace("_", " ") for ds in datasets]
        headers = ["Model", "FT Type"] + ds_headers + ["Avg", "Avg (excl. Sleep)"]
        rows = []
        # raw[row_idx][col_idx] = (clean_vals, corrupt_vals) | None for Model/FT cols and skipped cells
        raw = []
        for model in ALL_MODELS:
            for ft in valid_ft_types(model, include_lh=include_lh, ft_mode=ft_mode):
                row = [model, FINETUNE_DISPLAY[ft]]
                row_raw = [None, None]
                ds_means = []
                ds_means_excl_sleep = []
                ds_clean_corrupt = []   # parallel list of (clean_vals, corrupt_vals) per ds for this row
                for ds in datasets:
                    if group_key in DATASET_SKIP_GROUPS.get(ds.strip().lower(), set()):
                        row.append("skip")
                        row_raw.append(None)
                        ds_means.append(np.nan)
                        ds_clean_corrupt.append((None, None))
                        if not _is_sleep(ds):
                            ds_means_excl_sleep.append(np.nan)
                        continue
                    df = load_csv(csv_path_eval(model, ds, ft, cond_id, source=source))
                    vals = get_metric_vals(df, test_col)
                    df_clean = load_csv(csv_path_eval(model, ds, ft, "clean"))
                    if df_clean is None:
                        df_clean = load_csv(csv_path_train(model, ds, ft))
                    clean_vals = get_metric_vals(df_clean, test_col)
                    ds_clean_corrupt.append((clean_vals, vals))
                    row_raw.append((clean_vals, vals))
                    if relative:
                        row.append(fmt_drop(clean_vals, vals))
                        if clean_vals is not None and vals is not None:
                            n = min(len(clean_vals), len(vals))
                            drop_mean = float(np.nanmean((vals[:n] - clean_vals[:n]) * 100))
                        else:
                            drop_mean = np.nan
                        ds_means.append(drop_mean)
                        if not _is_sleep(ds):
                            ds_means_excl_sleep.append(drop_mean)
                    else:
                        row.append(fmt_mean_std(vals))
                        m = float(np.nanmean(vals)) if vals is not None else np.nan
                        ds_means.append(m)
                        if not _is_sleep(ds):
                            ds_means_excl_sleep.append(m)
                if relative:
                    row.append(fmt_drop_avg(ds_means))
                    row.append(fmt_drop_avg(ds_means_excl_sleep))
                else:
                    row.append(fmt_avg(ds_means))
                    row.append(fmt_avg(ds_means_excl_sleep))
                # Stash aggregate (clean,corrupt) for the Avg / Avg (excl. Sleep) columns
                avg_clean_corrupt = ds_clean_corrupt[:]               # for Avg
                excl_clean_corrupt = [cc for ds, cc in zip(datasets, ds_clean_corrupt)
                                      if not _is_sleep(ds)]            # for Avg excl. Sleep
                row_raw.append(("agg", avg_clean_corrupt))
                row_raw.append(("agg", excl_clean_corrupt))
                rows.append(row)
                raw.append(row_raw)

        # --- Optionally insert Rel column next to the rank target ---
        rank_idx = _resolve_rank_target(rank_col, headers) if rank_col else None
        if rank_idx is not None:
            rel_strs = []
            rel_means = []  # for sort by Rel
            for row_raw in raw:
                cell_raw = row_raw[rank_idx]
                if cell_raw is None:
                    rel_strs.append("   —   "); rel_means.append(np.nan); continue
                if isinstance(cell_raw, tuple) and len(cell_raw) == 2 and isinstance(cell_raw[0], str) and cell_raw[0] == "agg":
                    # Aggregate column: mean of per-dataset Rel means
                    per_ds = []
                    for cc in cell_raw[1]:
                        cl, co = cc
                        if cl is None or co is None:
                            per_ds.append(np.nan); continue
                        n = min(len(cl), len(co))
                        if n == 0:
                            per_ds.append(np.nan); continue
                        cl_a = np.asarray(cl[:n], dtype=float)
                        co_a = np.asarray(co[:n], dtype=float)
                        rel_a = np.where(cl_a > 0, (cl_a - co_a) / cl_a * 100.0, np.nan)
                        rel_a = rel_a[~np.isnan(rel_a)]
                        per_ds.append(float(np.mean(rel_a)) if len(rel_a) else np.nan)
                    rel_strs.append(fmt_rel_avg(per_ds))
                    valid = [r for r in per_ds if not np.isnan(r)]
                    rel_means.append(float(np.mean(valid)) if valid else np.nan)
                else:
                    cl, co = cell_raw
                    rel_strs.append(fmt_rel(cl, co))
                    if cl is None or co is None:
                        rel_means.append(np.nan)
                    else:
                        n = min(len(cl), len(co))
                        if n == 0:
                            rel_means.append(np.nan)
                        else:
                            cl_a = np.asarray(cl[:n], dtype=float)
                            co_a = np.asarray(co[:n], dtype=float)
                            r = np.where(cl_a > 0, (cl_a - co_a) / cl_a * 100.0, np.nan)
                            r = r[~np.isnan(r)]
                            rel_means.append(float(np.mean(r)) if len(r) else np.nan)
            # Insert Rel header and per-row cells
            insert_at = rank_idx + 1
            headers = headers[:insert_at] + ["Rel"] + headers[insert_at:]
            rows = [r[:insert_at] + [rel_strs[i]] + r[insert_at:] for i, r in enumerate(rows)]
            # Pre-sort here based on the requested key, then disable print_table's sort
            if rank_by_relative:
                # Smallest relative loss first → ascending Rel
                order = sorted(range(len(rows)),
                               key=lambda i: (np.inf if np.isnan(rel_means[i]) else rel_means[i]))
            else:
                # Existing behaviour: sort by absolute Δ of rank_col (descending — least
                # negative = smallest absolute loss = best, comes first).
                order = sorted(range(len(rows)),
                               key=lambda i: _sort_key(rows[i][rank_idx]),
                               reverse=True)
            rows = [rows[i] for i in order]
            effective_rank = None  # already sorted
        else:
            effective_rank = rank_col

        label_suffix = " (Δ% from clean)" if relative else ""
        print_table(f"Test {metric} — {cond_label} ({cond_id}){label_suffix}",
                    headers, rows, rank_col=effective_rank)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Print result tables per corruption scheme.")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="Base metric name (without _train/_test suffix). "
                             "E.g. bacc, accuracy, kappa, f1_weighted, f1_macro, roc_auc")
    parser.add_argument("--results_eval_dir",    default=RESULTS_EVAL_DIR)
    parser.add_argument("--results_train_dir",   default=RESULTS_TRAIN_DIR)
    parser.add_argument("--results_dropped_dir", default=RESULTS_DROPPED_DIR,
                        help="Source for true_region_dropout / true_random_dropout groups.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Filter models. Default: all (FOUNDATION_MODELS + EEGNet).")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Condition groups to print. Options: clean, "
                             + ", ".join(CONDITION_GROUPS) + ". Default: all.")
    parser.add_argument("--rank", default=None,
                        help="Column name to rank (sort descending) rows by. E.g. 'Avg'.")
    parser.add_argument("--no-lh", action="store_true",
                        help="Exclude large head variants for all models.")
    parser.add_argument("--lh-models", nargs="*", default=["REVE"],
                        help="Only these models get large-head variants; all others "
                             "are head-only. Default: REVE. Pass --no-lh to disable "
                             "large-head for every model.")
    parser.add_argument("--ft-mode", default="both",
                        choices=["both", "head_only", "finetuned"],
                        help="Filter finetune types: head_only, finetuned, or both (default).")
    parser.add_argument("--relative", action="store_true",
                        help="Show relative drop from clean (in percentage points) instead of absolute values.")
    parser.add_argument("--rank-by-relative", action="store_true",
                        help="Sort by Rel value of --rank (smallest relative loss first). "
                             "Default sorts by the absolute value of --rank (descending).")
    args = parser.parse_args()

    _update_paths(args.results_eval_dir, args.results_train_dir, args.results_dropped_dir)

    if args.models:
        global ALL_MODELS
        ALL_MODELS = [m for m in ALL_MODELS if m in args.models] or list(args.models)

    datasets   = args.datasets or discover_datasets()
    # Default groups exclude the true_* variants — opt-in via --groups.
    default_groups = ["clean"] + [g for g in CONDITION_GROUPS
                                  if g != "clean" and not g.startswith("true_")]
    groups     = args.groups or default_groups
    if args.no_lh:
        include_lh = False
    else:
        include_lh = set(args.lh_models)
    ft_mode    = args.ft_mode

    print(f"Metric: {args.metric}")
    print(f"Datasets: {[ds.replace('_', ' ') for ds in datasets]}")
    print(f"Groups: {groups}")
    if isinstance(include_lh, set):
        lh_desc = f"only {sorted(include_lh)}" if include_lh else "none"
    else:
        lh_desc = "yes" if include_lh else "no"
    print(f"FT mode: {ft_mode}, Large head: {lh_desc}")

    for g in groups:
        if g == "clean":
            print_clean_table(datasets, args.metric, rank_col=args.rank,
                              include_lh=include_lh, ft_mode=ft_mode)
        elif g in CONDITION_GROUPS:
            print_clean_table(datasets, args.metric, rank_col=args.rank, skip_train=True,
                              include_lh=include_lh, ft_mode=ft_mode)
            print_corruption_table(g, datasets, args.metric, rank_col=args.rank,
                                   include_lh=include_lh, ft_mode=ft_mode,
                                   relative=args.relative,
                                   rank_by_relative=args.rank_by_relative)
        else:
            print(f"\nUnknown group: {g}")


if __name__ == "__main__":
    main()
