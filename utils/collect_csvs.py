"""
Collect and concatenate CSVs within each result group.

Each group contains CSVs with the same schema (columns). Files within a group
are concatenated into a single CSV for easy plotting / analysis.

Groups
------
  results_train_folds     — results/results_train/**/*_folds.csv
  results_train_epochs    — results/results_train/**/*_epochs.csv
  results_eval_folds      — results/results_eval/**/*_folds.csv
  dropped_channels        — results/dropped-channels-eval/**/*_folds.csv
  attribution/<method>/*  — one merged CSV per (method, csv_type) pair
                            e.g. gradcam/channel, gradcam/channel_time
  attention               — results/attribution/attention/*_attention.csv
  block_exit_folds        — results/results_block_exit/**/*_folds.csv
  block_exit_epochs       — results/results_block_exit/**/*_epochs.csv
  probing                 — results/probing/*.csv

Output
------
  results/collected/<group_name>.csv

Usage
-----
  python collect_csvs.py                   # collect everything
  python collect_csvs.py --groups eval     # only results_eval
  python collect_csvs.py --dry_run         # show what would be collected
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

RESULTS_ROOT = "results"
OUT_DIR = os.path.join(RESULTS_ROOT, "collected")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_csvs(pattern):
    """Return sorted list of CSV paths matching a glob pattern."""
    return sorted(glob.glob(pattern, recursive=True))


LARGE_HEAD_MODELS  = ("REVE",)                 # keep only large_head rows
SMALL_HEAD_MODELS  = ("NeuroRVQ", "LaBraM")    # keep only small_head rows


def _rewrite_backbone_equivalent_heads(df):
    """Relabel REVE `head_only` rows as `head_only_large_head`.

    Rationale: head-only fine-tuning only trains the classifier; the backbone
    stays at the pretrained REVE weights, which are identical across head
    variants. So attention / LRP / probing features extracted from a REVE
    `head_only` run are the same as what a `head_only_large_head` run would
    produce, and can be reused as the canonical head variant.

    Only the `ft_type` column is rewritten (attribution/attention CSVs). The
    `large_head` column in probing CSVs is already handled upstream by the
    pretrained-row duplication flow.
    """
    if "model" not in df.columns or "ft_type" not in df.columns:
        return df
    mask = (df["model"] == "REVE") & (df["ft_type"].astype(str) == "head_only")
    n = int(mask.sum())
    if n:
        df = df.copy()
        df.loc[mask, "ft_type"] = "head_only_large_head"
        print(f"  [backbone-equivalent] relabeled {n} REVE head_only rows → head_only_large_head")
    return df


def _filter_canonical_heads(df):
    """Keep only the 'canonical' head variant per model.

    - REVE            → large_head=True
    - NeuroRVQ/LaBraM → large_head=False
    - everything else → unchanged

    Probing CSVs carry a bool-ish `large_head` column; attribution/attention
    CSVs encode head type inside `ft_type` (`full_ft` vs `full_ft_large_head`).
    If neither column exists (e.g. raw train/eval CSVs without head metadata)
    the frame is returned unchanged.
    """
    if "model" not in df.columns:
        return df

    # Reuse head-only (pretrained-backbone) rows as the large-head equivalent
    # for REVE before applying the canonical-heads drop filter.
    df = _rewrite_backbone_equivalent_heads(df)

    if "large_head" in df.columns:
        lh = df["large_head"].astype(str).isin(["True", "true", "1"])
    elif "ft_type" in df.columns:
        lh = df["ft_type"].astype(str).str.contains("large_head", na=False)
    else:
        return df

    drop = (df["model"].isin(LARGE_HEAD_MODELS) & ~lh) | \
           (df["model"].isin(SMALL_HEAD_MODELS) &  lh)
    return df[~drop].copy()


def _concat_and_save(csv_paths, out_path, dry_run=False, canonical_heads=False,
                     dedupe_on=None):
    """Read, concatenate, and save CSVs. Returns number of rows or 0.

    If ``dedupe_on`` is provided (list of column names), duplicate rows on that
    key are dropped after the merge (keep='last').
    """
    if not csv_paths:
        return 0

    if dry_run:
        print(f"  Would merge {len(csv_paths)} files → {out_path}")
        for p in csv_paths[:5]:
            print(f"    {p}")
        if len(csv_paths) > 5:
            print(f"    ... and {len(csv_paths) - 5} more")
        return -1

    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if len(df) > 0:
                if "augmentation" not in df.columns:
                    df["augmentation"] = "clean"
                dfs.append(df)
        except Exception as e:
            print(f"  WARNING: skipping {p}: {e}")

    if not dfs:
        print(f"  No data found — skipping {out_path}")
        return 0

    merged = pd.concat(dfs, ignore_index=True)
    # Drop header-in-body garbage rows (literal column names appearing as values)
    if "model" in merged.columns:
        merged = merged[merged["model"] != "model"].copy()
    if dedupe_on:
        keys = [k for k in dedupe_on if k in merged.columns]
        if keys:
            before = len(merged)
            merged = merged.drop_duplicates(subset=keys, keep="last")
            dropped = before - len(merged)
            if dropped:
                print(f"  [dedupe] dropped {dropped} duplicate rows on {keys}")
    if canonical_heads:
        before = len(merged)
        merged = _filter_canonical_heads(merged)
        dropped = before - len(merged)
        if dropped:
            print(f"  [canonical_heads] dropped {dropped} non-canonical head rows "
                  f"(REVE small-head, NeuroRVQ/LaBraM large-head)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  Saved {out_path}  ({len(merged)} rows from {len(dfs)} files)")
    return len(merged)


# ── Collectors ───────────────────────────────────────────────────────────────

def collect_train(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect results_train folds and epochs CSVs."""
    base = os.path.join(RESULTS_ROOT, "results_train")
    if not os.path.isdir(base):
        print(f"  {base} not found — skipping")
        return

    folds = _find_csvs(os.path.join(base, "**", "*_folds.csv"))
    epochs = _find_csvs(os.path.join(base, "**", "*_epochs.csv"))

    _concat_and_save(folds, os.path.join(out_dir, "results_train_folds.csv"), dry_run, canonical_heads)
    _concat_and_save(epochs, os.path.join(out_dir, "results_train_epochs.csv"), dry_run, canonical_heads)


def collect_eval(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect results_eval folds CSVs."""
    base = os.path.join(RESULTS_ROOT, "results_eval")
    if not os.path.isdir(base):
        print(f"  {base} not found — skipping")
        return

    folds = _find_csvs(os.path.join(base, "**", "*_folds.csv"))
    _concat_and_save(folds, os.path.join(out_dir, "results_eval_folds.csv"), dry_run, canonical_heads)


def collect_dropped_channels(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect dropped-channels-eval folds CSVs.

    Layout mirrors results_eval (model/benchmark/finetune/*_folds.csv) but the
    inputs here have channels physically removed from X (vs zero-filled in the
    canonical results_eval). Useful for drop-vs-zero comparisons.
    """
    base = os.path.join(RESULTS_ROOT, "dropped-channels-eval")
    if not os.path.isdir(base):
        print(f"  {base} not found — skipping")
        return

    folds = _find_csvs(os.path.join(base, "**", "*_folds.csv"))
    _concat_and_save(folds, os.path.join(out_dir, "dropped_channels_eval_folds.csv"),
                     dry_run, canonical_heads)


def collect_attribution(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect attribution CSVs, one merged file per (method, csv_type)."""
    base = os.path.join(RESULTS_ROOT, "attribution")
    if not os.path.isdir(base):
        print(f"  {base} not found — skipping")
        return

    for method_dir in sorted(os.listdir(base)):
        method_path = os.path.join(base, method_dir)
        if not os.path.isdir(method_path):
            continue

        if method_dir == "attention":
            # Attention has its own format
            continue

        # Discover distinct CSV types by suffix: *_channel.csv, *_channel_time.csv, *_channel_freq.csv
        # Recursive: CSVs live under {method}/{model}/{benchmark}/*.csv (old flat layout also matched).
        all_csvs = _find_csvs(os.path.join(method_path, "**", "*.csv"))
        # Group by the final suffix: channel, channel_time, channel_freq
        # Filename: {model}_{bench}_{ft}_{fold}[_{aug}]_{csv_type}.csv
        type_groups = {}
        for p in all_csvs:
            fname = os.path.basename(p)
            m = re.search(r'_(channel(?:_time|_freq)?|other)\.csv$', fname)
            if m:
                csv_type = m.group(1)
                type_groups.setdefault(csv_type, []).append(p)
            else:
                type_groups.setdefault("other", []).append(p)

        for csv_type, paths in sorted(type_groups.items()):
            out_name = f"attribution_{method_dir}_{csv_type}.csv"
            _concat_and_save(paths, os.path.join(out_dir, out_name), dry_run, canonical_heads)


def collect_attention(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect attention attribution CSVs."""
    attn_dir = os.path.join(RESULTS_ROOT, "attribution", "attention")
    if not os.path.isdir(attn_dir):
        print(f"  {attn_dir} not found — skipping")
        return

    csvs = _find_csvs(os.path.join(attn_dir, "**", "*_attention.csv"))
    _concat_and_save(csvs, os.path.join(out_dir, "attribution_attention.csv"), dry_run, canonical_heads)


def collect_block_exit(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect block exit folds and epochs CSVs."""
    base = os.path.join(RESULTS_ROOT, "results_block_exit")
    if not os.path.isdir(base):
        print(f"  {base} not found — skipping")
        return

    folds = _find_csvs(os.path.join(base, "**", "*_folds.csv"))
    epochs = _find_csvs(os.path.join(base, "**", "*_epochs.csv"))

    # Also include clean results from results_eval for the same models
    eval_base = os.path.join(RESULTS_ROOT, "results_eval")
    if os.path.isdir(eval_base):
        # Find models that have block exit results
        block_exit_models = set()
        for p in folds:
            # Path: results/results_block_exit/<model>/...
            parts = os.path.relpath(p, base).split(os.sep)
            if parts:
                block_exit_models.add(parts[0])

        for model in block_exit_models:
            model_eval = os.path.join(eval_base, model)
            if os.path.isdir(model_eval):
                clean_folds = _find_csvs(os.path.join(model_eval, "**", "train-clean_test-clean_folds.csv"))
                folds.extend(clean_folds)

    _concat_and_save(folds, os.path.join(out_dir, "block_exit_folds.csv"), dry_run, canonical_heads)
    _concat_and_save(epochs, os.path.join(out_dir, "block_exit_epochs.csv"), dry_run, canonical_heads)


PROBING_POOLING_DIRS = ("concat", "cls", "mean")


def collect_probing(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect probing CSVs from the per-pooling/per-model layout.

    Canonical layout: ``results/probing/{pooling}/{model}.csv`` where
    ``pooling`` is one of: ``concat``, ``cls``, ``mean``.  Other subfolders
    (``old_results``, ``figures``, etc.) are ignored by design.
    """
    base = os.path.join(RESULTS_ROOT, "probing")
    if not os.path.isdir(base):
        print(f"  {base} not found — skipping")
        return

    csvs = []
    for pooling in PROBING_POOLING_DIRS:
        d = os.path.join(base, pooling)
        if os.path.isdir(d):
            csvs.extend(_find_csvs(os.path.join(d, "*.csv")))

    _concat_and_save(
        csvs, os.path.join(out_dir, "probing.csv"), dry_run, canonical_heads,
        dedupe_on=["model","benchmark","ckpt_type","pooling","large_head","fold","block_idx"],
    )


def collect_head_ablations(dry_run=False, out_dir=OUT_DIR, canonical_heads=False):
    """Collect clean bacc_test_best for head ablation comparison.

    For REVE, NeuroRVQ, and LaBraM: small head vs large head,
    head_only vs full_finetuned.  Prefers results_eval, falls back
    to results_train.  Excludes Sleep_EDF.

    The ``canonical_heads`` flag is deliberately ignored here — this group
    is the one place where both head variants must be compared.
    """
    del canonical_heads  # unused by design
    import csv as csv_mod

    models = ["REVE", "NeuroRVQ", "LaBraM"]
    ft_types = ["head_only", "head_only_large_head",
                "full_finetuned", "full_finetuned_large_head"]
    eval_dir = os.path.join(RESULTS_ROOT, "results_eval")
    train_dir = os.path.join(RESULTS_ROOT, "results_train")
    metric = "bacc_test_best"
    keep_datasets = {"physionet_eyes", "ku_mi", "ku_erp", "high_gamma"}

    datasets = sorted(keep_datasets)

    rows = []
    for model in models:
        for dataset in datasets:
            for ft_type in ft_types:
                # Try eval first, fall back to train
                source = None
                folds = []
                for src_label, src_dir in [("eval", eval_dir), ("train", train_dir)]:
                    path = os.path.join(src_dir, model, dataset, ft_type,
                                        "train-clean_test-clean_folds.csv")
                    if not os.path.isfile(path):
                        continue
                    try:
                        with open(path) as f:
                            csv_rows = list(csv_mod.DictReader(f))
                        if csv_rows and metric in csv_rows[0]:
                            folds = [(int(r["fold"]), float(r[metric]))
                                     for r in csv_rows if r.get(metric)]
                            if folds:
                                source = src_label
                                break
                    except Exception:
                        continue

                for fold, bacc in sorted(folds):
                    rows.append({
                        "model": model,
                        "benchmark": dataset.replace("_", " "),
                        "ft_type": ft_type,
                        "fold": fold,
                        "bacc_test_best": bacc,
                        "source": source,
                    })

    if dry_run:
        print(f"  Would write {len(rows)} rows → head_ablations.csv")
        return

    if not rows:
        print("  No head ablation data found — skipping")
        return

    out_path = os.path.join(out_dir, "head_ablations.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["model", "benchmark", "ft_type", "fold", "bacc_test_best", "source"]
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {out_path}  ({len(rows)} rows)")


# ── Main ─────────────────────────────────────────────────────────────────────

GROUP_MAP = {
    "train":              collect_train,
    "eval":               collect_eval,
    "dropped_channels":   collect_dropped_channels,
    "attribution":        collect_attribution,
    "attention":          collect_attention,
    "block_exit":         collect_block_exit,
    "probing":            collect_probing,
    "head_ablations":     collect_head_ablations,
}


def main():
    parser = argparse.ArgumentParser(description="Collect and merge CSVs per result group.")
    parser.add_argument("--groups", nargs="+", default=None,
                        choices=list(GROUP_MAP.keys()),
                        help="Which groups to collect (default: all).")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR}).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be collected without writing files.")
    parser.add_argument("--canonical_heads", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Keep only the canonical head variant per model in merged outputs: "
                             "REVE large_head=True, NeuroRVQ/LaBraM large_head=False. "
                             "Other models are unaffected. head_ablations ignores this flag. "
                             "Pass --no-canonical_heads to disable (include all head variants).")
    args = parser.parse_args()

    out_dir = args.out_dir
    groups = args.groups or list(GROUP_MAP.keys())

    for group in groups:
        print(f"\n{'='*60}")
        print(f"Collecting: {group}")
        print(f"{'='*60}")
        GROUP_MAP[group](dry_run=args.dry_run, out_dir=out_dir,
                         canonical_heads=args.canonical_heads)

    print(f"\nDone. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
