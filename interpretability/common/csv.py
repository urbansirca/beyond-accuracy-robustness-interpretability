"""CSV export for both attribution (LRP / IxG / GradCAM) and attention outputs."""

import os
import numpy as np
import pandas as pd

from data.loaders import slug_for


def _aug_tag(augmentation):
    """Return filename-safe augmentation tag, or empty string for clean data."""
    if augmentation is None:
        return ""
    return f"_{augmentation}".replace(" ", "_")


def attribution_csv_exists(
    model_name: str,
    benchmark_name: str,
    method: str,
    fold: int | str = "all",
    head_only: bool = False,
    large_head: bool = False,
    out_root: str = "results/attribution",
    augmentation: str | None = None,
    skip_tokenizer: bool = False,
) -> bool:
    """Check whether the channel CSV for this fold already exists."""
    ft_type = "head_only" if head_only else "full_ft"
    if large_head:
        ft_type += "_large_head"
    if skip_tokenizer:
        ft_type += "_skip_tokenizer"
    fold_tag = f"fold{fold}" if fold != "all" else "all_folds"
    aug = _aug_tag(augmentation)
    out_dir = os.path.join(out_root, method, model_name, slug_for(benchmark_name))
    fname = f"{model_name}_{slug_for(benchmark_name)}_{ft_type}_{fold_tag}{aug}_channel.csv"
    return os.path.exists(os.path.join(out_dir, fname))


def export_attribution_csvs(
    relevance_3d: np.ndarray,
    y: np.ndarray,
    preds: np.ndarray,
    conf: np.ndarray,
    ch_names: list,
    model_name: str,
    benchmark_name: str,
    method: str,
    fold: int | str = "all",
    confidence_quantile: float = 0.75,
    sfreq: float = 200.0,
    patch_size: int = 200,
    head_only: bool = False,
    large_head: bool = False,
    conservative: bool = False,
    out_root: str = "results/attribution",
    augmentation: str | None = None,
    skip_tokenizer: bool = False,
):
    """
    Export attribution relevance to CSV files.

    Parameters
    ----------
    relevance_3d : (N, C, A) array — per-sample, per-channel, per-time-patch relevance
    y : (N,) ground truth labels
    preds : (N,) predicted labels
    conf : (N,) confidence scores
    ch_names : list of channel names (length C)
    model_name, benchmark_name, method : string identifiers
    fold : int or "all" — fold index, or "all" for aggregated across folds
    confidence_quantile : filter to top-k% most confident correct predictions
    sfreq : sampling frequency (Hz)
    patch_size : samples per time patch
    head_only, large_head, conservative : experiment flags
    out_root : base output directory
    """
    # ── Filter: high-confidence correct predictions ──
    correct_mask = preds == y
    if correct_mask.sum() == 0:
        print(f"  [export] No correct predictions (fold={fold}) — skipping CSV export.")
        return

    conf_threshold = np.quantile(conf[correct_mask], confidence_quantile)
    mask = correct_mask & (conf >= conf_threshold)
    if mask.sum() == 0:
        print(f"  [export] No high-confidence correct predictions (fold={fold}) — skipping.")
        return

    rel = relevance_3d[mask]       # (N_filt, C, A)
    labels = y[mask]               # (N_filt,)
    n_filt = int(mask.sum())

    # ── Metadata columns ──
    ft_type = "head_only" if head_only else "full_ft"
    if large_head:
        ft_type += "_large_head"
    if skip_tokenizer:
        ft_type += "_skip_tokenizer"
    meta = {
        "model": model_name,
        "benchmark": benchmark_name,
        "method": method,
        "ft_type": ft_type,
        "fold": fold,
        "augmentation": augmentation or "clean",
        "conservative": conservative,
        "n_samples": n_filt,
        "confidence_quantile": confidence_quantile,
    }

    fold_tag = f"fold{fold}" if fold != "all" else "all_folds"
    aug = _aug_tag(augmentation)
    out_dir = os.path.join(out_root, method, model_name, slug_for(benchmark_name))
    os.makedirs(out_dir, exist_ok=True)

    all_classes = sorted(np.unique(labels))
    patch_duration = patch_size / sfreq  # seconds per patch

    # ── 1. Channel × Time CSV ──
    n_patches = rel.shape[2]
    rows = []
    for cls in all_classes:
        cls_rel = np.abs(rel[labels == cls]).mean(axis=0)  # (C, A)
        for ci, ch in enumerate(ch_names):
            for ai in range(n_patches):
                t_start = ai * patch_duration
                t_end = (ai + 1) * patch_duration
                rows.append({
                    **meta,
                    "class": int(cls),
                    "channel": ch,
                    "time_patch": ai,
                    "time_start_s": round(t_start, 3),
                    "time_end_s": round(t_end, 3),
                    "relevance": float(cls_rel[ci, ai]),
                })

    df_ct = pd.DataFrame(rows)
    fname_ct = f"{model_name}_{slug_for(benchmark_name)}_{ft_type}_{fold_tag}{aug}_channel_time.csv"
    path_ct = os.path.join(out_dir, fname_ct)
    df_ct.to_csv(path_ct, index=False)
    print(f"  Saved channel×time CSV: {path_ct}  ({len(df_ct)} rows)")

    # ── 2. Channel CSV (time-averaged) ──
    rows = []
    for cls in all_classes:
        cls_rel = np.abs(rel[labels == cls]).mean(axis=(0, 2))  # (C,)
        for ci, ch in enumerate(ch_names):
            rows.append({
                **meta,
                "class": int(cls),
                "channel": ch,
                "relevance": float(cls_rel[ci]),
            })

    df_ch = pd.DataFrame(rows)
    fname_ch = f"{model_name}_{slug_for(benchmark_name)}_{ft_type}_{fold_tag}{aug}_channel.csv"
    path_ch = os.path.join(out_dir, fname_ch)
    df_ch.to_csv(path_ch, index=False)
    print(f"  Saved channel CSV: {path_ch}  ({len(df_ch)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export (attention-shaped — bespoke, not via export_attribution_csvs)
# ─────────────────────────────────────────────────────────────────────────────


def _ft_type(large_head, train_head_only):
    ft = "head_only" if train_head_only else "full_ft"
    if large_head:
        ft += "_large_head"
    return ft


def attention_csv_exists(out_root, model_name, benchmark_name, fold="all",
                          large_head=False, train_head_only=False,
                          augmentation=None):
    fold_tag = f"fold{fold}" if fold != "all" else "all_folds"
    fname = (f"{model_name}_{slug_for(benchmark_name)}_{_ft_type(large_head, train_head_only)}"
             f"_{fold_tag}{_aug_tag(augmentation)}_attention.csv")
    out_dir = os.path.join(out_root, "attention", model_name, slug_for(benchmark_name))
    return os.path.exists(os.path.join(out_dir, fname))


def export_attention_csvs(out_root,chan_scores, y_combined, plot_ch_names,
                            model_name, benchmark_name, all_classes,
                            large_head=False, train_head_only=False,
                            cls_row=None, cls_col=None,
                            fold="all", augmentation=None):
    """Export per-block, per-channel attention scores to one wide CSV."""
    fold_tag = f"fold{fold}" if fold != "all" else "all_folds"
    os.makedirs(os.path.join(out_root, "attention", model_name, slug_for(benchmark_name)), exist_ok=True)

    meta = {
        "model": model_name,
        "benchmark": benchmark_name,
        "method": "attention",
        "ft_type": _ft_type(large_head, train_head_only),
        "fold": fold,
        "augmentation": augmentation or "clean",
    }
    rows = []

    if chan_scores is not None:
        for branch, block_dict in sorted(chan_scores.items()):
            for block, scores in sorted(block_dict.items()):
                for cls in all_classes:
                    cls_mask = y_combined == cls
                    if cls_mask.sum() == 0:
                        continue
                    cls_mean = scores[cls_mask].mean(axis=0)
                    for ci, ch in enumerate(plot_ch_names):
                        rows.append({**meta, "class": int(cls), "block": block,
                                     "branch": branch, "channel": ch,
                                     "score_type": "channel_received",
                                     "value": float(cls_mean[ci])})

    for tag, by_branch in (("cls_attends_to", cls_row), ("attends_to_cls", cls_col)):
        if by_branch is None:
            continue
        for branch, block_dict in sorted(by_branch.items()):
            for block, scores in sorted(block_dict.items()):
                for cls in all_classes:
                    cls_mask = y_combined == cls
                    if cls_mask.sum() == 0:
                        continue
                    cls_mean = scores[cls_mask].mean(axis=0)
                    for ci, ch in enumerate(plot_ch_names):
                        rows.append({**meta, "class": int(cls), "block": block,
                                     "branch": branch, "channel": ch,
                                     "score_type": tag,
                                     "value": float(cls_mean[ci])})

    if rows:
        df = pd.DataFrame(rows)
        fname = (f"{model_name}_{slug_for(benchmark_name)}_{_ft_type(large_head, train_head_only)}"
                 f"_{fold_tag}{_aug_tag(augmentation)}_attention.csv")
        path = os.path.join(out_root, "attention", model_name, slug_for(benchmark_name), fname)
        df.to_csv(path, index=False)
        print(f"  Saved attention CSV: {path}  ({len(df)} rows)")
