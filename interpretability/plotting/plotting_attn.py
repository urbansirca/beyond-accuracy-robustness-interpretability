import os
import numpy as np
import matplotlib.pyplot as plt
import mne

from data.loaders import slug_for

from interpretability.plotting._plotting_common import (
    mne_info as _mne_info,
    valid_channel_indices as _valid_channel_indices,
    savefig as _savefig,
)


def _default_block_subset(n_blocks, max_cols=4):
    step = max(1, n_blocks // max_cols)
    subset = list(range(0, n_blocks, step))[:max_cols]
    if (n_blocks - 1) not in subset:
        subset.append(n_blocks - 1)
    return subset


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Block × head grid of attention heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_heads_grid(
    mean_attn_by_block,
    model_name,
    benchmark_name,
    block_subset=None,
    out_dir=None,
    tag="",
    title_prefix="Attention",
):
    """Plot a grid of mean attention heatmaps: rows = selected blocks, cols = heads.

    Parameters
    ----------
    mean_attn_by_block : dict[int, np.ndarray]
        ``{block_idx: mean_attn}`` where ``mean_attn`` has shape ``(H, N, N)``.
        Produced by ``AttentionAccumulator.get_mean_attn()``.
    block_subset : list[int] | None
        Which blocks to show.  Defaults to up to 6 evenly-spaced blocks.
    """
    if not mean_attn_by_block:
        print("[plot_heads_grid] No data. Skipping.")
        return

    n_blocks = max(mean_attn_by_block) + 1
    if block_subset is None:
        block_subset = _default_block_subset(n_blocks, max_cols=6)

    sample_block = next(iter(mean_attn_by_block.values()))
    n_heads = sample_block.shape[0]

    _out_dir = out_dir or f"plots/attention/{model_name}/{slug_for(benchmark_name)}"
    bench_slug = slug_for(benchmark_name)
    suffix = f"_{tag}" if tag else ""
    out_path = os.path.join(_out_dir, f"{bench_slug}_heads_grid{suffix}.png")

    n_rows = len(block_subset)
    fig, axes = plt.subplots(
        n_rows, n_heads,
        figsize=(2.2 * n_heads, 2.2 * n_rows),
        squeeze=False,
    )

    for row, block_idx in enumerate(block_subset):
        mean_attn = mean_attn_by_block.get(block_idx)
        for h in range(n_heads):
            ax = axes[row, h]
            if mean_attn is None:
                ax.axis("off")
                continue
            ax.imshow(mean_attn[h], cmap="hot", vmin=0, aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Head {h}", fontsize=8)
        axes[row, 0].set_ylabel(f"Block {block_idx}", fontsize=8)

    fig.suptitle(f"{title_prefix} — {model_name}, {benchmark_name}", fontsize=11)
    fig.tight_layout()
    _savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Shared topomap grid builder
# ─────────────────────────────────────────────────────────────────────────────

def _plot_topomap_grid(
    scores_by_block,
    ch_names,
    y_test,
    classes,
    class_names,
    model_name,
    benchmark_name,
    block_subset,
    out_path,
    suptitle,
):
    """rows = classes, cols = blocks.  scores_by_block: {block: (N, C)}."""
    info = _mne_info(ch_names)
    valid = _valid_channel_indices(info)
    if not valid:
        print(f"[_plot_topomap_grid] No valid channel positions. Skipping {out_path}.")
        return

    # The model may process fewer channels than ch_names.
    first_data = next(
        (scores_by_block[b] for b in block_subset if scores_by_block.get(b) is not None),
        None,
    )
    if first_data is not None:
        n_chans_data = first_data.shape[1]
        valid = [i for i in valid if i < n_chans_data]
    if not valid:
        print(f"[_plot_topomap_grid] No valid channels within data bounds. Skipping {out_path}.")
        return
    info_plot = mne.pick_info(info, valid)

    n_rows = len(classes)
    n_cols = len(block_subset)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for col, block_idx in enumerate(block_subset):
        by_chan = scores_by_block.get(block_idx)  # (N, C)
        for row, cls in enumerate(classes):
            ax = axes[row, col]
            if col == 0:
                lbl = (class_names or {}).get(cls, str(cls)) if cls is not None else "all"
                ax.set_ylabel(lbl, fontsize=10)
            if row == 0:
                ax.set_title(f"Block {block_idx}", fontsize=9)

            if by_chan is None:
                ax.axis("off")
                continue

            if y_test is not None and cls is not None:
                mask = y_test == cls
                if mask.sum() == 0:
                    ax.axis("off")
                    continue
                data = by_chan[mask].mean(axis=0)
            else:
                data = by_chan.mean(axis=0)

            mne.viz.plot_topomap(
                data[valid], info_plot, axes=ax, show=False,
                cmap='RdBu_r', contours=0, sensors=False,
                extrapolate='head', image_interp='cubic', res=300,
                sphere=(0, 0, 0, 0.095),
            )

    fig.suptitle(f"{suptitle} — {model_name}, {benchmark_name}", fontsize=11)
    fig.tight_layout()
    _savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLS-token attention topomaps  (LaBraM / NeuroRVQ)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cls_topomap(
    cls_row_by_block,
    cls_col_by_block,
    ch_names,
    model_name,
    benchmark_name,
    block_subset=None,
    n_blocks=12,
    out_dir=None,
    tag="",
    y_test=None,
    class_names=None,
    all_classes=None,
):
    """Plot CLS-row and CLS-column attention as per-channel topomaps.

    CLS row    (``_cls_attends_to``)  : what CLS attends to.
        The CLS representation used for classification is built by weighting
        all tokens by this row — maps directly to "what the model looks at".

    CLS column (``_attends_to_cls``)  : what attends to CLS.
        Tokens whose queries align with CLS as a key.

    Both are pre-aggregated to ``(N_samples, C)`` by ``AttentionAccumulator``.

    Parameters
    ----------
    cls_row_by_block : dict[int, np.ndarray]
        ``{block_idx: (N_samples, C)}`` — CLS row scores.
        Produced by ``AttentionAccumulator.get_cls_row()``.
    cls_col_by_block : dict[int, np.ndarray]
        ``{block_idx: (N_samples, C)}`` — CLS column scores.
        Produced by ``AttentionAccumulator.get_cls_col()``.
    y_test : np.ndarray | None
        Class labels aligned with the ``N_samples`` axis.
    """
    if not cls_row_by_block and not cls_col_by_block:
        print("[plot_cls_topomap] No data. Skipping.")
        return

    if block_subset is None:
        block_subset = _default_block_subset(n_blocks)

    _out_dir = out_dir or f"plots/attention/{model_name}/{slug_for(benchmark_name)}"
    bench_slug = slug_for(benchmark_name)
    suffix = f"_{tag}" if tag else ""

    classes = (
        sorted(all_classes) if all_classes is not None
        else (sorted(np.unique(y_test)) if y_test is not None else [None])
    )

    _plot_topomap_grid(
        cls_row_by_block, ch_names, y_test, classes, class_names,
        model_name, benchmark_name, block_subset,
        out_path=os.path.join(_out_dir, f"{bench_slug}_cls_attends_to{suffix}.png"),
        suptitle="CLS attends to (row)",
    )
    _plot_topomap_grid(
        cls_col_by_block, ch_names, y_test, classes, class_names,
        model_name, benchmark_name, block_subset,
        out_path=os.path.join(_out_dir, f"{bench_slug}_attends_to_cls{suffix}.png"),
        suptitle="Attends to CLS (column)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Mean attention received per channel  (all transformer models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_channel_topomap(
    chan_scores_by_block,
    ch_names,
    model_name,
    benchmark_name,
    block_subset=None,
    n_blocks=12,
    out_dir=None,
    tag="",
    title_prefix="Attention received",
    y_test=None,
    class_names=None,
    all_classes=None,
):
    """Plot mean attention received per channel as a topomap.

    Parameters
    ----------
    chan_scores_by_block : dict[int, np.ndarray]
        ``{block_idx: (N_samples, C)}`` — per-sample channel scores.
        Produced by ``AttentionAccumulator.get_chan_scores()``.
    y_test : np.ndarray | None
        Class labels aligned with the ``N_samples`` axis.
    """
    if not chan_scores_by_block:
        print("[plot_channel_topomap] No data. Skipping.")
        return

    if block_subset is None:
        block_subset = _default_block_subset(n_blocks)

    _out_dir = out_dir or f"plots/attention/{model_name}/{slug_for(benchmark_name)}"
    bench_slug = slug_for(benchmark_name)
    suffix = f"_{tag}" if tag else ""

    classes = (
        sorted(all_classes) if all_classes is not None
        else (sorted(np.unique(y_test)) if y_test is not None else [None])
    )

    _plot_topomap_grid(
        chan_scores_by_block, ch_names, y_test, classes, class_names,
        model_name, benchmark_name, block_subset,
        out_path=os.path.join(_out_dir, f"{bench_slug}_channel_topomap{suffix}.png"),
        suptitle=title_prefix,
    )

