import os

import matplotlib.pyplot as plt
import mne
import numpy as np

from data.loaders import slug_for

from interpretability.plotting._plotting_common import (
    mne_info, valid_channel_indices, savefig,
)


def plot_topomaps(relevance: np.ndarray, y_test: np.ndarray, ch_names, class_names: dict,
                  model_name: str, benchmark_name: str, suffix: str = 'gt',
                  conservative: bool = False, all_classes=None,
                  out_dir: str = None, fig_tag: str = None,
                  title_prefix: str = 'LRP relevance'):
    """Plot mean |relevance| topomap per class and save to out_path.

    Parameters
    ----------
    all_classes : array-like, optional
        Full set of class labels to show. Useful when a filtered subset may be
        missing some classes — those panels are shown as blank with a note.
        If None, classes are inferred from y_test.
    out_dir : str, optional
        Directory for output files.  Defaults to plots/lrp/{model}/{benchmark}.
    fig_tag : str, optional
        Tag appended to the filename in place of the conservative/standard suffix.
        If None, uses 'conservative' or 'standard' based on the conservative flag.
    title_prefix : str, optional
        Prefix for the figure suptitle.  Defaults to 'LRP relevance'.
    """
    bench_slug = slug_for(benchmark_name)
    method_tag = fig_tag if fig_tag is not None else ("conservative" if conservative else "standard")
    _out_dir = out_dir if out_dir is not None else f'plots/lrp/{model_name}/{bench_slug}'
    out_path = f'{_out_dir}/{bench_slug}_{suffix}_{method_tag}.png'

    classes = sorted(all_classes) if all_classes is not None else sorted(np.unique(y_test))
    n_classes = len(classes)

    info = mne_info(ch_names)
    valid_inds = valid_channel_indices(info)
    if not valid_inds:
        print(f"Warning: No channels with valid positions found for {benchmark_name}. Skipping plot.")
        return
    info_plot = mne.pick_info(info, valid_inds)

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        mask = y_test == cls
        title = class_names.get(cls, str(cls)) if class_names else str(cls)
        if mask.sum() == 0:
            ax.set_visible(False)
            ax.set_title(f'{title}\n(no samples)')
            ax.set_visible(True)
            ax.axis('off')
            ax.text(0.5, 0.5, 'no samples', ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            ax.set_title(title)
            continue
        # Always 3D (N, C, A) — average over time then samples.
        rel_topo = np.abs(relevance[mask]).mean(axis=2).mean(axis=0)
        rel_topo_plot = rel_topo[valid_inds]
        mne.viz.plot_topomap(rel_topo_plot, info_plot, axes=ax, show=False,
                             cmap='RdBu_r', contours=0, sensors=False,
                             extrapolate='head', image_interp='cubic', res=300,
                             sphere=(0, 0, 0, 0.095))
        ax.set_title(f'{title}\n(n={mask.sum()})')

    suffix_label = {'gt': 'GT label', 'gt_correct': 'GT label (correct)', 'gt_incorrect': 'GT label (incorrect)', 'predicted': 'predicted label'}.get(suffix, suffix)
    fig.suptitle(f'{title_prefix} — {model_name}, {benchmark_name} [{suffix_label}]', fontsize=13)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_topomaps_per_subject(
    relevance: np.ndarray, y_test: np.ndarray, subject_ids: np.ndarray,
    ch_names, class_names: dict, model_name: str, benchmark_name: str,
    suffix: str = 'gt', conservative: bool = False, all_classes=None,
    out_dir: str = None, fig_tag: str = None, title_prefix: str = 'LRP relevance',
):
    """Plot one topomap grid per subject (classes as columns) and save."""
    bench_slug = slug_for(benchmark_name)
    method_tag = fig_tag if fig_tag is not None else ("conservative" if conservative else "standard")
    _out_dir = out_dir if out_dir is not None else f'plots/lrp/{model_name}/{bench_slug}/per_subject'
    os.makedirs(_out_dir, exist_ok=True)

    classes = sorted(all_classes) if all_classes is not None else sorted(np.unique(y_test))
    n_classes = len(classes)

    info = mne_info(ch_names)
    valid_inds = valid_channel_indices(info)
    if not valid_inds:
        print(f"Warning: No channels with valid positions. Skipping per-subject plots.")
        return
    info_plot = mne.pick_info(info, valid_inds)

    for sbj in np.unique(subject_ids):
        sbj_mask = subject_ids == sbj
        rel_sbj = relevance[sbj_mask]
        y_sbj = y_test[sbj_mask]

        fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
        if n_classes == 1:
            axes = [axes]

        for ax, cls in zip(axes, classes):
            cls_mask = y_sbj == cls
            if cls_mask.sum() == 0:
                ax.set_visible(False)
                continue
            rel_topo = np.abs(rel_sbj[cls_mask]).mean(axis=2).mean(axis=0)
            mne.viz.plot_topomap(rel_topo[valid_inds], info_plot, axes=ax, show=False,
                                 cmap='RdBu_r', contours=0, sensors=False,
                                 extrapolate='head', image_interp='cubic', res=300,
                                 sphere=(0, 0, 0, 0.095))
            title = class_names.get(cls, str(cls)) if class_names else str(cls)
            ax.set_title(title)

        suffix_label = {'gt': 'GT label', 'gt_correct': 'GT (correct)', 'gt_incorrect': 'GT (incorrect)', 'predicted': 'predicted'}.get(suffix, suffix)
        fig.suptitle(f'Subject {sbj} — {model_name}, {benchmark_name} [{suffix_label}]', fontsize=13)
        fig.tight_layout()
        out_path = os.path.join(_out_dir, f'{bench_slug}_sbj{sbj}_{suffix}_{method_tag}.png')
        savefig(fig, out_path)


def plot_topomaps_time(
    relevance_3d: np.ndarray,
    y_test: np.ndarray,
    ch_names,
    class_names: dict,
    model_name: str,
    benchmark_name: str,
    suffix: str = 'gt',
    all_classes=None,
    sfreq: float = 200.0,
    patch_size: int = 200,
    out_dir: str = None,
    fig_tag: str = None,
    title_prefix: str = 'LRP relevance',
):
    """Plot time-resolved topomaps: rows = classes, columns = time patches.

    Parameters
    ----------
    relevance_3d : np.ndarray, shape (N, C, A)
        Per-sample, per-channel, per-time-patch relevance scores.
        A = number of time patches (e.g. 4 for a 4-second trial at 200 Hz
        with patch_size=200).
    y_test : np.ndarray, shape (N,)
        Class labels aligned with relevance_3d.
    sfreq : float
        Sampling frequency in Hz, used to compute time-axis labels.
    patch_size : int
        Number of samples per time patch, used to compute time-axis labels.
    out_dir, fig_tag, title_prefix
        Same semantics as plot_topomaps — see that function's docstring.
    """

    if model_name == "BrainOmni":
        sfreq = 256
        patch_size = 512

    n_time = relevance_3d.shape[2]
    bench_slug = slug_for(benchmark_name)
    method_tag = fig_tag if fig_tag is not None else "standard"
    _out_dir = out_dir if out_dir is not None else f'plots/lrp/{model_name}/{bench_slug}'
    out_path = f'{_out_dir}/{bench_slug}_{suffix}_{method_tag}_temporal.png'

    classes = sorted(all_classes) if all_classes is not None else sorted(np.unique(y_test))
    n_classes = len(classes)

    info = mne_info(ch_names, sfreq=sfreq)
    valid_inds = valid_channel_indices(info)
    if not valid_inds:
        print(f"Warning: No channels with valid positions found for {benchmark_name}. Skipping temporal plot.")
        return
    info_plot = mne.pick_info(info, valid_inds)

    # Time-patch axis labels: "0.0–1.0 s", "1.0–2.0 s", …
    patch_dur = patch_size / sfreq
    time_labels = [
        f"{i * patch_dur:.1f}–{(i + 1) * patch_dur:.1f} s"
        for i in range(n_time)
    ]

    # Grid: rows = classes, columns = time patches
    fig, axes = plt.subplots(
        n_classes, n_time,
        figsize=(2.8 * n_time, 3.2 * n_classes),
        squeeze=False,
    )

    for row, cls in enumerate(classes):
        mask = y_test == cls
        cls_label = class_names.get(cls, str(cls)) if class_names else str(cls)

        # Row label on leftmost axis
        axes[row, 0].set_ylabel(cls_label, fontsize=11, labelpad=6)

        for col in range(n_time):
            ax = axes[row, col]

            # Column header on top row only
            if row == 0:
                ax.set_title(time_labels[col], fontsize=9)

            if mask.sum() == 0:
                ax.axis('off')
                ax.text(0.5, 0.5, 'no samples', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='grey')
                continue

            # Mean over samples for this class × time patch
            rel_topo = relevance_3d[mask, :, col].mean(axis=0)  # (C,)
            rel_topo_plot = rel_topo[valid_inds]
            mne.viz.plot_topomap(rel_topo_plot, info_plot, axes=ax, show=False,
                                 cmap='RdBu_r', contours=0, sensors=False,
                                 extrapolate='head', image_interp='cubic', res=300,
                                 sphere=(0, 0, 0, 0.095))

            # Sample count on bottom-right corner of each cell
            ax.text(0.97, 0.03, f'n={mask.sum()}', ha='right', va='bottom',
                    transform=ax.transAxes, fontsize=7, color='dimgrey')

    suffix_label = {
        'gt': 'GT label', 'gt_correct': 'GT label (correct)',
        'gt_incorrect': 'GT label (incorrect)', 'predicted': 'predicted label',
    }.get(suffix, suffix)
    fig.suptitle(
        f'{title_prefix} (temporal) — {model_name}, {benchmark_name} [{suffix_label}]',
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, out_path)
