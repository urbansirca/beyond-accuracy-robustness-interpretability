"""The topomap cascade + per-fold stats shared by the attribution methods.

The actual matplotlib functions live in ``interpretability.plotting``; this
module is the per-method dispatch over them.
"""

import numpy as np

from interpretability.plotting.plotting_attr import (
    plot_topomaps,
    plot_topomaps_per_subject,
    plot_topomaps_time,
)
from interpretability.common.data import BENCHMARK_CLASSES


def plot_by_subset(
    relevance_3d, y_test, preds, conf, sbj_ids,
    plot_ch_names, all_classes, model_name, benchmark_name,
    target, confidence_quantile,
    out_standard, out_temporal,
    fig_tag, title_prefix,
):
    """Render the full topomap cascade for one method.

    ``target`` ∈ {'gt', 'predicted'}.  Renders:
      - grand average (gt or predicted)
      - correct / incorrect (gt only)
      - high-confidence (top quantile)
      - per-subject grid
    """
    class_names = BENCHMARK_CLASSES.get(benchmark_name, None)
    correct_mask = preds == y_test

    kw = dict(
        all_classes=all_classes,
        out_dir=out_standard, fig_tag=fig_tag, title_prefix=title_prefix,
    )
    time_kw = dict(
        all_classes=all_classes,
        sfreq=200.0, patch_size=200,
        out_dir=out_temporal, fig_tag=fig_tag, title_prefix=title_prefix,
    )

    def _plot_both(rel3d, labels, suffix):
        plot_topomaps(rel3d, labels, plot_ch_names, class_names,
                      model_name, benchmark_name, suffix=suffix, **kw)
        plot_topomaps_time(rel3d, labels, plot_ch_names, class_names,
                           model_name, benchmark_name, suffix=suffix, **time_kw)

    if target == 'predicted':
        _plot_both(relevance_3d, preds, 'predicted')

        conf_threshold = np.quantile(conf, confidence_quantile)
        hc_mask = conf >= conf_threshold
        if hc_mask.sum() > 0:
            print(f'\n  High-confidence ({confidence_quantile:.0%}): '
                  f'{hc_mask.sum()} / {len(conf)} samples (threshold={conf_threshold:.3f})')
            _plot_both(relevance_3d[hc_mask], preds[hc_mask], 'predicted_highconf')

        plot_topomaps_per_subject(
            relevance_3d, preds, sbj_ids, plot_ch_names, class_names,
            model_name, benchmark_name, suffix='predicted',
            **dict(kw, out_dir=f'{out_standard}/per_subject'),
        )
        return

    # target == 'gt'
    _plot_both(relevance_3d, y_test, 'gt')

    if correct_mask.sum() > 0:
        _plot_both(relevance_3d[correct_mask], y_test[correct_mask], 'gt_correct')
    else:
        print("  No correct predictions — skipping gt_correct plot.")

    incorrect_mask = ~correct_mask
    if incorrect_mask.sum() > 0:
        _plot_both(relevance_3d[incorrect_mask], y_test[incorrect_mask], 'gt_incorrect')
    else:
        print("  No incorrect predictions — skipping gt_incorrect plot.")

    if correct_mask.sum() > 0:
        conf_threshold = np.quantile(conf[correct_mask], confidence_quantile)
        hc_mask = correct_mask & (conf >= conf_threshold)
        if hc_mask.sum() > 0:
            print(f'\n  High-confidence correct ({confidence_quantile:.0%}): '
                  f'{hc_mask.sum()} / {correct_mask.sum()} correct samples '
                  f'(threshold={conf_threshold:.3f})')
            _plot_both(relevance_3d[hc_mask], y_test[hc_mask], 'gt_correct_highconf')

        plot_topomaps_per_subject(
            relevance_3d[correct_mask], y_test[correct_mask], sbj_ids[correct_mask],
            plot_ch_names, class_names, model_name, benchmark_name,
            suffix='gt_correct',
            **dict(kw, out_dir=f'{out_standard}/per_subject'),
        )


def print_fold_stats(fold, rel, conf):
    print(f"  Fold {fold} relevance stats: min={rel.min():.3f}, max={rel.max():.3f}, mean={np.abs(rel).mean():.4f}")
    print(f"  Confidence: min={conf.min():.3f}, median={np.median(conf):.3f}, max={conf.max():.3f}")
    if np.allclose(rel, 0):
        print("  WARNING: relevance is all zeros.")
    if np.isnan(rel).any():
        print("  WARNING: relevance contains NaNs.")
