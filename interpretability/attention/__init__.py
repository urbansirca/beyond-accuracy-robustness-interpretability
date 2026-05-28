import os

import numpy as np
import torch

from data.loaders import slug_for
from models.wrappers import get_model

from interpretability.common.csv import export_attention_csvs, attention_csv_exists
from interpretability.common.data import (
    BENCHMARK_CLASSES, build_test_set, load_data_with_augmentation, resolve_channels,
)
from interpretability.common.checkpoints import get_ckpt_path, unwrap_model
from interpretability.attention.capture import attention_context
from interpretability.attention.accumulator import AttentionAccumulator
from interpretability.attention.models import labram, neurorvq, reve
from interpretability.plotting.plotting_attn import (
    plot_channel_topomap,
    plot_cls_topomap,
    plot_heads_grid,
)

_RUNNERS = {
    "LaBraM":   labram.run,
    "NeuroRVQ": neurorvq.run,
    "REVE":     reve.run,
}
_SUPPORTED = set(_RUNNERS)

def run_analysis(out_root, data_root, model_name, benchmark_name, fold,
                 large_head=False, train_head_only=False,
                 overwrite=False, augmentation=None):
    if model_name not in _SUPPORTED:
        print(f"  Attention not supported for {model_name}; skipping.")
        return
    
    run_attn = _RUNNERS[model_name]

    # 1. Load data
    X, subject_ids, y, ch_names, _ = load_data_with_augmentation(
        model_name, benchmark_name, data_root, augmentation,
    )
    X = np.array(X, dtype=np.float32)
    n, c, t = X.shape
    all_classes = sorted(np.unique(y))
    n_outputs = len(all_classes)
    n_time_patches = t // 200
    has_cls = model_name in ("LaBraM", "NeuroRVQ")
    print(f"  {n} trials, {c} channels, {t} timepoints, {n_outputs} classes")

    # 2. Channel resolution + model wrapper
    channels = resolve_channels(model_name, ch_names)
    plot_ch_names = channels['plot_ch_names']
    print(f"  Channels for topomap: {len(plot_ch_names)}, time patches: {n_time_patches}")

    print(f'Initializing {model_name}...')
    wrapper = get_model(
        model_name, n_chans=len(ch_names), ch_names=ch_names, sfreq=200,
        n_times=t, n_outputs=n_outputs, sbj_ids=None,
        large_head=large_head, train_head_only=train_head_only,
    )

    # 3. Fold loop — one global accumulator + a per-fold one for fold-level CSVs
    acc = AttentionAccumulator(len(plot_ch_names), n_time_patches, has_cls)
    folds = range(10) if fold == -1 else ([fold] if isinstance(fold, int) else fold)
    batch_size = 8

    for f in folds:
        print(f'\n--- Processing Fold {f} ---')

        if not overwrite and attention_csv_exists(
            out_root,
            model_name, benchmark_name, fold=f,
            large_head=large_head, train_head_only=train_head_only,
            augmentation=augmentation,
        ):
            print(f'  CSV already exists for fold {f} — skipping (use --overwrite to recompute).')
            continue

        X_test, y_test, _ = build_test_set(X, subject_ids, y, f)
        print(f'  {len(X_test)} test samples')

        ckpt_path = get_ckpt_path(model_name, benchmark_name, f,
                                  large_head=large_head, train_head_only=train_head_only)
        if not os.path.exists(ckpt_path):
            print(f'  Checkpoint not found: {ckpt_path}. Skipping.')
            continue

        print(f'  Loading {ckpt_path}')
        wrapper.load_model(ckpt_path)
        model = unwrap_model(wrapper).cuda().eval()

        fold_acc = AttentionAccumulator(len(plot_ch_names), n_time_patches, has_cls)

        for start in range(0, len(X_test), batch_size):
            X_batch = X_test[start : start + batch_size]
            y_batch = y_test[start : start + batch_size]
            with attention_context(model_name, model) as storage:
                run_attn(model, X_batch, ch_names, channels)
            acc.consume(storage.maps, y_batch)
            fold_acc.consume(storage.maps, y_batch)

        # Per-fold CSV export
        fold_y = fold_acc.get_y()
        if fold_y is not None and len(fold_y) > 0:
            export_attention_csvs(
                out_root, fold_acc.get_chan_scores(), fold_y, plot_ch_names,
                model_name, benchmark_name, all_classes,
                large_head=large_head, train_head_only=train_head_only,
                cls_row=fold_acc.get_cls_row() if has_cls else None,
                cls_col=fold_acc.get_cls_col() if has_cls else None,
                fold=f, augmentation=augmentation,
            )
        del fold_acc
        model.cpu()
        del model
        torch.cuda.empty_cache()

    # 4. Aggregated
    y_combined = acc.get_y()
    if y_combined is None or len(y_combined) == 0:
        print("No attention data accumulated.")
        return

    n_blocks = acc.n_blocks()
    class_names  = BENCHMARK_CLASSES.get(benchmark_name, None)
    finetune_tag = ("_head" if train_head_only else "") + ("_large_head" if large_head else "")
    aug_tag      = f"/{augmentation}" if augmentation else ""
    out_dir      = f"plots/attention/{model_name}{finetune_tag}/{slug_for(benchmark_name)}{aug_tag}"
    print(f"\nAggregated over {len(y_combined)} samples, {n_blocks} blocks")

    # 5. Plot
    mean_attn = acc.get_mean_attn()
    if mean_attn:
        plot_heads_grid(mean_attn, model_name, benchmark_name, out_dir=out_dir)

    chan_scores_by_branch = acc.get_chan_scores()
    cls_row_by_branch = acc.get_cls_row()
    cls_col_by_branch = acc.get_cls_col()

    branches = sorted(chan_scores_by_branch)
    multi_branch = len(branches) > 1

    for branch in branches:
        tag = f"branch{branch}" if multi_branch else ""
        if has_cls:
            plot_cls_topomap(
                cls_row_by_branch.get(branch, {}),
                cls_col_by_branch.get(branch, {}),
                plot_ch_names, model_name, benchmark_name,
                n_blocks=n_blocks, out_dir=out_dir, tag=tag,
                y_test=y_combined, class_names=class_names,
                all_classes=all_classes,
            )
        plot_channel_topomap(
            chan_scores_by_branch.get(branch, {}),
            plot_ch_names, model_name, benchmark_name,
            n_blocks=n_blocks, out_dir=out_dir, tag=tag,
            y_test=y_combined, class_names=class_names,
            all_classes=all_classes,
        )

    if multi_branch:
        all_blocks = sorted({b for bd in chan_scores_by_branch.values() for b in bd})

        def _avg_branches(by_branch):
            return {
                block: np.mean(
                    [by_branch[br][block] for br in branches if block in by_branch.get(br, {})],
                    axis=0,
                )
                for block in all_blocks
                if any(block in by_branch.get(br, {}) for br in branches)
            }

        if has_cls:
            plot_cls_topomap(
                _avg_branches(cls_row_by_branch),
                _avg_branches(cls_col_by_branch),
                plot_ch_names, model_name, benchmark_name,
                n_blocks=n_blocks, out_dir=out_dir, tag="avg",
                y_test=y_combined, class_names=class_names,
                all_classes=all_classes,
            )
        plot_channel_topomap(
            _avg_branches(chan_scores_by_branch),
            plot_ch_names, model_name, benchmark_name,
            n_blocks=n_blocks, out_dir=out_dir, tag="avg",
            y_test=y_combined, class_names=class_names,
            all_classes=all_classes,
        )

    export_attention_csvs(
        out_root, chan_scores_by_branch, y_combined, plot_ch_names,
        model_name, benchmark_name, all_classes,
        large_head=large_head, train_head_only=train_head_only,
        cls_row=cls_row_by_branch if has_cls else None,
        cls_col=cls_col_by_branch if has_cls else None,
        fold="all", augmentation=augmentation,
    )
