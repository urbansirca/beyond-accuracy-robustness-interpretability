import importlib

import numpy as np

from data.loaders import slug_for
from models.wrappers import get_model

from interpretability.common.data import (
    build_test_set, load_data_with_augmentation, resolve_channels,
)
from interpretability.common.checkpoints import get_lrp_batch_size, load_fold_model
from interpretability.common.csv import attribution_csv_exists, export_attribution_csvs
from interpretability.common.plots import plot_by_subset, print_fold_stats


_SUPPORTED = {"EEGNet", "CBraMod", "REVE", "LaBraM", "NeuroRVQ", "BrainOmni"}
METHOD_TAG = "input_x_gradient"


def run_analysis(data_root, model_name, benchmark_name, fold,
                 target='gt', confidence_quantile=0.75, large_head=False,
                 overwrite=False, augmentation=None,
                 skip_brainomni_tokenizer=False,
                 out_root="results/attribution"):
    if model_name not in _SUPPORTED:
        print(f"  Grad × Input not supported for {model_name}; skipping.")
        return

    skip_tokenizer = (model_name == "BrainOmni" and skip_brainomni_tokenizer)
    # BrainOmni's tokenizer runs at 256 Hz / 512-sample patches.
    sfreq      = 256.0 if model_name == 'BrainOmni' else 200.0
    patch_size = 512   if model_name == 'BrainOmni' else 200

    run = importlib.import_module(f"{__package__}.models.{model_name.lower()}").run

    # 1. Load data
    X, subject_ids, y, ch_names, _ = load_data_with_augmentation(
        model_name, benchmark_name, data_root, augmentation,
    )
    n, c, t = X.shape
    all_classes = sorted(np.unique(y))
    n_outputs = len(all_classes)
    print(f'  {n} trials, {c} channels, {t} timepoints, {n_outputs} classes')

    # 2. Channel resolution + model wrapper
    channels = resolve_channels(model_name, ch_names)
    plot_ch_names = channels['plot_ch_names']
    print(f'  Channels used: {len(plot_ch_names)} / {len(ch_names)}')

    print(f'Initializing {model_name}...')
    wrapper = get_model(
        model_name, n_chans=c, ch_names=ch_names, sfreq=sfreq,
        n_times=t, n_outputs=n_outputs, sbj_ids=None, large_head=large_head,
        skip_tokenizer=skip_tokenizer,
    )
    batch_size = get_lrp_batch_size(model_name, benchmark_name, c, t, n_outputs, ch_names)

    # 3. Fold loop
    folds = range(10) if fold == -1 else [fold]
    all_relevance, all_y_test, all_preds, all_conf, all_sbj_ids = [], [], [], [], []

    for f in folds:
        print(f'\n--- Processing Fold {f} ---')

        if not overwrite and attribution_csv_exists(
            model_name, benchmark_name, method=METHOD_TAG,
            fold=f, head_only=False, large_head=large_head,
            augmentation=augmentation, skip_tokenizer=skip_tokenizer,
            out_root=out_root,
        ):
            print(f'  CSV already exists for fold {f} — skipping (use --overwrite to recompute).')
            continue

        X_test, y_test, sbj_ids_test = build_test_set(X, subject_ids, y, f)
        print(f'  {len(X_test)} test trials')

        model = load_fold_model(wrapper, model_name, benchmark_name, f,
                                large_head=large_head, skip_tokenizer=skip_tokenizer)
        if model is None:
            continue

        print(f'  Computing Grad × Input...')
        method_y = y_test if target == 'gt' else None
        rel, preds, conf = run(
            model, X_test, batch_size, ch_names, channels, y_test=method_y,
        )

        print_fold_stats(f, rel, conf)

        export_attribution_csvs(
            rel, y_test, preds, conf, plot_ch_names,
            model_name=model_name, benchmark_name=benchmark_name,
            method=METHOD_TAG, fold=f,
            confidence_quantile=confidence_quantile,
            sfreq=sfreq, patch_size=patch_size,
            head_only=False, large_head=large_head,
            augmentation=augmentation, skip_tokenizer=skip_tokenizer,
            out_root=out_root,
        )

        all_relevance.append(rel)
        all_y_test.append(y_test)
        all_preds.append(preds)
        all_conf.append(conf)
        all_sbj_ids.append(sbj_ids_test)

    if not all_relevance:
        print("No relevance computed.")
        return

    relevance_3d = np.concatenate(all_relevance, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    conf = np.concatenate(all_conf, axis=0)
    sbj_ids = np.concatenate(all_sbj_ids, axis=0)

    print(f'\nCombined relevance shape: {relevance_3d.shape}')
    print(f'Model accuracy on test set: {(preds == y_test).mean():.3f}')

    # 4. Plot cascade
    head_suffix = '_large_head' if large_head else ''
    aug_subdir  = f'/{augmentation}' if augmentation else ''
    out_base    = f'plots/input_x_gradient/{model_name}{head_suffix}/{slug_for(benchmark_name)}{aug_subdir}'
    plot_by_subset(
        relevance_3d, y_test, preds, conf, sbj_ids,
        plot_ch_names, all_classes, model_name, benchmark_name,
        target=target, confidence_quantile=confidence_quantile,
        out_standard=f'{out_base}/standard',
        out_temporal=f'{out_base}/temporal',
        fig_tag=METHOD_TAG, title_prefix='Grad × Input',
    )

    # 5. Aggregated CSV
    export_attribution_csvs(
        relevance_3d, y_test, preds, conf, plot_ch_names,
        model_name=model_name, benchmark_name=benchmark_name,
        method=METHOD_TAG, fold="all",
        confidence_quantile=confidence_quantile,
        sfreq=sfreq, patch_size=patch_size,
        head_only=False, large_head=large_head,
        augmentation=augmentation, skip_tokenizer=skip_tokenizer,
        out_root=out_root,
    )
