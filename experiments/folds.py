import os
import traceback
import numpy as np
import skorch
from sklearn import model_selection

from data.loaders import zero_pad_channels, VARIABLE_CHANNEL_MODELS
from models.wrappers import get_model
from experiments.config import ExperimentConfig


def _prepare_data(config, b_train, b_test):
    """Extract arrays from benchmark objects, applying zero-padding when needed.

    Returns:
        (X_train_full, X_test_full, sbj_id, y, ch_names, test_ch_names)
    """
    X_train_full, sbj_id, y, ch_names = b_train.get_data()
    X_test_full, _, _, test_ch_names = b_test.get_data()

    needs_padding = len(test_ch_names) < len(ch_names)
    if (needs_padding and config.model_name not in VARIABLE_CHANNEL_MODELS) or config.pad_channels: # only pad if model doesn't support variable channels and config doesn't disable padding
        X_test_full = zero_pad_channels(X_test_full, test_ch_names, ch_names)
        test_ch_names = ch_names
        print(f"Zero-padded test data to {len(ch_names)} channels")
        

    return X_train_full, X_test_full, sbj_id, y, ch_names, test_ch_names


def train_folds(config: ExperimentConfig, metrics: list, b_train, b_test, logger=None):
    """Fine-tune a model on each cross-validation fold and collect per-epoch metrics."""
    X_train_full, X_test_full, sbj_id, y, ch_names, test_ch_names = _prepare_data(config, b_train, b_test)

    sbj_id_unique = np.unique(sbj_id)
    n_outputs = len(np.unique(y))
    n, c, t = X_train_full.shape

    if config.overwrite_existing_checkpoints:
        for path in [config.get_folds_csv_path(), config.get_epochs_csv_path()]:
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleting existing CSV file at {path}")

    kf = model_selection.KFold(n_splits=config.n_folds, shuffle=True, random_state=99)

    for i_fold, (sbj_idx_train, sbj_idx_test) in enumerate(kf.split(sbj_id_unique)):
        
        if config.fold_filter is not None and i_fold not in config.fold_filter:
            print(f"FOLD {i_fold}... SKIPPED (not in fold_filter)")
            continue
        
        
        print(f"FOLD {i_fold}...")
        train_subjects = sbj_id_unique[sbj_idx_train].tolist()
        test_subjects = sbj_id_unique[sbj_idx_test].tolist()
        b_train.check_and_save_split(i_fold, train_subjects, test_subjects)

        # Skip if checkpoint already exists for this fold
        fold_ckpt_path = config.get_checkpoint_path(i_fold, ckpt_type='best')
        fold_ckpt_path_last = config.get_checkpoint_path(i_fold, ckpt_type='last')
        ckpt_exists = os.path.exists(fold_ckpt_path) or os.path.exists(fold_ckpt_path_last)

        if ckpt_exists and not config.overwrite_existing_checkpoints:
            print(f"Checkpoint already exists for fold {i_fold}, skipping")
            continue
        
        

        train_mask = np.isin(sbj_id, sbj_id_unique[sbj_idx_train])
        test_mask = np.isin(sbj_id, sbj_id_unique[sbj_idx_test])

        train_dataset = skorch.dataset.Dataset(X_train_full[train_mask], y[train_mask])
        test_dataset = skorch.dataset.Dataset(X_test_full[test_mask], y[test_mask])

        try:
            model = get_model(
                config.model_name,
                n_chans=c,
                ch_names=ch_names,
                sfreq=config.fs(),
                n_times=t,
                n_outputs=n_outputs,
                sbj_ids=sbj_id[train_mask],
                encoder_only=False,
                ckpt_path=None,
                train_head_only=config.train_head_only,
                large_head=config.large_head, # for labram and neuroRVQ
                exit_block=config.exit_block,
                skip_tokenizer=config.model_name == "BrainOmni" and config.skip_tokenizer
            )
            print(f"No. Trainable Parameters: {model.size()}")

            model.fit(
                train_dataset,
                test_dataset,
                batch_size=config.batch_size,
                epochs=config.n_epochs,
                early_stopping_patience=config.early_stopping_patience,
            )
        except Exception as fold_err:
            print(f"ERROR in fold {i_fold}: {fold_err}")
            traceback.print_exc()
            config.log_error(f"fold {i_fold}: {fold_err}")
            continue

        # Save fold results to CSV (append-only, crash-resilient)
        fold_train = {m: model.results[f'train_{m}'] for m in metrics}
        fold_test = {m: model.results[f'val_{m}'] for m in metrics}
        config.save_fold_result(i_fold, model.best_epoch, metrics, fold_train, fold_test)
        config.save_epoch_results(i_fold, metrics, fold_train, fold_test)

        if config.save_best_checkpoint or config.save_last_checkpoint:
            ckpt_dir = os.path.dirname(config.get_checkpoint_path(i_fold))
            os.makedirs(ckpt_dir, exist_ok=True)
            if config.save_last_checkpoint:
                model.save_model(config.get_checkpoint_path(i_fold, ckpt_type='last'))
            if config.save_best_checkpoint and model.best_state_dict is not None:
                model.save_best_model(config.get_checkpoint_path(i_fold, ckpt_type='best'))


        if logger is not None:
            for m in metrics:
                for epoch, val in enumerate(fold_train[m]):
                    logger.report_scalar(
                        title=f"{config.model_name} {config.benchmark_name} {m}",
                        series=f'train fold_{i_fold}', value=val, iteration=epoch)
                for epoch, val in enumerate(fold_test[m]):
                    logger.report_scalar(
                        title=f"{config.model_name} {config.benchmark_name} {m}",
                        series=f'test fold_{i_fold}', value=val, iteration=epoch)


def evaluate_folds(config: ExperimentConfig, metrics: list, b_train, b_test, logger=None):
    """Load per-fold checkpoints and evaluate on the test set (no training)."""
    X_train_full, X_test_full, sbj_id, y, ch_names, test_ch_names = _prepare_data(config, b_train, b_test)

    sbj_id_unique = np.unique(sbj_id)
    n_outputs = len(np.unique(y))
    n, c, t = X_train_full.shape

    if config.overwrite_existing_checkpoints:
        for path in [config.get_folds_csv_path(), config.get_epochs_csv_path()]:
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleting existing CSV file at {path}")

    completed_data = config.load_completed_folds(metrics)

    kf = model_selection.KFold(n_splits=config.n_folds, shuffle=True, random_state=99)

    for i_fold, (sbj_idx_train, sbj_idx_test) in enumerate(kf.split(sbj_id_unique)):
        
        if config.fold_filter is not None and i_fold not in config.fold_filter:
            print(f"FOLD {i_fold}... SKIPPED (not in fold_filter)")
            continue
        
        print(f"FOLD {i_fold}...")

        train_subjects = sbj_id_unique[sbj_idx_train].tolist()
        test_subjects = sbj_id_unique[sbj_idx_test].tolist()
        b_train.check_and_save_split(i_fold, train_subjects, test_subjects)

        # Skip if already evaluated (results in CSV)
        if not config.overwrite_existing_checkpoints and i_fold in completed_data['completed_folds']:
            print(f"Results already exist for fold {i_fold}, skipping evaluation")
            continue

        fold_ckpt_path = config.get_checkpoint_path(i_fold, ckpt_type=config.evaluate_on)
        if not os.path.exists(fold_ckpt_path):
            print(f"No checkpoint found for fold {i_fold} at {fold_ckpt_path}, skipping")
            continue

        train_mask = np.isin(sbj_id, sbj_id_unique[sbj_idx_train])
        test_mask = np.isin(sbj_id, sbj_id_unique[sbj_idx_test])
        test_dataset = skorch.dataset.Dataset(X_test_full[test_mask], y[test_mask])

        model = get_model(
            config.model_name,
            n_chans=c,
            ch_names=ch_names,
            sfreq=config.fs(),
            n_times=t,
            n_outputs=n_outputs,
            sbj_ids=sbj_id[train_mask],
            encoder_only=False,
            ckpt_path=None,
            train_head_only=config.train_head_only,
            large_head=config.large_head, # for labram and neuroRVQ
            exit_block=config.exit_block,
            skip_tokenizer=config.model_name == "BrainOmni" and config.skip_tokenizer
        )

        model.load_model(fold_ckpt_path)

        test_metrics = model.evaluate(test_dataset, batch_size=config.batch_size, ch_names=test_ch_names)

        # Save to CSV (append-only, crash-resilient)
        config.save_fold_result(i_fold, -1, metrics,
                                {m: [np.nan] for m in metrics},
                                {m: [test_metrics[m]] for m in metrics})

