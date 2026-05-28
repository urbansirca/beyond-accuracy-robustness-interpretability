"""Cross-product experiment runner shared by cli/train.py and cli/evaluate.py.

Iterates (model × benchmark × augmentation × finetune_mode × large_head × exit_block)
from a parsed config dict and dispatches each combination to train_folds() or
evaluate_folds() based on `phase`.
"""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict, replace
from datetime import datetime

import clearml
import pandas as pd
import torch

from data.loaders import AugmentationNotFoundError, load_benchmark, get_subdir, VARIABLE_CHANNEL_MODELS
from experiments.batch_sizing import make_batch_size_resolver
from experiments.config import ExperimentConfig
from experiments.folds import evaluate_folds, train_folds
from experiments.loggers import get_logger



_LARGE_HEAD_MODELS = {"LaBraM", "NeuroRVQ", "REVE"}
# Per-model natural head. None in the config sweep resolves to this value.
_PRIMARY_LARGE_HEAD = {"REVE": True, "LaBraM": False, "NeuroRVQ": False}

_EXIT_BLOCK_MODELS = {"NeuroRVQ", "REVE"} # only models that support intermediate exits
_NO_HEAD_ONLY_MODELS = {"EEGNet"}



def _resolve_large_head(model: str, value) -> bool:
    """None sentinel means: use each model's primary head."""
    if value is None:
        return _PRIMARY_LARGE_HEAD.get(model, False)
    return bool(value)


def _is_supported(model: str, ft_mode: str, large_head: bool, exit_block) -> bool:
    """Return False for unsupported model configurations to skip them."""
    if ft_mode == "head_only" and model in _NO_HEAD_ONLY_MODELS:
        return False
    if large_head and model not in _LARGE_HEAD_MODELS:
        return False
    if exit_block is not None and model not in _EXIT_BLOCK_MODELS:
        return False
    return True


def _evict_cache(cache: dict, remaining_configs: list) -> None:
    """Delete benchmark objects no longer needed by any remaining augmentation config."""
    needed = {a for c in remaining_configs for a in (c["train"], c["test"])}
    for k in list(cache.keys()):
        if k not in needed:
            print(f"  [cache] Releasing benchmark aug={k!r}")
            del cache[k]
    

def run_experiments(cfg: dict) -> None:
    phase = cfg["phase"]
    assert phase in {"train", "evaluate"}, f"Unknown phase: {phase}"

    models = cfg["models"]
    benchmarks = cfg["benchmarks"]
    augmentations = cfg["augmentations"]
    finetune_modes = cfg["finetune_modes"]
    large_head_options = cfg["large_head"]
    exit_block_options = cfg["exit_block"]
    metrics = cfg["metrics"]

    base = ExperimentConfig.init(cfg, phase)
    get_batch_size = make_batch_size_resolver(
        models, benchmarks, cfg.get("run_batch_size_finder")
    )

    total = (len(models) * len(benchmarks) * len(augmentations)
             * len(finetune_modes) * len(large_head_options) * len(exit_block_options))
    print(f"\nRunning {total} experiments  (phase={phase})")
    print(f"  Models: {models}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Augmentations: {len(augmentations)}")
    print(f"  Finetune modes: {finetune_modes}")
    print(f"  Large head: {large_head_options}")
    print(f"  Exit block: {exit_block_options}\n")

    completed, failed = [], []
    current = 0

    for benchmark_name in benchmarks:
        for model_name in models:
            # True channel dropout (pad_channels=False) feeds the model fewer
            # channels; fixed-channel models can't accept that, so skip them.
            if not base.pad_channels and model_name not in VARIABLE_CHANNEL_MODELS:
                print(f"\nSkipping {model_name} — true channel dropout requires a "
                      f"variable-channel model ({sorted(VARIABLE_CHANNEL_MODELS)})")
                continue

            subdir = get_subdir(model_name)
            bench_cache: dict = {}

            for i_aug, aug in enumerate(augmentations):
                train_aug, test_aug = aug["train"], aug["test"]

                for ft_mode in finetune_modes:
                    for large_head_opt in large_head_options:
                        large_head = _resolve_large_head(model_name, large_head_opt)
                        for exit_block in exit_block_options:
                            current += 1
                            if not _is_supported(model_name, ft_mode, large_head, exit_block):
                                print(f"\n[{current}/{total}] Skipping {model_name} "
                                      f"(ft={ft_mode}, large_head={large_head}, exit_block={exit_block}) "
                                      "— not supported")
                                continue

                            config = replace(
                                base,
                                model_name=model_name,
                                benchmark_name=benchmark_name,
                                train_augmentation=train_aug,
                                test_augmentation=test_aug,
                                train_head_only=(ft_mode == "head_only"),
                                large_head=large_head,
                                exit_block=exit_block,
                                batch_size=get_batch_size(model_name, benchmark_name),
                            )

                            if not config.overwrite_existing_checkpoints and config.all_folds_complete():
                                print(f"\n[{current}/{total}] {config.get_experiment_name()} — "
                                      "all folds complete, skipping")
                                completed.append((config.get_experiment_name(), config))
                                continue

                            try:
                                if train_aug not in bench_cache:
                                    bench_cache[train_aug] = load_benchmark(
                                        benchmark_name, cfg["data_root"], subdir, cfg["apply_car"],
                                        augmentation=train_aug
                                    )
                                if test_aug not in bench_cache:
                                    bench_cache[test_aug] = load_benchmark(
                                        benchmark_name, cfg["data_root"], subdir, cfg["apply_car"],
                                        augmentation=test_aug
                                    )
                            except AugmentationNotFoundError as e:
                                print(f"\n[{current}/{total}] {config.get_experiment_name()} — SKIPPED")
                                print(f"  Reason: {e}")
                                continue

                            b_train = bench_cache[train_aug]
                            b_test = bench_cache[test_aug] if train_aug != test_aug else b_train

                            exp_name = config.get_experiment_name()
                            print(f"\n{'='*60}\n[{current}/{total}] {exp_name}\n{'='*60}")

                            logger, task = (None, None)
                            if cfg.get("use_clearml", False):
                                logger, task = get_logger(
                                    logger_type="clearml",
                                    project_name="EEG-FM-robustness",
                                    task_name=exp_name,
                                    task_type=("testing" if config.evaluate_only else "training"),
                                    tags=config.get_experiment_tags(),
                                    config_dict=asdict(config),
                                    add_unique_id=cfg.get("add_unique_id", False),
                                )

                            try:
                                if phase == "evaluate":
                                    evaluate_folds(config, metrics, b_train, b_test, logger)
                                else:
                                    train_folds(config, metrics, b_train, b_test, logger)
                                completed.append((exp_name, config))
                                print(f"Results saved to: {config.get_folds_csv_path()}")
                            except Exception as e:
                                print(f"ERROR: {e}")
                                failed.append((exp_name, str(e)))
                                traceback.print_exc()
                                config.log_error(e)

                            if task:
                                task.close()

                _evict_cache(bench_cache, augmentations[i_aug + 1:])

    # ------------ Print summary ------------
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    print(f"Results saved in: {cfg['output_dir']}/<model>/<dataset>/<finetune_type>/")
    for exp, c in completed:
        print(f"  [OK]   {exp}")
        print(f"         {c.get_folds_csv_path()}")
    for exp, err in failed:
        print(f"  [FAIL] {exp}: {err}")

    if cfg.get("use_clearml", False) and completed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dfs = [pd.read_csv(c.get_folds_csv_path()) for _, c in completed
               if os.path.isfile(c.get_folds_csv_path())]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            summary_task = clearml.Task.init(
                project_name="EEG-FM-robustness",
                task_name=f"Benchmarking Summary {timestamp}",
                task_type=clearml.TaskTypes.data_processing,
                tags=["summary"],
                auto_connect_frameworks=False,
            )
            clearml.Logger.current_logger().report_table(
                title="Benchmarking Results",
                series="Per-Fold Results",
                table_plot=df,
            )
            summary_task.close()
            print("Summary table uploaded to ClearML")

    
    
