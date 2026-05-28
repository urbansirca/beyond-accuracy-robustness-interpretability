import os
from dataclasses import dataclass
from typing import Optional
import json

import csv
from datetime import datetime

from data.loaders import slug_for


@dataclass
class ExperimentConfig:
    # Model
    model_name: str
    train_head_only: bool = False
    large_head: bool = False # for labram and neuroRVQ to use the REVE head
    exit_block: Optional[int] = None  # Exit after N transformer blocks (1-based). None = use all blocks.
    ckpt_path: Optional[str] = None
    
    #brainomni-specific skip VQ step
    skip_tokenizer: bool = False # whether to skip the brainomni tokenizer
    
    overwrite_existing_checkpoints: bool = False  # Whether to skip training if checkpoint already exists for a fold
    early_stopping_patience: Optional[int] = None  # Number of epochs with no improvement after which training will be stopped. If None, no early stopping is applied.
    # Data
    benchmark_name: str = None
    data_root: str = None
    train_augmentation: Optional[str] = None  # e.g., "sensor_noise_pink_0db"
    test_augmentation: Optional[str] = None
    apply_car: bool = False
    pad_channels: bool = True  # Whether to apply zero-padding for variable channel models as well

    # Training
    n_folds: int = 10
    n_epochs: int = 20
    batch_size: int = 64
    
    fold_filter: Optional[list[int]] = None  # If set, only run these fold indices (0-based)

    # Output
    output_dir: str = "results"
    ckpt_root: str = "weights/finetuned"  # Root dir for saved finetuned ckpts; final path is {ckpt_root}/{model}/{dataset}/
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = False
    # Evaluation only (skip training, load from checkpoint)
    evaluate_only: bool = False
    evaluate_on: str = "best"  # "best" or "last", which checkpoint to load for evaluation when evaluate_only=True
    
    # Logging
    logger: str = "clearml"  # "clearml" or "csv"
    
    @classmethod
    def init(cls, cfg: dict, phase: str) -> "ExperimentConfig":
        return cls(
            model_name="",
            benchmark_name="",
            data_root=cfg["data_root"],
            n_folds=cfg["n_folds"],
            n_epochs=cfg["n_epochs"],
            output_dir=cfg["output_dir"],
            ckpt_root=cfg.get("ckpt_root", "weights/finetuned"),
            apply_car=cfg["apply_car"],
            evaluate_only=(phase == "evaluate"),
            overwrite_existing_checkpoints=cfg.get("overwrite", False),
            early_stopping_patience=cfg.get("early_stopping_patience"),
            save_best_checkpoint=cfg.get("save_best_checkpoint", True),
            save_last_checkpoint=cfg.get("save_last_checkpoint", True),
            evaluate_on=cfg.get("evaluate_on", "best"),
            skip_tokenizer=cfg.get("skip_brainomni_tokenizer", False),
            fold_filter=cfg.get("fold_filter", None),
            # "True" channel dropout: drop channels from the input instead of
            # zero-padding them back to the full montage (variable-channel models only).
            pad_channels=not cfg.get("true_channel_dropout", False),
        )
    
    def fs(self):
        """Determine sampling frequency based on model name."""
        if self.model_name == "BrainOmni":
            return 256
        return 200

    def get_experiment_name(self) -> str:
        """For ClearML task naming"""
        finetune_type = "head" if self.train_head_only else "full"
        if self.exit_block is not None:
            finetune_type += f"_block{self.exit_block}"
        train_data = self.train_augmentation or "clean"
        test_data = self.test_augmentation or "clean"
        return f"{self.model_name}_{self.benchmark_name}_{finetune_type}_train-{train_data}_test-{test_data}"


    def get_experiment_tags(self) -> list[str]:
        """Generate tags for ClearML."""
        return [
            f"model:{self.model_name}",
            f"benchmark:{self.benchmark_name}",
            f"finetune:{'head_only' if self.train_head_only else 'full'}",
            f"train:{self.train_augmentation or 'clean'}",
            f"test:{self.test_augmentation or 'clean'}",
        ] + ([f"exit_block:{self.exit_block}"] if self.exit_block is not None else [])

    def get_checkpoint_name(self, fold: int, ckpt_type="best") -> str:
        """Generate checkpoint filename for a fold."""
        finetune_type = "head" if self.train_head_only else "full"
        if self.large_head:
            finetune_type += "_large_head"
        if self.exit_block is not None:
            finetune_type += f"_block{self.exit_block}"
        train_data = self.train_augmentation or "clean"
        
        if self.skip_tokenizer:
            finetune_type += "_skip_tokenizer"
            
        return f"{finetune_type}_train-{train_data}_fold{fold}_{ckpt_type}.pt"
    
    def get_checkpoint_path(self, fold: int, ckpt_type="best") -> str:
        """Generate full checkpoint path for a fold."""
        ckpt_dir = os.path.join(
            self.ckpt_root,
            self.model_name,
            slug_for(self.benchmark_name),
        )
        return os.path.join(ckpt_dir, self.get_checkpoint_name(fold, ckpt_type))

    def get_results_path(self) -> str:
        """Generate organized path for results JSON file."""
        finetune_type = "head_only" if self.train_head_only else "full_finetuned"
        if self.large_head:
            finetune_type += "_large_head"
        if self.exit_block is not None:
            finetune_type += f"_block{self.exit_block}"
        # Create directory structure: results/{model}/{dataset}/{finetune_type}/
        results_dir = os.path.join(
            self.output_dir,
            self.model_name,
            slug_for(self.benchmark_name),
            finetune_type,
        )
        os.makedirs(results_dir, exist_ok=True)

        # Filename includes training and test conditions
        
        return results_dir
    
    def get_folds_csv_path(self) -> str:
        """Generate path for per run CSV file."""
        train_data = self.train_augmentation or "clean"
        test_data = self.test_augmentation or "clean"
        filename = f"train-{train_data}_test-{test_data}_folds.csv"
        return os.path.join(self.get_results_path(), filename)
    
    def get_epochs_csv_path(self) -> str:
        """Generate path for per epoch CSV file."""
        train_data = self.train_augmentation or "clean"
        test_data = self.test_augmentation or "clean"
        filename = f"train-{train_data}_test-{test_data}_epochs.csv"
        return os.path.join(self.get_results_path(), filename)
    
    def save_fold_result(self, fold: int, best_epoch: int, metrics: list,
                     train_metrics_this_fold: dict, test_metrics_this_fold: dict):
        """Append one fold's summary to the folds CSV.
        
        Args:
            fold: fold index
            best_epoch: best epoch index (-1 if N/A)
            metrics: list of metric names e.g. ["accuracy", "bacc"]
            train_metrics_this_fold: {metric_name: [per_epoch_values]}
            test_metrics_this_fold: {metric_name: [per_epoch_values]}
        """
        csv_path = self.get_folds_csv_path()
        n_epochs = len(train_metrics_this_fold[metrics[0]])
        ep = best_epoch if best_epoch >= 0 else -1
        head_only = "head_only" if self.train_head_only else "full_finetuned"
        if self.large_head:
            head_only += "_large_head"
        if self.exit_block is not None:
            head_only += f"_block{self.exit_block}"

        header = ["benchmark", "model", "head_only", "train_data", "test_data", "fold", "best_epoch", "n_epochs", "timestamp"]
        for m in metrics:
            header += [f'{m}_train_best', f'{m}_test_best', f'{m}_train_last', f'{m}_test_last']

        write_header = not os.path.isfile(csv_path)
        row = [self.benchmark_name, self.model_name, head_only, self.train_augmentation or "clean", self.test_augmentation or "clean", fold, best_epoch, n_epochs, datetime.now().isoformat()]
        for m in metrics:
            row.append(float(train_metrics_this_fold[m][ep]))
            row.append(float(test_metrics_this_fold[m][ep]))
            row.append(float(train_metrics_this_fold[m][-1]))
            row.append(float(test_metrics_this_fold[m][-1]))

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
            
    def save_epoch_results(self, fold: int, metrics: list,
                       train_metrics_this_fold: dict, test_metrics_this_fold: dict):
        """Append per-epoch metrics for one fold to the epochs CSV."""
        csv_path = self.get_epochs_csv_path()
        n_epochs = len(train_metrics_this_fold[metrics[0]])

        header = ["benchmark", "model", "head_only", "train_data", "test_data", "fold", "epoch", "timestamp"]
        for m in metrics:
            header += [f'train_{m}', f'val_{m}']
            
            
        head_only = "head_only" if self.train_head_only else "full_finetuned"
        if self.large_head:
            head_only += "_large_head"
        if self.exit_block is not None:
            head_only += f"_block{self.exit_block}"

        write_header = not os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            for ep_i in range(n_epochs):
                row = [self.benchmark_name, self.model_name, head_only, self.train_augmentation or "clean", self.test_augmentation or "clean", fold, ep_i, datetime.now().isoformat()]
                for m in metrics:
                    row.append(float(train_metrics_this_fold[m][ep_i]))
                    row.append(float(test_metrics_this_fold[m][ep_i]))
                writer.writerow(row)
                
    def load_completed_folds(self, metrics: list) -> dict:
        """Load completed fold data from CSV files.
        
        Returns dict with:
            'completed_folds': set of fold indices
            'best_epochs': {fold: best_epoch}
            'train_metrics': {metric: {fold: [epoch_values]}}
            'test_metrics': {metric: {fold: [epoch_values]}}
        """
        fold_path = self.get_folds_csv_path()
        epoch_path = self.get_epochs_csv_path()

        result = {
            'completed_folds': set(),
            'best_epochs': {},
            'train_metrics': {m: {} for m in metrics},
            'test_metrics': {m: {} for m in metrics},
        }

        if os.path.isfile(fold_path):
            with open(fold_path, 'r') as f:
                for row in csv.DictReader(f):
                    fold_i = int(row['fold'])
                    result['completed_folds'].add(fold_i)
                    result['best_epochs'][fold_i] = int(row['best_epoch'])

        if os.path.isfile(epoch_path):
            with open(epoch_path, 'r') as f:
                for row in csv.DictReader(f):
                    fold_i = int(row['fold'])
                    epoch_i = int(row['epoch'])
                    # epoch 0 resets arrays for this fold (last run wins)
                    if epoch_i == 0:
                        for m in metrics:
                            result['train_metrics'][m][fold_i] = []
                            result['test_metrics'][m][fold_i] = []
                    for m in metrics:
                        result['train_metrics'][m][fold_i].append(float(row[f'train_{m}']))
                        result['test_metrics'][m][fold_i].append(float(row[f'val_{m}']))

        return result

    def all_folds_complete(self) -> bool:
        """Check if all folds have both checkpoints and CSV results."""
        # Check CSV
        fold_path = self.get_folds_csv_path()
        if not os.path.isfile(fold_path):
            return False
        completed = set() # duplicates are ignored
        with open(fold_path, 'r') as f:
            for row in csv.DictReader(f):
                completed.add(int(row['fold']))
        if len(completed) < self.n_folds:
            return False

        # Check checkpoints
        for fold in range(self.n_folds):
            best_path = self.get_checkpoint_path(fold, ckpt_type='best')
            last_path = self.get_checkpoint_path(fold, ckpt_type='last')
            if not (os.path.exists(best_path) and os.path.exists(last_path)):
                return False

        return True
    
    def log_error(self, message):
        """Log an error message to json file in results directory."""
        
        log_path = os.path.join(self.get_results_path(), "error_log.json")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "benchmark": self.benchmark_name,
            "train_data": self.train_augmentation or "clean",
            "test_data": self.test_augmentation or "clean",
            "message": str(message),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

