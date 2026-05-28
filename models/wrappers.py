import math
import os
import shutil
import tempfile
import numpy as np
import skorch
import torch
from skorch.callbacks import LRScheduler, Checkpoint
from abc import ABC, abstractmethod
from braindecode.models import EEGNetv1, EEGInception
from braindecode import EEGClassifier
from skorch.helper import predefined_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from models import BIOTModule, BrainOmniModule, CBraModModule, LaBraMModule, NeuroRVQModule, REVEModule


PIN_MEMORY = True
NUM_WORKERS = 8


def _compute_metrics(targets: np.ndarray[int], probs: np.ndarray[float]) -> dict:
    """Compute common classification metrics
    Shape of targets: (n_samples,)
    Shape of probs: (n_samples, n_classes) or (n_samples,) for binary
    
    """
    accuracy = accuracy_score(targets, probs.argmax(axis=1))
    bacc = balanced_accuracy_score(targets, probs.argmax(axis=1))
    kappa = cohen_kappa_score(targets, probs.argmax(axis=1))
    f1_weighted = f1_score(targets, probs.argmax(axis=1), average='weighted')
    f1_macro = f1_score(targets, probs.argmax(axis=1), average='macro')
    
    try:
        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(targets, probs[:,1])
        else:
            roc_auc = roc_auc_score(targets, probs, multi_class='ovr', average="macro")
    except ValueError:
        roc_auc = float('nan')  # Unable to compute ROC AUC (e.g. only one class present)

    return {
        "accuracy": accuracy,
        "bacc": bacc,
        "kappa": kappa,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc
    }
    
class FinetuningWrapper(ABC):
    """
    Wrapper class for initializing model, fitting and evaluating on benchmark data, and storing results
    """
    def __init__(self):
        self.model = None
        self.results = {}
        self.best_state_dict = None
        self.best_epoch = -1
        self.best_val_bacc = - float('inf')

    @abstractmethod
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        print("fit function not implemented")

    def evaluate(self, dataset, batch_size, ch_names=None):
        """Generic evaluation only at the end of training (no per-epoch metrics) - can be overridden by specific models if needed"""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        preds, targets, probs = [], [], []

        self.model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(next(self.model.parameters()).device).float() 
                out = self.model(X)

                preds.append(out.argmax(dim=1).cpu())
                targets.append(y)
                probs.append(torch.softmax(out, dim=1).cpu())

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        probs = torch.cat(probs).numpy() # Shape (n_samples, n_classes)

        return _compute_metrics(targets, probs)

    def size(self):
        """ Returns number of trainable parameters in model """
        if self.model is None:
            print("model not initialised")
        else:
            return self.model.size()

    def save_model(self, path):
        """Generic model saving - saves model state dict to path"""
        if self.model is None:
            raise ValueError("Model not initialized")

        print(f'Saving checkpoint to {path}...')

        # Handle different model wrapper types
        if hasattr(self.model, 'state_dict'):
            # Direct PyTorch model (EEGNet, EEGInception)
            torch.save(self.model.state_dict(), path)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'state_dict'):
            # Custom module with nested model attribute
            torch.save(self.model.model.state_dict(), path)
        else:
            raise NotImplementedError(
                f"save_model not implemented for {type(self.model).__name__}. "
                "Please override this method in the wrapper class."
            )
        
    def save_best_model(self, path):
        """Saves the best model state dict to path"""
        if self.best_state_dict is None:
            print("No best model to save")
            return
        
        print(f'Saving best checkpoint from epoch {self.best_epoch} with val_bacc={self.best_val_bacc:.4f} to {path}...')
        torch.save(self.best_state_dict, path)

    def load_model(self, path):
        """
        Generic model loading - loads fine-tuned checkpoint from path.
        This is called after model initialization (via get_model) to load fine-tuned weights.
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f'Loading checkpoint from {path}...')

        # Load checkpoint
        state_dict = torch.load(path, map_location="cpu")

        state_dict = {k: v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v
                     for k, v in state_dict.items()}

        # Handle different model wrapper types
        if hasattr(self.model, 'load_state_dict'):
            # Direct PyTorch model (EEGNet, EEGInception)
            self.model.load_state_dict(state_dict)
            self.model.float()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'load_state_dict'):
            # Custom module with nested model attribute
            self.model.model.load_state_dict(state_dict)
            self.model.model.float()
        else:
            raise NotImplementedError(
                f"load_model not implemented for {type(self.model).__name__}. "
                "Please override this method in the wrapper class."
            )
        del state_dict
        import gc; gc.collect()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(self.model, 'cuda'):
                    self.model.cuda()
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'cuda'):
                    self.model.model.cuda()

            print(f'✓ Checkpoint loaded successfully')

        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            raise e


class EEGNetv1Wrapper(FinetuningWrapper):
    def __init__(self, n_chans, sfreq, n_times, n_outputs):
        super().__init__()
        self.model = EEGNetv1(
            n_chans=n_chans, 
            n_times=n_times, 
            input_window_seconds=n_times//sfreq, 
            n_outputs=n_outputs
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=None):
        tmp_dir = tempfile.mkdtemp()
        monitor = 'valid_loss_best'
        
        
        extra_callbacks = []
        
        extra_callbacks.append(("best_ckpt", Checkpoint(
            monitor=monitor,
            dirname=tmp_dir,
            f_params='best_params.pt',
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
        )))
        
        if early_stopping_patience is not None:
            extra_callbacks.append(("early_stop", skorch.callbacks.EarlyStopping(
                monitor="valid_loss",
                patience=early_stopping_patience,
                # restore_best=True,
                lower_is_better=True
            )))

        net = get_braindecode_net(
            self.model,
            batch_size=batch_size,
            train_split=predefined_split(validation_dataset),
            extra_callbacks=extra_callbacks,
            )
        net.fit(train_dataset, y=None, epochs=epochs)
        self.results['train_accuracy'] = net.history[:, 'train_accuracy']
        self.results['val_accuracy'] = net.history[:, 'valid_accuracy']
        self.results['train_bacc'] = net.history[:, 'train_balanced_accuracy']
        self.results['val_bacc'] = net.history[:, 'valid_balanced_accuracy']

        best_ckpt_path = os.path.join(tmp_dir, 'best_params.pt')
        if os.path.exists(best_ckpt_path):
            self.best_state_dict = torch.load(best_ckpt_path, map_location='cpu')
            val_losses = list(net.history[:, 'valid_loss'])
            best_epoch = int(np.argmin(val_losses))
            self.best_epoch = best_epoch
            self.best_val_bacc = net.history[best_epoch, 'valid_balanced_accuracy']
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

class EEGInceptionWrapper(FinetuningWrapper):
    def __init__(self, n_chans, sfreq, n_times, n_outputs):
        super().__init__()
        self.model = EEGInception(
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=n_times//sfreq,
            n_outputs=n_outputs,
            sfreq=sfreq
        )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=5):
        tmp_dir = tempfile.mkdtemp()
        monitor = 'valid_loss_best'
        extra_callbacks = []
        
        extra_callbacks.append(("best_ckpt", Checkpoint(
            monitor=monitor,
            dirname=tmp_dir,
            f_params='best_params.pt',
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
        )))
        
        if early_stopping_patience is not None:
            extra_callbacks.append(("early_stop", skorch.callbacks.EarlyStopping(
                monitor="valid_balanced_accuracy",
                patience=early_stopping_patience,
                restore_best=True,
                lower_is_better=False
            )))

        net = get_braindecode_net(
            self.model,
            batch_size=batch_size,
            train_split=predefined_split(validation_dataset),
            extra_callbacks=extra_callbacks,
            )
        net.fit(train_dataset, y=None, epochs=epochs)
        self.results['train_accuracy'] = net.history[:, 'train_accuracy']
        self.results['val_accuracy'] = net.history[:, 'valid_accuracy']
        self.results['train_bacc'] = net.history[:, 'train_balanced_accuracy']
        self.results['val_bacc'] = net.history[:, 'valid_balanced_accuracy']

        best_ckpt_path = os.path.join(tmp_dir, 'best_params.pt')
        if os.path.exists(best_ckpt_path):
            self.best_state_dict = torch.load(best_ckpt_path, map_location='cpu')
            val_losses = list(net.history[:, 'valid_loss'])
            best_epoch = int(np.argmin(val_losses))
            self.best_epoch = best_epoch
            self.best_val_bacc = net.history[best_epoch, 'valid_balanced_accuracy']
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class LaBraMWrapper(FinetuningWrapper):
    def __init__(self, ch_names, sfreq, n_times, n_outputs, ckpt_path, train_head_only, large_head):
        super().__init__()
        self.model  = LaBraMModule(
            ch_names=ch_names,
            sfreq=sfreq,
            n_times=n_times,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            concat_pool=large_head,
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=5):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience)
        self.results = self.model.results
        self.best_state_dict = self.model.best_state_dict
        self.best_epoch = self.model.best_epoch
        self.best_val_bacc = self.model.best_val_bacc

    def load_model(self, path):
        """
        Override load_model for LaBraM to preserve mixed precision compatibility.
        LaBraM uses torch.amp.autocast during evaluation, so we should not force float32.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f'Loading checkpoint from {path}...')

        state_dict = torch.load(path, map_location='cpu')

        self.model.model.load_state_dict(state_dict)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.model.to(device)

        print(f'✓ Checkpoint loaded successfully')

    def evaluate(self, dataset, batch_size, ch_names=None):
        """Override to use LaBraM's custom evaluation logic"""
        from models.LaBraM import engine_for_finetuning

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=False,
        )

        eval_ch_names = ch_names if ch_names is not None else self.model.chnames
        metrics = self.model.metrics
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        is_binary = self.model.nb_classes == 1

        results = engine_for_finetuning.evaluate(
            data_loader,
            self.model.model,
            device,
            header='Test:',
            ch_names=eval_ch_names,
            metrics=metrics,
            is_binary=is_binary
        )
        
        pred_raw = results.pop("_pred")
        true_raw = results.pop("_true")
        
        if is_binary:
            p = pred_raw.ravel()
            probs = np.column_stack([1 - p, p])       # shape (N, 2)
            targets = true_raw.astype(int).ravel()
        else:
            exp = np.exp(pred_raw - pred_raw.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            targets = true_raw.astype(int).ravel()

        return _compute_metrics(targets, probs)

class CBraModWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_times, sfreq, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        self.model = CBraModModule(
            ch_names=ch_names,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=None):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience)
        self.results = self.model.results
        self.best_state_dict = self.model.best_state_dict
        self.best_epoch = self.model.best_epoch
        self.best_val_bacc = self.model.best_val_bacc

    def evaluate(self, dataset, batch_size, ch_names=None):
        """Override evaluate since CBraModModule is not an nn.Module."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        preds, targets, probs = [], [], []
        nn_model = self.model.model  
        is_binary = self.model.n_outputs <= 2

        nn_model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(next(nn_model.parameters()).device).float()
                out = nn_model(X)

                if is_binary:
                    p = torch.sigmoid(out)
                    batch_probs = torch.stack([1 - p, p], dim=1)
                else:
                    batch_probs = torch.softmax(out, dim=1)

                targets.append(y)
                probs.append(batch_probs.cpu())

        targets = torch.cat(targets).numpy()
        probs = torch.cat(probs).numpy()

        return _compute_metrics(targets, probs)


class BIOTWrapper(FinetuningWrapper):
    def __init__(self, ch_names, n_outputs, ckpt_path, train_head_only):
        super().__init__()
        self.model = BIOTModule(
            ch_names=ch_names,
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only
            )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=None):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        extra_callbacks = []
        if early_stopping_patience is not None:
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                mode='min',
                verbose=True
            )
            extra_callbacks.append(early_stop_callback)
            
        trainer = pl.Trainer(accelerator='cuda',
                        max_epochs=epochs, 
                        min_epochs=5,
                        logger=None,
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                        benchmark=True,
                        callbacks=extra_callbacks)
        trainer.fit(self.model, train_dataloader, val_dataloader)
        self.results = self.model.results
        self.best_state_dict = self.model.best_state_dict
        self.best_epoch = self.model.best_epoch
        self.best_val_bacc = self.model.best_val_bacc

    def evaluate(self, dataset, batch_size, ch_names=None):
        """Override evaluate to handle BIOT's single-logit binary classification (n_classes=1)."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        targets, probs = [], []

        self.model.eval()
        device = next(self.model.parameters()).device

        # Use test ch_names if provided (e.g. when test data has dropped channels)
        original_ch_names = self.model.ch_names
        if ch_names is not None:
            self.model.ch_names = ch_names

        with torch.no_grad():
            for X, y in loader:
                X = X.to(device).float()
                out = self.model(X)

                if self.model.n_classes == 1:
                    # Binary: single logit → sigmoid → stack as [1-p, p]
                    p = torch.sigmoid(out).cpu().squeeze(-1)
                    prob = torch.stack([1 - p, p], dim=1)
                else:
                    # Multi-class: standard softmax
                    prob = torch.softmax(out, dim=1).cpu()

                targets.append(y)
                probs.append(prob)

        self.model.ch_names = original_ch_names

        targets = torch.cat(targets).numpy()
        probs = torch.cat(probs).numpy()

        return _compute_metrics(targets, probs)

class NeuroRVQWrapper(FinetuningWrapper):
    def __init__(self, n_time, ch_names, n_outputs, ckpt_path, train_head_only, large_head, exit_block=None):
        super().__init__()
        self.model = NeuroRVQModule(
            sample_length=n_time,
            chnames=ch_names,
            n_out=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            large_head=large_head,
            exit_block=exit_block,
            )

    # def prepare(self):
    #     """Prepare model for evaluation by initializing classifier head."""
    #     self.model.prepare()

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=None):
        self.model.fit(train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience)
        self.results = self.model.results
        self.best_state_dict = self.model.best_state_dict
        self.best_epoch = self.model.best_epoch
        self.best_val_bacc = self.model.best_val_bacc

    def save_model(self, path):
        print(f'Saving checkpoint to {path}...')
        torch.save(self.model.model.state_dict(), path)
    
    def evaluate(self, dataset, batch_size, ch_names=None):
        """Override evaluate to handle NeuroRVQModule which wraps the actual nn.Module."""
        return self.model.evaluate(dataset, batch_size, ch_names=ch_names)
    
    def load_model(self, path):
        """Override load_model to handle NeuroRVQModule."""
        # Set up the correct head structure (incl. large_head) before loading weights
        self.model.prepare()
        state_dict = torch.load(path, map_location='cpu')
        self.model.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model.model.cuda()

class BrainOmniWrapper(FinetuningWrapper):
    def __init__(self, ch_names, sfreq, n_outputs, ckpt_path, train_head_only, skip_tokenizer=False):
        super().__init__()
        self.model = BrainOmniModule(
            ch_names=ch_names,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            skip_tokenizer=skip_tokenizer,
        )

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=None):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader   = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        extra_callbacks = []
        if early_stopping_patience is not None:
            extra_callbacks.append(pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                mode='min',
                verbose=True,
            ))

        trainer = pl.Trainer(
            accelerator='cuda',
            max_epochs=epochs,
            min_epochs=5,
            logger=None,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            benchmark=True,
            callbacks=extra_callbacks,
        )
        trainer.fit(self.model, train_loader, val_loader)
        self.results         = self.model.results
        self.best_state_dict = self.model.best_state_dict
        self.best_epoch      = self.model.best_epoch
        self.best_val_bacc   = self.model.best_val_bacc

    def evaluate(self, dataset, batch_size, ch_names=None):
        self.model._set_eval_ch_names(ch_names)
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            targets, probs = [], []

            self.model.eval()
            device = next(self.model.parameters()).device

            with torch.no_grad():
                for X, y in loader:
                    X    = X.to(device).float()
                    out  = self.model(X)                              # (B, n_outputs)
                    prob = torch.softmax(out, dim=1).cpu()
                    targets.append(y)
                    probs.append(prob)

            targets = torch.cat(targets).numpy()
            probs   = torch.cat(probs).numpy()
            return _compute_metrics(targets, probs)
        finally:
            self.model._set_eval_ch_names(None)


class REVEWrapper(FinetuningWrapper):
    def __init__(self, ch_names, sfreq, n_times, n_outputs, ckpt_path=None, train_head_only=False, large_head=True, exit_block=None):
        super().__init__()
        self.large_head = large_head
        mean_pooling = not large_head  # large_head=True → flatten head; large_head=False → mean-pool head
        self.model = REVEModule(ch_names, sfreq, n_outputs, n_times, ckpt_path, train_head_only, mean_pooling, exit_block=exit_block)

    def load_model(self, path):
        ALLOWED_MISSING    = {"backbone._position_bank.embedding"}
        ALLOWED_UNEXPECTED = {"backbone.cls_query_token"}

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"Loading checkpoint from {path}...")
        state_dict = torch.load(path, map_location='cpu')
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        bad_missing    = [k for k in missing    if k not in ALLOWED_MISSING]
        bad_unexpected = [k for k in unexpected if k not in ALLOWED_UNEXPECTED]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Error(s) loading REVE state_dict:\n"
                + (f"  Unexpected missing keys: {bad_missing}\n" if bad_missing else "")
                + (f"  Unexpected extra keys:   {bad_unexpected}\n" if bad_unexpected else "")
            )

        if missing:
            print(f"  [load_model] known missing keys (using init values): {list(missing)}")
        if unexpected:
            print(f"  [load_model] known extra keys (ignored): {list(unexpected)}")
        self.model.float()

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience=None):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader   = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        extra_callbacks = []
        if early_stopping_patience is not None:
            extra_callbacks.append(pl.callbacks.EarlyStopping(
                monitor='val_loss', patience=early_stopping_patience, mode='min'))
        trainer = pl.Trainer(accelerator='cuda', max_epochs=epochs, min_epochs=5,
                             logger=None, enable_checkpointing=False,
                             num_sanity_val_steps=0, callbacks=extra_callbacks)
        trainer.fit(self.model, train_loader, val_loader)
        self.results         = self.model.results
        self.best_state_dict = self.model.best_state_dict
        self.best_epoch      = self.model.best_epoch
        self.best_val_bacc   = self.model.best_val_bacc

    def evaluate(self, dataset, batch_size, ch_names=None):
        self.model._set_eval_ch_names(ch_names)
        try:
            return super().evaluate(dataset, batch_size, ch_names=ch_names)
        finally:
            self.model._set_eval_ch_names(None)



def get_braindecode_net(model, lr=0.001, weight_decay=0, n_epochs=100, batch_size=64, train_split=None, extra_callbacks=None):
    callbacks = ["accuracy", "balanced_accuracy",
                 ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1))]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return EEGClassifier(
        model,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        train_split=train_split,
        callbacks=callbacks,
        device="cuda",
        max_epochs=n_epochs,
        iterator_train__num_workers=NUM_WORKERS,
        iterator_train__pin_memory=PIN_MEMORY,
        iterator_valid__num_workers=NUM_WORKERS,
        iterator_valid__pin_memory=PIN_MEMORY,
    )



def get_model(model_name, n_chans, ch_names, sfreq, n_times, n_outputs, sbj_ids, encoder_only=False, ckpt_path=None, train_head_only=False, large_head=False, exit_block=None, skip_tokenizer=False):
    """
    Returns: FinetuningWrapper for the specified model
    """
    if model_name == "EEGNet":
        return EEGNetv1Wrapper(
            n_chans=n_chans, 
            n_times=n_times, 
            sfreq=sfreq, 
            n_outputs=n_outputs
            )
    elif model_name == "EEGInception":
        return EEGInceptionWrapper(
            n_chans=n_chans, 
            n_times=n_times, 
            sfreq=sfreq, 
            n_outputs=n_outputs
        )
    elif model_name == "LaBraM":
        return LaBraMWrapper(
            ch_names=ch_names,
            sfreq=sfreq,
            n_times=n_times,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            large_head=large_head,
        )
    elif model_name == "CBraMod":
        return CBraModWrapper(
            ch_names=ch_names,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "BIOT":
        return BIOTWrapper(
            ch_names=ch_names,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "BrainOmni":
        return BrainOmniWrapper(
            ch_names=ch_names,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            skip_tokenizer=skip_tokenizer
        )
    elif model_name == "NeuroRVQ":
        return NeuroRVQWrapper(
            n_time=n_times,
            ch_names=ch_names,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            large_head=large_head, # use REVE head
            exit_block=exit_block,
        )
    elif model_name == "REVE":
        return REVEWrapper(
            ch_names=ch_names,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only,
            large_head=large_head,
            exit_block=exit_block,
        )
        
    else:
        print(f"Undefined model name: {model_name}")
