import os
import json

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from models.BrainOmni._repo import ensure_repo_on_path

import torch.nn.functional as F
from einops import rearrange
# ── Hyperparameters ──────────────────────────────────────────────────────────
LR = 1e-4
WEIGHT_DECAY = 1e-5
BRAINOMNI_SFREQ = 256



@torch.no_grad()
def _tokenize_no_vq(self, x, pos, sensor_type, overlap_ratio, **kwargs):
    self.eval()
    x = self.unfold(x, overlap_ratio=overlap_ratio)
    sensor_embedding = self.sensor_embed(pos, sensor_type)
    feature = self.encoder(x, sensor_embedding)
    feature = F.normalize(feature, p=2.0, dim=-1)
    B, C, N, T, D = feature.shape
    indices = torch.zeros(B, C, N, T, 1, dtype=torch.long, device=feature.device)
    feature = rearrange(feature, "B C N T D -> B C (N T) D")
    indices = rearrange(indices, "B C N T Q -> B C (N T) Q")
    return feature, indices



# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_brainomni(ckpt_path: str, skip_tokenizer: bool = False):
    """Load BrainOmni weights from a checkpoint directory."""
    ensure_repo_on_path()
    from brainomni.model import BrainOmni

    cfg_path = os.path.join(ckpt_path, 'model_cfg.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"model_cfg.json not found in {ckpt_path}. "
            "Download the checkpoint with: "
            "huggingface-cli download OpenTSLab/BrainOmni "
            "--local-dir weights/pretrained/BrainOmni"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = BrainOmni(**cfg)
    sd = torch.load(
        os.path.join(ckpt_path, 'BrainOmni.pt'),
        map_location='cpu',
        weights_only=True,
    )
    model.load_state_dict(sd, strict=False)
    # Freeze tokenizer — it is never updated during fine-tuning
    for p in model.tokenizer.parameters():
        p.requires_grad = False
        
    if skip_tokenizer:
        print("Skipping BrainOmni tokenizer")
        model.tokenizer.tokenize = _tokenize_no_vq.__get__(model.tokenizer, type(model.tokenizer))

    return model, cfg.get('dim', cfg.get('embed_dim', 768))


def _get_eeg_positions(ch_names) -> np.ndarray:
    """Return (C, 6) float32 array of electrode positions."""
    import mne
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_pos = montage.get_positions()['ch_pos']
    # Build a case-insensitive lookup
    pos_lookup = {k.upper(): v for k, v in ch_pos.items()}

    rows = []
    missing = []
    for ch in ch_names:
        xyz = pos_lookup.get(ch.upper(), None)
        if xyz is None:
            xyz = np.zeros(3, dtype=np.float32)
            missing.append(ch)
        rows.append(np.concatenate([xyz.astype(np.float32), np.zeros(3, dtype=np.float32)]))
    
    if missing:
        print(f"BrainOmni Warning: {len(missing)} channels have no known position (set to 0,0,0): {missing[:5]}...")
        
    return np.array(rows, dtype=np.float32)  # (C, 6)


# ── Module ───────────────────────────────────────────────────────────────────

class BrainOmniModule(pl.LightningModule):
    """PyTorch Lightning wrapper around BrainOmni for supervised fine-tuning."""

    def __init__(
        self,
        ch_names,
        sfreq: int,
        n_outputs: int,
        ckpt_path: str = None,
        train_head_only: bool = False,
        skip_tokenizer: bool = False,
    ):
        super().__init__()
        self.ch_names = list(ch_names)
        self.sfreq = sfreq
        self.n_outputs = n_outputs
        self.train_head_only = train_head_only
        self.skip_tokenizer = skip_tokenizer

        # ── Pretrained backbone ───────────────────────────────────────────
        if ckpt_path is None:
            ckpt_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '../../weights/pretrained/BrainOmni/base')
            )
            
        self.backbone, self.embed_dim = _load_brainomni(ckpt_path, skip_tokenizer=skip_tokenizer)

        pos_np = _get_eeg_positions(self.ch_names)                        # (C, 6)        
        xyz = pos_np[:, :3]
        xyz -= xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz ** 2, axis=1)))
        if scale > 0:
            xyz /= scale
        pos_np[:, :3] = xyz
        sensor_np = np.zeros(len(self.ch_names), dtype=np.int64)          # (C,) — EEG = 0
        self.register_buffer('_pos', torch.from_numpy(pos_np))
        self.register_buffer('_sensor_type', torch.from_numpy(sensor_np))
        self._eval_state = None

        # ── Downstream Model (Backbone + Classifier) ──────────────────────
        from downstream.model import DownstreamModel
        self.downstream_model = DownstreamModel(
            backbone=self.backbone,
            frozen=train_head_only,
            n_dim=self.embed_dim,
            num_classes=n_outputs,
        )

        self.results = {
            'train_accuracy': [], 'train_bacc': [],
            'val_accuracy': [],   'val_bacc': [],
        }
        self.best_state_dict = None
        self.best_epoch = -1
        self.best_val_loss = float('inf')
        self.best_val_bacc = -float('inf')

        self._train_preds: list = []
        self._train_gts: list = []
        self._val_preds: list = []
        self._val_gts: list = []

        self._init_lazy_layers()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _set_eval_ch_names(self, eval_ch_names):
        """Configure the module for an eval pass over a different channel set."""
        if eval_ch_names is None or list(eval_ch_names) == self.ch_names:
            self._eval_state = None
            return

        device = self._pos.device
        pos_np = _get_eeg_positions(list(eval_ch_names))
        xyz = pos_np[:, :3]
        xyz -= xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz ** 2, axis=1)))
        if scale > 0:
            xyz /= scale
        pos_np[:, :3] = xyz
        sensor_np = np.zeros(len(eval_ch_names), dtype=np.int64)

        self._eval_state = (
            torch.from_numpy(pos_np).to(device),
            torch.from_numpy(sensor_np).to(device),
        )

    def _prepare_input(self, X: torch.Tensor):
        """Normalize and prepare input dict for BrainOmni forward pass."""
        X = X - X.mean(dim=1, keepdim=True)
        std = X.std(dim=(1, 2), keepdim=True) + 1e-5
        X = X / std

        B = X.shape[0]
        eval_state = getattr(self, '_eval_state', None)
        if eval_state is None:
            pos_buf, st_buf = self._pos, self._sensor_type
        else:
            pos_buf, st_buf = eval_state
            pos_buf = pos_buf.to(X.device)
            st_buf = st_buf.to(X.device)
        pos = pos_buf.unsqueeze(0).expand(B, -1, -1).float()  # (B, C, 6)
        st  = st_buf.unsqueeze(0).expand(B, -1)               # (B, C)

        return {'x': X, 'pos': pos, 'sensor_type': st}


    def _init_lazy_layers(self):
        """Run a dummy forward pass to initialize LazyLinear weights."""
        C = len(self.ch_names)
        dummy_x = torch.zeros(1, C, BRAINOMNI_SFREQ * 4)
        dummy_pos = self._pos.unsqueeze(0).float()
        dummy_st = self._sensor_type.unsqueeze(0)
        input_dict = {'x': dummy_x, 'pos': dummy_pos, 'sensor_type': dummy_st}
        self.downstream_model.predict(input_dict)

    # ── nn.Module interface ───────────────────────────────────────────────
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        input_dict = self._prepare_input(X.float())
        logits, _ = self.downstream_model.predict(input_dict)
        return logits

    def size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx):
        X, y = batch
        input_dict = self._prepare_input(X.float())
        logits, loss = self.downstream_model(input_dict, y)
        self._train_preds.append(torch.softmax(logits, dim=1).detach())
        self._train_gts.append(y.detach())
        return loss

    def on_train_epoch_end(self):
        preds_t = torch.cat(self._train_preds).cpu()   # (N, n_outputs)
        gts_t   = torch.cat(self._train_gts).cpu()
        preds   = preds_t.argmax(dim=1).numpy()
        gts     = gts_t.numpy()
        self.results['train_accuracy'].append(accuracy_score(gts, preds))
        self.results['train_bacc'].append(balanced_accuracy_score(gts, preds))
        self._train_preds.clear()
        self._train_gts.clear()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self(X)
        self._val_preds.append(logits.detach())
        self._val_gts.append(y.detach())
        return logits, y

    def on_validation_epoch_end(self):
        val_logits = torch.cat(self._val_preds).cpu()  # (N, n_outputs)
        val_gts    = torch.cat(self._val_gts).cpu()

        preds   = val_logits.argmax(dim=1).numpy()
        gts     = val_gts.numpy()
        val_acc  = accuracy_score(gts, preds)
        val_bacc = balanced_accuracy_score(gts, preds)
        val_loss = nn.CrossEntropyLoss()(val_logits, val_gts.long()).item()

        self.results['val_accuracy'].append(val_acc)
        self.results['val_bacc'].append(val_bacc)
        self.log('val_bacc', val_bacc, prog_bar=True)
        self.log('val_loss', val_loss, prog_bar=True)

        if self.current_epoch >= 5 and val_loss < self.best_val_loss:
            self.best_val_loss  = val_loss
            self.best_val_bacc  = val_bacc
            self.best_epoch     = self.current_epoch
            self.best_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}

        self._val_preds.clear()
        self._val_gts.clear()

    def configure_optimizers(self):
        params = self.downstream_model.class_head.parameters() if self.train_head_only else self.parameters()
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY,
                                      betas=(0.9, 0.99), eps=1e-6)

        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.1 * total_steps))
        sched_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
        )
        sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=LR * 0.1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_steps],
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }
