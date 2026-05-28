from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import AutoModel
from .reve_braindecode import REVE as REVEBraindecode

LR = 1e-4
WEIGHT_DECAY = 1e-5


class REVEModule(pl.LightningModule):
    """PyTorch Lightning wrapper around REVE for supervised fine-tuning."""
    def __init__(
        self,
        ch_names,
        sfreq: int,
        n_outputs: int,
        n_times: int,
        ckpt_path: str = None,
        train_head_only: bool = False,
        mean_pooling: bool = False,
        exit_block: Optional[int] = None,
    ):
        super().__init__()
        self.ch_names = list(ch_names)
        self.sfreq = sfreq
        self.n_outputs = n_outputs
        self.train_head_only = train_head_only
        self.exit_block = exit_block

        # ── Position bank: maps electrode names → (C, 3) 3D coordinates ──────
        hf_id = ckpt_path or "brain-bzh/reve-base"
        pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)

        # pos_bank.mapping is a {name: index} dict of all known electrode names.
        canonical_map = {k.upper(): k for k in pos_bank.mapping.keys()}
        resolved_names = []
        valid_mask_list = []
        for name in self.ch_names:
            if name in pos_bank.mapping:
                resolved_names.append(name)
                valid_mask_list.append(True)
            elif name.upper() in canonical_map:
                resolved_names.append(canonical_map[name.upper()])
                valid_mask_list.append(True)
            else:
                valid_mask_list.append(False)

        valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool)
        pos_valid = pos_bank(resolved_names)   # (C_valid, 3)

        self.register_buffer("_pos", pos_valid)           # (C_valid, 3)
        self.register_buffer("_valid_mask", valid_mask)   # (C,) for channel selection


        self._train_resolved_upper = [n.upper() for n in resolved_names]
        self._canonical_to_bank_idx = {k.upper(): i for i, k in enumerate(pos_bank.mapping.keys())}
        all_canonical = list(pos_bank.mapping.keys())
        self.register_buffer("_all_known_pos", pos_bank(all_canonical).clone(), persistent=False)
        self._eval_state = None
        
        missing_names = [name for name, valid in zip(self.ch_names, valid_mask_list) if not valid]
        print(f"REVE: using {valid_mask.sum().item()}/{len(self.ch_names)} channels with known positions")
        if missing_names:
            print(f"Missing channels: {', '.join(missing_names)}")
        
        # ── Backbone ─────────────────────────────────────────────────────────
        C_valid = pos_valid.shape[0]
        self.backbone = REVEBraindecode(
            n_outputs=n_outputs, n_chans=C_valid, n_times=n_times, sfreq=200,
            embed_dim=512, depth=22, heads=8, head_dim=64,
            mlp_dim_ratio=2.66, use_geglu=True, freqs=4,
            patch_size=200, patch_overlap=20,
            attention_pooling=mean_pooling,   # builds the backbone's pooling layers (then disabled below) so checkpoint weights still load
        )
        hf_weights = AutoModel.from_pretrained(hf_id, trust_remote_code=True).state_dict()
        missing, unexpected = self.backbone.load_state_dict(hf_weights, strict=False)
        print(f"REVE HF weights loaded — missing: {missing}, unexpected: {unexpected}")
        self.backbone.float()  # ensure float32 regardless of checkpoint dtype

        # ── Replace final_layer with our classification head ─────────────────
        original_final = self.backbone.final_layer
        self.backbone.final_layer = nn.Identity()
        embed_dim = self._probe_embed_dim(pos_valid, n_times)
        self.backbone.final_layer = original_final
        if mean_pooling:
            print(f"Using mean pooling with embed_dim={self.backbone.embed_dim} and n_outputs={n_outputs}")
            # mean pool over all tokens: (B, C, T, E) → (B, E); simple linear head
            self.backbone.use_attention_pooling = False  # disable learned attention pooling
            self.backbone.final_layer = nn.Sequential(
                nn.LayerNorm(self.backbone.embed_dim),
                nn.Linear(self.backbone.embed_dim, n_outputs),
            )
            self._use_mean_pooling = True
        else:
            print(f"Using flatten pooling with embed_dim={embed_dim} and n_outputs={n_outputs}")
            # flatten all tokens: (B, C*T*512); large head
            self.backbone.final_layer = nn.Sequential(
                nn.Flatten(),
                nn.RMSNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, n_outputs),
            )
            self._use_mean_pooling = False            

        if train_head_only:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("final_layer"):
                    p.requires_grad = False
                    
        # print n params
        print("Total parameters of REVE:", sum(p.numel() for p in self.backbone.parameters()))
        print("Total trainable parameters of REVE:", sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))

        print("\n" + "="*60)
        print("REVE Architecture:")
        print(f"  {self.backbone.__class__.__name__} — mean_pooling={mean_pooling}")
        print(f"  final_layer: {self.backbone.final_layer}")
        print("="*60 + "\n")

        self.results = {
            "train_accuracy": [], "train_bacc": [],
            "val_accuracy":   [], "val_bacc":   [],
        }
        self.best_state_dict = None
        self.best_epoch = -1
        self.best_val_loss = float("inf")
        self.best_val_bacc = -float("inf")

        self._train_preds: list = []
        self._train_gts:   list = []
        self._val_preds:   list = []
        self._val_gts:     list = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_eval_ch_names(self, eval_ch_names):
        """Configure the module for an eval pass over a different channel set."""
        if eval_ch_names is None or list(eval_ch_names) == self.ch_names:
            self._eval_state = None
            return

        valid_mask, kept_bank_idx, train_slot = [], [], []
        for name in eval_ch_names:
            up = name.upper()
            bidx = self._canonical_to_bank_idx.get(up)
            if bidx is None or up not in self._train_resolved_upper:
                valid_mask.append(False)
                continue
            valid_mask.append(True)
            kept_bank_idx.append(bidx)
            train_slot.append(self._train_resolved_upper.index(up))

        device = self._all_known_pos.device
        valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        eval_pos = self._all_known_pos[torch.tensor(kept_bank_idx, dtype=torch.long, device=device)]
        train_slot_t = torch.tensor(train_slot, dtype=torch.long, device=device)
        self._eval_state = (valid_mask_t, eval_pos, train_slot_t)

    @torch.no_grad()
    def _probe_embed_dim(self, pos_valid: torch.Tensor, n_times: int) -> int:
        """Run a dummy forward through the backbone (with Identity final_layer)
        to determine the flattened embedding dimension for this channel/time config."""
        C_valid = pos_valid.shape[0]
        dummy_x   = torch.zeros(1, C_valid, n_times)
        dummy_pos = pos_valid.unsqueeze(0)        # (1, C_valid, 3)
        out = self.backbone(dummy_x, dummy_pos, exit_block=self.exit_block) 
        return out.flatten(start_dim=1).shape[-1]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, n_outputs) logits. Drops channels with unknown positions."""
        if self.training:
            self.backbone.eval()
            self.backbone.final_layer.train()

        B = X.shape[0]
        X = X.to(next(self.parameters()).dtype)  

        eval_state = getattr(self, '_eval_state', None)
        if eval_state is None:
            valid_mask, pos_buf, train_slot = self._valid_mask, self._pos, None
        else:
            valid_mask, pos_buf, train_slot = eval_state

        X_valid = X[:, valid_mask, :]                                       # (B, C_valid, T)
        pos = pos_buf.unsqueeze(0).expand(B, -1, -1).type_as(X_valid)       # (B, C_valid, 3)

        if getattr(self, '_use_mean_pooling', False):
            orig_final = self.backbone.final_layer
            self.backbone.final_layer = nn.Identity()
            x = self.backbone(X_valid, pos, exit_block=self.exit_block)
            self.backbone.final_layer = orig_final
            x = x.mean(dim=(1, 2))
            return self.backbone.final_layer(x)

        if train_slot is None:
            return self.backbone(X_valid, pos, exit_block=self.exit_block)

        orig_final = self.backbone.final_layer
        self.backbone.final_layer = nn.Identity()
        x = self.backbone(X_valid, pos, exit_block=self.exit_block)         # (B, C_eval, T, D)
        self.backbone.final_layer = orig_final

        C_train_valid = self._pos.shape[0]
        B_, _, T_, D_ = x.shape
        x_padded = torch.zeros(B_, C_train_valid, T_, D_, device=x.device, dtype=x.dtype)
        x_padded[:, train_slot] = x
        return self.backbone.final_layer(x_padded)

    def size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── PL hooks ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = nn.CrossEntropyLoss()(logits, y)
        self._train_preds.append(torch.softmax(logits, dim=1).detach().cpu())
        self._train_gts.append(y.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self._train_preds).argmax(dim=1).numpy()
        gts   = torch.cat(self._train_gts).numpy()
        self.results["train_accuracy"].append(accuracy_score(gts, preds))
        self.results["train_bacc"].append(balanced_accuracy_score(gts, preds))
        self._train_preds.clear()
        self._train_gts.clear()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self(X)
        self._val_preds.append(logits.cpu())
        self._val_gts.append(y.cpu())

    def on_validation_epoch_end(self):
        val_logits = torch.cat(self._val_preds)
        val_gts    = torch.cat(self._val_gts)

        preds    = val_logits.argmax(dim=1).numpy()
        gts      = val_gts.numpy()
        val_acc  = accuracy_score(gts, preds)
        val_bacc = balanced_accuracy_score(gts, preds)
        val_loss = nn.CrossEntropyLoss()(val_logits, val_gts.long()).item()

        self.results["val_accuracy"].append(val_acc)
        self.results["val_bacc"].append(val_bacc)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_bacc", val_bacc, prog_bar=True)

        if self.current_epoch >= 5 and val_loss < self.best_val_loss:
            self.best_val_loss   = val_loss
            self.best_val_bacc   = val_bacc
            self.best_epoch      = self.current_epoch
            self.best_state_dict = {
                k: v.cpu().contiguous().float() if v.is_floating_point() else v.cpu().contiguous()
                for k, v in self.state_dict().items()
            }

        self._val_preds.clear()
        self._val_gts.clear()

    def configure_optimizers(self):
        if self.train_head_only:
            params = self.backbone.final_layer.parameters()
        else:
            params = self.parameters()
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

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
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
