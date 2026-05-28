import pdb
import os
import sys
import torch
import warnings
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# sys.path.insert(0, '/home/usirca/workspace/NeuroRVQ')
# from NeuroRVQ import NeuroRVQFM

from models.NeuroRVQm.model import NeuroRVQFM

PIN_MEMORY = True
NUM_WORKERS = 8


# NeuroRVQ config (copied from flags_NeuroRVQ.yml and utils/functional.py)
EEG_size= 1600
patch_size= 200
n_patches= 256
n_code= 8192 # 8192
use_for_pretraining= False
in_chans_second_stage= 1
out_chans_second_stage= 8 # 8 (base), 16 (large), 32 (huge)
depth_second_stage= 12 # 12 (base), 24 (large), 48 (huge)
num_heads_second_stage= 10 # 10 (base), 16 (large), 16 (huge)
mlp_ratio_second_stage= 4.
qkv_bias_second_stage= True
drop_rate_second_stage= 0.
attn_drop_rate_second_stage= 0.
drop_path_rate_second_stage= 0.
init_values_second_stage= 1.e-5 # 0.1 (base), 1.e-5 (large), 1.e-6 (huge)
init_scale_second_stage= 0.001
embed_dim_second_stage= 200 # 200 (base), 400 (large), 800 (huge)
lr = 5e-4
layer_decay = 0.975
weight_decay = 1e-2
amp_dtype=torch.bfloat16
warmup_epochs = 4

ch_names_global = np.array([b'a1', b'a2', b'af3', b'af4', b'af7', b'af8', b'afz', b'c1', b'c2',
    b'c3', b'c4', b'c5', b'c6', b'ccp1', b'ccp2', b'ccp3', b'ccp4',
    b'ccp5', b'ccp6', b'ccp7', b'ccp8', b'cfc1', b'cfc2', b'cfc3',
    b'cfc4', b'cfc5', b'cfc6', b'cfc7', b'cfc8', b'cp1', b'cp2',
    b'cp3', b'cp4', b'cp5', b'cp6', b'cpz', b'cz', b'eog', b'f1',
    b'f10', b'f2', b'f3', b'f4', b'f5', b'f6', b'f7', b'f8', b'f9',
    b'fc1', b'fc2', b'fc3', b'fc4', b'fc5', b'fc6', b'fcz', b'fp1',
    b'fp2', b'fpz', b'ft7', b'ft8', b'fz', b'iz', b'loc', b'o1', b'o2',
    b'oz', b'p08', b'p1', b'p10', b'p2', b'p3', b'p4', b'p5', b'p6',
    b'p7', b'p8', b'p9', b'po1', b'po10', b'po2', b'po3', b'po4',
    b'po7', b'po8', b'po9', b'poz', b'pz', b'roc', b'sp1', b'sp2',
    b't1', b't10', b't2', b't3', b't4', b't5', b't6', b't7', b't8',
    b't9', b'tp10', b'tp7', b'tp8', b'tp9'])

def create_embedding_ix(n_time, max_n_patches, ch_names_sample, ch_names_global):
    """Creates temporal and spatial embedding indices for a sample with given regular shape.
    Args:
        n_time: Int. Number of patches along the time dimension
        max_n_patches: The maximum number of patches, for aligning the current time-point to the right.
        ch_names_sample (n_channels_sample,): The specific channel names of the sample
        ch_names_global (n_channels_global): The reference channel names of the model
    Returns:
        temp_embed_ix (1, n_patches): tensor
        spat_embed_ix (1, n_patches): tensor
    """

    # Temporal embedding ix
    temp_embed_ix = torch.arange(max_n_patches - n_time, max_n_patches)
    temp_embed_ix = temp_embed_ix.repeat(len(ch_names_sample))
    temp_embed_ix = temp_embed_ix.reshape(1, -1)

    # Spatial embedding ix
    spat_embed_ix = torch.tensor([np.where(ch_names_global == c)[0][0] for c in ch_names_sample])
    spat_embed_ix = torch.repeat_interleave(spat_embed_ix, n_time)
    spat_embed_ix = spat_embed_ix.reshape(1, -1)

    return temp_embed_ix, spat_embed_ix

def get_class_weights(y, n_cls):
    y = torch.Tensor(y)
    class_weights = torch.unique(y, return_counts=True)[1]
    class_weights = 1 / class_weights
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights * len(torch.unique(y))  # (n_classes,)
    if len(class_weights) < n_cls:
        tmp = class_weights
        class_weights = torch.zeros(n_cls)
        class_weights[:len(tmp)] = tmp
    class_weights = class_weights.cuda()
    return class_weights

class NeuroRVQModule():
    def __init__(self, sample_length, chnames, n_out, ckpt_path, train_head_only, large_head, exit_block=None):
        self.n_time = sample_length // patch_size
        chnames = np.array([c.lower().encode() for c in chnames])
        self.chmask = np.isin(chnames, ch_names_global)
        self.chnames = chnames[self.chmask]
        self.n_out = n_out
        self.model = NeuroRVQFM(n_patches=n_patches,
                                    patch_size=patch_size,
                                    in_chans=in_chans_second_stage, out_chans=out_chans_second_stage,
                                    num_classes=0,
                                    embed_dim=embed_dim_second_stage,
                                    depth=depth_second_stage,
                                    num_heads=num_heads_second_stage,
                                    mlp_ratio=mlp_ratio_second_stage, qkv_bias=qkv_bias_second_stage,
                                    qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=drop_rate_second_stage,
                                    attn_drop_rate=attn_drop_rate_second_stage,
                                    drop_path_rate=drop_path_rate_second_stage,
                                    init_values=init_values_second_stage,
                                    init_scale=init_scale_second_stage,
                                    n_global_electrodes=len(ch_names_global),
                                    use_as_encoder=True, vocab_size=n_code,
                                    use_for_pretraining=use_for_pretraining)
        
        if ckpt_path is None:
            ckpt_path = "weights/pretrained/neurorvq-base.pt"
        if ckpt_path is not None and os.path.exists(ckpt_path):
            model_state_dict = torch.load(ckpt_path, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            print(f"Missing keys: {missing_keys},\nUnexpected keys: {unexpected_keys}")
        else:
            print(f"Pretrained backbone not found at {ckpt_path} — proceeding with random init (assumed to be overwritten by finetuned weights).")

        self.train_head_only = train_head_only
        self.large_head = large_head
        self.exit_block = exit_block
        
        
        self.criterion = F.cross_entropy if self.n_out > 2 else F.binary_cross_entropy_with_logits

        self.results = {'train_accuracy' : [], 'val_accuracy' : [], 'train_bacc' : [], 'val_bacc' : []}
        self.best_state_dict = None
        self.best_epoch = -1
        self.best_val_bacc = -float('inf')
        self.best_val_loss = float('inf')
        
        self.prepare()
    
    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def prepare(self):
        """Initialize classifier head and move model to GPU."""
        d_out = self.n_out if self.n_out > 2 else 1
        self.model.reset_classifier(d_out)
        if self.large_head:
            n_tokens = len(self.chnames) * self.n_time
            in_features = n_tokens * embed_dim_second_stage * 4
            self.model.use_concat_pooling = True
            self.model.fc_norm = nn.Identity()
            self.model.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(0.1),
                nn.Linear(in_features, d_out),
            )
        if self.train_head_only:
            for name, param in self.model.named_parameters():
                if 'head.' in name or 'fc_norm.' in name:
                    continue
                else:
                    param.requires_grad = False

        self.model.cuda()

        # print("==="* 60)
        # print(self.model)
        # print("==="* 60)


    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience):
        self.prepare()
        # Set model parameter groups with layer_decay on the learning rate

        param_groups = {}
        for i_m, (p_name, param) in enumerate(self.model.named_parameters()):  # model layers
            if not param.requires_grad:
                continue
            if ('head.' in p_name) or ('fc_norm.' in p_name): # normal lr for classification head
                # head_lr = lr * 0.1 if self.large_head else lr
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': weight_decay}
                                        # 'lr': head_lr}
            else:
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': weight_decay,
                                        'lr': lr * layer_decay ** (
                                                len(list(self.model.named_parameters())) - i_m)}

        # Optimizer and lr_scheduler
        optimizer = torch.optim.AdamW(list(param_groups.values()))
        n_batches_train = int(np.ceil(len(train_dataset) / batch_size))
        if epochs < warmup_epochs + 1 :
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1,
                                                        total_iters=epochs * n_batches_train)
        else:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1,
                                                            total_iters=warmup_epochs * n_batches_train)
            scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-1,
                                                            total_iters=(epochs-warmup_epochs) * n_batches_train)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2],
                                                                    milestones=[warmup_epochs * n_batches_train])
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')
        # Prepare automatic mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        y_train = [ys for _,ys in train_dataset]
        y_val = [ys for _,ys in validation_dataset]
        y = y_train + y_val
        class_weights = get_class_weights(y, self.n_out)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        temp_embed_ix, spat_embed_ix = create_embedding_ix(self.n_time, n_patches, self.chnames, ch_names_global)
        
        # Loop over epochs
        for i_epoch in range(epochs):
            print(f"Epoch {i_epoch}")
            # Loop over training batches
            self.model.train()
            e_pred_train = []  # collect predictions
            y_true_train = [] # y in order seen
            for x_b, y_b in tqdm(train_dataloader):
                x_b = x_b[:,self.chmask,:]
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, patch_size).cuda()
                y_b = y_b.long() if self.n_out > 2 else y_b.float()
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    optimizer.zero_grad()
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix, exit_block=self.exit_block)
                    p = p.squeeze(-1)  # remove class dim if binary task
                    loss_weight = class_weights if p.ndim == 2 else class_weights[y_b.long()]
                    loss = self.criterion(p, y_b.cuda(), weight=loss_weight)

                scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                # Collect class predictions to compute metrics on the full epoch
                p = p.detach().cpu().float()
                p = p.argmax(dim=-1) if p.ndim == 2 else torch.round(torch.sigmoid(p))
                e_pred_train += [p.numpy()]
                y_true_train += [y_b.numpy()]

            # Loop over validation batches
            self.model.eval()
            e_pred_val = []  # collect predictions
            y_true_val = [] # y in order seen
            for x_b, y_b in tqdm(val_dataloader):
                x_b = x_b[:,self.chmask,:]
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, patch_size).cuda()
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix, exit_block=self.exit_block)
                    p = p.squeeze(-1)  # remove class dim if binary task        

                # Collect class predictions to compute metrics on the full epoch
                p = p.detach().cpu().float()
                p = p.argmax(dim=-1) if p.ndim == 2 else torch.round(torch.sigmoid(p))
                e_pred_val += [p.numpy()]
                y_true_val += [y_b.numpy()]

            # Compute accuracy and balanced accuracy
            e_pred_train = np.concatenate(e_pred_train)
            e_pred_val = np.concatenate(e_pred_val)
            y_true_train = np.concatenate(y_true_train)
            y_true_val = np.concatenate(y_true_val)
            
            self.results['train_accuracy'] += [accuracy_score(y_true_train, e_pred_train)]
            self.results['val_accuracy'] += [accuracy_score(y_true_val, e_pred_val)]
            self.results['train_bacc'] += [balanced_accuracy_score(y_true_train, e_pred_train)]
            val_bacc = balanced_accuracy_score(y_true_val, e_pred_val)
            self.results['val_bacc'] += [val_bacc]

            val_loss = loss.item()
            if i_epoch >= 5 and val_loss < self.best_val_loss:
                self.best_val_bacc = val_bacc
                self.best_val_loss = val_loss
                
                self.best_epoch = i_epoch
                self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if early_stopping_patience is not None:
                if i_epoch > 5 and i_epoch - self.best_epoch >= early_stopping_patience:
                    print(f"Early stopping at epoch {i_epoch}. Best epoch was {self.best_epoch} with val_bacc {self.best_val_bacc}")
                    break
            if len(validation_dataset) > 1:
                print(f"VAL ACC: {self.results['val_accuracy'][-1]}, VAL BACC: {self.results['val_bacc'][-1]}")
    
    def evaluate(self, dataset, batch_size, ch_names=None):
        """Evaluate the model on a dataset and return metrics.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            ch_names: Optional list of channel names in the dataset. If provided and different
                     from training channels, will create a new channel mask.
        """
        from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        # Handle potentially different channels in test data
        if ch_names is not None:
            ch_names_encoded = np.array([c.lower().encode() for c in ch_names])
            chmask = np.isin(ch_names_encoded, ch_names_global)
            chnames_eval = ch_names_encoded[chmask]
        else:
            chmask = self.chmask
            chnames_eval = self.chnames
        
        temp_embed_ix, spat_embed_ix = create_embedding_ix(self.n_time, n_patches,
                                                            chnames_eval, ch_names_global)
        
        self.model.eval()
        preds = []
        targets = []
        probs = []
        
        with torch.no_grad():
            for x_b, y_b in tqdm(dataloader, desc="Evaluating"):
                x_b = x_b[:, chmask, :].float()  # Ensure float32 dtype
                n, c, t = x_b.shape
                x_b = x_b.reshape(n, c, self.n_time, patch_size).cuda()
                
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    p, _ = self.model(x_b, temp_embed_ix, spat_embed_ix, exit_block=self.exit_block)
                    p = p.squeeze(-1)  # remove class dim if binary task
                
                p = p.detach().cpu().float()
                
                # Get probabilities
                if p.ndim == 2:  # multiclass
                    prob = torch.softmax(p, dim=-1)
                    pred = p.argmax(dim=-1)
                else:  # binary
                    prob = torch.sigmoid(p)
                    pred = torch.round(prob)
                    # Convert to 2-class probability format for metrics
                    prob = torch.stack([1 - prob, prob], dim=-1)
                
                preds.append(pred)
                targets.append(y_b)
                probs.append(prob)
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        probs = torch.cat(probs).numpy()
        
        # Compute metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['bacc'] = balanced_accuracy_score(targets, preds)
        metrics['kappa'] = cohen_kappa_score(targets, preds)
        
        # Handle different cases for F1 and ROC AUC
        n_classes = len(np.unique(targets))
        if n_classes > 2:
            metrics['f1_weighted'] = f1_score(targets, preds, average='weighted', zero_division=0)
            metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            except ValueError:
                metrics['roc_auc'] = np.nan
        else:  # binary
            metrics['f1_weighted'] = f1_score(targets, preds, average='weighted', zero_division=0)
            metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probs[:, 1])
            except ValueError:
                metrics['roc_auc'] = np.nan
        
        return metrics