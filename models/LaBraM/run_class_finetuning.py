# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import pdb
import torch
from torch import nn

from collections import OrderedDict
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy

from . import modeling_finetune
from . import optim_factory
from . import engine_for_finetuning
from . import utils

# original LaBraM hparams
MODEL = "labram_base_patch200_200"
DROP = 0.0
DROP_PATH = 0.1
ATTN_DROP_RATE = 0.0
USE_MEAN_POOLING = True
INIT_SCALE = 0.001
REL_POS_BIAS = False
ABS_POS_EMB = True
LAYER_SCALE_INIT_VALUE = 0.1
QKV_BIAS = False
MODEL_KEY = "model|module"
MODEL_FILTER_NAME = "gzp"
DEVICE = torch.device('cuda:0')
NUM_WORKERS = 8
PIN_MEM = True
UPDATE_FREQ = 1
LAYER_DECAY = 1
DISABLE_WEIGHT_DECAY_ON_REL_POS_BIAS = False
LR = 5e-4
MIN_LR = 1e-6
WARMUP_EPOCHS = 4
WARMUP_STEPS = -1
WEIGHT_DECAY = 0.05
SMOOTHING = 0.1
CLIP_GRAD = None
OPT_EPS = 1e-8
OPT_BETAS = None

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx] 


def get_models(nb_classes):
    model = create_model(
        MODEL,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=DROP,
        drop_path_rate=DROP_PATH,
        attn_drop_rate=ATTN_DROP_RATE,
        drop_block_rate=None,
        use_mean_pooling=USE_MEAN_POOLING,
        init_scale=INIT_SCALE,
        use_rel_pos_bias=REL_POS_BIAS,
        use_abs_pos_emb=ABS_POS_EMB,
        init_values=LAYER_SCALE_INIT_VALUE,
        qkv_bias=QKV_BIAS,
    )

    return model

class LaBraMModule():
    def __init__(self, ch_names, sfreq, n_times, n_outputs, ckpt_path, train_head_only=False, large_head=False, concat_pool=False):
        self.chnames = ch_names
        self.sample_rate = sfreq
        self.nb_classes = n_outputs if n_outputs > 2 else 1
        self.metrics = ["accuracy", "balanced_accuracy"]
        self.model = get_models(self.nb_classes)
        self.train_head_only = train_head_only
        self.large_head = large_head
        self.concat_pool = concat_pool

        if ckpt_path is None:
            ckpt_path = "weights/pretrained/labram-base.pth"
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % ckpt_path)
        checkpoint_model = None
        for model_key in MODEL_KEY.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (MODEL_FILTER_NAME != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = self.model.state_dict()
        
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(self.model, checkpoint_model, prefix='')
        
        # if self.large_head:
        #     self.model.head = nn.Sequential(
        #             nn.Flatten(),
        #             nn.RMSNorm(self.model.embed_dim),
        #             nn.Dropout(0.1),
        #             nn.Linear(self.model.embed_dim, self.nb_classes),
        #         )
        #     print("In the large head, embedding dim is:", self.model.embed_dim)

        if self.concat_pool:
            n_time = n_times // self.model.patch_size
            n_tokens = len(ch_names) * n_time
            in_features = n_tokens * self.model.embed_dim
            self.model.use_concat_pooling = True
            self.model.head = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(in_features),
                nn.Dropout(0.1),
                nn.Linear(in_features, self.nb_classes),
            )
            print(f"Concat pooling head: {n_tokens} tokens × {self.model.embed_dim} dim = {in_features} in_features")

        
        print("==="*60)
        print(self.model)
        print("==="*60)


        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.results = {"train_accuracy":[], "train_bacc":[], "val_accuracy":[], "val_bacc":[]}
        self.best_state_dict = None
        self.best_epoch = -1
        self.best_val_bacc = -float('inf')
        self.best_val_loss = float('inf')

    def size(self):
        """ Returns number of trainable parameters in model """
        return self.n_parameters
    
    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience):
        self.model.to(DEVICE)
        model_without_ddp = self.model
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.SequentialSampler(validation_dataset)

        data_loader_train = torch.utils.data.DataLoader(
            train_dataset, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEM,
            drop_last=True,
        )

        if validation_dataset is not None:
            data_loader_val = torch.utils.data.DataLoader(
                validation_dataset, sampler=sampler_val,
                batch_size=int(1.5 * batch_size),
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEM,
                drop_last=False
            )
        else:
            data_loader_val = None

        total_batch_size = batch_size * UPDATE_FREQ * utils.get_world_size()
        num_training_steps_per_epoch = max(1, len(train_dataset) // total_batch_size)
        print("LR = %.8f" % LR)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % UPDATE_FREQ)
        print("Number of training examples = %d" % len(train_dataset))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_layers = model_without_ddp.get_num_layers()
        if LAYER_DECAY < 1.0:
            assigner = optim_factory.LayerDecayValueAssigner(list(LAYER_DECAY ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = self.model.no_weight_decay()
        if DISABLE_WEIGHT_DECAY_ON_REL_POS_BIAS:
            for i in range(num_layers):
                skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)
        
        if self.train_head_only:
            for name, param in model_without_ddp.named_parameters():
                if not 'head' in name:
                    param.requires_grad = False

        optimizer = optim_factory.create_optimizer(
            model_without_ddp, weight_decay=WEIGHT_DECAY, lr=LR, opt_eps=OPT_EPS, opt_betas=OPT_BETAS, 
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = utils.NativeScalerWithGradNormCount()

        print("Use step level LR scheduler!")
        if epochs < WARMUP_EPOCHS + 1:
            lr_schedule_values = utils.cosine_scheduler(
                LR, MIN_LR, epochs, num_training_steps_per_epoch,
                warmup_epochs=0, warmup_steps=WARMUP_STEPS,
            )
        else:
            lr_schedule_values = utils.cosine_scheduler(
                LR, MIN_LR, epochs, num_training_steps_per_epoch,
                warmup_epochs=WARMUP_EPOCHS, warmup_steps=WARMUP_STEPS,
            )

        wd_schedule_values = utils.cosine_scheduler(
            WEIGHT_DECAY, WEIGHT_DECAY, epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        if self.nb_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))
        
        # Training
        print(f"Start training for {epochs} epochs")
        for epoch in range(epochs):
            train_stats = engine_for_finetuning.train_one_epoch(
                self.model, criterion, data_loader_train, optimizer,
                DEVICE, epoch, loss_scaler, CLIP_GRAD, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=UPDATE_FREQ, 
                ch_names=self.chnames, is_binary=self.nb_classes == 1
            )
    
            if data_loader_val is not None:
                val_stats = engine_for_finetuning.evaluate(data_loader_val, self.model, DEVICE, header='Val:', ch_names=self.chnames, metrics=self.metrics, is_binary=self.nb_classes == 1)
                print(f"Accuracy of the network on the {len(validation_dataset)} val EEG: {val_stats['accuracy']:.2f}%")
                val_loss = val_stats['loss']

            self.results['train_accuracy'].append(train_stats['class_acc'])
            self.results['val_accuracy'].append(val_stats['accuracy'])
            self.results['train_bacc'].append(train_stats['class_bacc'])
            val_bacc = val_stats['balanced_accuracy']
            self.results['val_bacc'].append(val_bacc)

            if epoch >= 5 and val_loss < self.best_val_loss:
                self.best_val_bacc = val_bacc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
                
            if early_stopping_patience is not None:
                if epoch > 5 and epoch - self.best_epoch >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} with best val bacc {self.best_val_bacc:.4f} at epoch {self.best_epoch}")
                    break