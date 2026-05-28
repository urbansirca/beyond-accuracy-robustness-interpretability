from torch.utils.data import DataLoader

from .model_for_finetuning import Model
from .finetune_trainer import Trainer


PIN_MEMORY = True
NUM_WORKERS = 8

class CBraModModule():
    def __init__(self, ch_names, n_times, sfreq, n_outputs, ckpt_path, train_head_only):
        n_chans = len(ch_names)
        self.n_outputs = n_outputs
        if ckpt_path is None:
            ckpt_path = "weights/pretrained/cbramod-base.pth"
        self.model = Model(n_chans=n_chans, input_length=n_times, sfreq=sfreq, n_outputs=n_outputs, load_path=ckpt_path)
        self.train_head_only=train_head_only

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs, early_stopping_patience):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        data_loader = {'train': train_dataloader, 'val': val_dataloader}

        trainer = Trainer(self.model, data_loader, self.n_outputs, epochs, self.train_head_only, early_stopping_patience)
        self.results = trainer.train()
        self.best_state_dict = trainer.best_state_dict
        self.best_epoch = trainer.best_epoch
        self.best_val_bacc = trainer.best_val_bacc