import torch
import torch.nn as nn

from . import cbramod


class Model(nn.Module):
    def __init__(self, n_chans, input_length, sfreq, n_outputs, load_path, use_pretrained_weights=True, cuda=0, dropout=0.1):
        super(Model, self).__init__()
        self.backbone = cbramod.CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        self.num_of_classes = 1 if n_outputs <=2 else n_outputs

        if use_pretrained_weights:
            map_location = torch.device(f'cuda:{cuda}')
            self.backbone.load_state_dict(torch.load(load_path, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        self.len_seconds = input_length // sfreq
        self.classifier = nn.Sequential(
            nn.Linear(n_chans * self.len_seconds * 200, self.len_seconds * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.len_seconds * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, self.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, t = x.shape
        x = x.reshape(bz, ch_num, self.len_seconds, -1)

        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num*self.len_seconds*200)
        if self.num_of_classes > 1:
            out = self.classifier(feats)
        else:
            out = self.classifier(feats).contiguous().view(bz)
        return out