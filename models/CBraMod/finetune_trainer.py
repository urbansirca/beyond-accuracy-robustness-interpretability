import pdb
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from timeit import default_timer as timer
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc

# CBraMod hparams
LABEL_SMOOTHING = 0.1
FROZEN = False
OPTIMIZER = 'AdamW'
MULTI_LR = False
LR=5e-4
WEIGHT_DECAY=5e-2
CLIP_VALUE=1

class Evaluator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.float().cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = accuracy_score(truths, preds)
        bacc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, bacc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.float().cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = accuracy_score(truths, preds)
        bacc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, bacc, pr_auc, roc_auc, cm

class Trainer(object):
    def __init__(self, model, data_loader, n_outputs, epochs, train_head_only, early_stopping_patience):
        self.epochs = epochs
        self.data_loader = data_loader
        self.early_stopping_patience = early_stopping_patience
        self.best_state_dict = None
        self.best_epoch = -1
        self.best_val_bacc = -float('inf')
        self.best_val_loss = float('inf')

        self.train_eval = Evaluator(self.data_loader['train'])
        self.val_eval = Evaluator(self.data_loader['val'])

        self.model = model.cuda()
        self.is_binary = (n_outputs < 3)
        if self.is_binary:
            self.criterion = BCEWithLogitsLoss().cuda()
        else:
            self.criterion = CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).cuda()

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if FROZEN:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if train_head_only:
            for param in model.backbone.parameters():
                param.requires_grad = False
            backbone_params=[]
            
        if OPTIMIZER == 'AdamW':
            if MULTI_LR: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': LR},
                    {'params': other_params, 'lr': LR * 5}
                ], weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR,
                                                   weight_decay=WEIGHT_DECAY)
        else:
            if MULTI_LR:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': LR},
                    {'params': other_params, 'lr': LR * 5}
                ],  momentum=0.9, weight_decay=WEIGHT_DECAY)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=0.9,
                                                 weight_decay=WEIGHT_DECAY)
     
        

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train(self):
        
        if self.is_binary:
            return self.train_for_binaryclass()
        else:
            return self.train_for_multiclass()

    def train_for_multiclass(self):
        results = {'train_accuracy':[], 'train_bacc':[], 'val_accuracy':[], 'val_bacc':[]}
        for epoch in range(self.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                # pdb.set_trace()
                x = x.float().cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_VALUE)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                train_acc, train_bacc, _, _, _ = self.train_eval.get_metrics_for_multiclass(self.model)
                acc, val_bacc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                results['train_accuracy'].append(train_acc)
                results['val_accuracy'].append(acc)
                results['train_bacc'].append(train_bacc)
                results['val_bacc'].append(val_bacc)

                val_losses = []
                for x_val, y_val in self.data_loader['val']:
                    x_val = x_val.float().cuda()
                    y_val = y_val.cuda()
                    val_losses.append(self.criterion(self.model(x_val), y_val).cpu().numpy())
                val_loss = np.mean(val_losses)
                if epoch >= 5 and val_loss < self.best_val_loss:
                    self.best_val_bacc = val_bacc
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

                if self.early_stopping_patience is not None:
                    if epoch > 5 and epoch - self.best_epoch >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} with best val bacc {self.best_val_bacc:.4f} at epoch {self.best_epoch}")
                        break
                print(cm)
        return results

    def train_for_binaryclass(self):
        results = {'train_accuracy':[], 'train_bacc':[], 'val_accuracy':[], 'val_bacc':[]}
        for epoch in range(self.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.float().cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y.float())

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_VALUE)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                train_acc, train_bacc, _, _, _ = self.train_eval.get_metrics_for_binaryclass(self.model)
                acc, val_bacc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )

                print(cm)
                results['train_accuracy'].append(train_acc)
                results['val_accuracy'].append(acc)
                results['train_bacc'].append(train_bacc)
                results['val_bacc'].append(val_bacc)


                val_losses = []
                for x_val, y_val in self.data_loader['val']:
                    x_val = x_val.float().cuda()
                    y_val = y_val.cuda()
                    val_losses.append(self.criterion(self.model(x_val), y_val.float()).cpu().numpy())
                val_loss = np.mean(val_losses)
                if epoch >= 5 and val_loss < self.best_val_loss:
                    self.best_val_bacc = val_bacc
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

                if self.early_stopping_patience is not None:
                    if epoch > 5 and epoch - self.best_epoch >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} with best val bacc {self.best_val_bacc:.4f} at epoch {self.best_epoch}")
                        break
        return results