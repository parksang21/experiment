from sys import setdlopenflags
from pytorch_lightning import LightningModule
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics import AUROC
import wandb

from data.cifar10 import SplitCifar10, train_transform, val_transform, OpenTestCifar10
from data.capsule_split import get_splits
from utils import accuracy
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.apply(weights_init)
        
    def block0(self, x):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)
        return x
    
    def block1(self, x):
        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x) 
        return x
    
    def block2(self, x):
        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        
        return x
        
    def forward(self, x):
        l1 = self.block0(x)
        l2 = self.block1(l1)
        l3 = self.block2(l2)
        
        y = self.fc1(l3)
        
        return {
            'logit': y,
            'l3': l3,
            'l2': l2,
            'l1': l1,
        }
        
    def forward_l2(self, x):
        l3 = self.block2(x)
        
        logit = self.fc1(l3)
        return logit

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channel: int=128,
                 out_channel: int=128,
                 kernal_size: int=3,
                 stride: int=1,
                 padding: int=1,
                 **kwargs):
        super(self.__class__, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernal_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU(0.2)
        
        self.apply(weights_init)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.activation(out)
        
        return x + out
        
class NoiseGeneration(LightningModule):
    def __init__(self,
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 data_dir: str = '/datasets',
                 dataset: str = 'cifar10',
                 batch_size: int = 128,
                 num_workers: int = 48,
                 class_split: int = 0,
                 latent_size: int = 128,
                 alpha: float = 0.5,
                 log_dir: str = './log',
                 **kwargs):
        
        super(self.__class__, self).__init__()
        
        self.save_hyperparameters()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_split = class_split
        self.latent_size = latent_size
        self.alpha = alpha
        self.log_dir = log_dir
        
        self.splits = get_splits(dataset, class_split)
        self.num_classes = len(self.splits['known_classes'])
        
        self.model = classifier32(num_classes=len(self.splits['known_classes']))
        # self.memory = nn.Parameter(torch.randn(6, 128, 8, 8, requires_grad=True))
        self.convB = ConvBlock(in_channel=128, out_channel=128, kernal_size=3, stride=1, padding=1)
        self.oFC = nn.Linear(128, 3)

        self.criterion = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss()
        self.auroc1 = AUROC(pos_label=1)
        self.auroc2 = AUROC(pos_label=1)
        self.auroc3 = AUROC(pos_label=1)
        
        # self.automatic_optimization = False

    def on_train_start(self) -> None:
        wandb.save(self.log_dir+f"/{__file__.split('/')[-1]}")    
    
    def forward(self, x):
        out = self.model(x)
        return out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        out = self(x)
        
        
        c_loss = self.criterion(out['logit'], y)
        
        acc = accuracy(out['logit'], y)[0]
        
        new_f = self.convB(out['l2'])
        new_out = self.model.block2(new_f)
        
        new_logit = self.oFC(new_out)
        new_y = new_logit.max(-1)[1]
        
        open_loss = self.criterion(torch.cat([new_logit, out['logit']], dim=1), new_y)
        
        
        log_dict = {
            'classification loss': c_loss,
            'open loss': open_loss,
            # 'gen loss': gen_loss,
            # 'kld loss': kld_loss,
            'acc': acc
        }
        
        self.log_dict(log_dict, on_step=True)
        
        loss = c_loss + open_loss
        return loss
        
        
    def kldiv(self, P, Q):
        return (P * (P / Q).log()).sum(-1)
    
    def cal_index(self, y):
        batch_size = y.size(0)
        new_index = torch.randperm(batch_size).type_as(y)
        newY = y[new_index]
        mask = (newY == y)
        while mask.any().item():
            newY[mask] = torch.randint(0, self.num_classes, (torch.sum(mask),)).type_as(y)
            mask = (newY == y)
        return newY
       
    
    def validation_step(self, batch, batch_idx):
        x, train_y, y, known_idxs = batch
       
        out = self(x)
        new_f = self.convB(out['l2'])
        new_out = self.model.block2(new_f)
        new_logit = self.oFC(new_out)

        total_logit = torch.cat([out['logit'], new_logit], dim=1)
        total_softmax = total_logit.softmax(-1)[:, :self.num_classes]
        
        loss = self.criterion(out['logit'][known_idxs], train_y[known_idxs])
        
        soft_max_logit = torch.softmax(out['logit'], dim=-1)
        soft_max_auroc = self.auroc1(soft_max_logit.max(-1)[0], known_idxs.long())
        logit_auroc = self.auroc2(out['logit'].max(-1)[0], known_idxs.long())
        total_softmax_auroc = self.auroc3(total_softmax.max(-1)[0], known_idxs.long())
        
        # softmin_auroc = self.auroc(logit.max(-1)[0], (known_idxs).long())
        
        top1k = accuracy(out['logit'][known_idxs], train_y[known_idxs], topk=(1,))[0]
                
        log_dict = {
            'val_loss': loss,
            'val_acc': top1k,
            'softmax': soft_max_auroc,
            'logit': logit_auroc,
            'total_softmax': total_softmax_auroc,
            }
        
        self.log_dict(log_dict)

        return loss
    
    def on_validation_end(self) -> None:
        return super().on_validation_end()
    
    def configure_optimizers(self):
        # params = self.parameters()
        
        # optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, 
        #                              weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(self.parameters(),
                                 lr=self.lr, weight_decay=self.weight_decay)
        
        # REDUCE LR ON PLATEAU
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.5, 
        #     patience=10
        #     )
        
        # COSINE ANNEALING
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        # MULTI SETP LR
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50],
            gamma=0.1
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                # 'monitor': 'val_loss'
            }
        }
    
    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,):
        # skip the first 500 steps
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
        
    def train_dataloader(self):
        
        if self.dataset == 'cifar10':
            dataset = SplitCifar10(self.data_dir, train=True,
                                   transform=train_transform)
            dataset.set_split(self.splits['known_classes'])
        
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        if self.dataset == 'cifar10':
            dataset = OpenTestCifar10(self.data_dir, train=False,
                                      transform=val_transform, split=self.splits['known_classes'])
            
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--latent_dim", type=int, default=128)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=48)
        parser.add_argument("--data_dir", type=str, default="/datasets")
        parser.add_argument("--class_split", type=int, default=0)
        
        parser.add_argument("--alpha", type=float, default=1.0)

        return parser
