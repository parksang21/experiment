'''
s2-0-cifar10-s0-v3
/workspace/experiment/main.py --model NoiseInjection 
                            --lr 0.1 
                            --gpus 0, 
                            --accelerator ddp 
                            --max_epoch 100 
                            --class_split 0 
                            --alpha 0.5 
                            --lr 0.001
'''

from typing import Optional
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

class NoiseLayer(nn.Module):
    def __init__(self, alpha, num_classes):
        super(NoiseLayer, self).__init__()
        self.alpha = alpha
        self.means = None
        self.std = None
        self.num_classes = torch.arange(num_classes)
        
    def calculate_class_mean(self, 
                           x: torch.Tensor, 
                           y: torch.Tensor):
        """calculate the variance of each classes' noise

        Args:
            x (torch.Tensor): [input tensor]
            y (torch.Tensor): [target tensor]

        Returns:
            [Tensor]: [returns class dependent noise variance]
        """
        self.num_classes = self.num_classes.type_as(y)
        idxs = y.unsqueeze(0) == self.num_classes.unsqueeze(1)
        mean = []
        std = []
        for i in range(self.num_classes.shape[0]):
            x_ = x[idxs[i]]
            mean.append(x_.mean(0))
            std.append(x_.std(0))
        
        self.means = torch.stack(mean)
        self.std = torch.stack(std)
    
    def forward(self, x, newY):
        #TODO: sampling different noise for each input not per classes
    
        # class_noise = torch.normal(mean=0, std=self.std[newY]).type_as(x).detach()
        class_noise = torch.normal(mean=0, std=self.std[newY]).type_as(x)

        return (x + self.alpha * class_noise)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32(nn.Module):
    def __init__(self, num_classes=2, alpha=0.5, **kwargs):
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
        self.noise0 = NoiseLayer(alpha, num_classes)
        self.noise1 = NoiseLayer(alpha, num_classes)
        self.noise2 = NoiseLayer(alpha, num_classes)
        self.noise3 = NoiseLayer(alpha, num_classes)
        
    def cal_index(self, x, y):
        batch_size = x.size(0)
        
        #TODO: matching indexes with other classes reduce iteration
        new_index = torch.randperm(batch_size).type_as(y)
        newY = y[new_index]
        mask = (newY == y)
        while mask.any().item():
            newY[mask] = torch.randint(0, 6, (torch.sum(mask),)).type_as(y)
            mask = (newY == y)
        return newY

    def forward(self, x, y,  noise=[]):
        batch_size = len(x)
        
        if len(noise) > 0:
            newY = self.cal_index(x, y)
        else:
            newY = None

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.conv3(x)
        x = self.bn3(x)
        l1 = nn.LeakyReLU(0.2)(x)
        
        self.noise0.calculate_class_mean(l1, y)
        if 0 in noise:
            l1 = self.noise0(l1, newY)

        x = self.dr2(l1)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        l2 = nn.LeakyReLU(0.2)(x)
        
        self.noise1.calculate_class_mean(l2, y)
        if 1 in noise:
            l2 = self.noise1(l2, newY)

        x = self.dr3(l2)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        l3 = nn.LeakyReLU(0.2)(x)
        
        l3 = self.avgpool(l3)
        l3 = l3.view(batch_size, -1)
        
        self.noise2.calculate_class_mean(l3, y)
        if 2 in noise:
            l3 = self.noise2(l3, newY)
        
        y = self.fc1(l3)
        
        return {
            'logit': y,
            'l3': l3,
            'l2': l2,
            'l1': l1,
            'newY': newY
        }


class NoiseInjection(LightningModule):
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
                 **kwargs):
        
        super().__init__()
        
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
        self.splits = get_splits(dataset, class_split)
        
        self.model = classifier32(num_classes=len(self.splits['known_classes']), alpha=self.alpha)

        self.criterion = nn.CrossEntropyLoss()
        self.auroc = AUROC(pos_label=1)
        
    def setup(self, stage: Optional[str] = None) -> None:
        wandb.save(__file__)
    
    def forward(self, x, y, noise=[]):
        out = self.model(x, y, noise)
        return out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # forward without noise
        out = self(x, y, [0, 1, 2])
        
        closs = self.criterion(out['logit'], y)

        # noise out
        # nout = self(x, y, [0, 1, 2])
        # noise_loss = self.criterion(nout['logit'], nout['newY']) * 0.3
        
        top1k = accuracy(out['logit'], y, topk=(1,))[0]
        
        # loss = closs + noise_loss
        loss = closs

        log_dict = {
            'classification loss': closs,
            'train acc': top1k,
            # 'ranking loss': mrl_loss,
            # 'noiseLoss': noise_loss,
            'total loss': loss
        }
        
        self.log_dict(log_dict, on_step=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, train_y, y, known_idxs = batch
       
        out = self(x, train_y, [])
        
        loss = self.criterion(out['logit'][known_idxs], train_y[known_idxs])
        
        soft_max_logit = torch.softmax(out['logit'], dim=-1)
        soft_max_auroc = self.auroc(soft_max_logit.max(-1)[0], known_idxs.long())
        logit_auroc = self.auroc(out['logit'].max(-1)[0], known_idxs.long())
        
        top1k = accuracy(out['logit'][known_idxs], train_y[known_idxs], topk=(1,))[0]
                
        log_dict = {
            'val_loss': loss, 
            'val_acc': top1k,
            'softmax': soft_max_auroc,
            'logit': logit_auroc, 
            }
        
        self.log_dict(log_dict)
        
        return loss
    
    def on_validation_end(self) -> None:
        return super().on_validation_end()
    
    def configure_optimizers(self):
        params = self.parameters()
        # optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, 
        #                              weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=20
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
        # return optimizer
        
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
        
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--latent_dim", type=int, default=128)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=48)
        parser.add_argument("--data_dir", type=str, default="/datasets")
        parser.add_argument("--class_split", type=int, default=0)
        
        parser.add_argument("--alpha", type=float, default=1.0)

        return parser
