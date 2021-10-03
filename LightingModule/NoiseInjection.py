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


##############################################################
# noise Layer
##############################################################

class NoiseEncoder(nn.Module):
    def __init__(self, 
                 channel_in:  int, 
                 channel_out: int, 
                 kernel_size: int, 
                 stride:      int, 
                 padding:     int, 
                 bias:        bool=False, 
                 num_classes: int=6,
                 alpha: float=1.0):
        
        super(self.__class__, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(channel_out)
        self.bn_noise = nn.BatchNorm2d(channel_out)
        self.activation = nn.LeakyReLU(0.2)
        self.num_classes = num_classes
        self.register_buffer('buffer', None)
        self.alpha = alpha
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
    def forward_clean(self, x, class_mask):
        x = self.conv(x)
        self.cal_class_per_std(x, class_mask)
        x = self.bn(x)
        x = self.activation(x)
        return x
        
    def forward_noise(self, x, newY):
        x = self.conv(x)
        x = x + self.alpha * torch.normal(mean=0, std=self.buffer[newY]).type_as(x)
        x = self.bn_noise(x)
        x = self.activation(x)
        return x
    
    def cal_class_per_std(self, x, idxs):
        std = []
        for i in range(self.num_classes):
            x_ = x[idxs[i]].detach().clone()
            
            std.append(x_.std(0))
        self.buffer = torch.stack(std)

class Noiseclassifier32(nn.Module):
    def __init__(self, 
                 num_classes: int=6,
                 alpha: int=1.0,
                 **kwargs):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = NoiseEncoder(3,     64,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv2 = NoiseEncoder(64,    64,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv3 = NoiseEncoder(64,   128,    3, 2, 1, bias=False, num_classes=num_classes)
        
        self.conv4 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv5 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv6 = NoiseEncoder(128,  128,    3, 2, 1, bias=False, num_classes=num_classes)
        
        self.conv7 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv8 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv9 = NoiseEncoder(128,  128,    3, 2, 1, bias=False, num_classes=num_classes)
        
        self.fc = nn.Linear(128, num_classes + 1)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.apply(self.weights_init)
        
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        batch_size = len(x)
        x = self.dr1(x)
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.dr2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.dr3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        
        logit = self.fc(x)
        
        return logit
    
    def forward_clean(self, x, y):
        batch_size = len(x)
        class_mask = y.unsqueeze(0) == torch.arange(self.num_classes).type_as(y).unsqueeze(1)
        x = self.dr1(x)
        x = self.conv1.forward_clean(x, class_mask)
        x = self.conv2.forward_clean(x, class_mask)
        x = self.conv3.forward_clean(x, class_mask)
        
        x = self.dr2(x)
        x = self.conv4.forward_clean(x, class_mask)
        x = self.conv5.forward_clean(x, class_mask)
        x = self.conv6.forward_clean(x, class_mask)
        
        x = self.dr3(x)
        x = self.conv7.forward_clean(x, class_mask)
        x = self.conv8.forward_clean(x, class_mask)
        x = self.conv9.forward_clean(x, class_mask)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        
        logit = self.fc(x)
        
        return {
            "logit": logit,
            "emb": x
        }
    
    def forward_noise(self, x, y, noise_layer=[]):
        batch_size = len(x)
        newY = self.cal_index(y)
        # newY = y
        
        x = self.dr1(x)
        x = self.conv1.forward_noise(x, newY)
        x = self.conv2.forward_noise(x, newY)
        x = self.conv3.forward_noise(x, newY)
        
        x = self.dr2(x)
        x = self.conv4.forward_noise(x, newY)
        x = self.conv5.forward_noise(x, newY)
        x = self.conv6.forward_noise(x, newY)
        
        x = self.dr3(x)
        x = self.conv7.forward_noise(x, newY)
        x = self.conv8.forward_noise(x, newY)
        x = self.conv9.forward_noise(x, newY)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        
        logit = self.fc(x)
        
        return {
            'logit': logit,
            'newY': newY,
            'emb': x
        }
        
    def cal_index(self, y):
        batch_size = y.size(0)
        new_index = torch.randperm(batch_size).type_as(y)
        newY = y[new_index]
        mask = (newY == y)
        while mask.any().item():
            newY[mask] = torch.randint(0, self.num_classes, (torch.sum(mask),)).type_as(y)
            mask = (newY == y)
        return newY

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
                 log_dir: str = './log',
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
        self.log_dir = log_dir
        
        self.splits = get_splits(dataset, class_split)
        
        self.model = Noiseclassifier32(num_classes=len(self.splits['known_classes']), alpha=self.alpha)
        # self.dummy_fc = nn.Linear(self.latent_size, 1)

        self.criterion = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.auroc = AUROC(pos_label=1)
        self.mrl = nn.MarginRankingLoss(margin=3)
        
    def on_train_start(self) -> None:
        wandb.save(self.log_dir+f"/{__file__.split('/')[-1]}")
    
    def forward(self, x, y, noise=[]):
        out = self.model(x, y, noise)
        return out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        batch_size = len(x)
        clean_out = self.model.forward_clean(x, y)
        closs = self.criterion(clean_out['logit'], y)
        
        noise_out = self.model.forward_noise(x, y)
        # open_loss = self.criterion(noise_out['logit'], torch.ones_like(y) * 6)
        # open_loss = self.kldiv(noise_out['logit'].softmax(-1), clean_out['logit'].softmax(-1).detach()).mean()
        open_loss = self.criterion(noise_out['logit'], y) * 0.5 + \
            self.criterion(noise_out['logit'], noise_out['newY']) * 0.5
        top1k = accuracy(clean_out['logit'], y, topk=(1,))[0]

        
        loss = closs + open_loss
        
        log_dict = {
            'classification loss': closs,
            'train acc': top1k,
            'open loss': open_loss,
            # 'ranking': mrl,
            'total loss': loss
        }
        
        
        self.log_dict(log_dict, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, train_y, y, known_idxs = batch
       
        out = self.model(x)
        
        loss = self.criterion(out[known_idxs], train_y[known_idxs])
        
        # print(out.softmax(-1).shape)
        # exit()
        
        soft_max_logit = torch.softmax(out, dim=-1)
        soft_max_auroc = self.auroc(soft_max_logit.max(-1)[0], known_idxs.long())
        logit_auroc = self.auroc(out.max(-1)[0], known_idxs.long())
        
        # print(kld_result.shape)
        # exit()
        
        top1k = accuracy(out[known_idxs], train_y[known_idxs], topk=(1,))[0]
        
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
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, 
                                     weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        
        # # REDUCE LR ON PLATEAU
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.5,
        #     patience=20
        #     )
        
        # # COSINE ANNEALING
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-3)

        # MULTI SETP LR
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[70],
            gamma=0.5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                # 'monitor': 'val_loss'
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
    
    def kldiv(self, P, Q):
        return (P * (P / Q).log()).sum(-1)
    