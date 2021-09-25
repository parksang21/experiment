##################################################
# Imports
##################################################

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics import AUROC

from data.cifar10 import SplitCifar10, train_transform, val_transform, OpenTestCifar10
from data.capsule_split import get_splits
from utils import accuracy


##################################################
# Utils
##################################################

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


##################################################
# ResNet34 Model
##################################################

class ResNet34(nn.Module):
    def __init__(self, in_channels=3, inplanes=64, alpha=0.5, num_classes=10,):
        super(ResNet34, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, 
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.NL = NoiseLayer(alpha=alpha, num_classes=num_classes)

        # Init layers
        self._init_layers()
        
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1, downsample=None):
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers += [BasicBlock(planes, planes)]
        return nn.Sequential(*layers)

    def forward(self, x, y, noise=[]):
        ny = None
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        if 0 in noise:
            x1, ny = self.NL(x1, y)
        x2 = self.layer2(x1)
        if 1 in noise:
            x2, ny = self.NL(x2, y)
        x3 = self.layer3(x2)
        if 2 in noise:
            x3, ny = self.NL(x3, y)
        x = self.layer4(x3)
        if 3 in noise:
            x, ny = self.NL(x, y)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = {
            'x_l1': x1,
            'x_l2': x2,
            'x_l3': x3,
            'x_f': x,
            'ny': ny
        }
        return out
    
    
##################################################
# pytorch_lightning ResNet34 Model
##################################################

class NoiseLayer(nn.Module):
    def __init__(self, alpha, num_classes):
        super(NoiseLayer, self).__init__()
        self.alpha = alpha
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
        
        return torch.stack(mean), torch.stack(std)
    
    def forward(self, x, y):
        batch_size = x.size(0)
        class_mean, class_var = self.calculate_class_mean(x, y)
        
        class_noise = torch.normal(mean=class_mean, std=class_var).type_as(x).detach()
        # class_noise = torch.normal(mean=0., std=class_var).type_as(x).detach()

        index = torch.randperm(batch_size).type_as(y)
        newY = y[index]
        mask = y != newY
        if x.dim() == 2:
            mask = mask.unsqueeze(1).expand_as(x).type_as(x)
        else:
            mask = mask[...,None,None,None].expand_as(x).type_as(x)
        
        return ((1 - self.alpha) * x + self.alpha * class_noise[newY]), newY
    
    
class NoiseInj_resnet(LightningModule):
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
        
        self.model = ResNet34(alpha=self.alpha, num_classes=len(self.splits['known_classes']))
        self.fc = nn.Linear(512, len(self.splits['known_classes']))
        self.dummyFC = nn.Linear(512, len(self.splits['known_classes']))
        self.criterion = nn.CrossEntropyLoss()
        self.auroc = AUROC(pos_label=1)
        self.mrl = nn.MarginRankingLoss(margin=5.0)
    
    def forward(self, x, y, noise=[]):
        logit, features = self.model(x, y, noise)
        return logit, features
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x, y, noise=[0,1,2,3])
        logit = self.fc(out['x_f'])
        dummy_origin = self.dummyFC(out['x_f'])
        
        closs = self.criterion(torch.cat([logit, dummy_origin], dim=1), y)
        
        uo = self.model(x, y, noise=[1,])
        
        logit2 = self.fc(uo['x_f'])
        noise_logit = self.dummyFC(uo['x_f'])
        
        max_noise_logit = torch.max(noise_logit, dim=1)[0]
        max_fc_logit = torch.max(logit2, dim=1)[0]
        
        # mrl_loss = self.mrl(max_noise_logit, max_fc_logit, torch.ones_like(max_noise_logit))
        
        # c2loss = self.criterion(logit2, y)
        # noise_loss = self.criterion(noise_logit, y)
        noise_loss = self.criterion(torch.cat([noise_logit, logit2], dim=1), y)
        
        top1k = accuracy(logit, y, topk=(1,))[0]
        
        # loss = closs + mrl_loss * 0.01 + noise_loss
        loss = closs + noise_loss

        log_dict = {
            'classification loss': closs,
            'train acc': top1k,
            'noiseLoss': noise_loss,
            'total loss': loss
        }
        
        self.log_dict(log_dict, on_step=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, train_y, y, known_idxs = batch
       
        out = self.model(x, y)
        logit = self.fc(out['x_f'])
        dummy_out = self.dummyFC(out['x_f'])
        
        total_logit = torch.cat([logit, dummy_out], dim=0)
        loss = self.criterion(logit[known_idxs], train_y[known_idxs])
        
        soft_max_logit = torch.softmax(logit, dim=-1)
        soft_max_auroc = self.auroc(soft_max_logit.max(-1)[0], known_idxs.long())
        logit_auroc = self.auroc(logit.max(-1)[0], known_idxs.long())
        
        top1k = accuracy(logit[known_idxs], train_y[known_idxs], topk=(1,))[0]
        
        max_ = logit.max(1)[0] - dummy_out.max(1)[0]
        
        max_auroc = self.auroc(max_, known_idxs.long())
        
        log_dict = {'val_loss': loss, 'val_acc': top1k,
                    'softmax': soft_max_auroc, 'logit': logit_auroc, 'sub': max_auroc}
        
        self.log_dict(log_dict)
        
        return loss
    
    def on_validation_end(self) -> None:
        return super().on_validation_end()
    
    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.dummyFC.parameters()) + \
                 list(self.fc.parameters())
        # optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, 
        #                              weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
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
