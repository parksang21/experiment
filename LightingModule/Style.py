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
from matplotlib import pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
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

class classifier32(nn.Module):
    def __init__(self, num_classes=2, alpha=1.0, **kwargs):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.conv1 = NoiseEncoder(3,     64,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv2 = NoiseEncoder(64,    64,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv3 = NoiseEncoder(64,   128,    3, 2, 1, bias=False, num_classes=num_classes)
        
        self.conv4 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv5 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv6 = NoiseEncoder(128,  128,    3, 2, 1, bias=False, num_classes=num_classes)
        
        self.conv7 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv8 = NoiseEncoder(128,  128,    3, 1, 1, bias=False, num_classes=num_classes)
        self.conv9 = NoiseEncoder(128,  128,    3, 2, 1, bias=False, num_classes=num_classes)
        
        self.fc = nn.Linear(128, num_classes * 2)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.buffer = None
        
        self.apply(weights_init)
        
    def block0(self, x):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    def block1(self, x):
        x = self.dr2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x
    
    def block2(self, x):
        x = self.dr3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        
        return x
        
    def forward(self, x):
        l1 = self.block0(x)
        l2 = self.block1(l1)
        l3 = self.block2(l2)
        
        y = self.fc(l3)
        
        return {
            'logit': y,
            'l3': l3,
            'l2': l2,
            'l1': l1,
        }
        
    def cal_mean_std(self, x, eps=1e-5):
        size = x.size()
        assert (len(size) == 4)
        B, C = size[:2]
        variance = x.view(B, C, -1).var(dim=2) + eps
        std = variance.sqrt().view(B, C, 1, 1)
        mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        
        return mean, std
    
    def mix_(self, x1, x2):
        size = x1.size()
        x1_mean, x1_std = self.cal_mean_std(x1)
        x2_mean, x2_std = self.cal_mean_std(x2)
        
        x1_normalize = (x1 - x1_mean.expand(size)) / x1_std.expand(size)
        
        return x1_normalize * x1_std.expand(size) + x2_mean.expand(size)
    
    def normalize(self, x, mean, std):
        size = x.size()
        return (x - mean.expand(size)) / std.expand(size)
    
    def unnormalize(self, x, mean, std):
        size = x.size()
        return x * std.expand(size) + mean.expand(size)
    
    def cal_std(self, x, y, mean, std):
        stds = []
        normalize = (x - mean.expand(x.size())) / std.expand(x.size())
        idxs = y.unsqueeze(0) == torch.arange(self.num_classes).unsqueeze(1).type_as(y)
        for i in range(self.num_classes):
            x_ = normalize[idxs[i]].detach().clone()
            
            stds.append(x_.std(0))
            
        self.buffer = torch.stack(stds)

    def forward_norm_std(self, x, y):
        newY = self.cal_index(y)
        x = self.block0(x)
        x = self.block1(x)
        
        mean, std = self.cal_mean_std(x)
        self.cal_std(x, y, mean, std)
        x_norm = self.normalize(x, mean, std)
        x_new = x_norm + self.alpha * torch.normal(mean=0, std=torch.ones_like(x_norm)).type_as(x)
        x_new = self.unnormalize(x_new, mean, std)
        
        clean_x = self.block2(x)
        clean_logit = self.fc(clean_x)
        
        noise_x = self.block2(x_new)
        noise_logit = self.fc(noise_x)
        
        return {
            'clean_x': x,
            'clean_logit': clean_logit,
            'noise_x': x_new,
            'noise_logit': noise_logit,
            'mean': mean,
            'std': std
        }
        
    def forward_l2(self, x):
        l3 = self.block2(x)
        
        logit = self.fc(l3)
        return {
            'logit': logit,
            'l3': l3
        }
    
    def forward_mix(self, x, y):
        newY = self.cal_index(y)

        l1 = self.block0(x)
        l2 = self.block1(l1)
        
        m_l2 = self.mix_(l2, l2[newY])
        l3 = self.block2(l2)
        m_l3 = self.block2(m_l2)
        
        logit = self.fc(l3)
        m_logit = self.fc(m_l3)
        
        return {
            'logit': logit,
            'l2': l2,
            'm_l2': m_l2,
            'm_logit': m_logit,
            'newY': newY
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
    
    def channel_swap(self, x, y):
        # newY = self.cal_index(y)
        B, C, H, W = x.size()
        channel_select = torch.randint(0, C, (int(C * self.alpha),)).type_as(y)
        x[:, channel_select, :, :] = x[y][:,channel_select, :, :].clone().detach()
        return x
    
    def forward_swap(self, x, y):
        l1 = self.block0(x)
        l2 = self.block1(l1)
        l3 = self.block2(l2)
        newY = self.cal_index(y)
        m_l2 = self.channel_swap(l2, newY)
        m_l3 = self.block2(m_l2)
        
        logit = self.fc(l3)
        m_logit = self.fc(m_l3)
        
        return {
            'clean_logit': logit,
            'l2': l2,
            'noise_logit': m_logit,
            'm_l2': m_l2,
            'newY': newY
        }

class Style(LightningModule):
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
    
    def on_train_epoch_start(self) -> None:
        self.log_img = True
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        out = self.model.forward_swap(x, y)
        
        # if self.log_img:
        #     img_dict = dict()
        #     img1 = out['clean_x'][0].detach().cpu().numpy()
        #     img2 = out['noise_x'][0].detach().cpu().numpy()
        #     imgs = np.concatenate([img1, img2], axis=1)
        #     for i in range(imgs.shape[0]):
        #         or_img = wandb.Image(imgs[i], caption=f"c_{i}")
        #         img_dict[f"c_{i}"] = or_img
        #     wandb.log(img_dict)
        #     self.log_img = False
        
        c_loss = self.criterion(out['clean_logit'], y)
        
        acc = accuracy(out['clean_logit'], y)[0]
        
        open_loss = self.criterion(out['noise_logit'], y + self.num_classes)
        
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
       
    def validation_epoch_end(self, outputs) -> None:
        
        return 
        
    def validation_step(self, batch, batch_idx):
        x, train_y, y, known_idxs = batch
        
        out = self.model(x)
        
        pred = out['logit'].max(-1)[1]
        known = pred < 6
        unknown = pred >= 6
        
        pred[known] = 1
        pred[unknown] = 0
         
        open_acc = pred.eq(known_idxs.long()).sum().item() / known_idxs.size(0)    
        
        loss = self.criterion(out['logit'][known_idxs], train_y[known_idxs])
        
        soft_max_logit = torch.softmax(out['logit'][:,:6], dim=-1)
        soft_max_auroc = self.auroc1(soft_max_logit.max(-1)[0], known_idxs.long())
        logit_auroc = self.auroc2(out['logit'][:, :6].max(-1)[0], known_idxs.long())
        
        # softmin_auroc = self.auroc(logit.max(-1)[0], (known_idxs).long())
        
        top1k = accuracy(out['logit'][known_idxs], train_y[known_idxs], topk=(1,))[0]
        
        log_dict = {
            'val_loss': loss,
            'val_acc': top1k,
            'softmax': soft_max_auroc,
            'logit': logit_auroc,
            'open acc': open_acc
            # 'total_softmax': total_softmax_auroc,
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
        
        # # REDUCE LR ON PLATEAU
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.5, 
        #     patience=10
        #     )
        
        # COSINE ANNEALING
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        # # MULTI SETP LR
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[50],
        #     gamma=0.1
        # )
        
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     # 'monitor': 'val_loss'
            # }
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
        parser.add_argument("--num_workers", type=int, default=10)
        parser.add_argument("--data_dir", type=str, default="/datasets")
        parser.add_argument("--class_split", type=int, default=0)
        
        parser.add_argument("--alpha", type=float, default=0.5)

        return parser
