from sys import setdlopenflags
from pytorch_lightning import LightningModule
from argparse import ArgumentParser
import random

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics import AUROC
from torchmetrics.functional import f1
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
        
def cal_index(num_classes, y):
        batch_size = y.size(0)
        new_index = torch.randperm(batch_size).type_as(y)
        newY = y[new_index]
        mask = (newY == y)
        while mask.any().item():
            newY[mask] = torch.randint(0, num_classes, (torch.sum(mask),)).type_as(y)
            mask = (newY == y)
        return newY
        
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
    
    def forward_clean(self, x, y, class_mask):
        newY = cal_index(self.num_classes, y)
        x = self.conv(x)
        self.cal_class_per_std(x, class_mask)
        noise = torch.normal(mean=0, std=self.buffer[newY]).type_as(x)
        x_n = x.detach().clone() + self.alpha * noise
        x = self.bn(x)
        x = self.activation(x)
        
        x_n = self.bn(x_n)
        x_n = self.activation(x_n)
        return x, x_n
        
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
    def __init__(self, num_classes=2, **kwargs):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
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
        
        self.apply(weights_init)
        
    def forward(self, x):
        l1 = self.block1(x)
        l2 = self.block2(l1)
        y = self.block3(l2)
        
        
        return y
        
    def block1(self, x):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x
    
    def block1_n(self, x, y):
        class_mask = y.unsqueeze(0) == torch.arange(
            self.num_classes).type_as(y).unsqueeze(1)
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        clean, noise = self.conv3.forward_clean(x, y, class_mask)
        return clean, noise
    
    def block2(self, x):
        x = self.dr2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x
    
    def block2_n(self, x, y):
        class_mask = y.unsqueeze(0) == torch.arange(
            self.num_classes).type_as(y).unsqueeze(1)
        x = self.dr2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        clean, noise = self.conv6.forward_clean(x, y, class_mask)
        return clean, noise

    def block3(self, x):
        x = self.dr3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        logit = self.fc(x)
        return logit
    
class Generator(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(self.__class__, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            )
        
    def forward(self, x):
        output = self.main(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(self.__class__, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(out_channel, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        output = self.main(x)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output).flatten()
        output = self.activation(output)
        return output

class FeatureGeneration(LightningModule):
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
                 beta: float = 0.5,
                 gan_lr: float = 0.0002,
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
        self.beta = beta
        self.gan_lr = gan_lr
        self.log_dir = log_dir
        
        self.splits = get_splits(dataset, class_split)
        self.num_classes = len(self.splits['known_classes'])
        
        self.model = classifier32(num_classes=len(self.splits['known_classes']))
        # self.NewFC = nn.Linear(self.latent_size, self.num_classes * self.num_classes)
        self.G = Generator(128, 128)
        self.D = Discriminator(128, 128)
        self.criterionD = nn.BCELoss()

        self.criterion = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss()
        self.auroc1 = AUROC(pos_label=1)
        self.auroc2 = AUROC(pos_label=1)
        self.auroc3 = AUROC(pos_label=1)
        
        # a = [list(range(6)) for _ in range(6)]
        # start = 6
        # for i in range(6):
        #     for j in range(i, 6):
        #         if i == j:
        #             a[i][j] = i
        #         else:
        #             a[i][j] = start
        #             a[j][i] = start
        #             start += 1
        # self.YM = torch.tensor(a, dtype=torch.long).to(self.device)
        
        self.automatic_optimization = False

    def on_train_start(self) -> None:
        wandb.save(self.log_dir+f"/{__file__.split('/')[-1]}")
        
        # print(f"load state_dict")
        # weight_path = f"./result/checkpoints/s{self.class_split}.ckpt"
        # state_dict = torch.load(weight_path)['state_dict']
        
        # self.load_state_dict(state_dict)
    
        # model_state_dict = self.model.state_dict()
        # for name, param in state_dict.items():
        #     n = name.replace('model.', '')
        #     if n in model_state_dict.keys():
        #         model_state_dict[n].copy_(param)
        
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    
    def on_train_epoch_start(self) -> None:
        self.log_img = True
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        batch_size = x.size(0)
        gen_size = batch_size // 6

        opt_C, opt_G, opt_D = self.optimizers()
        
        # scheduler = self.lr_schedulers()
        
        requires_grad(self.model, True)
        requires_grad(self.G, True)
        
        l1_clean, l1_noise = self.model.block1_n(x, y)
        # l2_clean, l2_noise = self.model.block2_n(l1, y)
        fake_feature = self.G(l1_noise)
        
        ##################################################
        # Train Discriminator
        ##################################################
        requires_grad(self.D, True)
        requires_grad(self.G, False)
        requires_grad(self.model, False)

        Dreal = self.D(l1_clean.detach())
        Dfake = self.D(fake_feature.detach())

        Dreal_loss = self.criterionD(Dreal, torch.ones(Dreal.size()).to(self.device))
        Dfake_loss = self.criterionD(Dfake, torch.zeros(Dfake.size()).to(self.device))
        
        D_loss = Dreal_loss + Dfake_loss
        
        opt_D.zero_grad()
        self.manual_backward(D_loss)
        opt_D.step()
        
        ##################################################
        # Train Generator & Classifier
        ##################################################
        requires_grad(self.D, False)
        requires_grad(self.G, True)
        requires_grad(self.model, True)
        
        Greal = self.D(fake_feature)
        
        Greal_loss = self.criterionD(Greal, torch.ones(Greal.size()).to(self.device))
        
        l2_clean = self.model.block2(l1_clean)
        Ologit = self.model.block3(l2_clean)
        
        Nlogit = self.model.block2(fake_feature)
        noise_logit = self.model.block3(Nlogit)
        
        # Gclass_loss = self.criterion(noise_logit, torch.ones_like(y) * 6)
        fake_distribution = torch.ones_like(noise_logit) * -2
        
        indexs = y.unsqueeze(1) == torch.arange(self.num_classes+1).type_as(y).unsqueeze(0)
        fake_distribution[indexs] = 0.2
        fake_distribution[:,-1] = 1
        
        Oclass_loss = self.criterion(Ologit, y)
        # temper_loss = self.criterion(noise_logit, y) * 0.3
        Gclass_loss = self.kldiv(noise_logit.softmax(-1), fake_distribution.softmax(-1)).mean()
        G_loss = Greal_loss + Gclass_loss
        
        # G_loss = Gclass_loss 
        C_loss = Oclass_loss
        update = G_loss + C_loss
        opt_C.zero_grad()
        opt_G.zero_grad()
        self.manual_backward(update)
        opt_G.step()
        opt_C.step()
        
        
        log_dict = {
            'G/classify': Gclass_loss,
            'G/generation': Greal_loss,
            
            'D/real': Dreal_loss,
            'D/fake': Dfake_loss,
            
            'C/class': Oclass_loss, 
        }
        
        self.log_dict(log_dict, on_step=True)
        
        loss = G_loss + D_loss + C_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, train_y, y, known_idxs = batch
        
        out = self.model(x)
        
        pred = out.max(-1)[1]
        known = pred < 6
        unknown = pred >= 6
        
        pred[known] = 1
        pred[unknown] = 0
         
        # open_acc = pred.eq(known_idxs.long()).sum().item() / known_idxs.size(0)
        f1_score = f1(pred, known_idxs.long(), num_classes=2, average='macro')
        
        loss = self.criterion(out[known_idxs], train_y[known_idxs])
        
        soft_max_logit = torch.softmax(out[:,:6], dim=-1)
        soft_max_auroc = self.auroc1(soft_max_logit.max(-1)[0], known_idxs.long())
        logit_auroc = self.auroc2(out[:, :6].max(-1)[0], known_idxs.long())
        
        # softmin_auroc = self.auroc(logit.max(-1)[0], (known_idxs).long())
        
        top1k = accuracy(out[known_idxs], train_y[known_idxs], topk=(1,))[0]
        
        # unknown_acc = f1(pred[unknown], torch.zeros_like(pred[unknown]), num_classes=1, average='macro')
        unknown_acc = pred[~known_idxs].eq(torch.zeros_like(pred[~known_idxs])).sum().item() / len(pred[~known_idxs])
        
        log_dict = {
            'val_loss': loss,
            'val_acc': top1k,
            'softmax': soft_max_auroc,
            'logit': logit_auroc,
            'open f1': f1_score,
            'unknown acc': unknown_acc
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
        optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=self.lr, weight_decay=self.weight_decay)
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.gan_lr, 
                                 betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.gan_lr,
                                 betas=(0.5, 0.999))
        
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
        
        # return {
        #     'optimizer': [optimizer, opt_G, opt_D]
        #     # 'lr_scheduler': {
        #     #     'scheduler': scheduler,
        #     #     # 'monitor': 'val_loss'
        #     # }
        # }
        return [optimizer, opt_G, opt_D], [scheduler]
    
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

        
    def kldiv(self, P, Q):
        return (P * (P / Q).log()).sum(-1)
       
    def validation_epoch_end(self, outputs) -> None:
        
        return 
            
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
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--gan_lr", type=float, default=0.0001)

        return parser
    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        