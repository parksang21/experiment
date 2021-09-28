# from pytorch_lightning import LightningModule
# from argparse import ArgumentParser

# import torch
# from torch import nn
# from torch.utils.data.dataloader import DataLoader
# from torchmetrics import AUROC
# from torch.autograd import Variable

# from data.cifar10 import SplitCifar10, train_transform, val_transform, OpenTestCifar10
# from data.capsule_split import get_splits
# from utils import accuracy

# class NoiseLayer(nn.Module):
#     def __init__(self, alpha, num_classes):
#         super(NoiseLayer, self).__init__()
#         self.alpha = alpha
#         self.num_classes = torch.arange(num_classes)
        
#     def calculate_class_mean(self, 
#                            x: torch.Tensor, 
#                            y: torch.Tensor):
#         """calculate the variance of each classes' noise

#         Args:
#             x (torch.Tensor): [input tensor]
#             y (torch.Tensor): [target tensor]

#         Returns:
#             [Tensor]: [returns class dependent noise variance]
#         """
#         self.num_classes = self.num_classes.type_as(y)
#         idxs = y.unsqueeze(0) == self.num_classes.unsqueeze(1)
#         mean = []
#         std = []
#         for i in range(self.num_classes.shape[0]):
#             x_ = x[idxs[i]]
#             mean.append(x_.mean(0))
#             std.append(x_.std(0))
        
#         return torch.stack(mean), torch.stack(std)
    
#     def forward(self, x, y):
#         batch_size = x.size(0)
#         class_mean, class_var = self.calculate_class_mean(x, y)
        
#         class_noise = torch.normal(mean=class_mean, std=class_var).type_as(x).detach()
#         # class_noise = torch.normal(mean=0., std=class_var).type_as(x).detach()

#         index = torch.randperm(batch_size).type_as(y)
#         newY = y[index]
#         mask = y != newY
#         if x.dim() == 2:
#             mask = mask.unsqueeze(1).expand_as(x).type_as(x)
#         else:
#             mask = mask[...,None,None,None].expand_as(x).type_as(x)
        
#         return ((1 - self.alpha) * x + self.alpha * class_noise[newY]), newY

# import torch
# from torch import nn, optim
# import torch.nn.functional as F

# class VAE(nn.Module):
#     def __init__(self, label, image_size, channel_num, kernel_num, z_size):
#         # configurations
#         super().__init__()
#         self.label = label
#         self.image_size = image_size
#         self.channel_num = channel_num
#         self.kernel_num = kernel_num
#         self.z_size = z_size

#         # encoder
#         self.encoder = nn.Sequential(
#             self._conv(channel_num, kernel_num // 4),
#             self._conv(kernel_num // 4, kernel_num // 2),
#             self._conv(kernel_num // 2, kernel_num),
#         )

#         # encoded feature's size and volume
#         self.feature_size = image_size // 8
#         self.feature_volume = kernel_num * (self.feature_size ** 2)

#         # q
#         self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
#         self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

#         # projection
#         self.project = self._linear(z_size, self.feature_volume, relu=False)

#         # decoder
#         self.decoder = nn.Sequential(
#             self._deconv(kernel_num, kernel_num // 2),
#             self._deconv(kernel_num // 2, kernel_num // 4),
#             self._deconv(kernel_num // 4, channel_num),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # encode x
#         encoded = self.encoder(x)

#         # sample latent code z from q given x.
#         mean, logvar = self.q(encoded)
#         z = self.z(mean, logvar)
#         z_projected = self.project(z).view(
#             -1, self.kernel_num,
#             self.feature_size,
#             self.feature_size,
#         )

#         # reconstruct x from z
#         x_reconstructed = self.decoder(z_projected)

#         # return the parameters of distribution of q given x and the
#         # reconstructed image.
#         return (mean, logvar), x_reconstructed

#     # ==============
#     # VAE components
#     # ==============

#     def q(self, encoded):
#         unrolled = encoded.view(-1, self.feature_volume)
#         return self.q_mean(unrolled), self.q_logvar(unrolled)

#     def z(self, mean, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = (
#             Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
#             Variable(torch.randn(std.size()))
#         )
#         return eps.mul(std).add_(mean)

#     def reconstruction_loss(self, x_reconstructed, x):
#         return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

#     def kl_divergence_loss(self, mean, logvar):
#         return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

#     # =====
#     # Utils
#     # =====

#     @property
#     def name(self):
#         return (
#             'VAE'
#             '-{kernel_num}k'
#             '-{label}'
#             '-{channel_num}x{image_size}x{image_size}'
#         ).format(
#             label=self.label,
#             kernel_num=self.kernel_num,
#             image_size=self.image_size,
#             channel_num=self.channel_num,
#         )

#     def sample(self, size):
#         z = Variable(
#             torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
#             torch.randn(size, self.z_size)
#         )
#         z_projected = self.project(z).view(
#             -1, self.kernel_num,
#             self.feature_size,
#             self.feature_size,
#         )
#         return self.decoder(z_projected).data

#     def _is_on_cuda(self):
#         return next(self.parameters()).is_cuda

#     # ======
#     # Layers
#     # ======

#     def _conv(self, channel_size, kernel_num):
#         return nn.Sequential(
#             nn.Conv2d(
#                 channel_size, kernel_num,
#                 kernel_size=4, stride=2, padding=1,
#             ),
#             nn.BatchNorm2d(kernel_num),
#             nn.ReLU(),
#         )

#     def _deconv(self, channel_num, kernel_num):
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 channel_num, kernel_num,
#                 kernel_size=4, stride=2, padding=1,
#             ),
#             nn.BatchNorm2d(kernel_num),
#             nn.ReLU(),
#         )

#     def _linear(self, in_size, out_size, relu=True):
#         return nn.Sequential(
#             nn.Linear(in_size, out_size),
#             nn.ReLU(),
#         ) if relu else nn.Linear(in_size, out_size)
# # KL Divergence
# def kl_div(p_mean, p_var, t_mean, t_var):
#     """
#     Compute the KL-Divergence between two Gaussians p and q:
#         p ~ N(p_mean, diag(p_var))
#         t ~ N(t_mean, diag(t_var))

#     Args:
#         p_mean: tensor of shape [bs(, ...), dim]
#         p_var: tensor of shape [bs(, ...), dim]
#         t_mean: tensor of shape [bs(, ...), dim]
#         t_var: tensor of shape [bs(, ...), dim]

#     Output:
#         kl: tensor of shape [bs(, ...)]
#     """
#     # if torch.is_tensor(p_mean):
#     kl = - 0.5 * (p_var - t_var + 1 - p_var / t_var - (p_mean - t_mean).pow(2) / t_var ).sum(-1)
#     # else:
#     #     kl = - 0.5 * (np.log(p_var) - np.log(t_var) + 1 - p_var / t_var - ((p_mean - t_mean) ** 2) / t_var).sum(-1)
#     return kl

# class CVAE(LightningModule):
#     def __init__(self,
#                  lr: float = 0.01,
#                  momentum: float = 0.9,
#                  weight_decay: float = 1e-4,
#                  data_dir: str = '/datasets',
#                  dataset: str = 'cifar10',
#                  batch_size: int = 128,
#                  num_workers: int = 48,
#                  class_split: int = 0,
#                  latent_size: int = 512,
#                  alpha: float = 0.5,
#                  **kwargs):
        
#         super().__init__()
        
#         self.save_hyperparameters()
#         self.lr = lr
#         self.momentum = momentum
#         self.weight_decay = weight_decay
#         self.data_dir = data_dir
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.class_split = class_split
#         self.latent_size = latent_size
#         self.alpha = alpha
#         self.splits = get_splits(dataset, class_split)
        
#         self.model = VAE(
#             label = 'cifar10',
#             image_size=32,
#             channel_num=3,
#             kernel_num=128,
#             z_size=128
#         )
#         self.class_mu = nn.Embedding(len(self.splits['known_classes']), self.latent_size)
#         self.class_logvar = nn.Embedding(len(self.splits['known_classes']), self.latent_size)
#         self.fc = nn.Linear(self.latent_size, len(self.splits['known_classes']))
#         self.bce = nn.BCELoss(size_average=False)
#         self.criterion = nn.CrossEntropyLoss()
#         self.auroc = AUROC(pos_label=1)
    
#     def forward(self, x, return_features=[]):
#         logit, features = self.model(x, return_features)
#         return logit, features
        
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         (mean, logvar), x_reconstructed = self.model(x)
        
#         reconstruction_loss = self.model.reconstruction_loss(x_reconstructed, x)

#         y_mu = self.class_mu(y)
#         y_var = self.class_logvar(y)
        
#         kl = kl_div(out['mu'], out['logvar'], y_mu, y_var)
        
#         loss = recon_loss + kl

#         log_dict = {
#             'total loss': loss,
#             'recon loss': recon_loss,
#             'kld loss': kl,
#         }
        
#         self.log_dict(log_dict, on_step=True)
        
#         return loss
        
    
#     def validation_step(self, batch, batch_idx):
        
        
        
#         return 0
    
#     def on_validation_end(self) -> None:
#         return super().on_validation_end()
    
#     def configure_optimizers(self):
#         params = self.parameters()
#         # optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, 
#         #                              weight_decay=self.weight_decay)
#         optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'monitor': 'val_loss'
#             }
#         }
#         # return optimizer
        
#     def train_dataloader(self):
        
#         if self.dataset == 'cifar10':
#             dataset = SplitCifar10(self.data_dir, train=True,
#                                    transform=train_transform)
#             dataset.set_split(self.splits['known_classes'])
        
#         return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
#     def val_dataloader(self):
#         if self.dataset == 'cifar10':
#             dataset = OpenTestCifar10(self.data_dir, train=False,
#                                       transform=val_transform, split=self.splits['known_classes'])
            
#         return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
#         parser.add_argument("--lr", type=float, default=1e-4)
#         parser.add_argument("--latent_dim", type=int, default=128)
#         parser.add_argument("--momentum", type=float, default=0.9)
#         parser.add_argument("--weight_decay", type=float, default=1e-4)

#         parser.add_argument("--batch_size", type=int, default=256)
#         parser.add_argument("--num_workers", type=int, default=48)
#         parser.add_argument("--data_dir", type=str, default="/datasets")
#         parser.add_argument("--class_split", type=int, default=0)
        
#         parser.add_argument("--alpha", type=float, default=1.0)

#         return parser
