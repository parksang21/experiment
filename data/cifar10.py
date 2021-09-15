from typing import *
from numpy.core.defchararray import split
import torchvision
from torchvision import transforms as T
import torch

from PIL import Image
import numpy as np

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class SeperateDatasets(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, train: bool, transform: Optional[Callable], target_transform: Optional[Callable], download: bool) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        x_divider = int(img.size(1) / 2)
        y_divider = int(img.size(2) / 2)
        idx_shuffle = torch.randint(0, 4, (4,))

        seperate_list = [img[:, :x_divider, :y_divider],  # top left
                         img[:, x_divider:, :y_divider],  # top right
                         img[:, :x_divider, y_divider:],  # bottom left
                         img[:, x_divider:, y_divider:]]  # bottom right

        seperate_image = torch.stack([seperate_list[i] for i in idx_shuffle], dim=0)
        return (img, target), (seperate_image, idx_shuffle)



class SplitCifar10(torchvision.datasets.CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 split: list = None
                 ) -> None:

        super(SplitCifar10, self).__init__(root, train=train, transform=transform,
                                           target_transform=target_transform, download=download,)

        self.split_idx = None
        self.split = split
        if not self.split is None:
            self.set_split()

    def set_split(self, split):
        self.split = split
        np_target = np.array(self.targets)
        split_idxs = np.isin(np_target, split)
        np_target = np.where(split_idxs == True)[0]

        self.split_idx = np_target

    def __getitem__(self, index):
        assert self.split_idx is not None
        index = self.split_idx[index]

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, self.split.index(target)
    
    def __len__(self) -> int:
        return len(self.split_idx)

class SplitPatchCifar10(torchvision.datasets.CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:

        super(SplitPatchCifar10, self).__init__(root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        self.split_idx = None
        self.split = None

    def set_split(self, split):
        self.split = split
        np_target = np.array(self.targets)
        split_idxs = np.isin(np_target, split)
        np_target = np.where(split_idxs == True)[0]

        self.split_idx = np_target

    def __getitem__(self, index):
        assert self.split_idx is not None
        index = self.split_idx[index]

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        x_divider = int(img.size(1) / 2)
        y_divider = int(img.size(2) / 2)
        idx_shuffle = torch.randint(0, 4, (4,))

        seperate_list = [img[:, :x_divider, :y_divider],  # top left
                         img[:, x_divider:, :y_divider],  # top right
                         img[:, :x_divider, y_divider:],  # bottom left
                         img[:, x_divider:, y_divider:]]  # bottom right

        seperate_image = torch.stack([seperate_list[i] for i in idx_shuffle], dim=0)
        return (img, self.split.index(target)), (seperate_image, idx_shuffle)
    
    def __len__(self) -> int:
        return len(self.split_idx)
