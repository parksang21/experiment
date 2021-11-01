from typing import *
from numpy.core.defchararray import split
import torchvision
from torchvision import transforms as T
import torch

from PIL import Image
import numpy as np


class OpenTestCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, 
                 root: str='/datasets',
                 train: bool=False,
                 transform: Optional[Callable]=None,
                 target_tarsnform: Optional[Callable]=None,
                 download: bool=False,
                 split: list=None) -> None:
        super(OpenTestCifar100, self).__init__(root, train=train, transform=transform, 
                                              target_transform=target_tarsnform, 
                                              download=download)
        self.split_idx = None
        assert split, 'split is requried'
        self.set_split(split)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        known_target = True if target in self.split else False
        tTarget = self.split.index(target) if target in self.split else 9999
        
        return img, tTarget, target, known_target
    
    def set_split(self, split):
        self.split = split
        np_target = np.array(self.targets)
        split_idxs = np.isin(np_target, split)
        np_target = np.where(split_idxs == True)[0]

        self.split_idx = np_target
        
        
class SplitCifar100(torchvision.datasets.CIFAR100):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 split: list = None
                 ) -> None:

        super(SplitCifar100, self).__init__(root, train=train, transform=transform,
                                           target_transform=target_transform, download=download,)

        self.split_idx = None
        self.split = split
        if not self.split is None:
            self.set_split(self.split)

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
