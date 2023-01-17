import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import Dataset, DataLoader

class KvasirDataset(Dataset):
    def __init__(self, path="data/Kvasir-SEG", transform='') -> None:
        super().__init__()
        self.path = path
        self.transform =transform
        self.images = sorted(glob(path + '/images/*'))
        self.masks = sorted(glob(path + '/masks/*'))
        self.json = glob(path + '/*.json')
    
    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
    
    def __len__(self):
        return len(self.images)
