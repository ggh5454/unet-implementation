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
    def __init__(self, path="data/Kvasir-SEG", transform='', mode="train") -> None:
        super().__init__()
        self.path = path
        self.transform =transform
        self.images = sorted(glob(path + '/images/*'))
        self.masks = sorted(glob(path + '/masks/*'))
        self.json = glob(path + '/*.json')
        size = len(self.images) * 0.8
        if mode == "train":
            self.images = self.images[:int(size)]
            self.masks = self.masks[:int(size)]
        else:
            self.images = self.images[int(size):]
            self.masks = self.masks[int(size):]

        
        
    
    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img)
        img = np.array(img, dtype=np.float32)
        mask = self.masks[index]
        mask = Image.open(mask).convert("L")
        mask = np.array(mask, dtype=np.float32)


        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            
            mask = transformed['mask']

        return img, mask.unsqueeze(0)
    
    def __len__(self):
        return len(self.images)
