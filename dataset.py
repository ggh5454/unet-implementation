import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import Dataset, DataLoader
from custom_transform import ElasticTransform

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
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = np.array(img, dtype=np.float32) / 255
        mask = self.masks[index]
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        mask = np.array(mask, dtype=np.float32) / 255

        sample = {
            "image": img,
            "mask" : mask
        }

        if self.transform:
            sample = self.transform(sample)
            img = sample["image"]
            mask = sample["mask"]

        return img, mask
    
    def __len__(self):
        return len(self.images)

class MembraneDataset(Dataset):
    def __init__(self, path="data/membrane_dataset", transform='', mode="train") -> None:
        super().__init__()
        self.path = path
        self.transform =transform
        if mode == "train":
            self.images = sorted(glob(path + '/train/input*'))
            self.masks = sorted(glob(path + '/train/label*'))
        else:
            self.images = sorted(glob(path + '/test/input*'))
            self.masks = sorted(glob(path + '/test/label*'))

        
    def __getitem__(self, index):
        img = np.load(self.images[index])
        img = np.array(img, dtype=np.float32) / 255
        mask = np.load(self.masks[index])
        mask = np.array(mask, dtype=np.float32) / 255
        sample = {
            "image": img,
            "mask" : mask
        }

        if self.transform:
            sample = self.transform(sample)
            img = sample["image"]
            mask = sample["mask"]
            # sigma  = torch.randint(6, 12, (1,)).item()
            # elastic = ElasticTransform(alpha=10, sigma=sigma, random_seed=42)
            # elastic.requires_grad_(False)
            # img = elastic(img)
            # mask= elastic(mask)

            # transformed = self.transform(image=img, mask=mask)
            # img = transformed['image']
            
            # mask = transformed['mask']

        return img, mask
    
    def __len__(self):
        return len(self.images)



### membrane dataset을 npy로 바꾸기

# https://dacon.io/en/codeshare/4245
## 라이브러리 불러오기
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# ## 데이터 불러오기
# dir_data = './data/membrane_dataset/' 

# name_label = 'train-labels.tif'
# name_input = 'train-volume.tif'

# img_label = Image.open(os.path.join(dir_data, name_label))
# img_input = Image.open(os.path.join(dir_data, name_input))

# ny, nx = img_label.size
# nframe = img_label.n_frames

# ## train/test/val 폴더 생성
# nframe_train = 26
# nframe_test = 4

# dir_save_train = os.path.join(dir_data, 'train')
# dir_save_test = os.path.join(dir_data, 'test')

# if not os.path.exists(dir_save_train):
#     os.makedirs(dir_save_train)


# if not os.path.exists(dir_save_test):
#     os.makedirs(dir_save_test)

# ## 전체 이미지 30개를 섞어줌
# id_frame = np.arange(nframe)
# np.random.shuffle(id_frame)

# ## 선택된 train 이미지를 npy 파일로 저장
# offset_nframe = 0

# for i in range(nframe_train):
#     img_label.seek(id_frame[i + offset_nframe])
#     img_input.seek(id_frame[i + offset_nframe])

#     label_ = np.asarray(img_label)
#     input_ = np.asarray(img_input)

#     np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
#     np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# ## 선택된 test 이미지를 npy 파일로 저장
# offset_nframe = nframe_train

# for i in range(nframe_test):
#     img_label.seek(id_frame[i + offset_nframe])
#     img_input.seek(id_frame[i + offset_nframe])

#     label_ = np.asarray(img_label)
#     input_ = np.asarray(img_input)

#     np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
#     np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

# ## 이미지 시각화
# plt.subplot(122)
# plt.imshow(label_, cmap='gray')
# plt.title('label')

# plt.subplot(121)
# plt.imshow(input_, cmap='gray')
# plt.title('input')

# plt.show()