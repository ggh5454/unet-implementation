import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import UNet
from dataset import KvasirDataset, MembraneDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS = 200
CLASS = 1
CHANNELS = 1
IMAGE_SIZE = (572, 572)
DATASET = "membrane_dataset"
SAVE = True
tr_img = transforms.ToPILImage()
PATH = "model/membrane_best.pt"

transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    ToTensorV2(transpose_mask=True),
])

if DATASET == "data/Kvasir-SEG":
    test_dataset = KvasirDataset(path= "data/Kvasir-SEG", transform=transform, mode="test")
else:
    test_dataset = MembraneDataset(path= "data/membrane_dataset", transform=transform, mode="test")
    
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(CHANNELS, CLASS).to(device)
model.load_state_dict(torch.load(PATH))

with torch.no_grad():
    for imgs, masks in test_dataloader:
        outputs = model(imgs.to(device)).detach().cpu()

        outputs = torch.where(outputs>0.5, 1.0, 0.0)

        for i in range(BATCH_SIZE):
            plt.subplot(1,3,1)
            plt.imshow(tr_img(imgs[i]), cmap='gray')
            plt.subplot(1,3,2)

            plt.imshow(tr_img(masks[i]), cmap='gray')
            plt.subplot(1,3,3)

            plt.imshow(tr_img(outputs[i]), cmap='gray')
            plt.show()