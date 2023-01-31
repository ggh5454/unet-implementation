import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_transform import ElasticTransform, ToTensor, ToPILImage, Resize, RandomHorizontalFlip, RandomVerticalFlip
from model import UNet
from dataset import KvasirDataset, MembraneDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from loss import EnergyLoss, dice_loss, MixedLoss, FocalLoss
import os
import numpy as np
import matplotlib.pyplot as plt
import metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS = 200
CLASS = 1
CHANNELS = 1
IMAGE_SIZE = (572, 572)
DATASET = "membrane_dataset"
SAVE = False
VISUALIZATION = True

# transform = A.Compose([
#     A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
#     A.HorizontalFlip(),
#     A.VerticalFlip(),
#     ToTensorV2(transpose_mask=True),
# ])

transform = transforms.Compose([
    ToTensor(),
    Resize(IMAGE_SIZE),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ElasticTransform(alpha=2, sigma=15, random_seed=42),
])

if DATASET == "data/Kvasir-SEG":
    train_dataset = KvasirDataset(path= "data/Kvasir-SEG", transform=transform)
    test_dataset = KvasirDataset(path= "data/Kvasir-SEG", transform=transform, mode="test")
else:
    train_dataset = MembraneDataset(path= "data/membrane_dataset", transform=transform)
    test_dataset = MembraneDataset(path= "data/membrane_dataset", transform=transform, mode="test")
    
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(CHANNELS, CLASS).to(device)

lr = 3e-4

optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.BCEWithLogitsLoss()


test_img, test_mask = next(iter(test_dataloader))
tr_img = transforms.ToPILImage()

model.train()
loss_list = []
Iou = []
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for img, mask in train_dataloader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        outputs = model(img)

        loss = criterion(outputs, mask)

        loss.backward(retain_graph=True)

        optimizer.step()
        iou = metrics.IoU(torch.where(outputs>0.5, 1.0, 0.0), mask)
        Iou.append(iou)
        total_loss += loss.item()
    
    with torch.no_grad():
        imgs, masks = test_img, test_mask
        outputs = model(imgs.to(device)).detach().cpu()
        outputs = torch.where(outputs>0.5, 1.0, 0.0)
        plt.subplot(1,3,1)
        plt.imshow(tr_img(imgs[0]), cmap='gray')
        plt.subplot(1,3,2)

        plt.imshow(tr_img(masks[0]), cmap='gray')
        plt.subplot(1,3,3)

        plt.imshow(tr_img(outputs[0]), cmap='gray')
        plt.show()

    
    loss_list.append(total_loss / len(train_dataloader))
    print(f"[{epoch}] loss : {total_loss / len(train_dataloader)}, Iou : {iou}")

if SAVE:    
    torch.save(model.state_dict(), "model/membrane_best.pt")

if VISUALIZATION:
    plt.subplot(121)
    plt.plot(loss_list, label="loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(Iou, label="IOU")
    plt.legend()
    plt.show()





