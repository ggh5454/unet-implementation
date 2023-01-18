import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import KvasirDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS = 200

transform = A.Compose([
    A.Resize(572, 572),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.ElasticTransform(),
    ToTensorV2(transpose_mask=True),
])

train_dataset = KvasirDataset(path= "data/Kvasir-SEG", transform=transform)
test_dataset = KvasirDataset(path= "data/Kvasir-SEG", transform=transform, mode="test")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(3, 1).to(device)

lr = 3e-4

optimizer = optim.SGD(model.parameters(), lr=lr)

criterion = nn.BCEWithLogitsLoss()


model.train()
loss_list = []
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for img, mask in train_dataloader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        output = model(img)

        loss = criterion(output, mask)

        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    
    loss_list.append(total_loss / len(train_dataloader))
    print(f"[{epoch}] loss : {total_loss / len(train_dataloader)}")

    


import matplotlib.pyplot as plt
model.eval()

imgs, masks = next(iter(test_dataloader))
outputs = model(imgs.to(device)).detach().cpu()
tr_img = transforms.ToPILImage()
for i in range(BATCH_SIZE):
    plt.subplot(1,3,1)
    plt.imshow(tr_img(imgs[i]))
    plt.subplot(1,3,2)

    plt.imshow(tr_img(masks[i]))
    plt.subplot(1,3,3)

    plt.imshow(tr_img(outputs[i]))
    plt.show()


