import torch
import torch.nn as nn
import torch.functional as F


def crop(x, coord, shape):
    return x[:,:,coord[-1]:shape[-2] +coord[-1] ,coord[0]:shape[-1]+coord[0]]

def convblock(in_feature, out_feature):
    conv = nn.Sequential(
        nn.Conv2d(in_feature, out_feature, stride=1, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_feature, out_feature, stride=1, kernel_size=3),
        nn.ReLU()
    )

    if out_feature == 2:
        conv = nn.Sequential(
        nn.Conv2d(in_feature, out_feature, stride=1, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_feature, out_feature, stride=1, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_feature, 2, stride=1, kernel_size=1)
    )

    return conv
def findxy(shape1, shape2):
    x = torch.randint(0, shape1[-1] - shape2[-1], (1,))
    y = torch.randint(0, shape1[-2] - shape2[-2], (1,))
    return x, y

class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        features = [64, 128, 256, 512, 1024]

        # contracting path
        self.convblock1 = convblock(1, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.convblock2 = convblock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)

        self.convblock3 = convblock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)

        self.convblock4 = convblock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.convblock5 = convblock(features[3], features[4])

        # expansive path
        self.upconv1 = nn.ConvTranspose2d(features[4], features[3], stride=2, kernel_size=2)
        self.convblock6 = convblock(features[4], features[3])

        self.upconv2 = nn.ConvTranspose2d(features[3], features[2], stride=2, kernel_size=2)
        self.convblock7 = convblock(features[3], features[2])

        self.upconv3 = nn.ConvTranspose2d(features[2], features[1], stride=2, kernel_size=2)
        self.convblock8 = convblock(features[2], features[1])

        self.upconv4 = nn.ConvTranspose2d(features[1], features[0], stride=2, kernel_size=2)
        self.convblock9 = convblock(features[1], 2)

    def forward(self, x):
        x1 = self.convblock1(x)
        x = self.pool1(x1)
        x2 = self.convblock2(x)
        x = self.pool2(x2)
        x3 = self.convblock3(x)
        x = self.pool3(x3)
        x4 = self.convblock4(x)
        x = self.pool4(x4)
        x = self.convblock5(x)
        x = self.upconv1(x)
        coord = findxy(x4.shape, x.shape)
        x4 = crop(x4, coord, x.shape)
        x = self.convblock6(torch.cat((x4, x), 1))
        x = self.upconv2(x)
        coord = findxy(x3.shape, x.shape)
        x3 = crop(x3, coord, x.shape)
        x = self.convblock7(torch.cat((x3, x), 1))
        x = self.upconv3(x)
        coord = findxy(x2.shape, x.shape)
        x2 = crop(x2, coord, x.shape)
        x = self.convblock8(torch.cat((x2, x), 1))
        x = self.upconv4(x)
        coord = findxy(x1.shape, x.shape)
        x1 = crop(x1, coord, x.shape)
        x = self.convblock9(torch.cat((x1, x), 1))
        return x





