import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetDepth(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1)   
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        x = F.interpolate(x, size=(180,180))
        return x
