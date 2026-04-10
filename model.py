import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,  
            stride=stride,  
            padding=1,       
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None


        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = x  


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

  
        out = out + identity
        out = self.relu(out)

        return out


class ResNetDepth(nn.Module):
    def __init__(self):
        super().__init__()

 
        self.in_channels = 64

  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.conv_x3 = nn.Conv2d(256, 512, 1)
        self.conv_x2 = nn.Conv2d(128, 256, 1)
        self.conv_x1 = nn.Conv2d(64, 128, 1)

        self.reduce_x4 = nn.Conv2d(512, 256, 1)
        self.reduce_d2 = nn.Conv2d(256, 128, 1)



        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, 1)
        )


    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(
            BasicBlock(self.in_channels, out_channels, stride)
        )

        self.in_channels = out_channels

  
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.in_channels, out_channels)
            )

        return nn.Sequential(*layers)


    def forward(self, x):


        x = self.conv1(x)   
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)  

        x1 = self.layer1(x)    
        x2 = self.layer2(x1)   
        x3 = self.layer3(x2)   
        x4 = self.layer4(x3)    


        
        x3 = self.conv_x3(x3)
        d = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d = d + x3

        x2 = self.conv_x2(x2)
        d = self.reduce_x4(d)
        d = F.interpolate(d, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d = d + x2

        x1 = self.conv_x1(x1)
        d = self.reduce_d2(d)
        d = F.interpolate(d, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d = d + x1


        d = self.head(d)
        d = F.interpolate(d, size=(180, 180), mode='bilinear', align_corners=False)

        return d

        

