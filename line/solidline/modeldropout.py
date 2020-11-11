import torch.nn as nn
import torch.nn.functional as F
import torch

def m_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.Dropout(p=0.5),
        nn.ReLU(inplace=True)
    )   


class FoInternNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = m_conv(3, 64)
        self.down2 = m_conv(64, 128)
        self.down3 = m_conv(128, 256)
        self.down4 = m_conv(256, 512)

        self.maxpooling = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up3 = m_conv(256 + 512, 256)
        self.up2 = m_conv(128 + 256, 128)
        self.up1 = m_conv(128 + 64, 64)

        self.last = nn.Conv2d(64, n_classes, 1)
        
        

    def forward(self, x):
        conv1 = self.down1(x)
        x = self.maxpooling(conv1)

        conv2 = self.down2(x)
        x = self.maxpooling(conv2)

        conv3 = self.down3(x)
        x = self.maxpooling(conv3)

        x = self.down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.up1(x)

        out_layer = self.last(x)


        return out_layer



if __name__ == '__main__':
    model = FoInternNet(n_classes=2)