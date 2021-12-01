import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ConvBlock import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, conv_depth=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        if conv_depth is not None:
            self.dconv = nn.Conv2d(conv_depth[0], conv_depth[1], kernel_size=1)
        else:
            self.dconv = None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.dconv is not None:
            x2 = self.dconv(x2)
            x = x1 + x2
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, cfg, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.cfg = cfg
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # If use 2 more reductions
        self.down5 = Down(512, 1024 // factor)
        self.down6 = Down(512, 1024 // factor)
        


        # Original code
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        # Version that keeps large depth in the up part
        # "plus"
        # self.up1 = Up(512, 512, bilinear)
        # self.up2 = Up(512, 512, bilinear, [256, 512])
        # self.up3 = Up(512, 256, bilinear, [128, 512])
        # self.up4 = Up(256, 256, bilinear, [64, 256])
        # self.outc = OutConv(256, 256)

        # Version that keeps large depth in the up part
        # concat
        self.up1 = Up(512+512, 512, bilinear)
        self.up2 = Up(512+256, 512, bilinear)
        self.up3 = Up(512+128, 256, bilinear)
        self.up4 = Up(256+64, 256, bilinear)
        self.outc = OutConv(256, 256)

        # If use 2 more reductions
        self.up0 = Up(512+512, 512, bilinear)
        self.up00 = Up(512+512, 512, bilinear)



        # Version compatible if we use first a base part like SHG        
        # self.inc = DoubleConv(n_channels, 256)
        # self.down1 = Down(256, 256)
        # self.down2 = Down(256, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512, bilinear)
        # self.up2 = Up(512+256, 512, bilinear)
        # self.up3 = Up(512+256, 256, bilinear)
        # self.up4 = Up(256+256, 256, bilinear)
        # self.outc = OutConv(256, 256)
      

        # SHG base part
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if self.cfg["model"]["norm"] == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.cfg["model"]["norm"] == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.cfg["model"]["hg_down"] == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.cfg["model"]["norm"])
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.cfg["model"]["hg_down"] == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.cfg["model"]["norm"])
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.cfg["model"]["hg_down"] == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.cfg["model"]["norm"])
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.cfg["model"]["norm"])
        self.conv4 = ConvBlock(128, 256, self.cfg["model"]["norm"])
        """
    def forward(self, x):

        # SHG base part
        """
        # [1, 3, 512, 512]
        x = F.relu(self.bn1(self.conv1(x)), True)
        # [1, 64, 256, 256]
        tmpx = x
        # conv2 => [1, 128, 256, 256]
        if self.cfg["model"]["hg_down"] == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
            # [1, 128, 128, 128]
        elif self.cfg["model"]["hg_down"] in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')
        
        normx = x
        # [1, 128, 128, 128]
        x = self.conv3(x)
        # [1, 128, 128, 128]
        x = self.conv4(x)
        # [1, 256, 128, 128]
        """
        
        
        # Just basic resolution reduction
        # x = F.avg_pool2d(x, 2, stride=2)
        # x = F.avg_pool2d(x, 2, stride=2)

        # x 2, 3, 512, 512
        x1 = self.inc(x)
        # x1 2, 64, 512, 512
        x2 = self.down1(x1)
        # x2 2, 128, 256, 256
        x3 = self.down2(x2)
        # x3 2, 256, 128, 128
        x4 = self.down3(x3)
        # x4 2, 512, 64, 64
        x5 = self.down4(x4)
        # x5 2, 512, 32, 32 / x4 2, 512, 64, 64
        

        # If use 2 more reductions        
        # x6 = self.down5(x5)
        # x7 = self.down6(x6)
        # x = self.up0(x7, x6)
        # x = self.up00(x6, x)

        # Use only Encoding part of Unet : if multiscale_mlp for example
        # return [x1,x2,x3,x4,x5]

        x = self.up1(x5, x4)
        # x 2, 512, 64, 64 / x3 2, 256, 128, 128
        x = self.up2(x4, x3)
        # x 2, 512, 128, 128 / x2 2, 128, 256, 256
        x = self.up3(x, x2)
        # x 2, 256, 256, 256 / x1 2, 64, 512, 512
        x = self.up4(x, x1)
        # x 2, 256, 512, 512
        logits = self.outc(x)
        # logits 2, 256, 512, 512
        return [logits]