import torch.nn as nn
import torch.nn.functional as F
from model.ConvBlock import *

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)

        r = up1 + up2
        return r

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):
    def __init__(self, cfg):
        super(HGFilter, self).__init__()
        self.num_modules = cfg["model"]["num_stack"]

        self.cfg = cfg

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)

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

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, cfg["model"]["num_hourglass"], 256, self.cfg["model"]["norm"]))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.cfg["model"]["norm"]))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.cfg["model"]["norm"] == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.cfg["model"]["norm"] == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(256, cfg["model"]["hourglass_dim"], kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(cfg["model"]["hourglass_dim"],
                                                                 256, kernel_size=1, stride=1, padding=0))
            if hg_module == (self.num_modules-1) and self.cfg["model"]["recover_dim"]:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(cfg["model"]["hourglass_dim"], 256, kernel_size=1, stride=1, padding=0))

        # recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512
        if self.cfg["model"]["recover_dim"]:
            self.recover_dim_match_fea_1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
            self.recover_dim_conv_1      = ConvBlock(256, 256, self.cfg["model"]["norm"])
            self.recover_dim_match_fea_2 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0)
            self.recover_dim_conv_2      = ConvBlock(256, 256, self.cfg["model"]["norm"])
            

    def forward(self, x):
        raw_x = x

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

        previous = x

        # Multiscale version
        # l = [self.inception(self.conv(F.relu(previous)))]
        # for i in range(1, 4):
        #     tmp = F.avg_pool2d(l[-1], 2, stride=2)
        #     l.append(F.upsample(x, size=l[0].shape[2:], scale_factor=None, mode='bilinear'))


        outputs = []
        for i in range(self.num_modules):
            
            # Multiscale version
            # previous = previous + self.inceptions_2[i](self.inceptions_1[i](l[3-i]))
            
            
            hg = self._modules['m' + str(i)](previous)
            # [1, 256, 128, 128]

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)
            # [1, 256, 128, 128]
            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)
            # [1, 256, 128, 128]

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            # [1, 256, 128, 128]
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                # [1, 256, 128, 128]
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                # [1, 256, 128, 128]
                previous = previous + ll + tmp_out_
                # [1, 256, 128, 128]
                # recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512
            if i == (self.num_modules-1) and self.cfg["model"]["recover_dim"]:

                # merge features
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                fea_upsampled = previous + ll + tmp_out_ # (BV,256,128,128)

                # upsampling: (BV,256,128,128) to (BV,256,256,256)
                if self.cfg["model"]["upsample_mode"] == "bicubic":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='bicubic', align_corners=True) # (BV,256,256,256)
                elif self.cfg["model"]["upsample_mode"] == "nearest":
                    
                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='nearest') # (BV,256,256,256)
                else:
                    print("Error: undefined self.upsample_mode")
                    
                fea_upsampled = fea_upsampled + self.recover_dim_match_fea_1(tmpx)
                fea_upsampled = self.recover_dim_conv_1(fea_upsampled) # (BV,256,256,256)

                # upsampling: (BV,256,256,256) to (BV,256,512,512)
                if self.cfg["model"]["upsample_mode"] == "bicubic":

                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='bicubic', align_corners=True) # (BV,256,512,512)
                elif self.cfg["model"]["upsample_mode"] == "nearest":
                    
                    fea_upsampled = F.interpolate(fea_upsampled, scale_factor=2, mode='nearest') # (BV,256,512,512)
                else:
                    print("Error: undefined self.upsample_mode")

                fea_upsampled = fea_upsampled + self.recover_dim_match_fea_2(raw_x)
                fea_upsampled = self.recover_dim_conv_2(fea_upsampled) # (BV,256,512,512)

                outputs.append(fea_upsampled)

        return outputs, tmpx.detach(), normx
