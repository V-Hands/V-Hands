import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_bn=True, with_relu=True):
        super(Conv2dBNRelu, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]
        if with_bn:       
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        if with_relu:
            layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 

class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)

class GRU(nn.Module):
    def __init__(self, channels):
        super(GRU, self).__init__()
        self.zx = Conv2dBNRelu(channels + 2, channels, 3, padding=1, with_relu=False)
        self.zh = Conv2dBNRelu(channels + 2, channels, 3, padding=1, with_relu=False)
        self.rx = Conv2dBNRelu(channels + 2, channels, 3, padding=1, with_relu=False)
        self.rh = Conv2dBNRelu(channels + 2, channels, 3, padding=1, with_relu=False)

        self.hx = Conv2dBNRelu(channels, channels, 3, padding=1, with_relu=False)
        self.hh = Conv2dBNRelu(channels, channels, 3, padding=1, with_relu=False)

        self.h_init = Conv2dBNRelu(channels, channels, 3, padding=1, with_relu=False)

    def forward(self, x, h, img):
        if h == None:
            h_init = F.tanh(
                self.h_init(x)
            )
            return h_init
        else:
            x_img = torch.cat([x, img], dim=1)
            h_img = torch.cat([h, img], dim=1)
            z = F.sigmoid(self.zx(x_img) + self.zh(h_img))
            r = F.sigmoid(self.rx(x_img) + self.rh(h_img))
            h_update = F.tanh(
                self.hx(x) + self.hh(r * h)
            )
            h = (1 - z) * h + z * h_update
            return h

class DOWN(nn.Module):
    def __init__(self, cin, cout, with_relu):
        super(DOWN, self).__init__()
        self.layer_update = Conv2dBNRelu(cin + 2, cout, 3, padding=1, stride=2, with_relu=with_relu)

    def forward(self, x, img):
        return self.layer_update(torch.cat([x, img], dim=1))

class UP(nn.Module):
    def __init__(self, cin, cout):
        super(UP, self).__init__()
        self.layer_update = Conv2dBNRelu(cin + 2, cout, 3, padding=1, with_relu=False)

    def forward(self, h, img, res=None):
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.layer_update(torch.cat([h, img], dim=1))
        if not res == None:
            x = x + res
        return F.leaky_relu(x)

class PoseNet(nn.Module):
    def __init__(self, z_m=-1.2, z_M=1.2):
        super(PoseNet, self).__init__()

        self.initlayer = Conv2dBNRelu(2, 96, 3, padding=1, with_relu=True)
        
        self.DOWN2 = DOWN(96, 96, False)
        self.DOWN4 = DOWN(96, 96, False)
        self.DOWN8 = DOWN(96, 128, False)
        self.DOWN16 = DOWN(128, 256, False)
        self.DOWN32 = DOWN(256, 512, True)

        self.se_block = SEBlock(512, 512, reduction=4)

        self.UP16 = UP(512, 256)
        self.UP8 = UP(256, 128)
        self.UP4 = UP(128, 96)
        self.UP2 = UP(96, 96)
        self.UP1 = UP(96, 96)

        self.uh32 = GRU(512)
        self.uh16 = GRU(256)
        self.uh8 = GRU(128)
        self.uh4 = GRU(96)
        self.uh2 = GRU(96)
        self.uh1 = GRU(96)

        self.heatmap_layer = nn.Conv2d(96, 21*2, 3, padding=1)
        self.existence_conv32 = Conv2dBNRelu(512, 128, 3, padding=1)
        self.existence_conv1 = Conv2dBNRelu(96, 128, 3, padding=1)
        self.existence_linear = nn.Linear(256, 2)

        self.depth_conv32 = Conv2dBNRelu(512, 128, 3, padding=1)
        self.depth_conv1 = Conv2dBNRelu(96, 128, 3, padding=1)
        self.depth_linear = nn.Linear(256, 21*2*48)

        depth_base = []
        for i in range(48):
            depth_base.append(z_m + (z_M - z_m) * i / 47.)
        self.depth_base = depth_base

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self._init_layer(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                self._init_norm(m)

    def forward(self, img_pad, hs):
        img_pad2x = F.interpolate(img_pad, scale_factor=1/2, mode='bilinear', align_corners=False)
        img_pad4x = F.interpolate(img_pad, scale_factor=1/4, mode='bilinear', align_corners=False)
        img_pad8x = F.interpolate(img_pad, scale_factor=1/8, mode='bilinear', align_corners=False)
        img_pad16x = F.interpolate(img_pad, scale_factor=1/16, mode='bilinear', align_corners=False)
        img_pad32x = F.interpolate(img_pad, scale_factor=1/32, mode='bilinear', align_corners=False)

        if hs == None:
            uh32, uh16, uh8, uh4, uh2, uh1 = None, None, None, None, None, None
        else:
            uh32, uh16, uh8, uh4, uh2, uh1 = hs[0], hs[1], hs[2], hs[3], hs[4], hs[5]

        down1 = self.initlayer(img_pad)
        down2 = self.DOWN2(down1, img_pad)
        down4 = self.DOWN4(F.leaky_relu(down2), img_pad2x)
        down8 = self.DOWN8(F.leaky_relu(down4), img_pad4x)
        down16 = self.DOWN16(F.leaky_relu(down8), img_pad8x)
        down32 = self.DOWN32(F.leaky_relu(down16), img_pad16x)
        
        down32 = self.se_block(down32)

        uh32o = self.uh32(down32, uh32, img_pad32x)
        up16 = self.UP16(uh32o, img_pad16x, down16)
        uh16o = self.uh16(up16, uh16, img_pad16x)
        up8 = self.UP8(uh16o, img_pad8x, down8)
        uh8o = self.uh8(up8, uh8, img_pad8x)
        up4 = self.UP4(uh8o, img_pad4x, down4)
        uh4o = self.uh4(up4, uh4, img_pad4x)
        up2 = self.UP2(uh4o, img_pad2x, down2)
        uh2o = self.uh2(up2, uh2, img_pad2x)
        up1 = self.UP1(uh2o, img_pad)
        uh1o = self.uh1(up1, uh1, img_pad)

        heatmap = self.heatmap_layer(uh1o)

        existence_feature32 = self.existence_conv32(uh32o)
        existence_feature1 = self.existence_conv1(uh1o)
        existence_feature32 = existence_feature32.mean(dim=(2,3))
        existence_feature1 = existence_feature1.mean(dim=(2,3))
        existence_feature = torch.cat([existence_feature32, existence_feature1], dim=1)
        existence_output = self.existence_linear(existence_feature)

        depth_feature32 = self.depth_conv32(uh32o)
        depth_feature1 = self.depth_conv1(uh1o)
        depth_feature32 = depth_feature32.mean(dim=(2,3))
        depth_feature1 = depth_feature1.mean(dim=(2,3))
        depth_feature = torch.cat([depth_feature32, depth_feature1], dim=1)
        depth_weight = self.depth_linear(depth_feature).view(-1, 21*2, 48)
        depth_weight = F.softmax(depth_weight, dim=2)
        depth_base = torch.tensor(self.depth_base, device='cuda', dtype=torch.float32).view(1, 1, 48)
        depth = torch.sum(depth_weight * depth_base, dim=2)

        return heatmap, depth, F.sigmoid(existence_output), [uh32o,uh16o,uh8o,uh4o,uh2o,uh1o]

    def _init_layer(self, layer):
        nn.init.kaiming_uniform_(
            layer.weight, a=0, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

