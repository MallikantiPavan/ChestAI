import math

import torch
from torch import nn
from torch.nn import functional as F

from model.utils import get_norm


class Conv2dNormRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 bias=True, norm_type='Unknown'):
        super(Conv2dNormRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
            get_norm(norm_type, out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class CAModule(nn.Module):
    def __init__(self, num_channels, reduc_ratio=2):
        super(CAModule, self).__init__()
        self.num_channels = num_channels
        self.reduc_ratio = reduc_ratio

        self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio,
                             bias=True)
        self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels,
                             bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        gap_out = feat_map.view(feat_map.size()[0], self.num_channels,
                                -1).mean(dim=2)
        fc1_out = self.relu(self.fc1(gap_out))
        fc2_out = self.sigmoid(self.fc2(fc1_out))
        fc2_out = fc2_out.view(fc2_out.size()[0], fc2_out.size()[1], 1, 1)
        feat_map = torch.mul(feat_map, fc2_out)

        return feat_map



class ECAModule(nn.Module):
    def __init__(self, num_channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log2(num_channels) / gamma) + b))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y

class SAModule(nn.Module):

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = self.conv1(feat_map).view(batch_size, -1,
                                               width * height).permute(0, 2, 1)

        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = F.softmax(relation_map, dims=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map


class FPAModule(nn.Module):
    def __init__(self, num_channels, norm_type):
        super(FPAModule, self).__init__()

        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dNormRelu(num_channels, num_channels, kernel_size=1,
                           norm_type=norm_type))
        self.mid_branch = Conv2dNormRelu(num_channels, num_channels,
                                         kernel_size=1, norm_type=norm_type)

        self.downsample1 = Conv2dNormRelu(num_channels, 1, kernel_size=7,
                                          stride=2, padding=3,
                                          norm_type=norm_type)

        self.downsample2 = Conv2dNormRelu(1, 1, kernel_size=5, stride=2,
                                          padding=2, norm_type=norm_type)

        self.downsample3 = Conv2dNormRelu(1, 1, kernel_size=3, stride=2,
                                          padding=1, norm_type=norm_type)

        self.scale1 = Conv2dNormRelu(1, 1, kernel_size=7, padding=3,
                                     norm_type=norm_type)
        self.scale2 = Conv2dNormRelu(1, 1, kernel_size=5, padding=2,
                                     norm_type=norm_type)
        self.scale3 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1,
                                     norm_type=norm_type)

    def forward(self, feat_map):
        height, width = feat_map.size(2), feat_map.size(3)
        gap_branch = self.gap_branch(feat_map)
        gap_branch = nn.Upsample(size=(height, width), mode='bilinear',
                                 align_corners=False)(gap_branch)

        mid_branch = self.mid_branch(feat_map)

        scale1 = self.downsample1(feat_map)
        scale2 = self.downsample2(scale1)
        scale3 = self.downsample3(scale2)

        scale3 = self.scale3(scale3)
        scale3 = nn.Upsample(size=(height // 4, width // 4), mode='bilinear',
                             align_corners=False)(scale3)
        scale2_ = self.scale2(scale2)
        if scale2_.shape != scale3.shape:
            scale3 = F.interpolate(scale3, size=scale2_.shape[2:], mode='bilinear', align_corners=False)
        scale2 = scale2_ + scale3
        scale2 = nn.Upsample(size=(height // 2, width // 2), mode='bilinear',
                             align_corners=False)(scale2)
        scale1_ = self.scale1(scale1)
        if scale1_.shape != scale2.shape:
            scale2 = F.interpolate(scale2, size=scale1_.shape[2:], mode='bilinear', align_corners=False)
        scale1 = scale1_ + scale2
        scale1 = nn.Upsample(size=(height, width), mode='bilinear',
                             align_corners=False)(scale1)

        feat_map = torch.mul(scale1, mid_branch) + gap_branch

        return feat_map


class AttentionMap(nn.Module):

    def __init__(self, cfg, num_channels):
        super(AttentionMap, self).__init__()
        self.cfg = cfg
        self.channel_attention = CAModule(num_channels)
        self.eca_attention = ECAModule(num_channels)
        self.spatial_attention = SAModule(num_channels)
        self.pyramid_attention = FPAModule(num_channels, cfg.norm_type)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map):
        if self.cfg.attention_map == "CAM":
            return self.channel_attention(feat_map)
        elif self.cfg.attention_map == "SAM":
            return self.spatial_attention(feat_map)
        elif self.cfg.attention_map == "FPA":
            return self.pyramid_attention(feat_map)
        elif self.cfg.attention_map == "ECA":
            return self.eca_attention(feat_map)
        elif self.cfg.attention_map == "None":
            return feat_map
        else:
            Exception('Unknown attention type : {}'
                      .format(self.cfg.attention_map))
