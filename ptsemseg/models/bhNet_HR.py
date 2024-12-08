from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.submodule import *
from ptsemseg.models.utils import unetUpsimple, unetConv2, unetUp, unetUpC


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 确保卷积层的输出通道数至少为 1
        self.fc1 = nn.Conv2d(in_planes, max(in_planes // ratio, 1), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(in_planes // ratio, 1), in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add residual connection
        out = self.relu(out)
        return out


class Uencoder(nn.Module):
    def __init__(
            self, feature_scale=4, is_deconv=True, in_channels=3, is_batchnorm=True, filters=[64, 128, 256, 512, 1024]
    ):
        super(Uencoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels)

        # filters = [int(x / self.feature_scale) for x in filters]
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.residual1 = ResidualBlock(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.residual2 = ResidualBlock(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.residual3 = ResidualBlock(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.residual4 = ResidualBlock(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.residual_center = ResidualBlock(filters[4], filters[4])

    def forward(self, inputs):
        # 通道注意力机制
        inputs = self.channel_attention(inputs) * inputs

        # inputs
        conv1 = self.conv1(inputs)
        conv1 = self.residual1(conv1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        conv2 = self.residual2(conv2)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        conv3 = self.residual3(conv3)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        conv4 = self.residual4(conv4)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.residual_center(center)

        return conv1, conv2, conv3, conv4, center


class Udecoder(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, filters=[64, 128, 256, 512, 1024]):
        super(Udecoder, self).__init__()
        self.is_deconv = is_deconv
        self.feature_scale = feature_scale
        self.filters = filters

        # 空间注意力机制
        self.spatial_attention = SpatialAttention()

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, conv1, conv2, conv3, conv4, center):
        # 空间注意力机制
        center = self.spatial_attention(center) * center

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


class UdecoderC(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, filters=[64, 128, 256, 512, 1024]):
        super(UdecoderC, self).__init__()
        self.is_deconv = is_deconv
        self.feature_scale = feature_scale
        self.filters = filters

        # 空间注意力机制
        self.spatial_attention = SpatialAttention()

        # upsampling
        self.up_concat4 = unetUpC(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, conv1, conv2, conv3, conv4, center):
        # 空间注意力机制
        center = self.spatial_attention(center) * center

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


class BHNet_HR(nn.Module):
    def __init__(
            self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(BHNet_HR, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.uencoder1 = Uencoder(self.feature_scale, self.is_deconv, 4, self.is_batchnorm, filters)
        self.uencoder2 = Uencoder(self.feature_scale, self.is_deconv, 3, self.is_batchnorm, filters)
        self.uencoder3 = Uencoder(self.feature_scale, self.is_deconv, 2, self.is_batchnorm, filters)
        self.uencoder4 = Uencoder(self.feature_scale, self.is_deconv, 8, self.is_batchnorm, filters)
        self.uencoder5 = Uencoder(self.feature_scale, self.is_deconv, 4, self.is_batchnorm, filters)

        # upsampling
        self.udecoder1 = Udecoder(self.feature_scale, n_classes, self.is_deconv, filters)
        self.udecoder2 = UdecoderC(self.feature_scale, n_classes, self.is_deconv, filters)
        self.udecoder3 = Udecoder(self.feature_scale, n_classes, self.is_deconv, filters)
        self.udecoder4 = Udecoder(self.feature_scale, n_classes, self.is_deconv, filters)
        self.udecoder5 = Udecoder(self.feature_scale, n_classes, self.is_deconv, filters)

        # final layer
        self.final = nn.Conv2d(5, n_classes, 1)  # height_tlc, height_mux, height_VV_VH, height_S2

    def forward(self, inputs):
        # encoder 1 & 2 & 3
        conv10, conv11, conv12, conv13, center1 = self.uencoder1(inputs[:, :4, :, :])  # mux
        conv20, conv21, conv22, conv23, center2 = self.uencoder2(inputs[:, 4:7, :, :])  # tlc
        conv30, conv31, conv32, conv33, center3 = self.uencoder3(inputs[:, 7:9, :, :])  # VV+VH
        conv40, conv41, conv42, conv43, center4 = self.uencoder4(inputs[:, 9:17, :, :])  # S2
        conv50, conv51, conv52, conv53, center5 = self.uencoder5(inputs[:, 17:, :, :])  # POI

        # decoder 1 & 2 & 3
        com_center = torch.cat([center2, center1], 1)
        final1 = self.udecoder1(conv10, conv11, conv12, conv13, center1)  # mux height
        final2 = self.udecoder2(conv20, conv21, conv22, conv23, com_center)  # tlc height
        final3 = self.udecoder3(conv30, conv31, conv32, conv33, center3)  # S1 height
        final4 = self.udecoder4(conv40, conv41, conv42, conv43, center4)  # S2 height
        final5 = self.udecoder5(conv50, conv51, conv52, conv53, center5)  # POI height
        final6 = self.final(torch.cat([final1, final2, final3, final4, final5], 1))  # tlc+mux+S1+S2+POI height

        # deep supervision
        if self.training:
            return final1, final2, final3, final4, final5, final6
        else:
            return final6
