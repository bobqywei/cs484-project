from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from layers import *


class ResNetMultiImageInput(models.ResNet):

    def __init__(self, block, layers, num_classes, num_input_images):
        
        super(ResNetMultiImageInput, self).__init__(block, layers)
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images,
            64,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
            
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm2d):
            
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PoseDecoder(nn.Module):

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for, stride):
    
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
    
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))


    def forward(self, input_features):
    
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        num_input_images=6,
        num_ch_enc,
        num_input_features,
        num_frames_to_predict_for=None,
        stride=1
    ):
        
        self.model = nn.Sequential(
            ResNetMultiImageInput(
                block,
                layers,
                num_classes,
                num_input_images
            ),
            PoseDecoder(
                num_ch_enc,
                num_input_features,
                num_frames_to_predict_for,
                stride
            )
        )
    

    def forward(self, input_image):

        return self.model(input_image)
