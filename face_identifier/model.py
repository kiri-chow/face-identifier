#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:14:35 2024

@author: anthony

The identifier model.

"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)


class FaceIdentifier(nn.Module):
    """
    Face Identifier Model

    params
    ------
    output_dim : int,
        The number of output dimensions (size of face ID)
    pretrain : bool,
        Load the pretrain weight or not.
    lock_num : int,
        Number of convolution layers to lock.

    """
    @classmethod
    def load(cls, path, **kwargs):
        """
        load model from path
    
        """
        model = cls(pretrain=False, **kwargs)
        model.load_state_dict(torch.load(path))
        return model

    def __init__(self, output_dim=500, pretrain=True, lock_num=0):
        super().__init__()

        weights = None
        if pretrain:
            weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)

        # lock the convolution layers
        for i in range(1, lock_num + 1):
            layer = getattr(self.model, f'layer{i}')
            for parameters in layer.parameters():
                parameters.requires_grad = False

        # add the final layer
        self.out_features = output_dim
        self.fc = nn.Sequential(OrderedDict([
            ("relu1", nn.ReLU()),
            ("ln2", nn.Linear(self.model.fc.out_features, output_dim)),
        ]))

    def forward(self, x):
        "forward propagation"
        x = self.model(x)
        x = self.fc(x)

        # normalization
        x = F.normalize(x)
        return x


class FaceDetector(nn.Module):
    """
    detect bbox of face

    params
    ------
    pretrain : bool,
        Load the pretrain weight or not.

    """
    @classmethod
    def load(cls, path, **kwargs):
        """
        load model from path
    
        """
        model = cls(pretrain=False, **kwargs)
        model.load_state_dict(torch.load(path))
        return model

    def __init__(self, pretrain=True):
        super().__init__()

        weights = None
        if pretrain:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)

        last_output = self.model.classifier[-1].out_features
        self.has_face = nn.Sequential(nn.Linear(last_output, 1), nn.Sigmoid())
        self.bbox = nn.Sequential(nn.ReLU(), nn.Linear(last_output, 4))

    def forward(self, x):
        "forward propagation"
        x = self.model(x)
        has_face = self.has_face(x)
        bbox = self.bbox(x) * has_face
        to_return = torch.cat([has_face, bbox], dim=1)
        return to_return
