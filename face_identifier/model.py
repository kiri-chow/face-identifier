#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:14:35 2024

@author: anthony

The identifier model.

"""
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, ResNet34_Weights


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
    def __init__(self, output_dim=500, pretrain=True, lock_num=3):
        super().__init__()
        weights = None
        if pretrain:
            weights = ResNet34_Weights.DEFAULT
        self.model = resnet34(weights=weights)

        # lock the convolution layers
        for i in range(1, lock_num + 1):
            layer = getattr(self.model, f'layer{i}')
            for parameters in layer.parameters():
                parameters.requires_grad = False

        # adjust the final layers
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        "forward propagation"
        x = self.model(x)

        # normalization
        x = F.normalize(x)
        return x
