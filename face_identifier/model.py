#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:14:35 2024

@author: anthony

The identifier model.

"""
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


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

        # adjust the final layers
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        "forward propagation"
        x = self.model(x)
        x = self.linear1(F.relu(x))
        x = self.linear2(F.relu(x))

        # normalization
        x = F.normalize(x)
        return x
