#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:14:35 2024

@author: anthony

The identifier model.

"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)


class TwinModel(nn.Module):
    """
    Twin model wrapping the face identifier.
    This model is for training or evalation.
    The trainling labels should be N * 0 or 1 (different people, same person)

    """

    def __init__(self, model, threshold=0.8):
        super().__init__()
        self.model = model
        self.threshold = threshold
        self.comp_threshold = 1 - threshold

    def forward(self, x, y):
        """
        forward propagation

        params
        ------
        x, y : tensor,
            images for binary classification.

        """
        x = self.model(x)
        y = self.model(y)

        pred = F.cosine_similarity(x, y)
        pred = F.sigmoid((pred - self.threshold) / self.comp_threshold)
        return pred


class TripletModel(TwinModel):
    """
    Triplet model wrapping the face identifier.
    This model is for training.
    The training labels should always be N * [1, 0].

    """

    def forward(self, x, y, z):
        """
        forward propagation

        params
        ------
        x, y, z : tensor,
            x : base images.
            y : images from the same people as x.
            z : images from different people from x.

        """
        x = self.model(x)
        y = self.model(y)
        z = self.model(z)

        same = F.cosine_similarity(x, y)
        diff = F.cosine_similarity(x, z)

        pred = torch.cat([same.view(-1, 1), diff.view(-1, 1)], dim=1)

        # consider as a classification task for gradient
        pred = F.sigmoid((pred - self.threshold) / self.comp_threshold)
        return pred


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

    def __init__(self, output_dim=1000, pretrain=True, lock_num=0):
        super().__init__()

        # load the pretrain weights
        weights = None
        if pretrain:
            weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=weights)

        # modify the final layer
        self.out_features = output_dim
        final_layer = self.model.classifier[-1]
        self.model.classifier[-1] = nn.Linear(
            in_features=final_layer.in_features,
            out_features=self.out_features,
        )

    def forward(self, x):
        "forward propagation"
        y = self.model(x)

        # test time augmentation
        if not self.training:
            # x.dim : N, C, H, W, flip the last dim is a horizontal flipping
            y2 = self.model(x.flip(-1))
            y += y2

        # normalization
        return F.normalize(y)

    def predict(self, *images):
        """
        return vectors for images tensor.

        """
        img0 = images[0]
        images = torch.cat(images).view(-1, *img0.shape)
        with torch.no_grad():
            self.eval()
            vectors = self.forward(images)
        return vectors
