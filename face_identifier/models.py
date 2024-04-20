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
    tta : bool,
        Use test time augmentation or not

    """
    @classmethod
    def load(cls, path, **kwargs):
        """
        load model from path

        """
        model = cls(pretrain=False, **kwargs)
        model.load_state_dict(torch.load(path))
        return model

    def __init__(self, output_dim=500, pretrain=True, lock_num=0, tta=False):
        super().__init__()
        self.tta = tta

        # load the pretrain weights
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
        y = self._base_forward(x)

        # test time augmentation
        if not self.training and self.tta:
            # x.dim : N, C, H, W, flip the last dim is a horizontal flipping
            y2 = self._base_forward(x.flip(-1))
            y += y2

        # normalization
        return F.normalize(y)

    def _base_forward(self, x):
        y = self.model(x)
        y = self.fc(y)
        return y

    def predict(self, *args):
        img0 = args[0]
        x = torch.cat(args).view(-1, *img0.shape)
        with torch.no_grad():
            self.eval()
            return self(x)


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
        bbox = self.bbox(x)

        # mask bbox
        bbox[(has_face <= 0.5).flatten()] = 0
        to_return = torch.cat([has_face, bbox], dim=1)
        return to_return
