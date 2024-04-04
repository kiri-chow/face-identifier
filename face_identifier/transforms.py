#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:12:49 2024

@author: anthony
"""
import torch
from torch import nn
from face_identifier.model import FaceDetector
from face_identifier.dataset import TENSOR2IMG, TENSORIZER


class CropFace(nn.Module):
    """
    Crop face area after detecting

    params
    ------
    path : path,
        path of face-detector's state dict

    """

    def __init__(self, path, threshold=0.5, size=256, scalar=1.1):
        super().__init__()
        self.model = FaceDetector.load(path)
        self.threshold = threshold
        self.size = size
        self.scalar = scalar

    def forward(self, *args):
        "crop tensor"
        x = args[0]
        with torch.no_grad():
            # detect face
            self.model.eval()
            prediction = self.model(x.view(1, *x.shape))[0]
            prediction = prediction.cpu().detach().numpy()
            has_face = prediction[0]

            # no face
            if has_face <= self.threshold:
                return torch.zeros_like(x)

            # crop image
            bbox = prediction[1:] * self.size
            x1, y1, x2, y2 = bbox
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            x1, y1 = (x / self.scalar for x in (x1, y1))
            x2, y2 = (x * self.scalar for x in (x2, y2))
            image = TENSOR2IMG(x).crop(
                (x1, y1, x2, y2)).resize((self.size, self.size))
            out = TENSORIZER(image) / 256

        return out
