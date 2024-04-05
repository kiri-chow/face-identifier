#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:12:49 2024

@author: anthony
"""
import os
import torch
from PIL import Image
import numpy as np
from torch import nn
from face_identifier.models import FaceDetector
from face_identifier.datasets import TENSOR2IMG, TENSORIZER, read_image


class CropFace(nn.Module):
    """
    Crop face area after detecting

    params
    ------
    path : path,
        path of face-detector's state dict.
    threshold : [0, 1),
        threshold determining the image has face or not.
    size : int,
        size of output image
    scalar : float,
        scaling detected bbox

    """

    def __init__(self, path, threshold=0.5, size=256, scalar=1.1):
        super().__init__()
        self.model = FaceDetector.load(path)
        self.threshold = threshold
        self.size = size
        self.scalar = scalar

    def crop_images(self, paths, save_dir="cropped"):
        "crop and save images"
        for path in paths:
            image = Image.open(path)
            bbox, has_face = self.get_bbox(read_image(path, self.size))
            if not has_face:
                continue

            # crop
            bbox = (np.array(bbox).reshape(2, 2) * image.size).flatten()
            image_cropped = image.crop(bbox)

            # save
            _, fp = os.path.split(path)
            os.makedirs(os.path.join(save_dir), exist_ok=True)
            image_cropped.save(os.path.join(save_dir, fp))

    def get_bbox(self, x):
        """
        return face bbox from tensor

        returns
        -------
        bbox : [x1, y1, x2, y2],
        has_face : bool

        """
        with torch.no_grad():
            # detect face
            self.model.eval()
            prediction = self.model(x.view(1, *x.shape))[0]
            prediction = prediction.cpu().detach().numpy()
            has_face = prediction[0]

            # no face
            if has_face <= self.threshold:
                return [0, 0, 0, 0], False

            # crop image
            bbox = prediction[1:]
            x1, y1, x2, y2 = bbox
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            x1, y1 = (x / self.scalar for x in (x1, y1))
            x2, y2 = (x * self.scalar for x in (x2, y2))
            return [x1, y1, x2, y2], True

    def forward(self, *args):
        "crop tensor"
        x = args[0]
        bbox, has_face = self.get_bbox(x)
        if not has_face:
            return torch.zeros_like(x)

        bbox = [b * self.size for b in bbox]
        image = TENSOR2IMG(x).crop(bbox).resize((self.size, self.size))
        out = TENSORIZER(image) / 256
        return out
