#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:12:49 2024

@author: anthony
"""
from collections import OrderedDict
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.models import mobilenet_v2


class LandmarksDetector(nn.Module):
    """
    The face landmarks detector

    """
    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model.load_state_dict(torch.load(path))
        return obj

    def __init__(self):
        super().__init__()
        model = mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 10)
        self.model = model

    def forward(self, x):
        # normalization
        # means = x.mean([0, 2, 3])
        # stds = x.std([0, 2, 3])
        # norm = v2.Normalize(means, stds)
        norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # x = norm(x)
        return self.model(x)


class CropFace(nn.Module):
    """
    Crop face area by detecting landmarks

    params
    ------
    model : nn.Module,
        the landmark model return 5 points for eyes, nose, and mouth
    size : int,
        size of output image
    scalar : float,
        scaling detected bbox

    """

    def __init__(self, model, in_size=224, out_size=256, scalar=2):
        super().__init__()
        self.model = model
        self.in_size = in_size
        self.out_size = out_size
        self.scalar = scalar

        self.in_resizer = v2.Resize((self.in_size, self.in_size))
        self.out_resizer = v2.Resize((self.out_size, self.out_size))

    def forward(self, imgs):
        """
        crop the tensors

        params
        ------
        imgs : torch.Tensor,
            dims should be NCHW for batch images
                or CHW for a single image,
                or HW for a grayscale image.

        returns
        -------
        tensors : torch.Tensor,
            images with faces, dims NCHW

        """
        x_new = self.in_resizer(imgs)
        bboxes = self.get_bboxes(x_new)

        # crop
        n = imgs.size(0)
        imgs_new = torch.cat([self._crop_one_image(img, bbox)
                              for (img, bbox) in zip(imgs, bboxes)]
                             ).view(n, -1, self.out_size, self.out_size)
        return imgs_new

    def _crop_one_image(self, img, bbox):
        height = img.size(-2)
        width = img.size(-1)
        scaler = torch.Tensor([width, height])

        # scalr the bbox
        (xmin, xmax), (ymin, ymax) = (
            bbox.view(2, 2) * scaler).transpose(0, 1).long()
        img = img[..., ymin: ymax, xmin: xmax]
        return self.out_resizer(img)

    def get_bboxes(self, x):
        """
        return face bboxes from tensors

        returns
        -------
        bboxes : n * [x1, y1, x2, y2],

        """
        with torch.no_grad():
            # detect landmarks
            self.model.eval()
            x = self._convert_x_shape(x)
            landmarks = self.model(x)

            # compute bboxes
            bboxes = self._compute_bboxes(landmarks)
            return bboxes

    def _convert_x_shape(self, x):
        x_shape = x.shape
        x_dims = len(x_shape)
        if x_dims < 4:
            new_shape = [-1, 1, self.in_size, self.in_size]
            new_shape[-x_dims:] = x_shape
            x = x.view(*new_shape)
        return x

    def _compute_bboxes(self, landmarks):
        n = landmarks.size(0)
        landmarks = landmarks.view(n, 5, 2)
        noses = landmarks[:, [2], :]

        # scale the cectors
        vectors = (landmarks[:, [0, 1, 3, 4], :] - noses) * self.scalar
        noses_offset = -vectors.mean(1) / 5

        new_noses = noses.view(n, 2) + noses_offset
        anchors = torch.cat([(vectors + noses).view(n, -1) , new_noses], dim=1)
        anchors = anchors.view(n, 5, 2)

        xymin = anchors.min(1)[0].view(n, 2)
        xymax = anchors.max(1)[0].view(n, 2)
        bboxes = torch.cat([xymin, xymax], dim=1)
        return bboxes
