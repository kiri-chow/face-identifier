#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:12:49 2024

@author: anthony
"""
import torch
from torch import nn
from torchvision.transforms import v2
from .models import FaceDetector


class CropFace(nn.Module):
    """
    Crop face area by detecting face

    params
    ------
    model : nn.Module,
        the face-detector
    size : int,
        size of output image
    scalar : float,
        scaling detected bbox

    """

    def __init__(self, model, in_size=256, out_size=256, scale=1.5, device='cpu'):
        super().__init__()
        self.model = model
        self.in_size = in_size
        self.out_size = out_size
        self.scale = scale

        self.in_resizer = v2.Resize((self.in_size, self.in_size))
        self.out_resizer = v2.Resize((self.out_size, self.out_size))

        self.device = device

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
        self.to(self.device)
        x_new = self.in_resizer(imgs)
        has_face, bboxes = self.get_bboxes(x_new)

        # crop
        n = imgs.size(0)
        imgs_new = torch.cat(
            [self._crop_one_image(img, hf, bbox)
             for img, hf, bbox in zip(imgs, has_face, bboxes)
             ]).view(n, -1, self.out_size, self.out_size)
        return imgs_new

    def _crop_one_image(self, img, has_face, bbox):
        if has_face == 1:
            height = img.size(-2)
            width = img.size(-1)
            scaler = torch.Tensor([width, height]).to(self.device)

            # scalr the bbox
            (xmin, xmax), (ymin, ymax) = (
                bbox.view(2, 2) * scaler).transpose(0, 1).long()
            img = img[..., ymin: ymax, xmin: xmax]

            return self.out_resizer(img)
        return torch.zeros((3, self.out_size, self.out_size), dtype=img.dtype).to(self.device)

    def get_bboxes(self, x):
        """
        return face bboxes from tensors

        returns
        -------
        has_face : n * 0 or 1
        bboxes : n * [x1, y1, x2, y2],

        """
        with torch.no_grad():
            # detect landmarks
            self.model.eval()
            x = self._convert_x_shape(x)
            predictions = self.model(x)

            # compute bboxes
            has_face, bboxes = self._compute_bboxes(predictions)
            return has_face, bboxes

    def _convert_x_shape(self, x):
        x_shape = x.shape
        x_dims = len(x_shape)
        if x_dims < 4:
            new_shape = [-1, 1, self.in_size, self.in_size]
            new_shape[-x_dims:] = x_shape
            x = x.view(*new_shape)
        return x

    def _compute_bboxes(self, predictions):
        has_face = predictions[:, 0]
        bboxes = predictions[:, 1:].view(-1, 2, 2)

        # sort as xmin, ymin, xmax, ymax
        bboxes = bboxes.sort(-2)[0]
        centers = bboxes.mean(-2).view(-1, 1, 2)

        # scale the bboxes
        vectors = bboxes - centers
        bboxes = (vectors * self.scale) + centers

        # limit
        bboxes[:, 0][bboxes[:, 0] < 0] = 0
        bboxes[:, 1][bboxes[:, 1] > 1] = 1

        bboxes = bboxes.view(-1, 4)
        return has_face, bboxes


class CropFaceByLandmarks(CropFace):
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
        scaler = torch.Tensor([width, height]).to(self.device)

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
        vectors = (landmarks[:, [0, 1, 3, 4], :] - noses) * self.scale
        noses_offset = -vectors.mean(1) / 5

        new_noses = noses.view(n, 2) + noses_offset
        anchors = torch.cat([(vectors + noses).view(n, -1) , new_noses], dim=1)
        anchors = anchors.view(n, 5, 2)

        xymin = anchors.min(1)[0].view(n, 2)
        xymin[xymin < 0] = 0
        xymax = anchors.max(1)[0].view(n, 2)

        bboxes = torch.cat([xymin, xymax], dim=1)
        return bboxes
