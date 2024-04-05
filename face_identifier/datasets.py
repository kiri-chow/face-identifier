#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:10:31 2024

@author: anthony

Functions and components to read data needed. 

"""
import re
import os
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import tv_tensors


_REG_SPLIT = re.compile(r'\s+')
_REG_SLASH = re.compile(r'[\\/]+')
TENSORIZER = transforms.PILToTensor()
TENSOR2IMG = transforms.ToPILImage()


def read_identities(path, path_images, n_images=2):
    """
    read the CelebA Identity annotations and group the images by people.

    params
    ------
    path : str,
        path of identity annotations of CelebA dataset.
    path_images : str,
        path of the images.
    n_images : int,
        number of images of each people.

    returns
    -------
    data : dict,
        format in person_id(int): images(list of image path)

    """
    with open(path, 'r') as f:
        data = f.readlines()
    images, identites = zip(*filter(
        lambda x: os.path.exists(os.path.join(path_images, x[0])),
        map(_split_line_id, data)))
    images = np.array(images, object)
    identites = np.array(identites, int)

    grouped_data = {}
    for person in np.unique(identites):
        index = identites == person
        if index.sum() < n_images:
            continue
        grouped_data[person] = images[index].tolist()
    return grouped_data


def split_train_valid_test(data, ratios=(0.7, 0.2, 0.1)):
    """
    split the dataset into training, validation, and test sets.

    params
    ------
    data : dict,
        data returned from `read_identities`
    ratios : [float, float, float],
        ratios of training, validation, and test sets.

    returns
    -------
    training_index, validation_index, test_index : list,
        identities of 3 sets.

    """
    assert len(ratios) == 3

    index = list(data)
    random.shuffle(index)
    length = len(index)

    cut_1, cut_2 = [int(length * r / sum(ratios)) for r in ratios[:2]]
    cut_2 += cut_1

    training_index = index[:cut_1]
    validation_index = index[cut_1: cut_2]
    test_index = index[cut_2:]
    return training_index, validation_index, test_index


def read_image(path, size=256):
    "return image tensor of path"
    image = Image.open(path)
    if size:
        image = image.resize((size, size))
    image = TENSORIZER(image) / 256
    return image


class _BaseImageReader:

    def _read_image(self, path):
        path = os.path.join(self.path_images, path)
        return read_image(path, self.size)

    def _transform(self, data):
        if self.transform:
            return self.transform(data)
        return data


class IdentityDataset(Dataset, _BaseImageReader):
    """
    Data Reader for grouped CelebA Identity Data

    params
    ------
    grouped_data : dict,
        grouped_data returned from `read_identities`
    path_images : str,
        directory of images
    index : list or None,
        identities of samples. None means using all samples in data.
    size : int,
        size of the image.
    transform : callable,
        transform pipeline.

    items
    -----
    img1, img2, img3 : Tensor,
        the tensor of images.
        img1 is the base.
        img2 is from the same person.
        img3 is from another person.

    """

    def __init__(self, grouped_data, path_images, index=None, size=256,
                 transform=None):
        self.data = grouped_data
        if index is None:
            self.index = dict(enumerate(grouped_data))
        else:
            self.index = dict(enumerate(index))
        self.path_images = path_images
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        # image from the same person
        index = self.index[index]
        img_1, img_2 = random.sample(self.data[index], 2)
        img_1 = self._transform(self._read_image(img_1))
        img_2 = self._transform(self._read_image(img_2))

        # image from different people
        oth_index = index
        while oth_index == index:
            oth_index = random.choice(self.index)
        img_3 = random.choice(self.data[oth_index])
        img_3 = self._transform(self._read_image(img_3))

        return img_1, img_2, img_3

    def draw(self, index):
        "visualize data"
        img_1, img_2, img_3 = self[index]
        fig, (axes) = plt.subplots(1, 3)
        titles = ['base', 'same', 'diff']
        for ax, img, t in zip(axes, [img_1, img_2, img_3], titles):
            draw_tensor_image(img, ax)
            ax.set_title(t)
            ax.set_xticks([])
            ax.set_yticks([])
        return ax


class FaceDataset(Dataset, _BaseImageReader):
    """
    Data Reader for face bbox detection.
    every line of the annotation should be as the format below:
        image_path has_face(0 or 1) x1(int) y1(int) x2(int) y2(int)

    params
    ------
    path_anno : path,
        path of annotation.
    path_images : str,
        directory of images.
    size : int,
        width and height of the output image.
    transform : callable,
        transform pipeline.

    items
    -----
    image : Tensor,
        the tensor of image.
    has_face : 0 or 1,
        whether the image includes a face.
    bbox : tv_tensors.BoundingBoxes,
        bounding box that can be transformed.

    """

    def __init__(self, path_anno, path_images, size=256, transform=None):
        self.path_images = path_images
        self.size = size
        self.__read_data(path_anno)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __read_data(self, path):
        with open(path) as f:
            data = f.read().split('\n')
        self.data = list(map(self.__read_line, data))

    def __read_line(self, line):
        line = _REG_SPLIT.split(line)
        path_img = line[0]
        path_img = os.path.join(*_REG_SLASH.split(path_img))
        has_face = float(line[1])
        bbox = list(map(float, line[2:6]))
        label = (has_face, bbox)
        return path_img, label

    def __getitem__(self, index):
        # read data
        line = self.data[index]
        path_img, (has_face, bbox) = line

        # read image
        image = Image.open(os.path.join(self.path_images, path_img))

        # read bbox
        bbox = self.__convet_bbox(image, bbox)

        image = TENSORIZER(image.resize((self.size, self.size))) / 256
        if image.shape[0] == 1:
            image = torch.cat([image] * 3)

        # construct data
        has_face = torch.Tensor([has_face])[0]
        bbox = tv_tensors.BoundingBoxes(
            [bbox], format='XYXY', canvas_size=[1, 1])
        data = (image, has_face, bbox)

        data = self._transform(data)

        return data

    def draw(self, index, prediction=None):
        "visualize data"
        image, has_face, bbox = self[index]
        ax = draw_tensor_image(image)

        if has_face:
            x1, y1, x2, y2 = bbox.numpy()[0] * self.size
            ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])

        if prediction is not None:
            prediction = prediction.cpu().detach().numpy()
            has_face = prediction[0]
            if has_face > 0.5:
                # x1, y1, x2, y2 = prediction[1:] / has_face * self.size
                x1, y1, x2, y2 = prediction[1:] * self.size
                ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])
                ax.set_xticks([])
                ax.set_yticks([])
        return ax

    def __convet_bbox(self, image, bbox):
        x1, y1, x2, y2 = bbox
        # x1, x2, y1, y2 = bbox
        width, height = image.size
        x1 /= width
        x2 /= width
        y1 /= height
        y2 /= height
        return [x1, y1, x2, y2]


def draw_tensor_image(tensor, ax=None):
    "Show tensor as an image"
    if ax is None:
        _, ax = plt.subplots()
    image = TENSOR2IMG(tensor)
    ax.imshow(image, vmin=0, vmax=1)
    return ax


def _split_line_id(line):
    image, identity = _REG_SPLIT.split(line)[:2]
    return image.strip(), int(identity.strip())


def _flatten_key_value(x):
    key, value = x
    return [(key, v) for v in value]
