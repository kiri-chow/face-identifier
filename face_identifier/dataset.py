#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:10:31 2024

@author: anthony

Functions and components to read identity annotations of CelebA dataset. 

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


_REG_SPLIT = re.compile(r'\s+')
_REG_SLASH = re.compile(r'[\\/]+')
TENSORIZER = transforms.PILToTensor()
TENSOR2IMG = transforms.ToPILImage()


def read_identities(path, n_images=2):
    """
    group the images by peopleval_loader

    params
    ------
    path : str,
        path of identity annotations of CelebA dataset
    n_images : int,
        number of images of each people.

    returns
    -------
    data : dict,
        format in person_id(int): images(list of image path)

    """
    with open(path, 'r') as f:
        data = f.readlines()
    images, identites = zip(*map(_split_line_id, data))
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

    index = list(data.keys())
    random.shuffle(index)
    length = len(index)

    cut_1, cut_2 = [int(length * r / sum(ratios)) for r in ratios[:2]]
    cut_2 += cut_1

    training_index = index[:cut_1]
    validation_index = index[cut_1: cut_2]
    test_index = index[cut_2:]
    return training_index, validation_index, test_index


class IdentityDataSet(Dataset):
    """
    Data Reader for CelebA Identity Data

    params
    ------
    data : dict,
        data returned from `read_identities`
    path_images : str,
        directory of images
    index : list or None,
        identities of samples. None means using all samples in data.
    size : int,
        size of the image.
    n_images : int,
        number of images of each people.
    transform : transform pipeline

    """

    def __init__(self, data, path_images, index=None, size=256, n_images=2,
                 transform=None):
        self.data = data
        if index is None:
            self.index = dict(enumerate(data))
        else:
            self.index = dict(enumerate(index))
        self.path_images = path_images
        self.size = size
        self.n_images = n_images
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __read_and_convert(self, path):
        path = os.path.join(self.path_images, path)
        image = Image.open(path)
        image = image.resize((self.size, self.size))
        image = TENSORIZER(image) / 256
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        # return 2 pictures of the given person
        paths = random.sample(self.data[self.index[index]], 2)
        images = torch.concat([self.__read_and_convert(pth) for pth in paths])
        images = images.view(2, -1, self.size, self.size)
        return images, 0

    def draw(self, index):
        "visualize data"
        images = self[index]
        fig, axes = plt.subplots(1, self.n_images)
        for ax, img in zip(axes, images):
            draw_tensor_image(img, ax)
        return ax


class FaceDataset(Dataset):
    """
    Dataset for face detection

    params
    ------
    path_data : path,
        path of data
    path_images : str,
        directory of images

    """

    def __init__(self, path_data, path_images, size=256, transform=None):
        self.path_images = path_images
        self.size = size
        self.__read_data(path_data)
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
        x1, y1, x2, y2 = self.__convet_bbox(image, bbox)

        image = TENSORIZER(image.resize((self.size, self.size))) / 256
        if image.shape[0] == 1:
            image = torch.cat([image] * 3)
        label = torch.Tensor([has_face, x1, y1, x2, y2])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def draw(self, index, prediction=None):
        "visualize data"
        image, label = self[index]
        ax = draw_tensor_image(image)

        has_face = label[0].item()
        if has_face:
            x1, y1, x2, y2 = label[1:].numpy() * self.size
            ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])

        if prediction is not None:
            prediction = prediction.cpu().detach().numpy()
            has_face = prediction[0]
            if has_face > 0.5:
                x1, y1, x2, y2 = prediction[1:] / has_face * self.size
                ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])
        return ax

    def __convet_bbox(self, image, bbox):
        # x1, y1, x2, y2 = bbox
        y1, x1, y2, x2 = bbox
        # y1, y2, x1, x2 = bbox
        # x1, x2, y1, y2 = bbox
        width, height = image.size
        x1 /= width
        x2 /= width
        y1 /= height
        y2 /= height
        return [x1, x2, y1, y2]


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
