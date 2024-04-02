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
    test_index = index[cut_2: ]
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

    """

    def __init__(self, data, path_images, index=None, size=256, n_images=2):
        self.data = data
        if index is None:
            self.index = dict(enumerate(data))
        else:
            self.index = dict(enumerate(index))
        self.path_images = path_images
        self.size = size
        self.n_images = n_images
        self.tensorizer = transforms.PILToTensor()

    def __len__(self):
        return len(self.index)

    def __read_and_convert(self, path):
        path = os.path.join(self.path_images, path)
        image = Image.open(path)
        image = image.resize((self.size, self.size))
        image = self.tensorizer(image) / 256
        return image

    def __getitem__(self, index):
        # return 2 pictures of the given person
        paths = random.sample(self.data[self.index[index]], 2)
        images = torch.concat([self.__read_and_convert(pth) for pth in paths])
        images = images .view(2, -1, self.size, self.size)
        return images


def draw_tensor_image(tensor, ax=None):
    "Show tensor as an image"
    if ax is None:
        _, ax = plt.subplots()
    image = transforms.ToPILImage()(tensor)
    ax.imshow(image, vmin=0, vmax=1)
    return ax


def _split_line_id(line):
    image, identity = _REG_SPLIT.split(line)[:2]
    return image.strip(), int(identity.strip())
