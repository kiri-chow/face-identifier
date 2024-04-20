#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:22:07 2024

@author: anthony
"""
import random
from itertools import combinations, product, groupby
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from face_identifier.models import TwinModel
from face_identifier.datasets import draw_tensor_image, _BaseImageReader


def evaluate_classification(model, dataset, n_samples=200,
                            threshold=0.8, transform=None,
                            batch_size=64, device='cpu'):
    """
    evaluate the model as a classification task.

    params
    ------
    model : FaceIdentifier,
        The Face re-ID model.
    dataset : IdentityDataset,
        The dataset to test.
    n_samples : int,
        Number of samples to test.
    threshold : float,
        threshold determining Positive or Negative.
    batch_size : int,
        size of mini batch
    device : "cpu" or "cuda"

    """
    # build dataset to classify
    index = random.sample(list(dataset.index.values()), n_samples)
    grouped_data = dict(map(lambda x: (x, dataset.data[x]), index))
    clf_dataset = BinaryDataset(
        grouped_data, dataset.path_images,
        size=dataset.size, transform=dataset.transform,
    )

    # classification
    result = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    keys = {(True, True): 'tp', (False, False): 'tn',
            (False, True): 'fp', (True, False): 'fn'}
    clf_loader = DataLoader(clf_dataset, batch_size=batch_size)
    evl_model = TwinModel(model, threshold)

    with torch.no_grad():
        evl_model.to(device)
        evl_model.eval()
        for imgs1, imgs2, labels in tqdm(clf_loader, 'Testing'):
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)

            if transform:
                imgs1 = transform(imgs1)
                imgs2 = transform(imgs2)

            pred = evl_model(imgs1, imgs2)

            pred = pred.cpu().numpy() > 0.5
            labels = labels.numpy().astype(bool)

            for key, value in groupby(zip(labels, pred)):
                result[keys[key]] += len(list(value))

    result['total'] = sum(result.values())
    result['accuracy'] = (result['tp'] + result['tn']) / result['total']
    result['precision'] = result['tp'] / (result['tp'] + result['fp'])
    result['recall'] = result['tp'] / (result['tp'] + result['fn'])
    result['f1'] = 1 / (1 / result['precision'] + 1 / result['recall']) * 2
    return result


class BinaryDataset(Dataset, _BaseImageReader):
    """
    Dataset for evaluation.

    params
    ------
    grouped_data : dict,
        grouped_data returned from `read_identities`
    path_images : str,
        directory of images
    size : int,
        size of the image.
    transform : callable,
        transform pipeline.

    items
    -----
    img1, img2 : Tensor,
        the tensor of images.
    label : 0 or 1,
        the images are from the same person or not.

    """

    def __init__(self, grouped_data, path_images, size=256, transform=None):
        self.data = _generate_binary_data(grouped_data)
        self.path_images = path_images
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (img1, img2), label = self.data[index]
        img1 = self._transform(self._read_image(img1))
        img2 = self._transform(self._read_image(img2))
        return img1, img2, label

    def draw(self, index):
        "show the images"
        img1, img2, label = self[index]
        fig, axes = plt.subplots(1, 2)
        for img, ax in zip([img1, img2], axes):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('same' if label else "diff")


def visualize(model, dataset, index, transform=None,
              device='cpu'):
    """
    Visualize the result.

    params
    ------
    model : FaceIdentifier,
        The Face re-ID model.
    dataset : IdentityDataset,
        The dataset to test.
    index : int,
        The index of data to show.

    """
    # vectorization
    images = dataset[index]
    images = torch.cat(images).view(-1, *images[0].shape).to(device)
    if transform:
        images = transform(images)

    model.to(device)
    model.eval()
    with torch.no_grad():
        vectors = model(images)
        vector1, vector2, vector3 = vectors.cpu().numpy()

    fig, axes = plt.subplots(1, 3)
    titles = [
        'base',
        f'same {vector1.dot(vector2):.5f}',
        f'diff {vector1.dot(vector3):.5f}'
    ]

    for img, ax, title in zip(images, axes, titles):
        ax.set_title(title)
        draw_tensor_image(img, ax)
        ax.set_xticks([])
        ax.set_yticks([])


def _generate_binary_data(grouped_data):
    """
    generate binary dataset from grouped data.

    returns
    -------
    data : [((img_path1, img_path2), same_person?), ...]

    """
    # images from the same person
    data_same = []
    for person, images in tqdm(grouped_data.items(), 'Preparing'):
        comb = list(combinations(images, 2))
        data_same.extend(comb)

    # images from different people
    data_diff = []
    for images_1, images_2 in tqdm(combinations(grouped_data.values(), 2),
                                   'Preparing'):
        comb = list(product(images_1, images_2))
        data_diff.extend(comb)

    # cut lists
    random.shuffle(data_same)
    random.shuffle(data_diff)
    length = min(len(data_same), len(data_diff))
    data_same = data_same[:length]
    data_diff = data_diff[:length]

    # merge as dataset
    to_return = (list(map(lambda x: (x, 1), data_same)) +
                 list(map(lambda x: (x, 0), data_diff)))
    return to_return
