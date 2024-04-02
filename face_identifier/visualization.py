#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:22:07 2024

@author: anthony
"""
import random
import torch
from matplotlib import pyplot as plt
from face_identifier.dataset import draw_tensor_image


def visualize_test(model, test_set):
    """
    Visualize the test result

    """
    # get samples
    length = len(test_set)
    samples = random.sample(range(length), 4)
    images = list(test_set[samples[0]])
    for ind in samples[1:]:
        images.append(test_set[ind][0])

    # compute face id
    with torch.no_grad():
        model.to('cpu')
        model.eval()
        inputs = torch.concat(images).view(5, 3, test_set.size, test_set.size)
        outputs = model(inputs)
        base = outputs[0].view(1, -1)
        dots = base.matmul(outputs[1:].T).numpy().flatten()

    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(8, 2)
    fig.suptitle("Cosine Similarities")
    draw_tensor_image(images[0], axes[0])
    axes[0].set_title("Base")

    for img, dot, ax in zip(images[1:], dots, axes[1:]):
        ax.set_title(f'{dot:.5f}')
        draw_tensor_image(img, ax)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
