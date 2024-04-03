#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:35:17 2024

@author: anthony

"""
from itertools import combinations as get_comb
import torch
from torch import nn


class IdentifyLoss(nn.Module):
    """
    Compute the loss of face identifier.

    `loss = similarity_oth / similarity_self`
    Minimizing loss equals to 
    minimizing similarity_oth and maximizing similarity_self.

    params
    ------
    n_images : int,
        number of images of each people.

    """
    
    def __init__(self, n_images=2):
        super().__init__()
        self.n_images = n_images

    def forward(self, x, y=None):
        "compute the loss"
        # combinations
        length = len(x)
        combinations = set(get_comb(range(length), 2))
    
        # cosine similarities
        dots = {(i1, i2): x[i1].dot(x[i2])
                for i1, i2 in combinations}
    
        # compute losses
        losses = []
        for person in range(length // self.n_images):
            # pictures from the same person
            index_self = set(get_comb(range(person, person + self.n_images), 2))
            similarity_self = min((dots[ind] for ind in index_self))
    
            # pictures from other people
            index_oth = combinations - index_self
            similarity_oth = max((dots[ind] for ind in index_oth))
    
            # loss to minimize
            loss = similarity_oth / similarity_self
            losses.append(loss)
    
        losses = torch.Tensor(losses)
        losses.requires_grad = True
        return losses.mean()
