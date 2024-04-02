#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:07:34 2024

@author: anthony
"""
from math import inf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from face_identifier.loss import identify_loss


class ModelTrainer:
    """
    Class to train FaceIdentifier model.

    params
    ------
    model : FaceIdentifier,
        the model to train.
    training_set, val_set : Dataset,
        training and validation data sets.
    optimizer : torch.optim.*
    batch_size : int,
        size of mini batch.
    device : "cpu" or "cuda"
    save_path : str,
        path to save the model's state dict.
    log_dir : str,
        log for tensorboard

    """

    def __init__(self, model, training_set, val_set, optimizer,
                 batch_size=10, device='cpu',
                 save_path="face-identifier.pt", log_dir=None):
        self.model = model

        self.training_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False)

        self.n_images_training = training_set.n_images
        self.n_images_val = val_set.n_images
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir)

        self.epochs_trained = 0
        self.best_loss = inf

    def save_model(self):
        "save the model's state dict"
        torch.save(self.model.state_dict(), self.save_path)

    def train(self, epochs=50):
        """
        train the model

        params
        ------
        epochs : int,
            epochs to train.

        """
        # initial
        if not self.epochs_trained:
            self.best_loss = test(
                self.model, tqdm(self.val_loader, 'Valing'),
                device=self.device, n_images=self.n_images_val)
            self.save_model()
            self.writer.add_scalars(
                'Face ID Loss', {'validation': self.best_loss}, self.epochs_trained)

        total_epoches = self.epochs_trained + epochs
        for _ in range(epochs):
            self.epochs_trained += 1

            # training
            training_loss = train(
                self.model,
                tqdm(self.training_loader,
                     f"Training[{self.epochs_trained}/{total_epoches}]"),
                self.optimizer,
                device=self.device, n_images=self.n_images_training)

            # validation
            val_loss = test(
                self.model, tqdm(self.val_loader, desc="Valing"),
                device=self.device, n_images=self.n_images_val)

            # save the best model
            if val_loss <= self.best_loss:
                self.best_loss = val_loss
                self.save_model()

            # logging
            self.writer.add_scalars(
                'Face ID Loss', {
                    'training': training_loss,
                    'validation': val_loss,
                }, self.epochs_trained)


def train(model, data_loader, optimizer, device="cpu", n_images=2):
    """
    training process of face identifier

    params
    ------
    model : FaceIdentifier,
        the model to train.
    data_loader : torch.utils.data.DataLoader,
        data_loader of training set.
    optimizer : torch.optim.*
    device : "cpu" or "cuda"
    n_images : int,
        number of images of each people.

    returns
    -------
    loss : float,
        mean training loss

    """
    model.to(device)
    model.train()
    losses = []
    for inputs in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs.flatten(end_dim=1).to(device))
        loss = identify_loss(outputs, n_images)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss


def test(model, data_loader, device="cpu", n_images=2):
    """
    test process of face identifier

    params
    ------
    model : FaceIdentifier,
        the model to test.
    data_loader : torch.utils.data.DataLoader,
        data_loader of test set.
    device : "cpu" or "cuda"
    n_images : int,
        number of images of each people.

    returns
    -------
    loss : float,
        mean test loss

    """
    model.to(device)
    with torch.no_grad():
        model.eval()
        losses = []
        for inputs in data_loader:
            outputs = model(inputs.flatten(end_dim=1).to(device))
            loss = identify_loss(outputs, n_images)
            losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss
