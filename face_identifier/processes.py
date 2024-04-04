#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:07:34 2024

@author: anthony
"""
from math import inf
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from face_identifier.loss import IdentifyLoss


LOSS_FUNC = IdentifyLoss(2)


class ClassificationModel(nn.Module):
    "Classfication model wrapping the face identifier"

    def __init__(self, model, out_features):
        self.model = model
        self.clf = nn.Sequential(
            nn.Linear(self.model.out_features, out_features),
            nn.Softmax(),
        )

    def forward(self, x):
        "forward propagation"
        x = self.model(x)
        x = self.clf(x)
        return x


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
                 batch_size=10, loss_func=None, device='cpu',
                 save_path="face-identifier.pt",
                 log_dir=None, log_name='Loss'):
        self.model = model

        self.training_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False)

        if loss_func:
            self.loss_func = loss_func
        else:
            self.loss_func = LOSS_FUNC

        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir)
        self.log_name = log_name

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
                loss_func=self.loss_func,
                device=self.device)
            self.save_model()
            self.writer.add_scalars(
                self.log_name, {'validation': self.best_loss}, self.epochs_trained)

        total_epoches = self.epochs_trained + epochs
        for _ in range(epochs):
            self.epochs_trained += 1

            # training
            training_loss = train(
                self.model,
                tqdm(self.training_loader,
                     f"Training[{self.epochs_trained}/{total_epoches}]"),
                self.optimizer, loss_func=self.loss_func,
                device=self.device)

            # validation
            val_loss = test(
                self.model, tqdm(self.val_loader, desc="Valing"),
                loss_func=self.loss_func,
                device=self.device)

            # save the best model
            if val_loss <= self.best_loss:
                self.best_loss = val_loss
                self.save_model()

            # logging
            self.writer.add_scalars(
                self.log_name, {
                    'training': training_loss,
                    'validation': val_loss,
                }, self.epochs_trained)


def train(model, data_loader, optimizer, loss_func, device="cpu"):
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

    returns
    -------
    loss : float,
        mean training loss

    """
    model.to(device)
    model.train()
    losses = []
    for images, has_faces, bboxes in data_loader:
        images = _convert_shape(images).to(device)
        labels = _merge_label(has_faces, bboxes).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss


def test(model, data_loader, loss_func, device="cpu", return_outputs=False):
    """
    test process of face identifier

    params
    ------
    model : FaceIdentifier,
        the model to test.
    data_loader : torch.utils.data.DataLoader,
        data_loader of test set.
    device : "cpu" or "cuda"
    return_outputs : bool,
        return the outputs from model.

    returns
    -------
    loss : float,
        mean test loss
    outputs? : Tensor.
        model's outputs

    """
    model.to(device)
    with torch.no_grad():
        model.eval()
        losses = []
        for images, has_faces, bboxes in data_loader:
            images = _convert_shape(images).to(device)
            labels = _merge_label(has_faces, bboxes).to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)
            losses.append(loss.item())
    loss = sum(losses) / len(losses)

    if return_outputs:
        return loss, outputs
    return loss


def _convert_shape(images):
    return images.view(-1, *images.shape[-3:])


def _merge_label(has_faces, bboxes):
    # identifier case
    if has_faces.shape == bboxes.shape:
        return torch.Tensor([0])[0]

    # detector case
    has_faces = has_faces.view(-1, 1)
    bboxes = bboxes.view(-1, 4)
    return torch.cat([has_faces, bboxes], dim=1)
