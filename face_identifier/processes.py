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


def _call_identifier(model, data, device):
    x, y, z = data
    outputs = model(x.to(device), y.to(device), z.to(device))
    labels = torch.zeros_like(outputs)
    labels[:, 0] = 1
    return outputs, labels.float().to(device)


def _call_detector(model, data, device):
    images, has_faces, bboxes = data

    has_faces = has_faces.view(-1, 1)
    bboxes = bboxes.view(-1, 4)
    labels = torch.cat([has_faces, bboxes], dim=1)

    outputs = model(images.to(device))
    return outputs, labels.to(device)


def _convert_shape(images):
    return images.view(-1, *images.shape[-3:])


MODEL_TO_FUNC = {
    'detector': _call_detector,
    'identifier': _call_identifier,
}


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
    model_type : "detector" or "identifier",
        model to train
    log_dir : str,
        log for tensorboard

    """

    def __init__(self, model, training_set, val_set, optimizer,
                 loss_func, batch_size=10, device='cpu',
                 save_path="face-identifier.pt",
                 model_type='identifier', test_func=None,
                 log_dir=None, log_name='Loss'):
        self.model = model

        self.training_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False)

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir)
        self.log_name = log_name
        self.call_func = MODEL_TO_FUNC[model_type]
        self.test_func = test_func

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
                call_func=(
                    self.test_func if self.test_func else self.call_func),
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
                self.optimizer,
                call_func=self.call_func,
                loss_func=self.loss_func,
                device=self.device)

            # validation
            val_loss = test(
                self.model, tqdm(self.val_loader, desc="Valing"),
                call_func=(
                    self.test_func if self.test_func else self.call_func),
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


def train(model, data_loader, optimizer, call_func, loss_func, device="cpu"):
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
    for data in data_loader:
        optimizer.zero_grad()
        outputs, labels = call_func(model, data, device)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss


def test(model, data_loader, call_func, loss_func, device="cpu", return_outputs=False):
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
        for data in data_loader:
            outputs, labels = call_func(model, data, device)
            loss = loss_func(outputs, labels)
            losses.append(loss.item())
    loss = sum(losses) / len(losses)

    if return_outputs:
        return loss, outputs
    return loss
