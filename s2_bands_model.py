import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
from IPython.display import clear_output

from dataloader.augmentations import MixUp

import torch
from torch import nn, optim, Tensor
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101 


class S2BandModel(nn.Module):
    """
    Model for yield prediction based of S2 Santinel bands data.
    """
    def __init__(self, bands_list: list, cfg: Dict):
        super().__init__()
        
        n_bands = 12 // cfg['data_loader']['s2_avg_by'] * len(bands_list) 
        n_indexes = 12 // cfg['data_loader']['s2_avg_by'] * len(cfg['data_loader']['indexes']) 
        num_in_channels = n_bands + n_indexes

        if cfg['s2_model_params']['backbone'] in ['resnet18', 'resnet34']:
            self.backbone = resnet18(pretrained=True)
            backbone_out_dim = 512
        elif cfg['s2_model_params']['backbone'] in ['resnet50', 'resnet101']:
            self.backbone = resnet50(pretrained=True)
            backbone_out_dim = 2048
        else:
            raise NotImplementedError('Such backbone model is not defined.')

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False
        )

        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, cfg['s2_model_params']['n_head']),
            nn.ReLU(),
            nn.Linear(cfg['s2_model_params']['n_head'], 1)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        logits = self.head(x)

        return logits


def forward(model, data, criterion, device):
    s2_bands = data['s2_bands'].to(device)
    yields = data['yield'].to(device)
    prediction = model(s2_bands).reshape(yields.shape)
    loss = criterion(yields, prediction)
    return loss, prediction


def evaluate(model, dataloader, device, criterion):
    losses = []
    model.eval()
    with torch.no_grad():
      val_iter = iter(dataloader)
      for i in range(len(dataloader)):
          data = next(val_iter)
          loss, _ = forward(model, data, criterion, device)
          losses.append(loss.item())
    return np.mean(losses)


def train_epoch(model, dataloader, device, optimizer, criterion, cfg):
    p_mixup = cfg['augmentations']['p_mixup']
    losses = []
    progress_bar = tqdm(range(len(dataloader)))
    dataloader_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(dataloader_iter)

        if p_mixup > 0 and np.random.rand() < p_mixup:
          data = MixUp(cfg)(data)

        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward(model, data, criterion, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_description(f'loss: {loss.item()}, avg loss: {np.mean(losses)}')

    return losses


def train_model(model, train_dataloader, val_dataloader, device, optimizer, criterion, cfg):
    losses_train = []
    losses_val = []
    n_epochs = cfg['s2_train_params']['n_epochs']

    progress_bar = tqdm(range(n_epochs))
    for _ in progress_bar:
        epoch_loss_train = train_epoch(model, train_dataloader, device, optimizer, criterion, cfg)
        epoch_loss_val = evaluate(model, val_dataloader, device, criterion)

        train_loss = np.mean(epoch_loss_train)
        losses_train.append(train_loss)
        losses_val.append(epoch_loss_val)
        progress_bar.set_description(f'train loss: {train_loss:.4f}, val loss: {epoch_loss_val:.4f}')
        clear_output(True)
        if cfg['s2_train_params']['plot_mode']:
            train_monitor(losses_train, losses_val)


def train_monitor(losses_train, losses_val):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    iters = np.arange(len(losses_train))
    ax[0].plot(iters, losses_train, linewidth=2, alpha=0.8,
               c='tab:blue', label='train loss')
    ax[0].plot(iters, losses_val, linewidth=2, alpha=0.8,
               c='tab:red', label='validation loss')

    ax[1].plot(iters, losses_train, linewidth=2, alpha=0.8,
               c='tab:blue', label='train loss')
    ax[1].plot(iters, losses_val, linewidth=2, alpha=0.8,
               c='tab:red', label='validation loss')
    ax[1].set_yscale('log')

    for i in [0, 1]:
        ax[i].set_ylabel('MSE loss')
        ax[i].set_xlabel('Iteration')
        ax[i].legend()
        ax[i].grid()

    plt.show()