from typing import Dict

import torch
from torch import nn, optim, Tensor
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101


class S2BandModel(nn.Module):
    """
    Model for yield prediction based of S2 Santinel bands data.
    """

    def __init__(self, bands_list: list, cfg_dl: Dict, cfg_model: Dict):
        super().__init__()

        n_bands = 12 // cfg_dl['data_loader']['s2_avg_by'] * len(bands_list)
        n_indexes = 12 // cfg_dl['data_loader']['s2_avg_by'] * len(cfg_dl['data_loader']['indexes'])
        num_in_channels = n_bands + n_indexes

        if cfg_model['s2_model']['backbone'] in ['resnet18', 'resnet34']:
            self.backbone = resnet18(pretrained=True)
            backbone_out_dim = 512
        elif cfg_model['s2_model']['backbone'] in ['resnet50', 'resnet101']:
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
            nn.Linear(backbone_out_dim, cfg_model['s2_model']['n_head']),
            nn.ReLU(),
            nn.Dropout(cfg_model['s2_model']['p_dropout']),
            nn.Linear(cfg_model['s2_model']['n_head'], 1)
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


class BandCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(12, 36, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(36, 72, (2, 2), (2, 2), (0, 0)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(72, 144, (2, 2), (2, 2), (0, 0)),
            nn.AvgPool2d(2, 2)
        )

        self.weights = nn.Sequential(
            nn.Linear(144, 12),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        xw = self.layer1(x)
        xw = torch.flatten(xw, 1)
        w = self.weights(xw).reshape(-1, 12, 1, 1)
        w = w.expand_as(x)
        xw = (x * w).sum(dim=1)
        return xw


class BandsCNN(nn.Module):
    def __init__(self, cfg_dl: Dict, cfg_model: Dict, device):
        super().__init__()
        n_bands = 13 + len(cfg_dl['data_loader']['indexes'])
        self.band_weighters = [BandCNN().to(device) for _ in range(n_bands)]
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_bands, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 128, (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 256, (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )

        self.head = nn.Sequential(
            nn.Linear(256, cfg_model['BandsCNN']['n_head']),
            nn.ReLU(),
            nn.Dropout(cfg_model['BandsCNN']['p_dropout']),
            nn.Linear(cfg_model['BandsCNN']['n_head'], 1)
        )

    def forward(self, x):
        bands = torch.split(x, 12, dim=1)
        weighted_bands = [weighter(band).reshape(-1, 1, 40, 40) for weighter, band in zip(self.band_weighters, bands)]
        weighted_bands = torch.cat(weighted_bands, dim=1)

        x = self.layer1(weighted_bands)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


class HybridBandsModel(nn.Module):
    def __init__(self, cfg_dl, cfg_model, device):
        super().__init__()
        n_bands = 13 + len(cfg_dl['data_loader']['indexes'])
        self.band_weighters = [BandCNN().to(device) for _ in range(n_bands)]

        self.backbone = resnet18(pretrained=True)
        backbone_out_dim = 512

        self.backbone.conv1 = nn.Conv2d(
            n_bands,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False
        )

        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, cfg_model['s2_model']['n_head']),
            nn.ReLU(),
            nn.Linear(cfg_model['s2_model']['n_head'], 1)
        )

    def forward(self, x):
        bands = torch.split(x, 12, dim=1)
        wb = [weighter(band).reshape(-1, 1, 40, 40) for weighter, band in zip(self.band_weighters, bands)]
        wb = torch.cat(wb, dim=1)

        x = self.backbone.conv1(wb)
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


class ShallowAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(12, 36, (3, 3), (2, 2), (1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(36, 72, (3, 3), (2, 2), (1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(72, 36, (4, 4), (2, 2), (0, 0), bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(36, 12, (4, 4), (3, 3), (0, 0), bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, 4, (4, 4), (2, 2), (0, 0), bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, (224, 224))
        return x


class AutoS2BandNetShallow(nn.Module):
    """
    Model for yield prediction based of S2 Santinel bands data.
    """

    def __init__(self, cfg_dl: Dict, cfg_model: Dict, device: str):
        super().__init__()

        n_bands = 13 + len(cfg_dl['data_loader']['indexes'])
        num_in_channels = 4 * n_bands

        self.autoencoders = [ShallowAutoEncoder().to(device) for _ in range(n_bands)]
        self.backbone = resnet18(pretrained=True)
        backbone_out_dim = 512

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False
        )

        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, cfg_model['s2_model']['n_head']),
            nn.ReLU(),
            nn.Dropout(cfg_model['s2_model']['p_dropout']),
            nn.Linear(cfg_model['s2_model']['n_head'], 1)
        )

    def forward(self, x):
        bands = torch.split(x, 12, dim=1)
        aue_bands = [aue(band) for aue, band in zip(self.autoencoders, bands)]
        aue_bands = torch.cat(aue_bands, dim=1)

        x = self.backbone.conv1(aue_bands)
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
