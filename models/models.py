from typing import Dict

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


class BandCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(12, 1, (1, 1), (1, 1), (0, 0)),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (2, 2), (1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((5, 5))
        )
        self.weights = nn.Sequential(
            nn.Linear(32, 12),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        xw = self.layer1(x)
        xw = self.layer2(xw)
        xw = torch.flatten(xw, 1)
        w = self.weights(xw).reshape(-1, 12, 1, 1)
        w = w.expand_as(x)
        xw = (x * w).sum(dim=1)
        return xw


class BandsCNN(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        n_bands = 12 + len(cfg['data_loader']['indexes'])
        self.band_weighters = [BandCNN(cfg).to(device) for _ in range(n_bands)]
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

        self.head = nn.Linear(256, 1)

    def forward(self, x):
        bands = torch.split(x, 12, dim=1)
        weighted_bands = [weighter(band).reshape(-1, 1, 40, 40) for weighter, band in zip(self.band_weighters, bands)]
        weighted_bands = torch.cat(weighted_bands, dim=1)

        x = self.layer1(weighted_bands)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


class HybridBandsModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        n_bands = 12 + len(cfg['data_loader']['indexes'])
        self.band_weighters = [BandCNN(cfg).to(device) for _ in range(n_bands)]

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
            nn.Linear(backbone_out_dim, cfg['s2_model_params']['n_head']),
            nn.ReLU(),
            nn.Linear(cfg['s2_model_params']['n_head'], 1)
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



