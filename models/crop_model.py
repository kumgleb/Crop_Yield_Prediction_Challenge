from typing import Dict

import torch
from torch import nn, optim, Tensor
from torchvision.models.resnet import resnet18


class S2CNN(nn.Module):
    """
    Model for images representation.
        Input: [bs, n_ch, w, h]
        Outputs: [bs, 512]
    """

    def __init__(self, cfg_data: Dict, cfg_model: Dict):
        super().__init__()

        num_in_channels = (len(cfg_data['data_loader']['s2_bands']) +
                           len(cfg_data['data_loader']['indexes']))

        self.bn_first = cfg_model['CropNet']['s2_cnn']['bn_first']
        self.backbone = resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(num_in_channels)

    def forward(self, x):
        if self.bn_first:
            x = self.bn(x)
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
        return x


class S2Seq(nn.Module):
    """
     Model for processing sequential months representation from S2CNN model.
        Input: [bs, seq, features]
        Output: [bs, n_features]
    """
    def __init__(self, cfg_model: Dict):
        super().__init__()
        k = 2 if cfg_model['CropNet']['s2_lstm']['bi'] else 1
        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=cfg_model['CropNet']['s2_lstm']['hidden_size'],
                            num_layers=cfg_model['CropNet']['s2_lstm']['n_layers'],
                            dropout=cfg_model['CropNet']['s2_lstm']['dropout'],
                            bidirectional=cfg_model['CropNet']['s2_lstm']['bi'],
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(k * cfg_model['CropNet']['s2_lstm']['hidden_size'],
                      cfg_model['CropNet']['s2_lstm']['n_head']),
            nn.ReLU()
            )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.head(x)
        return x


class ClimNet(nn.Module):
    """
    Model process sequential climate data.
        Input: [bs, seq, features]
        Output: [bs, n_features]
    """
    def __init__(self, cfg_data: Dict, cfg_model: Dict):
        super().__init__()
        input_dim = len(cfg_data['data_loader']['clim_bands'])
        k = 2 if cfg_model['CropNet']['clim_sltm']['bi'] else 1
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=cfg_model['CropNet']['clim_sltm']['hidden_size'],
                            num_layers=cfg_model['CropNet']['clim_sltm']['n_layers'],
                            dropout=cfg_model['CropNet']['clim_sltm']['dropout'],
                            bidirectional=cfg_model['CropNet']['clim_sltm']['bi'],
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(k * cfg_model['CropNet']['clim_sltm']['hidden_size'],
                      cfg_model['CropNet']['clim_sltm']['n_head']),
            nn.ReLU()
            )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.head(x)
        return x


class CropNet(nn.Module):
    def __init__(self, cfg_data: Dict, cfg_model: Dict, device: str):
        super().__init__()

        self.cnn = S2CNN(cfg_data, cfg_model).to(device)
        self.s2_sequential = S2Seq(cfg_model).to(device)
        self.clim = ClimNet(cfg_data, cfg_model).to(device)

        self.n_splits = 12 // cfg_data['data_loader']['s2_avg_by']
        n_input = cfg_model['CropNet']['s2_lstm']['n_head'] + cfg_model['CropNet']['clim_sltm']['n_head']
        self.logits = nn.Sequential(
            nn.Linear(n_input, cfg_model['CropNet']['n_head']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(cfg_model['CropNet']['n_head'], 1)
        )

    def forward(self, s2, clim):
        # put sequential first
        s2 = s2.permute(1, 0, 2, 3)
        # split by months
        s2 = torch.split(s2, self.n_splits, dim=0)
        # stack tensors from different channels in the same month
        s2_ch_seq = [torch.stack(channel) for channel in zip(*s2)]
        # perform months representation
        s2_feat_seq = [self.cnn(ch.permute(1, 0, 2, 3)) for ch in s2_ch_seq]
        # stack temporal representation
        s2_feat_seq = torch.stack(s2_feat_seq, dim=1)
        # evaluate S2 bands representation
        s2_feat = self.s2_sequential(s2_feat_seq)

        # evaluate CLIM representation
        clim_feat = self.clim(clim)

        # concatenate S2 and CLIM representations
        f = torch.cat([s2_feat, clim_feat], dim=1)

        # evaluate yield
        y = self.logits(f)
        return y
