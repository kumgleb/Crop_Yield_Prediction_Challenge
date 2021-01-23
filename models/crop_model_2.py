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
        self.attention = cfg_model['CropNet']['s2_cnn']['attention']
        self.backbone = resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False
        )
        self.attention_block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
            nn.Softmax(1)
        )
        self.bn = nn.BatchNorm2d(num_in_channels)

        self.head = nn.Sequential(
            nn.Linear(512, cfg_model['CropNet']['s2_cnn']['n_out']),
            nn.BatchNorm1d(cfg_model['CropNet']['s2_cnn']['n_out'])
        )

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

        if self.attention:
            x_att = self.backbone.avgpool(x)
            x_att = torch.flatten(x_att, 1)
            w = self.attention_block(x_att)
            w = w.view(-1, 1, 7, 7).expand_as(x)
            x = (x * w).sum(dim=(2, 3))
        else:
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)

        x = self.head(x)
        return x


class S2Seq(nn.Module):
    """
     Model for processing sequential months representation from S2CNN model.
        Input: [bs, features, seq]
        Output: [bs, n_features]
    """

    def __init__(self,
                 cfg_data: Dict,
                 cfg_model: Dict):
        super().__init__()

        seq_length = 12 // cfg_data['data_loader']['s2_avg_by']
        n_in_ch = cfg_model['CropNet']['s2_cnn']['n_out']

        self.layers = nn.Sequential(
            nn.Conv1d(n_in_ch, 2 * n_in_ch, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * n_in_ch),
            nn.AdaptiveAvgPool1d(max(1, seq_length // 2)),
            nn.Conv1d(2 * n_in_ch, 4 * n_in_ch, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(4 * n_in_ch),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * n_in_ch, cfg_model['CropNet']['s2_seq']['n_head']),
            nn.BatchNorm1d(cfg_model['CropNet']['s2_seq']['n_head'])
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ClimNet(nn.Module):
    """
    Model process sequential climate data.
        Input: [bs, features, seq]
        Output: [bs, n_features]
    """

    def __init__(self,
                 cfg_data: Dict,
                 cfg_model: Dict):
        super().__init__()

        seq_length = 12
        n_in_ch = len(cfg_data['data_loader']['clim_bands'])

        self.layers = nn.Sequential(
            nn.Conv1d(n_in_ch, 4 * n_in_ch, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(4 * n_in_ch),
            nn.AdaptiveAvgPool1d(max(1, seq_length // 2)),
            nn.Conv1d(4 * n_in_ch, 16 * n_in_ch, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(16 * n_in_ch),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * n_in_ch, cfg_model['CropNet']['clim_seq']['n_head']),
            nn.BatchNorm1d(cfg_model['CropNet']['clim_seq']['n_head'])
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class CropNet2(nn.Module):
    def __init__(self, cfg_data: Dict, cfg_model: Dict, device: str):
        super().__init__()

        self.cnn = S2CNN(cfg_data, cfg_model).to(device)
        self.s2_sequential = S2Seq(cfg_data, cfg_model).to(device)
        self.clim = ClimNet(cfg_data, cfg_model).to(device)

        self.n_splits = 12 // cfg_data['data_loader']['s2_avg_by']
        n_input = cfg_model['CropNet']['s2_seq']['n_head'] + cfg_model['CropNet']['clim_seq']['n_head']
        self.logits = nn.Sequential(
            nn.Linear(n_input, cfg_model['CropNet']['n_head']),
            nn.ReLU(),
            nn.Dropout(cfg_model['CropNet']['dropout']),
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
        s2_feat = self.s2_sequential(s2_feat_seq.permute(0, 2, 1))

        # evaluate CLIM representation
        clim_feat = self.clim(clim.permute(0, 2, 1))

        # concatenate S2 and CLIM representations
        f = torch.cat([s2_feat, clim_feat], dim=1)

        # evaluate yield
        y = self.logits(f)
        return y