import torch
from torch import nn, optim, Tensor


class S2CilmNet(nn.Module):
    def __init__(self, s2_model, clim_model, cfg):
        super(S2CilmNet, self).__init__()

        s2_out_dim = clim_model.lstm.hidden_size
        clim_out_dim = s2_model.backbone.fc.in_features

        self.s2_model = s2_model
        self.clim_model = clim_model
        self.head = nn.Sequential(
            nn.Linear(s2_out_dim + clim_out_dim, cfg['S2ClimNet']['n_head']),
            nn.ReLU(),
            nn.Dropout(cfg['S2ClimNet']['p_dropout']),
            nn.Linear(cfg['S2ClimNet']['n_head'], 1)
        )

    def forward(self, s2, clim):
        s2 = self.s2_model.backbone.conv1(s2)
        s2 = self.s2_model.backbone.bn1(s2)
        s2 = self.s2_model.backbone.relu(s2)
        s2 = self.s2_model.backbone.maxpool(s2)

        s2 = self.s2_model.backbone.layer1(s2)
        s2 = self.s2_model.backbone.layer2(s2)
        s2 = self.s2_model.backbone.layer3(s2)
        s2 = self.s2_model.backbone.layer4(s2)

        s2 = self.s2_model.backbone.avgpool(s2)
        s2 = torch.flatten(s2, 1)

        clim, _ = self.clim_model.lstm(clim)
        clim = clim[:, self.clim_model.month_to_proc, :]

        x = torch.cat([s2, clim], dim=1)
        x = self.head(x)

        return x
