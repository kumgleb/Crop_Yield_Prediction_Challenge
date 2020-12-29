import torch
from typing import Dict
from torch import nn, optim, Tensor


class ClimNet(nn.Module):
    """
    [bs, seq, features]
    """
    def __init__(self, cfg: Dict):
        super(ClimNet, self).__init__()
        self.month_to_proc = cfg['ClimNet']['month_to_proc']
        k = 2 if cfg['ClimNet']['bi'] else 1
        self.lstm = nn.LSTM(input_size=cfg['ClimNet']['input_dim'],
                            hidden_size=cfg['ClimNet']['hidden_dim'],
                            num_layers=cfg['ClimNet']['lstm_num_layers'],
                            dropout=cfg['ClimNet']['lstm_dropout'],
                            bidirectional=cfg['ClimNet']['bi'],
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(k * cfg['ClimNet']['hidden_dim'], cfg['ClimNet']['head_dim']),
            nn.ReLU(),
            nn.Dropout(cfg['ClimNet']['head_dropout']),
            nn.Linear(cfg['ClimNet']['head_dim'], 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, self.month_to_proc, :]
        x = self.head(x)
        return x


class ClimCNN(nn.Module):
  def __init__(self, cfg):
    super(ClimCNN, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv1d(13, 26, 2, 1, 1),
        nn.ReLU(),
        nn.Conv1d(26, 52, 2, 1, 0),
        nn.ReLU(),
        nn.AvgPool1d(2),
        nn.Conv1d(52, 128, 2, 1, 1),
        nn.ReLU(),
        nn.AvgPool1d(2),
        nn.Conv1d(128, 256, 2, 1, 0),
        nn.ReLU(),
        nn.AvgPool1d(2)
    )
    self.head = nn.Sequential(
        nn.Linear(256, cfg['ClimCNN']['n_head']),
        nn.ReLU(),
        nn.Dropout(cfg['ClimCNN']['p_dropout']),
        nn.Linear(cfg['ClimCNN']['n_head'], 1)
    )


  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.layer1(x)
    x = torch.flatten(x, 1)
    x = self.head(x)
    return x