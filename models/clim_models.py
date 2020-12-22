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
