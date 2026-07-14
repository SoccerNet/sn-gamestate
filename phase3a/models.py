import torch
import torch.nn as nn

class GRUForecaster(nn.Module):
    def __init__(self, in_feats=4, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_feats, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 6))

    def forward(self, x):
        _, h = self.gru(x)              # h: (layers,B,hidden)
        out = self.head(h[-1])          # (B,6)
        return out.view(-1, 3, 2)