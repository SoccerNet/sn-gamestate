import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

def _adj(n, edge, device):
    a = torch.eye(n, device=device)                 # self-loops
    if edge.numel():
        a[edge[0], edge[1]] = 1.0
    d = a.sum(1).clamp(min=1).pow(-0.5)
    return d[:, None] * a * d[None, :]               # symmetric-normalised

class GCN(nn.Module):
    def __init__(self, in_feats=4, hidden=32, n_classes=3):
        super().__init__()
        self.l1 = nn.Linear(in_feats, hidden); self.l2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, nodes, edge):
        an = _adj(nodes.size(0), edge, nodes.device)
        h = F.relu(an @ self.l1(nodes))
        h = F.relu(an @ self.l2(h))
        return self.head(h.mean(0))                  # graph-level logits

class GAT(nn.Module):
    def __init__(self, in_feats=4, hidden=32, n_classes=3):
        super().__init__()
        self.l1 = nn.Linear(in_feats, hidden); self.l2 = nn.Linear(hidden, hidden)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Linear(hidden, n_classes)

    def _gat(self, h, edge, n):
        if edge.numel() == 0:
            return h
        src, dst = edge[0], edge[1]
        e = F.leaky_relu(self.att(torch.cat([h[dst], h[src]], 1)).squeeze(1))
        e = e - e.max()
        w = e.exp()
        denom = torch.zeros(n, device=h.device).index_add_(0, dst, w).clamp(min=1e-9)
        msg = torch.zeros_like(h).index_add_(0, dst, w[:, None] * h[src])
        return msg / denom[:, None]

    def forward(self, nodes, edge):
        n = nodes.size(0)
        h = F.relu(self.l1(nodes)); h = F.relu(self._gat(h, edge, n))
        h = F.relu(self.l2(h))
        return self.head(h.mean(0))

def majority_predict(train_labels, n_test):
    c = np.bincount(np.asarray(train_labels)).argmax()
    return np.full(n_test, c, dtype=int)