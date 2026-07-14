import numpy as np, torch
from phase3b.metrics import macro_f1, confusion
from phase3b.models import GCN, GAT, majority_predict

def test_macro_f1_perfect_and_confusion():
    y = np.array([0,1,2,0,1,2])
    assert abs(macro_f1(y, y) - 1.0) < 1e-9
    assert confusion(y, y).trace() == 6

def test_forward_shapes():
    nodes = torch.randn(5, 4); edge = torch.tensor([[0,1,2,3],[1,2,3,4]])
    for M in (GCN, GAT):
        out = M()(nodes, edge)
        assert out.shape == (3,)

def test_majority():
    pred = majority_predict(np.array([1,1,1,2,0]), 4)
    assert (pred == 1).all() and len(pred) == 4