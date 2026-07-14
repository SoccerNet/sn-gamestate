import numpy as np
from phase3b.eval import model_macro_f1

def test_perfect_model(monkeypatch):
    items = [{"nodes": np.zeros((3,4),np.float32), "edge": np.zeros((2,0),np.int64), "label": 1}]
    # a stub model that always predicts class 1
    class Stub:
        def eval(self): pass
        def __call__(self, *_): import torch; return torch.tensor([0.,9.,0.])
    assert abs(model_macro_f1(Stub(), items) - 1.0) < 1e-9