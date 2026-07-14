import numpy as np
from phase3a.metrics import ade_fde

def test_zero_error():
    a = np.random.RandomState(0).randn(4, 75, 2).astype(np.float32)
    m = ade_fde(a, a)
    for k in m: assert abs(m[k]) < 1e-6

def test_constant_offset():
    gt = np.zeros((1, 75, 2), np.float32)
    pred = np.zeros((1, 75, 2), np.float32); pred[..., 0] = 3.0  # 3 m off in x, all frames
    m = ade_fde(pred, gt)
    assert abs(m["fde_3s"] - 3.0) < 1e-5
    assert abs(m["ade_1s"] - 3.0) < 1e-5