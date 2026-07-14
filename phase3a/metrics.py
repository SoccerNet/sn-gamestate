import numpy as np

HORIZONS = {"1s": 25, "2s": 50, "3s": 75}

def ade_fde(pred, gt):
    err = np.linalg.norm(pred - gt, axis=2)   # (B,75) per-frame euclidean error
    out = {}
    for name, h in HORIZONS.items():
        out[f"ade_{name}"] = float(err[:, :h].mean())
        out[f"fde_{name}"] = float(err[:, h - 1].mean())
    return out