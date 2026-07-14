import numpy as np
import pandas as pd
import os

from phase3a.windows import build_windows

def _toy_csv(tmp_path):
    n = 130
    df = pd.DataFrame({
        "seq": 21, "frame": np.arange(1, n+1), "track_id": 1,
        "x_pitch": 0.1*np.arange(n), "y_pitch": 0.0,
        "x_s": 0.1*np.arange(n), "y_s": 0.0,
        "vx_s": 2.5, "vy_s": 0.0, "team": "left", "role": "player",
        "jersey": "7", "conf": 1.0, "vx": 2.5, "vy": 0.0, "speed": 2.5,
    })
    p = os.path.join(tmp_path, "f.csv"); df.to_csv(p, index=False); return p

def test_shapes_and_normalization(tmp_path):
    w = build_windows(_toy_csv(tmp_path))
    assert w["x"].shape[1:] == (50, 4)
    assert w["y"].shape[1:] == (3, 2)
    assert w["gt"].shape[1:] == (75, 2)
    assert len(w["x"]) >= 1

    np.testing.assert_allclose(w["x"][0, 49, :2], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(w["y"][0, 0], [2.5, 0.0], atol=1e-5)

def test_gap_excluded(tmp_path):
    df = pd.DataFrame({
        "seq":21,"frame":np.arange(1,60),"track_id":9,
        "x_s":0.0,"y_s":0.0,"vx_s":0.0,"vy_s":0.0,"role":"player","team":"left",
        "x_pitch":0.0,"y_pitch":0.0,"jersey":"1","conf":1.0,"vx":0.0,"vy":0.0,"speed":0.0
    })
    p = os.path.join(tmp_path, "s.csv"); df.to_csv(p, index=False)
    assert len(build_windows(p)["x"]) == 0