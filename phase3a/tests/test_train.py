import numpy as np, pandas as pd, os, torch
from phase3a.train import make_loader

def test_loader_batches(tmp_path):
    n = 200
    df = pd.DataFrame({"seq":21,"frame":np.arange(1,n+1),"track_id":1,
        "x_s":0.05*np.arange(n),"y_s":0.0,"vx_s":1.25,"vy_s":0.0,
        "role":"player","team":"left","x_pitch":0.0,"y_pitch":0.0,
        "jersey":"1","conf":1.0,"vx":1.25,"vy":0.0,"speed":1.25})
    p=os.path.join(tmp_path,"f.csv"); df.to_csv(p,index=False)
    from phase3a.windows import build_windows
    w = build_windows(p)
    dl = make_loader(w, batch=4)
    xb, yb, p0b, gtb, vb = next(iter(dl))
    assert xb.shape[1:] == (50, 4) and yb.shape[1:] == (3, 2)
    assert gtb.shape[1:] == (75, 2) and vb.shape[1:] == (2,)