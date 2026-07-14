import numpy as np
from phase3a.eval import rows_from_windows

def test_baseline_rows_present():
    # synthetic windows: pure constant velocity -> CV baseline should be ~0 error
    B = 5
    w = {"x": np.zeros((B,50,4),np.float32), "gt": np.zeros((B,75,2),np.float32)}
    w["x"][:, :, 2] = 2.0                     # 2 m/s x
    t = (np.arange(1,76)/25.0)
    w["gt"][:, :, 0] = 2.0 * t                # true CV future
    df = rows_from_windows(w, model=None)     # baselines only
    assert "const_velocity" in df.index
    assert df.loc["const_velocity", "fde_3s"] < 1e-3