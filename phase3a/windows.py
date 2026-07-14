import numpy as np
import pandas as pd

IN_LEN = 50
HORIZONS = (25, 50, 75)
MAXH = 75
STRIDE = 25

WIN = IN_LEN + MAXH
ATH = ("player", "goalkeeper")
FPS, MAX_SPEED = 25.0, 12.0   # drop windows whose position jumps imply >MAX_SPEED (calibration blowups)

def _runs(frames):
    frames = np.sort(np.unique(frames))
    brk = np.where(np.diff(frames) != 1)[0] + 1
    return np.split(frames, brk)

def build_windows(csv_path, seqs=None, exclude_seqs=None):
    df = pd.read_csv(csv_path)
    df = df[df["role"].isin(ATH)]
    if seqs is not None:
        df = df[df["seq"].isin(seqs)]
    if exclude_seqs is not None:
        df = df[~df["seq"].isin(exclude_seqs)]
    X, Y, P0, GT, S, T = [], [], [], [], [], []
    for (seq, tid), g in df.groupby(["seq", "track_id"]):
        g = g.sort_values("frame")
        pos = g.set_index("frame")[["x_s", "y_s"]]
        vel = g.set_index("frame")[["vx_s", "vy_s"]].fillna(0.0)
        for run in _runs(g["frame"].values):
            if len(run) < WIN:
                continue
            for i in range(0, len(run) - WIN + 1, STRIDE):
                fr = run[i:i + WIN]
                P = pos.loc[fr].values.astype(np.float32)      # (125,2)
                if (np.linalg.norm(np.diff(P, axis=0), axis=1) * FPS).max() > MAX_SPEED:
                    continue                                   # calibration-garbage window -> skip
                V = vel.loc[fr].values.astype(np.float32)      # (125,2)
                p0 = P[IN_LEN - 1]                             # last input frame
                x = np.concatenate([P[:IN_LEN] - p0, V[:IN_LEN]], axis=1)  # (50,4)
                gt = P[IN_LEN:] - p0                           # (75,2)
                y = gt[[h - 1 for h in HORIZONS]]              # (3,2)
                X.append(x); Y.append(y); P0.append(p0); GT.append(gt)
                S.append(seq); T.append(tid)
    f32 = lambda a: np.asarray(a, np.float32)
    return {"x": f32(X).reshape(-1, IN_LEN, 4), "y": f32(Y).reshape(-1, 3, 2),
            "p0": f32(P0).reshape(-1, 2), "gt": f32(GT).reshape(-1, MAXH, 2),
            "seq": np.asarray(S), "tid": np.asarray(T)}