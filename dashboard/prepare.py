import io, os, zipfile, pickle, numpy as np, pandas as pd
import torch
from phase3b.graphs import build_dataset
from phase3b.models import GAT
from phase3a.models import GRUForecaster

STATE = {"valid": "sweep/stitch_c0.30.pklz", "test": "sweep/stitch_test_c0.30.pklz"}
ATH = ("player", "goalkeeper")
WIN = 25
ID2LABEL = {0: "low", 1: "mid", 2: "high"}
IN_LEN, FPS = 50, 25.0

def _frame(i): return int(str(i)[-6:])

def player_frame_table(seq, split="valid"):
    z = zipfile.ZipFile(STATE[split])
    det = pickle.load(io.BytesIO(z.read(f"{int(seq):03d}.pkl")))
    det = det[det.track_id.notna()].copy()
    det = det[det.role.isin(ATH) & det.team.isin(["left", "right"])]
    b = np.stack(det["bbox_ltwh"].values)
    img = pd.DataFrame({
        "frame": det["image_id"].map(_frame).astype(int),
        "track_id": det["track_id"].astype(int),
        "bx": b[:, 0], "by": b[:, 1], "bw": b[:, 2], "bh": b[:, 3],
    })
    feat = pd.read_csv(f"thesis_results/phase2/features_{split}_smoothed.csv")
    feat = feat[(feat.seq == int(seq)) & feat.role.isin(ATH)].copy()
    feat["track_id"] = feat.track_id.astype(int)
    sm = feat[["frame", "track_id", "x_s", "y_s", "team", "role", "jersey"]].rename(
        columns={"x_s": "x_pitch", "y_s": "y_pitch"})
    out = img.merge(sm, on=["frame", "track_id"], how="inner")
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)

def attach_pressing(table, seq, split, weights="phase3b/gat.pt"):
    items = [it for it in build_dataset(split) if int(it["seq"]) == int(seq)]
    model = GAT(); model.load_state_dict(torch.load(weights, map_location="cpu")); model.eval()
    pred, gt = {}, {}
    with torch.no_grad():
        for it in items:
            logit = model(torch.tensor(it["nodes"]), torch.tensor(it["edge"]))
            pred[(it["team"], it["window"])] = ID2LABEL[int(logit.argmax())]
            gt[(it["team"], it["window"])] = ID2LABEL[it["label"]]
    t = table.copy()
    t["window"] = (t.frame - 1) // WIN
    key = list(zip(t.team, t.window))
    t["press_pred"] = [pred.get(k, "") for k in key]
    t["press_gt"] = [gt.get(k, "") for k in key]
    return t

def forecast_table(seq, split, weights="phase3a/gru.pt"):
    feat = pd.read_csv(f"thesis_results/phase2/features_{split}_smoothed.csv")
    feat = feat[(feat.seq == int(seq)) & feat.role.isin(ATH)].copy()
    model = GRUForecaster(); model.load_state_dict(torch.load(weights, map_location="cpu")); model.eval()
    rows = []
    for tid, g in feat.groupby("track_id"):
        g = g.sort_values("frame")
        fr = g.frame.values
        P = g[["x_s", "y_s"]].values.astype(np.float32)
        V = g[["vx_s", "vy_s"]].fillna(0).values.astype(np.float32)
        pos = {int(f): (P[i], V[i]) for i, f in enumerate(fr)}
        for a in range(int(fr.min()) + IN_LEN - 1, int(fr.max()) + 1, WIN):
            win = list(range(a - IN_LEN + 1, a + 1))
            if not all(f in pos for f in win):        # need 50 contiguous frames
                continue
            Pw = np.stack([pos[f][0] for f in win]); Vw = np.stack([pos[f][1] for f in win])
            p0 = Pw[-1]
            x = np.concatenate([Pw - p0, Vw], axis=1).astype(np.float32)
            with torch.no_grad():
                d = model(torch.tensor(x)[None])[0].numpy()   # (3,2) displacements
            abspt = p0[None, :] + d
            rows.append((a, int(tid), *p0, *abspt[0], *abspt[1], *abspt[2]))
    return pd.DataFrame(rows, columns=["anchor_frame", "track_id", "x0", "y0",
                                       "h1x", "h1y", "h2x", "h2y", "h3x", "h3y"])

def bake(seq, split="valid"):
    os.makedirs("dashboard/data", exist_ok=True)
    t = attach_pressing(player_frame_table(seq, split), seq, split)
    t.to_parquet(f"dashboard/data/{int(seq):03d}.parquet")
    forecast_table(seq, split).to_parquet(f"dashboard/data/{int(seq):03d}_forecast.parquet")
    print(f"baked dashboard/data/{int(seq):03d}.parquet ({len(t)} rows) + forecast")

if __name__ == "__main__":
    import sys
    bake(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "valid")
