import glob, os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from phase3b.pitch import dist_to_own_goal
from phase3b.gt_labels import pressing_labels

WIN, MIN_NODES = 25, 3
ATH = ("player", "goalkeeper")
LABEL2ID = {"low": 0, "mid": 1, "high": 2}

def knn_edges(P, k):
    k = min(k, len(P) - 1)
    if k < 1:
        return np.zeros((2, 0), np.int64)
    _, nb = cKDTree(P).query(P, k=k + 1)
    src = np.repeat(np.arange(len(P)), k)
    dst = nb[:, 1:].reshape(-1)
    return np.stack([src, dst]).astype(np.int64)

def build_dataset(split, gt_root="data/SoccerNetGS"):
    feat = pd.read_csv(f"thesis_results/phase2/features_{split}_smoothed.csv")
    feat = feat[feat.role.isin(ATH) & feat.team.isin(["left", "right"])].copy()
    feat["window"] = (feat.frame - 1) // WIN
    # GT labels for every sequence in this split
    labs = []
    for jp in sorted(glob.glob(f"{gt_root}/{split}/SNGS-*/Labels-GameState.json")):
        seq = os.path.basename(os.path.dirname(jp)).replace("SNGS-", "")
        labs.append(pressing_labels(jp, int(seq)))
    lab = pd.concat(labs, ignore_index=True)
    lkey = {(r.seq, r.team, r.window): r.label for r in lab.itertuples()}
    items = []
    for (seq, team, win), g in feat.groupby(["seq", "team", "window"]):
        key = (int(seq), team, int(win))
        if key not in lkey:
            continue
        gp = g.groupby("track_id").agg(x=("x_s", "mean"), y=("y_s", "mean"),
                                       vx=("vx_s", "mean"), vy=("vy_s", "mean")).reset_index()
        if len(gp) < MIN_NODES:
            continue
        dg = dist_to_own_goal(gp.x.values, np.array([team] * len(gp)))
        nodes = np.stack([dg, gp.y.values, gp.vx.fillna(0).values, gp.vy.fillna(0).values], 1).astype(np.float32)
        edge = knn_edges(np.stack([gp.x.values, gp.y.values], 1), k=4)
        items.append({"nodes": nodes, "edge": edge, "label": LABEL2ID[lkey[key]],
                      "seq": str(seq), "team": team, "window": int(win)})
    return items