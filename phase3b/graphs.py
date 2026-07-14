import glob, os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from phase3b.pitch import dist_to_own_goal
from phase3b.gt_labels import pressing_labels

WIN, MIN_NODES = 25, 3
ATH = ("player", "goalkeeper")
LABEL2ID = {"low": 0, "mid": 1, "high": 2}
FORMATION2ID = {"4-4-2": 0, "4-3-3": 1, "3-5-2": 2}

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


def build_formation_dataset(split):
    """Build one graph per team and clip for formation-proxy classification.

    Inputs are tracker/calibration-derived, smoothed pitch coordinates.  The
    targets in ``formation_<split>.csv`` are the existing GT-position-derived
    proxy labels, not official team-sheet formations.
    """
    labels_path = f"phase3b/formation_{split}.csv"
    labels = pd.read_csv(labels_path, dtype={"seq": str})
    labels["seq"] = labels["seq"].astype(int).astype(str)
    lkey = {(r.seq, r.team): r.formation for r in labels.itertuples()}

    feat = pd.read_csv(f"thesis_results/phase2/features_{split}_smoothed.csv")
    feat = feat[(feat.role == "player") & feat.team.isin(["left", "right"])].copy()
    feat = feat.dropna(subset=["x_s", "y_s"]) 
    items = []
    for (seq, team), group in feat.groupby(["seq", "team"]):
        key = (str(int(seq)), team)
        if key not in lkey:
            continue
        # One representative node per tracked outfield player over the clip.
        players = group.groupby("track_id").agg(
            x=("x_s", "mean"), y=("y_s", "mean"),
            vx=("vx_s", "mean"), vy=("vy_s", "mean"),
        ).dropna().reset_index()
        if len(players) < 3:
            continue
        own_depth = dist_to_own_goal(players.x.values, np.full(len(players), team))
        nodes = np.stack([
            own_depth / 105.0,
            players.y.values / 68.0,
            players.vx.values / 10.0,
            players.vy.values / 10.0,
        ], axis=1).astype(np.float32)
        edge = knn_edges(nodes[:, :2], k=4)
        items.append({
            "nodes": nodes, "edge": edge,
            "label": FORMATION2ID[lkey[key]], "formation": lkey[key],
            "seq": key[0], "team": team,
        })
    return items
