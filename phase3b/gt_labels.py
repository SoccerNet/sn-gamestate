import json, numpy as np, pandas as pd
from phase3b.pitch import pressing_third, dist_to_own_goal

ATH = ("player", "goalkeeper")
WIN = 25
IN_X, IN_Y = 60.0, 45.0     # pitch-bounds filter (drop calibration outliers)

FORMATIONS = {"4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "3-5-2": (3, 5, 2)}

def _gt_frame(json_path):
    d = json.load(open(json_path))
    rows = []
    for a in d["annotations"]:
        bp = a.get("bbox_pitch"); at = a.get("attributes", {})
        if not bp or bp.get("x_bottom_middle") is None:
            continue
        if at.get("role") not in ATH or at.get("team") not in ("left", "right"):
            continue
        rows.append((int(str(a["image_id"])[-6:]), at["team"],
                     bp["x_bottom_middle"], bp["y_bottom_middle"],
                     at["role"], a.get("track_id")))
    return pd.DataFrame(rows, columns=["frame", "team", "x", "y", "role", "track_id"])

def pressing_labels(json_path, seq):
    df = _gt_frame(json_path)
    df = df[(df.x.abs() <= IN_X) & (df.y.abs() <= IN_Y)]
    df["window"] = (df.frame - 1) // WIN
    # robust per-(frame,team) centroid x -> per-frame class
    fc = df.groupby(["frame", "team"]).agg(cx=("x", "median")).reset_index()
    fc["cls"] = pressing_third(fc.cx.values, fc.team.values)
    fc["window"] = (fc.frame - 1) // WIN
    out = []
    for (team, win), g in fc.groupby(["team", "window"]):
        vc = g["cls"].value_counts()
        if vc.iloc[0] / vc.sum() >= 0.8:          # stability filter
            out.append((seq, team, int(win), vc.index[0]))
    return pd.DataFrame(out, columns=["seq", "team", "window", "label"])

def _match_formation(counts):
    return min(FORMATIONS, key=lambda k: sum(abs(a - b) for a, b in zip(FORMATIONS[k], counts)))

def formation_label(json_path):
    df = _gt_frame(json_path)
    df = df[(df.x.abs() <= IN_X) & (df.y.abs() <= IN_Y)]
    out = {}
    for team, g in df.groupby("team"):
        outf = g[g.role == "player"]
        if outf.track_id.nunique() < 6:          # too few outfield seen -> skip clip/team
            continue
        outf = outf.assign(dg=dist_to_own_goal(outf.x.values, np.array([team] * len(outf))))
        per = np.sort(outf.groupby("track_id")["dg"].mean().values)   # depth per outfield player
        if len(per) < 6:
            continue
        # split into 3 lines at the 2 largest depth gaps (defenders / mid / forwards)
        gaps = np.diff(per)
        cuts = np.sort(np.argsort(gaps)[-2:])          # indices of the 2 biggest gaps
        lines = [per[:cuts[0]+1], per[cuts[0]+1:cuts[1]+1], per[cuts[1]+1:]]
        counts = tuple(len(L) for L in lines)          # actual (def, mid, fwd) sizes
        s = sum(counts) or 1
        counts = tuple(int(round(c * 10 / s)) for c in counts)
        out[team] = _match_formation(counts)
    return out