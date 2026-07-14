"""Apply offline tracklet stitching and write a new tracker-state .pklz.

Remaps track_id in each video's detections so merged fragments share one id,
keeps everything else (embeddings, bbox_pitch, attributes, _image.pkl,
summary.json) byte-identical, and repackages the zip so tracklab can load it.

Then re-score to verify: AssA should RISE vs the 47.24 baseline if merges are
correct (falls if we're over-merging).

Usage: python3 stitch_and_write.py [in.pklz] [out.pklz] [cos_thresh]
"""
import sys, io, zipfile, pickle
import numpy as np
import pandas as pd

IN   = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase1/valid_state.pklz"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "thesis_results/phase1/valid_state_stitched.pklz"
COS  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.40   # operating point from the sweep
FPS, MAX_SPEED = 25.0, 12.0

def frame_of(i): return int(str(i)[-6:])
def pitch_xy(bp):
    if isinstance(bp, dict) and np.isfinite(bp.get("x_bottom_middle", np.nan)):
        return np.array([bp["x_bottom_middle"], bp["y_bottom_middle"]], float)
    return None
def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

def build(det):
    det = det[det["track_id"].notna()].copy()
    det["frame"] = det["image_id"].map(frame_of)
    T = []
    for tid, g in det.groupby("track_id"):
        g = g.sort_values("frame")
        E = np.stack([np.asarray(e).reshape(-1) for e in g["embeddings"]])
        emb = unit(np.mean(E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9), axis=0))
        pos = {int(f): xy for f, bp in zip(g["frame"], g["bbox_pitch"])
               if (xy := pitch_xy(bp)) is not None for f in [int(f)]}
        fr = set(int(f) for f in g["frame"])
        T.append({"ids": {tid}, "frames": fr, "emb": emb,
                  "start": min(fr), "end": max(fr), "pos": pos})
    return T

def can_merge(a, b, c):
    if a["frames"] & b["frames"]: return False
    e, l = (a, b) if a["end"] <= b["start"] else (b, a) if b["end"] <= a["start"] else (None, None)
    if e is None: return False
    if 1.0 - float(a["emb"] @ b["emb"]) > c: return False
    pe, pl = e["pos"].get(e["end"]), l["pos"].get(l["start"])
    if pe is not None and pl is not None:
        gap = max((l["start"] - e["end"]) / FPS, 1e-3)
        if np.linalg.norm(pe - pl) / gap > MAX_SPEED: return False
    return True

def merge(a, b):
    fr = a["frames"] | b["frames"]; na, nb = len(a["frames"]), len(b["frames"])
    return {"ids": a["ids"] | b["ids"], "frames": fr,
            "emb": unit(a["emb"] * na + b["emb"] * nb),
            "start": min(fr), "end": max(fr), "pos": {**a["pos"], **b["pos"]}}

def stitch(T, c):
    T = [dict(t) for t in T]
    while True:
        best, bi, bj = c + 1, -1, -1
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                if can_merge(T[i], T[j], c):
                    d = 1.0 - float(T[i]["emb"] @ T[j]["emb"])
                    if d < best: best, bi, bj = d, i, j
        if bi < 0: break
        T[bi] = merge(T[bi], T[bj]); T.pop(bj)
    # old_track_id -> representative (min id of its final cluster)
    m = {}
    for t in T:
        rep = min(t["ids"])
        for old in t["ids"]: m[old] = rep
    return m, len(T)

def main():
    zin = zipfile.ZipFile(IN)
    vids = sorted(n[:-4] for n in zin.namelist()
                  if n.endswith(".pkl") and not n.endswith("_image.pkl"))
    before = after = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            raw = zin.read(name)
            if name.endswith(".pkl") and not name.endswith("_image.pkl"):
                det = pickle.loads(raw)
                m, k = stitch(build(det), COS)
                before += det.loc[det["track_id"].notna(), "track_id"].nunique()
                after  += k
                det["track_id"] = det["track_id"].map(lambda x: m.get(x, x))
                zout.writestr(name, pickle.dumps(det))
            else:                                   # summary.json + *_image.pkl: verbatim
                zout.writestr(name, raw)
    print(f"cos_thresh={COS}  IDs {before} -> {after}  ({before/after:.2f}x fewer)")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()