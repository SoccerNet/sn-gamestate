#!/usr/bin/env python3
"""Offline tracklet stitching for the sn-gamestate baseline (Phase 2 prototype).

Adapts the offline-merge idea from Maglo et al. (SoccerNet-Tracking 2023 winners,
Sensors 2023): merge short StrongSORT fragments into long, identity-consistent
tracks using ReID-embedding similarity, gated by two physical sanity checks:
  - temporal:  two tracklets that share any frame are simultaneous -> never merge.
  - teleport:  implied speed between the end of one and start of the next must be
               <= MAX_SPEED (m/s) in PITCH metres (we have bbox_pitch, so this is
               metres, not their image-space pixels).

Runs OFFLINE on a saved tracker state (.pklz) -- no pipeline rerun, no test-time
ReID fine-tuning. Uses the PRTReID embeddings already stored in the state.

NOTE: cosine-distance threshold is model-specific (PRTReID space != Maglo's MGN
space) so it must be tuned -- this script sweeps a few values and reports the
resulting identity counts vs the known GT (~1348 IDs over valid, ~23/clip).

Usage: python3 stitch_tracklets.py [state.pklz]
"""
import sys, io, zipfile, pickle
import numpy as np
import pandas as pd

STATE = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase1/valid_state.pklz"
FPS = 25.0
MAX_SPEED = 12.0          # m/s; sprint ~10, slack for pitch/calibration noise
GT_TOTAL = 1348           # valid-set GT identities (from Phase-1 eval)
COS_THRESHOLDS = [0.20, 0.30, 0.40, 0.50]

def frame_of(image_id):            # last 6 digits of image_id encode the frame
    return int(str(image_id)[-6:])

def pitch_xy(bp):
    if isinstance(bp, dict) and np.isfinite(bp.get("x_bottom_middle", np.nan)):
        return np.array([bp["x_bottom_middle"], bp["y_bottom_middle"]], float)
    return None

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def build_tracklets(det):
    """One dict per track_id: frames set, boundary frames+positions, mean unit embedding."""
    det = det[det["track_id"].notna()].copy()
    det["frame"] = det["image_id"].map(frame_of)
    tracklets = []
    for tid, g in det.groupby("track_id"):
        g = g.sort_values("frame")
        embs = np.stack([np.asarray(e).reshape(-1) for e in g["embeddings"]])
        emb = unit(np.mean(embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9), axis=0))
        pos = {int(f): xy for f, bp in zip(g["frame"], g["bbox_pitch"])
               if (xy := pitch_xy(bp)) is not None for f in [f]}
        frames = set(int(f) for f in g["frame"])
        tracklets.append({"frames": frames, "emb": emb,
                          "start": min(frames), "end": max(frames), "pos": pos})
    return tracklets

def can_merge(a, b, cos_thresh):
    if a["frames"] & b["frames"]:                      # temporal overlap -> different players
        return False
    e, l = (a, b) if a["end"] <= b["start"] else (b, a) if b["end"] <= a["start"] else (None, None)
    if e is None:                                      # interleaved without shared frame: reject (unsafe)
        return False
    if 1.0 - float(a["emb"] @ b["emb"]) > cos_thresh:  # appearance gate
        return False
    pe, pl = e["pos"].get(e["end"]), l["pos"].get(l["start"])
    if pe is not None and pl is not None:              # teleportation gate (metres), skip if no calib
        gap_s = max((l["start"] - e["end"]) / FPS, 1e-3)
        if np.linalg.norm(pe - pl) / gap_s > MAX_SPEED:
            return False
    return True

def merge(a, b):
    frames = a["frames"] | b["frames"]
    pos = {**a["pos"], **b["pos"]}
    na, nb = len(a["frames"]), len(b["frames"])
    emb = unit(a["emb"] * na + b["emb"] * nb)          # count-weighted mean, renormalised
    return {"frames": frames, "emb": emb, "start": min(frames), "end": max(frames), "pos": pos}

def stitch(tracklets, cos_thresh):
    T = [dict(t) for t in tracklets]
    while True:
        best, bi, bj = cos_thresh + 1, -1, -1
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                if can_merge(T[i], T[j], cos_thresh):
                    d = 1.0 - float(T[i]["emb"] @ T[j]["emb"])
                    if d < best:
                        best, bi, bj = d, i, j
        if bi < 0:
            break
        T[bi] = merge(T[bi], T[bj]); T.pop(bj)
    return len(T)

def main():
    z = zipfile.ZipFile(STATE)
    vids = sorted(n[:-4] for n in z.namelist() if n.endswith(".pkl") and not n.endswith("_image.pkl"))
    rows, before_tot = [], 0
    after_tot = {t: 0 for t in COS_THRESHOLDS}
    for v in vids:
        det = pickle.load(io.BytesIO(z.read(f"{v}.pkl")))
        tk = build_tracklets(det)
        before = len(tk)
        before_tot += before
        res = {t: stitch(tk, t) for t in COS_THRESHOLDS}
        for t in COS_THRESHOLDS:
            after_tot[t] += res[t]
        rows.append((v, before, *[res[t] for t in COS_THRESHOLDS]))

    print(f"\nState: {STATE}   videos: {len(vids)}   GT identities (valid): {GT_TOTAL}\n")
    hdr = f"{'seq':<8}{'raw':>6}" + "".join(f"{'c=%.2f'%t:>8}" for t in COS_THRESHOLDS)
    print(hdr); print("-" * len(hdr))
    for r in rows[:8]:
        print(f"{r[0]:<8}{r[1]:>6}" + "".join(f"{x:>8}" for x in r[2:]))
    print("  ... (per-seq truncated) ...")
    print("-" * len(hdr))
    print(f"{'TOTAL':<8}{before_tot:>6}" + "".join(f"{after_tot[t]:>8}" for t in COS_THRESHOLDS))
    print(f"{'/GT':<8}{before_tot/GT_TOTAL:>6.1f}" + "".join(f"{after_tot[t]/GT_TOTAL:>8.1f}" for t in COS_THRESHOLDS))
    print("\n(ratio = predicted IDs / GT IDs; target ~1.0. raw is the un-stitched StrongSORT count.)")

if __name__ == "__main__":
    main()
