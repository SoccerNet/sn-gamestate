#!/usr/bin/env python3
"""Phase-2 feature extraction: tidy per-detection table + per-track velocity.

Reads a (stitched) tracker-state .pklz and emits one row per detection:
  seq, frame, track_id, x_pitch, y_pitch, team, role, jersey, conf,
  vx, vy, speed        (velocity in m/s, per track, along frames)

This is the base table every Phase-2/3 feature reads from (distances, spatial
graphs, trajectory sequences). Pure pandas/numpy -- runs on the Mac, no tracklab.

Usage: python3 build_features.py [state.pklz] [out.parquet]
"""
import sys, io, zipfile, pickle
import numpy as np
import pandas as pd

import os
STATE = sys.argv[1] if len(sys.argv) > 1 else "sweep/stitch_c0.30.pklz"
SPLIT = "test" if "test" in os.path.basename(STATE).lower() else "valid"
OUT   = sys.argv[2] if len(sys.argv) > 2 else f"thesis_results/phase2/features_{SPLIT}.csv"
FPS = 25.0
MAX_SPEED = 12.0   # m/s; velocities above this are calibration jitter -> NaN

def frame_of(i): return int(str(i)[-6:])

def pitch_xy(bp):
    if isinstance(bp, dict) and np.isfinite(bp.get("x_bottom_middle", np.nan)):
        return bp["x_bottom_middle"], bp["y_bottom_middle"]
    return np.nan, np.nan

def main():
    z = zipfile.ZipFile(STATE)
    seqs = sorted(n[:-4] for n in z.namelist()
                  if n.endswith(".pkl") and not n.endswith("_image.pkl"))
    frames = []
    for s in seqs:
        det = pickle.load(io.BytesIO(z.read(f"{s}.pkl")))
        det = det[det["track_id"].notna()].copy()
        xy = det["bbox_pitch"].map(pitch_xy)
        out = pd.DataFrame({
            "seq": s,
            "frame": det["image_id"].map(frame_of).astype(int),
            "track_id": det["track_id"].astype(int),
            "x_pitch": [p[0] for p in xy],
            "y_pitch": [p[1] for p in xy],
            "team": det.get("team"),
            "role": det.get("role"),
            "jersey": det.get("jersey_number"),
            "conf": det.get("bbox_conf"),
        })
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["seq", "track_id", "frame"]).reset_index(drop=True)

    # per-track velocity (m/s), along actual frame gaps
    g = df.groupby(["seq", "track_id"], sort=False)
    dt = g["frame"].diff() / FPS
    vx = g["x_pitch"].diff() / dt
    vy = g["y_pitch"].diff() / dt
    speed = np.hypot(vx, vy)
    bad = speed > MAX_SPEED                      # calibration-jitter guard
    df["vx"] = vx.where(~bad)
    df["vy"] = vy.where(~bad)
    df["speed"] = speed.where(~bad)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    saved = OUT

    print(f"rows: {len(df):,}   seqs: {df['seq'].nunique()}   tracks: {df.groupby(['seq','track_id']).ngroups}")
    print(f"pitch coverage: {df['x_pitch'].notna().mean()*100:.1f}% of detections have a pitch position")
    print(f"speed (m/s): median={df['speed'].median():.2f}  p95={df['speed'].quantile(.95):.2f}  (>{MAX_SPEED} dropped)")
    print(f"roles: {df['role'].value_counts().to_dict()}")
    print(f"teams: {df['team'].value_counts(dropna=False).to_dict()}")
    print(f"saved -> {saved}")
    print("\nsample:")
    print(df.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
