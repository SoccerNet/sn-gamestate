#!/usr/bin/env python3
"""Phase-2: smooth per-track pitch trajectories and recompute clean velocity.

Frame-to-frame calibration noise makes raw velocity jittery (spikes of several
m/s between consecutive frames). We smooth each track's (x,y) with a
Savitzky-Golay filter (order-2, ~0.4 s window) which de-noises position while
preserving real motion shape, then recompute velocity from the smoothed path
using ACTUAL frame gaps (np.gradient handles missing frames / non-uniform dt).

Adds columns: x_s, y_s, vx_s, vy_s, speed_s. Keeps the raw ones for comparison.

Usage: python3 smooth_tracks.py [features.csv] [out.csv]
"""
import sys
import numpy as np
import pandas as pd

import os
IN  = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase2/features_valid.csv"
SPLIT = "test" if "test" in os.path.basename(IN).lower() else "valid"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"thesis_results/phase2/features_{SPLIT}_smoothed.csv"
FPS = 25.0
MAX_SPEED = 12.0
WIN, POLY = 11, 2            # ~0.44 s window, quadratic (captures acceleration)

try:
    from scipy.signal import savgol_filter
    def smooth(a, w):
        return savgol_filter(a, w, POLY)
except Exception:            # dep-free fallback: centered rolling mean
    def smooth(a, w):
        return pd.Series(a).rolling(w, center=True, min_periods=1).mean().values

def odd_window(n):
    w = min(WIN, n if n % 2 == 1 else n - 1)
    return w if (w > POLY and w % 2 == 1) else 0     # 0 -> too short, skip

def process(sub):
    sub = sub.sort_values("frame")
    x, y = sub["x_pitch"].values.astype(float), sub["y_pitch"].values.astype(float)
    t = sub["frame"].values / FPS
    w = odd_window(len(sub))
    xs, ys = (smooth(x, w), smooth(y, w)) if w else (x.copy(), y.copy())
    if len(sub) >= 2:
        vx, vy = np.gradient(xs, t), np.gradient(ys, t)   # central diff, true dt
    else:
        vx = vy = np.full(len(sub), np.nan)
    sp = np.hypot(vx, vy)
    bad = sp > MAX_SPEED
    sub = sub.copy()
    sub["x_s"], sub["y_s"] = xs, ys
    sub["vx_s"], sub["vy_s"] = np.where(bad, np.nan, vx), np.where(bad, np.nan, vy)
    sub["speed_s"] = np.where(bad, np.nan, sp)
    return sub

def accel(df, col):          # |Δspeed|/Δt within each track, for jitter comparison
    g = df.groupby(["seq", "track_id"], sort=False)
    a = (g[col].diff().abs()) / (g["frame"].diff() / FPS)
    return a

def main():
    df = pd.read_csv(IN)
    df = df.groupby(["seq", "track_id"], group_keys=False, sort=False).apply(process)
    df = df.sort_values(["seq", "track_id", "frame"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    a_raw, a_sm = accel(df, "speed"), accel(df, "speed_s")
    print(f"rows: {len(df):,}   saved -> {OUT}")
    print("\njitter (|acceleration| m/s^2, lower = smoother):")
    print(f"  raw      median={np.nanmedian(a_raw):6.2f}   p95={np.nanpercentile(a_raw.dropna(),95):7.2f}")
    print(f"  smoothed median={np.nanmedian(a_sm):6.2f}   p95={np.nanpercentile(a_sm.dropna(),95):7.2f}")
    print("\nspeed (m/s):")
    print(f"  raw      median={df['speed'].median():.2f}  p95={df['speed'].quantile(.95):.2f}")
    print(f"  smoothed median={df['speed_s'].median():.2f}  p95={df['speed_s'].quantile(.95):.2f}")
    print("\nsample (track 021/1):")
    cols = ["frame", "x_pitch", "x_s", "speed", "speed_s"]
    print(df[(df.seq.astype(str) == "021") & (df.track_id == 1)][cols].head(8).to_string(index=False))

if __name__ == "__main__":
    main()
