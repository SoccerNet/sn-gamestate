#!/usr/bin/env python3
"""Phase-2: distance / spatial features from smoothed pitch positions.

Two outputs:
  1) per-detection (players + GKs): nearest-teammate, nearest-opponent,
     distance-to-ball  -> features_valid_dist.csv
  2) per-frame per-team shape: n, centroid, spread, width, length, hull area
     -> team_shape_valid.csv   (tactical primitives for 3B; edges for the graph)

Uses x_s/y_s (smoothed). Pure numpy/scipy, runs on the Mac.

Usage: python3 build_distances.py [features_smoothed.csv]
"""
import sys, os
import numpy as np
import pandas as pd
try:
    from scipy.spatial.distance import cdist
except Exception:                                  # numpy fallback (n is tiny per frame)
    def cdist(A, B):
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

IN   = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase2/features_valid_smoothed.csv"
SPLIT = "test" if "test" in os.path.basename(IN).lower() else "valid"
OUT_DET   = f"thesis_results/phase2/features_{SPLIT}_dist.csv"
OUT_SHAPE = f"thesis_results/phase2/team_shape_{SPLIT}.csv"
ATH = ("player", "goalkeeper")

def hull_area(P):
    if len(P) < 3 or ConvexHull is None:
        return np.ptp(P[:, 0]) * np.ptp(P[:, 1]) if len(P) >= 2 else np.nan
    try:
        return ConvexHull(P).volume          # 2D: .volume == area
    except Exception:
        return np.ptp(P[:, 0]) * np.ptp(P[:, 1])

def main():
    df = pd.read_csv(IN)
    df["nn_team"] = np.nan; df["nn_opp"] = np.nan; df["d_ball"] = np.nan
    shape = []

    for (seq, frame), g in df.groupby(["seq", "frame"], sort=False):
        ball = g[g["role"] == "ball"]
        bxy = ball[["x_s", "y_s"]].values[0] if len(ball) else None
        a = g[g["role"].isin(ATH) & g["team"].isin(["left", "right"])]
        if len(a):
            P = a[["x_s", "y_s"]].values
            tm = a["team"].values
            idx = a.index.values
            D = cdist(P, P); np.fill_diagonal(D, np.inf)
            same = tm[:, None] == tm[None, :]
            nn_t = np.where(same, D, np.inf).min(1)
            nn_o = np.where(~same, D, np.inf).min(1)
            df.loc[idx, "nn_team"] = np.where(np.isinf(nn_t), np.nan, nn_t)
            df.loc[idx, "nn_opp"] = np.where(np.isinf(nn_o), np.nan, nn_o)
            if bxy is not None:
                df.loc[idx, "d_ball"] = np.hypot(*(P - bxy).T)
            for t in ("left", "right"):
                Pt = P[tm == t]
                if len(Pt):
                    c = Pt.mean(0)
                    spread = np.hypot(*(Pt - c).T).mean() if len(Pt) > 1 else 0.0
                    shape.append((seq, frame, t, len(Pt), c[0], c[1], spread,
                                  np.ptp(Pt[:, 0]), np.ptp(Pt[:, 1]), hull_area(Pt)))

    sh = pd.DataFrame(shape, columns=["seq", "frame", "team", "n", "cx", "cy",
                                      "spread", "width", "length", "hull_area"])
    os.makedirs(os.path.dirname(OUT_DET), exist_ok=True)
    df.to_csv(OUT_DET, index=False)
    sh.to_csv(OUT_SHAPE, index=False)

    ath = df[df["role"].isin(ATH)]
    med = lambda s: np.nanmedian(s)
    print(f"per-detection: {len(df):,} rows -> {OUT_DET}")
    print(f"  nearest teammate (m): median={med(ath['nn_team']):.2f}  p95={np.nanpercentile(ath['nn_team'],95):.2f}")
    print(f"  nearest opponent (m): median={med(ath['nn_opp']):.2f}  p95={np.nanpercentile(ath['nn_opp'],95):.2f}")
    print(f"  distance to ball (m): median={med(ath['d_ball']):.2f}  (frames with ball: {ath['d_ball'].notna().mean()*100:.0f}%)")
    print(f"\nteam-shape: {len(sh):,} frame-team rows -> {OUT_SHAPE}")
    print(f"  players/team: median={med(sh['n']):.0f}")
    print(f"  spread (mean dist to centroid, m): median={med(sh['spread']):.1f}")
    print(f"  width  (lateral, m): median={med(sh['width']):.1f}")
    print(f"  length (depth,  m): median={med(sh['length']):.1f}")
    print(f"  hull area (m^2): median={med(sh['hull_area']):.0f}")
    print("\nsample team-shape rows:")
    print(sh.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
