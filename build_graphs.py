#!/usr/bin/env python3
"""Phase-2: per-frame spatial graphs from smoothed positions.

Nodes  = visible players + goalkeepers in a frame (features live in the
         per-detection table: x_s,y_s,vx_s,vy_s,team,role -> join on seq,frame,track_id).
Edges  = directed k-nearest-neighbour (k=5) between athletes, with:
           dist       euclidean pitch distance (m)
           same_team  1 if both nodes same team else 0
This is the standard GNN input for the 3B tactical classifier; distances also
give 3A neighbour context. A radius graph = filter edges by dist <= R.

Outputs:
  graph_edges_valid.parquet   seq, frame, src, dst, dist, same_team
  graph_summary_valid.csv     per-frame: n_nodes, n_edges, mean_edge_len, intra_frac

Usage: python3 build_graphs.py [features_dist.csv] [k]
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

IN = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase2/features_valid_dist.csv"
K  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SPLIT = "test" if "test" in os.path.basename(IN).lower() else "valid"
OUT_E = f"thesis_results/phase2/graph_edges_{SPLIT}.parquet"
OUT_S = f"thesis_results/phase2/graph_summary_{SPLIT}.csv"
ATH = ("player", "goalkeeper")

def main():
    df = pd.read_csv(IN)
    a = df[df["role"].isin(ATH) & df["team"].isin(["left", "right"])]
    edges, summ = [], []
    for (seq, frame), g in a.groupby(["seq", "frame"], sort=False):
        if len(g) < 2:
            continue
        P = g[["x_s", "y_s"]].values
        tid = g["track_id"].values
        tm = g["team"].values
        k = min(K, len(g) - 1)
        d, nb = cKDTree(P).query(P, k=k + 1)      # +1 = self
        elen, intra, ne = [], 0, 0
        for i in range(len(g)):
            for j_pos in range(1, k + 1):          # skip col 0 (self)
                j = nb[i, j_pos]
                st = int(tm[i] == tm[j])
                edges.append((seq, frame, int(tid[i]), int(tid[j]), float(d[i, j_pos]), st))
                elen.append(d[i, j_pos]); intra += st; ne += 1
        summ.append((seq, frame, len(g), ne, float(np.mean(elen)), intra / ne if ne else np.nan))

    E = pd.DataFrame(edges, columns=["seq", "frame", "src", "dst", "dist", "same_team"])
    S = pd.DataFrame(summ, columns=["seq", "frame", "n_nodes", "n_edges", "mean_edge_len", "intra_frac"])
    os.makedirs(os.path.dirname(OUT_E), exist_ok=True)
    try:
        E.to_parquet(OUT_E); e_saved = OUT_E
    except Exception:
        e_saved = OUT_E.replace(".parquet", ".csv"); E.to_csv(e_saved, index=False)
    S.to_csv(OUT_S, index=False)

    print(f"edges: {len(E):,} -> {e_saved}")
    print(f"frames (graphs): {len(S):,} -> {OUT_S}")
    print(f"  nodes/frame:      median={S['n_nodes'].median():.0f}  (visible athletes)")
    print(f"  mean edge len (m): median={S['mean_edge_len'].median():.2f}")
    print(f"  intra-team edge fraction: mean={S['intra_frac'].mean():.2f}  (share of k-NN links to a teammate)")
    print("\nsample edges:")
    print(E.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
