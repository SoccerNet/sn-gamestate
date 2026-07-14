"""Windowed, template-based formation-proxy prediction from tracked positions.

SoccerNet-GSR clips contain 750 frames (about 25 seconds), so one complete
clip is the longest uninterrupted window available in this project.  A player
node is the mean calibrated position for a sufficiently persistent track.
Nodes are clustered by goal-relative X depth into defender/midfielder/forward
bands and matched to the nearest supported formation template.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

from phase3b.gt_labels import FORMATIONS
from phase3b.metrics import macro_f1
from phase3b.pitch import dist_to_own_goal

FORMATION2ID = {name: index for index, name in enumerate(FORMATIONS)}
VAL_SEQS = [78, 79, 80, 81, 82, 83]


def _nearest_template(counts):
    """Scale visible band counts to an 10-outfield-player template."""
    total = sum(counts)
    scaled = tuple(int(round(count * 10 / total)) for count in counts)
    return min(FORMATIONS, key=lambda name: sum(
        abs(actual - expected) for actual, expected in zip(scaled, FORMATIONS[name])
    ))


def cluster_formation(team_rows, algorithm="kmeans", min_track_frames=30):
    """Predict one formation from a single uninterrupted team window.

    ``min_track_frames=30`` keeps tracks visible for at least one second,
    filtering momentary false detections while retaining enough players in a
    broadcast view. Returns ``None`` if fewer than three usable player nodes
    are available.
    """
    players = team_rows.groupby("track_id").agg(
        x=("x_s", "mean"), observed_frames=("frame", "nunique")
    ).dropna()
    players = players[players.observed_frames >= min_track_frames]
    if len(players) < 3:
        return None

    team = team_rows.team.iloc[0]
    depth = dist_to_own_goal(players.x.values, np.full(len(players), team))
    values = depth.reshape(-1, 1)
    if algorithm == "kmeans":
        cluster_id = KMeans(n_clusters=3, n_init=20, random_state=7).fit_predict(values)
    elif algorithm == "agglomerative":
        cluster_id = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(values)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    # Cluster IDs are arbitrary. Sort clusters from own goal to opponent goal.
    centers = np.array([depth[cluster_id == group].mean() for group in range(3)])
    ordered_groups = np.argsort(centers)
    counts = tuple(int((cluster_id == group).sum()) for group in ordered_groups)
    return _nearest_template(counts), counts, len(players)


def predict_split(split, algorithm="kmeans", min_track_frames=30):
    labels = pd.read_csv(f"phase3b/formation_{split}.csv", dtype={"seq": str})
    labels["seq"] = labels.seq.astype(int)
    tracks = pd.read_csv(f"thesis_results/phase2/features_{split}_smoothed.csv")
    tracks = tracks[(tracks.role == "player") & tracks.team.isin(["left", "right"])]

    rows = []
    for (seq, team), group in tracks.groupby(["seq", "team"]):
        result = cluster_formation(group, algorithm, min_track_frames)
        if result is None:
            continue
        formation, counts, nodes = result
        rows.append({"seq": int(seq), "team": team, "formation_pred": formation,
                     "band_counts_visible": "-".join(map(str, counts)), "player_nodes": nodes})
    return labels.merge(pd.DataFrame(rows), on=["seq", "team"], how="inner")


def evaluate(split="valid", algorithm="kmeans", min_track_frames=30):
    result = predict_split(split, algorithm, min_track_frames)
    evaluated = result[result.seq.isin(VAL_SEQS)] if split == "valid" else result
    y = evaluated.formation.map(FORMATION2ID).to_numpy()
    p = evaluated.formation_pred.map(FORMATION2ID).to_numpy()
    report = {
        "method": f"{algorithm} X-depth bands + template matching",
        "window": "one complete uninterrupted 750-frame / ~25-second clip",
        "min_track_frames": min_track_frames,
        f"{split}_examples": len(evaluated),
        f"{split}_macro_f1": round(macro_f1(y, p, len(FORMATION2ID)), 6),
        f"{split}_accuracy": round(float((y == p).mean()), 6),
        "target": "GT-derived geometric formation proxy",
    }
    out = Path(f"phase3b/formation_{algorithm}_{split}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    if split == "test":
        evaluated.to_csv(f"phase3b/formation_{algorithm}_test_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    return result, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=("kmeans", "agglomerative"), nargs="?", default="kmeans")
    parser.add_argument("--min-track-frames", type=int, default=30)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    evaluate(args.split, args.algorithm, args.min_track_frames)
