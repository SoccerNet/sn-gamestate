"""Fixed-length team-shape features for formation-proxy prediction."""
import numpy as np
import pandas as pd

from phase3b.graphs import FORMATION2ID
from phase3b.pitch import dist_to_own_goal


def _shape_features(team_rows):
    """Summarise a team's tracked outfield-player distribution for one clip."""
    players = team_rows.groupby("track_id").agg(
        x=("x_s", "mean"), y=("y_s", "mean"),
        observed_frames=("frame", "size"),
    ).dropna()
    depth = dist_to_own_goal(
        players.x.values, np.full(len(players), team_rows.team.iloc[0])
    ) / 105.0
    lateral = (players.y.values + 34.0) / 68.0
    quantiles = np.linspace(0, 1, 11)
    depth_gaps = np.diff(np.sort(depth))
    largest_gaps = np.pad(
        np.sort(depth_gaps)[-4:], (max(0, 4 - len(depth_gaps)), 0)
    )[-4:]
    return np.concatenate(([
        len(players), players.observed_frames.mean() / 750.0,
    ], np.quantile(depth, quantiles), np.quantile(lateral, quantiles),
        np.histogram(depth, bins=8, range=(0, 1), density=True)[0], largest_gaps))


def build_formation_features(split):
    """Return tracker-derived features paired with GT-derived proxy labels."""
    labels = pd.read_csv(f"phase3b/formation_{split}.csv", dtype={"seq": str})
    labels["seq"] = labels.seq.astype(int).astype(str)
    label_key = {(r.seq, r.team): r.formation for r in labels.itertuples()}

    tracks = pd.read_csv(f"thesis_results/phase2/features_{split}_smoothed.csv")
    tracks = tracks[(tracks.role == "player") & tracks.team.isin(["left", "right"])]
    rows = []
    for (seq, team), group in tracks.groupby(["seq", "team"]):
        key = (str(int(seq)), team)
        if key not in label_key:
            continue
        features = _shape_features(group)
        rows.append({
            "seq": key[0], "team": team, "formation": label_key[key],
            "label": FORMATION2ID[label_key[key]], "features": features,
        })
    return rows
