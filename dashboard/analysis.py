"""Data transforms for the dashboard's frame and clip analysis views."""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

FPS = 25.0


def player_trails(table, frame, seconds=3):
    """Return observed history ending at the selected frame."""
    start = max(int(table.frame.min()), int(frame - seconds * FPS))
    return table[(table.frame >= start) & (table.frame <= frame)].copy()


def proximity_edges(players, k=3):
    """Undirected within-team k-nearest-neighbour edges for explanation only."""
    rows = []
    for team, group in players.groupby("team"):
        points = group[["x_pitch", "y_pitch"]].to_numpy(float)
        if len(points) < 2:
            continue
        neighbours = min(k + 1, len(points))
        _, indices = cKDTree(points).query(points, k=neighbours)
        indices = np.atleast_2d(indices)
        for source, nearby in enumerate(indices):
            for target in np.atleast_1d(nearby)[1:]:
                target = int(target)
                if source < target:
                    rows.append({
                        "team": team,
                        "x1": points[source, 0], "y1": points[source, 1],
                        "x2": points[target, 0], "y2": points[target, 1],
                    })
    return pd.DataFrame(rows, columns=["team", "x1", "y1", "x2", "y2"])


def forecast_validation(table, forecast_row):
    """Match one forecast to known future tracked positions and calculate errors."""
    if forecast_row is None or len(forecast_row) == 0:
        return None, {}
    row = forecast_row.iloc[0]
    anchor, track_id = int(row.anchor_frame), int(row.track_id)
    actual = table[(table.track_id == track_id) & table.frame.isin([anchor + 25, anchor + 50, anchor + 75])]
    lookup = actual.set_index("frame")
    values, errors = {"track_id": track_id}, []
    for index, offset in enumerate((25, 50, 75), start=1):
        prediction = np.array([row[f"h{index}x"], row[f"h{index}y"]], dtype=float)
        if anchor + offset not in lookup.index:
            values[f"h{index}x"] = np.nan
            values[f"h{index}y"] = np.nan
            continue
        point = lookup.loc[anchor + offset]
        truth = np.array([point.x_pitch, point.y_pitch], dtype=float)
        values[f"h{index}x"], values[f"h{index}y"] = truth
        errors.append(float(np.linalg.norm(prediction - truth)))
    metrics = {
        "ADE (available horizons)": float(np.mean(errors)) if errors else np.nan,
        "FDE (3 s)": errors[2] if len(errors) == 3 else np.nan,
        "available_horizons": len(errors),
    }
    return pd.DataFrame([values]), metrics


def team_shape_metrics(table):
    """Frame-by-frame team width, depth, area and own-goal-relative centroid."""
    rows = []
    for (frame, team), group in table.groupby(["frame", "team"]):
        points = group[["x_pitch", "y_pitch"]].to_numpy(float)
        if not len(points):
            continue
        hull_area = np.nan
        if len(points) >= 3:
            try:
                hull_area = float(ConvexHull(points).volume)
            except Exception:
                pass
        centroid_x = float(points[:, 0].mean())
        rows.append({
            "frame": int(frame), "team": team,
            "width_m": float(np.ptp(points[:, 1])),
            "depth_m": float(np.ptp(points[:, 0])),
            "hull_area_m2": hull_area,
            "centroid_from_own_goal_m": centroid_x + 52.5 if team == "left" else 52.5 - centroid_x,
        })
    return pd.DataFrame(rows)
