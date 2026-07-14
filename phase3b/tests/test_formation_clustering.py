import numpy as np
import pandas as pd

from phase3b.formation_clustering import cluster_formation


def test_cluster_formation_orders_depth_bands():
    rows = []
    for track_id, x in enumerate((-45, -40, -35, -5, 0, 5, 25, 30, 35), start=1):
        for frame in range(1, 31):
            rows.append((track_id, frame, x, "left"))
    team_rows = pd.DataFrame(rows, columns=["track_id", "frame", "x_s", "team"])
    formation, counts, nodes = cluster_formation(team_rows, "kmeans", min_track_frames=30)
    assert nodes == 9
    assert counts == (3, 3, 3)
    assert formation in {"4-4-2", "4-3-3", "3-5-2"}


def test_cluster_formation_needs_three_nodes():
    team_rows = pd.DataFrame({
        "track_id": [1] * 30 + [2] * 30,
        "frame": list(range(1, 31)) * 2,
        "x_s": np.r_[[0.0] * 30, [10.0] * 30],
        "team": ["left"] * 60,
    })
    assert cluster_formation(team_rows, min_track_frames=30) is None
