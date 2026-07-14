import numpy as np
import pandas as pd

from dashboard.analysis import forecast_validation, player_trails, proximity_edges, team_shape_metrics


def _table():
    rows = []
    for frame in (50, 75, 100, 125):
        rows.extend([
            (frame, 1, -10 + frame / 25, 0., "left"),
            (frame, 2, 10., 5., "right"),
            (frame, 3, 15., -5., "right"),
        ])
    return pd.DataFrame(rows, columns=["frame", "track_id", "x_pitch", "y_pitch", "team"])


def test_player_trails_end_at_selected_frame():
    trail = player_trails(_table(), 100, seconds=2)
    assert trail.frame.max() == 100
    assert trail.frame.min() == 50


def test_proximity_edges_are_within_team():
    players = _table()[lambda x: x.frame == 50]
    edges = proximity_edges(players)
    assert len(edges) == 1 and edges.team.iloc[0] == "right"


def test_forecast_validation_matches_future_positions():
    forecast = pd.DataFrame({"anchor_frame":[50], "track_id":[1],
                             "h1x":[-7.], "h1y":[0.], "h2x":[-6.], "h2y":[0.],
                             "h3x":[-5.], "h3y":[0.]})
    actual, metrics = forecast_validation(_table(), forecast)
    assert actual.h1x.iloc[0] == -7.
    assert metrics["available_horizons"] == 3
    assert np.isclose(metrics["ADE (available horizons)"], 0)


def test_team_shape_metrics_has_both_teams():
    metrics = team_shape_metrics(_table())
    assert set(metrics.team) == {"left", "right"}
    assert {"width_m", "depth_m", "hull_area_m2"}.issubset(metrics.columns)
