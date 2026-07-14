import numpy as np
import pandas as pd

from dashboard.video import render_broadcast_frame, render_composite_frame


def test_render_broadcast_frame_draws_overlays():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    players = pd.DataFrame({
        "track_id": [1], "bx": [20], "by": [15], "bw": [25], "bh": [45],
        "team": ["left"], "jersey": ["7"], "press_pred": ["mid"],
    })
    rendered = render_broadcast_frame(image, players, 50)
    assert rendered.shape == image.shape
    assert rendered.sum() > 0


def test_render_composite_frame_contains_broadcast_and_minimap():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    players = pd.DataFrame({
        "track_id": [1, 2, 3, 4], "bx": [10, 30, 50, 70], "by": [10, 15, 20, 25],
        "bw": [10] * 4, "bh": [20] * 4, "team": ["left", "left", "right", "right"],
        "jersey": ["1", "2", "3", "4"], "press_pred": ["mid"] * 4,
        "x_pitch": [-20., -10., 10., 20.], "y_pitch": [-10., 10., -10., 10.],
    })
    forecasts = pd.DataFrame(columns=["anchor_frame", "track_id", "x0", "y0", "h1x", "h1y", "h2x", "h2y", "h3x", "h3y"])
    rendered = render_composite_frame(image, players, forecasts, 1)
    assert rendered.shape[0] > 0 and rendered.shape[1] > image.shape[1]
