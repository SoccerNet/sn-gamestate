import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from dashboard.render import _format_identifier, _player_label, broadcast, minimap

def _players():
    return pd.DataFrame({"track_id":[1,2],"bx":[10,20],"by":[10,20],"bw":[30,30],"bh":[80,80],
        "x_pitch":[-10.0,5.0],"y_pitch":[0.0,4.0],"team":["left","right"],
        "role":["player","player"],"jersey":["7","9"],"press_pred":["low","high"],"press_gt":["low","mid"]})

def test_broadcast_returns_figure():
    img = np.zeros((100,100,3), np.uint8)
    assert isinstance(broadcast(img, _players()), Figure)

def test_minimap_returns_figure():
    fig = minimap(_players())
    assert isinstance(fig, Figure)
    assert fig.axes[0].yaxis_inverted()

def test_minimap_draws_full_forecast_path():
    forecast = pd.DataFrame({"track_id": [1], "x0": [-10.0], "y0": [0.0],
                             "h1x": [-9.0], "h1y": [0.5], "h2x": [-8.0], "h2y": [1.0],
                             "h3x": [-7.0], "h3y": [1.5]})
    fig = minimap(_players(), forecast=forecast)
    assert len(fig.axes[0].lines) >= 2  # pitch markings plus the forecast trajectory
    assert fig.axes[0].get_xlim() == (0.0, 105.0)
    assert fig.axes[0].get_ylim() == (68.0, 0.0)

def test_minimap_supports_legacy_forecast_without_start_position():
    forecast = pd.DataFrame({"track_id": [1], "h1x": [-9.0], "h1y": [0.5],
                             "h2x": [-8.0], "h2y": [1.0], "h3x": [-7.0], "h3y": [1.5]})
    assert isinstance(minimap(_players(), forecast=forecast), Figure)

def test_player_label_falls_back_to_track_id_for_missing_jersey():
    player = _players().iloc[0].copy()
    player["jersey"] = np.nan
    assert _player_label(player) == "1"

def test_player_label_removes_float_suffix_from_identifiers():
    player = _players().iloc[0].copy()
    player["jersey"] = 4.0
    assert _player_label(player) == "4"
    assert _format_identifier(10.0) == "10"
