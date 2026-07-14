import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from dashboard.render import _format_identifier, _player_label, _voronoi_cells, broadcast, minimap, occupancy_figure, shape_timeline_figure

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

def test_voronoi_cells_are_clipped_to_pitch():
    cells = _voronoi_cells(np.array([[10., 10.], [90., 10.], [10., 58.], [90., 58.]]))
    assert len(cells) == 4
    for _, polygon in cells:
        assert polygon[:, 0].min() >= 0 and polygon[:, 0].max() <= 105
        assert polygon[:, 1].min() >= 0 and polygon[:, 1].max() <= 68

def test_minimap_can_draw_voronoi_territory():
    players = pd.concat([_players(), pd.DataFrame({
        "track_id":[3,4], "bx":[30,40], "by":[30,40], "bw":[30,30], "bh":[80,80],
        "x_pitch":[-25.,25.], "y_pitch":[-15.,15.], "team":["left","right"],
        "role":["player","player"], "jersey":["3","4"],
        "press_pred":["low","high"], "press_gt":["low","mid"],
    })], ignore_index=True)
    assert isinstance(minimap(players, show_voronoi=True), Figure)


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

def test_minimap_draws_trails_edges_and_actual_future():
    players = _players()
    trails = pd.concat([players.assign(frame=1), players.assign(frame=2, x_pitch=lambda x: x.x_pitch + 1)])
    edges = pd.DataFrame({"team":["left"], "x1":[-10.], "y1":[0.], "x2":[5.], "y2":[4.]})
    actual = pd.DataFrame({"track_id":[1], "h1x":[-9.], "h1y":[0.], "h2x":[-8.], "h2y":[0.], "h3x":[-7.], "h3y":[0.]})
    assert isinstance(minimap(players, trails=trails, edges=edges, actual_future=actual), Figure)

def test_occupancy_and_shape_timeline_return_figures():
    table = pd.concat([_players().assign(frame=1), _players().assign(frame=2)])
    metrics = pd.DataFrame({"frame":[1,1,2,2], "team":["left","right"] * 2,
                            "width_m":[1,2,2,3], "depth_m":[2,3,3,4],
                            "hull_area_m2":[1,2,3,4], "centroid_from_own_goal_m":[40,30,41,31]})
    assert isinstance(occupancy_figure(table), Figure)
    assert isinstance(shape_timeline_figure(metrics), Figure)

def test_player_label_falls_back_to_track_id_for_missing_jersey():
    player = _players().iloc[0].copy()
    player["jersey"] = np.nan
    assert _player_label(player) == "1"

def test_player_label_removes_float_suffix_from_identifiers():
    player = _players().iloc[0].copy()
    player["jersey"] = 4.0
    assert _player_label(player) == "4"
    assert _format_identifier(10.0) == "10"
