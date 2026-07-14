from dashboard.prepare import player_frame_table, attach_pressing, forecast_table

def test_player_frame_table_valid_021():
    df = player_frame_table(21, "valid")
    assert len(df) > 0
    assert set(["frame","track_id","bx","by","bw","bh","x_pitch","y_pitch","team","role","jersey"]).issubset(df.columns)
    assert df.team.isin(["left","right"]).all()
    assert (df.bw > 0).all() and (df.bh > 0).all()

def test_attach_pressing_021():
    df = attach_pressing(player_frame_table(21, "valid"), 21, "valid")
    assert "press_pred" in df.columns and "press_gt" in df.columns
    labelled = df[df.press_gt != ""]
    assert len(labelled) > 0
    assert labelled.press_pred.isin(["low","mid","high"]).all()

def test_forecast_table_021():
    fc = forecast_table(21, "valid")
    assert set(["anchor_frame", "track_id", "x0", "y0", "h1x", "h1y", "h3x", "h3y"]).issubset(fc.columns)
    # predicted points should be on-pitch (within generous bounds)
    if len(fc):
        assert fc[["h1x","h2x","h3x"]].abs().max().max() < 80
