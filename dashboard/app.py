import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import glob, numpy as np, pandas as pd, streamlit as st
from PIL import Image
from dashboard.render import broadcast, minimap

st.set_page_config(page_title="TactPipeline", layout="wide")

DATA_ROOT = "dashboard/data"

def clip_paths(seq, split):
    """Return the split-specific baked data paths, with legacy valid fallback."""
    base = os.path.join(DATA_ROOT, split, seq)
    players, forecast = f"{base}.parquet", f"{base}_forecast.parquet"
    if os.path.exists(players) and os.path.exists(forecast):
        return players, forecast
    if split == "valid":
        legacy = os.path.join(DATA_ROOT, seq)
        return f"{legacy}.parquet", f"{legacy}_forecast.parquet"
    return players, forecast

def available_seqs(split):
    pattern = os.path.join(DATA_ROOT, split, "*.parquet")
    seqs = sorted(os.path.basename(p)[:-8] for p in glob.glob(pattern)
                  if not p.endswith("_forecast.parquet"))
    if split == "valid":
        # Keep the original single-clip bake usable until its replacement exists.
        seqs.extend(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(DATA_ROOT, "*.parquet"))
                    if not p.endswith("_forecast.parquet"))
    return sorted(set(seqs))

@st.cache_data
def load(seq, split):
    players, forecast = clip_paths(seq, split)
    df = pd.read_parquet(players)
    fc = pd.read_parquet(forecast)
    return df, fc

def frame_path(seq, split, frame):
    return f"data/SoccerNetGS/{split}/SNGS-{seq}/img1/{frame:06d}.jpg"

def useful_forecast_frame(forecasts, fallback):
    """Choose an anchor with several visible three-second trajectories on first load."""
    if not {"anchor_frame", "x0", "y0", "h3x", "h3y"}.issubset(forecasts.columns):
        return fallback
    f = forecasts.copy()
    f["d3"] = np.hypot(f.h3x - f.x0, f.h3y - f.y0)
    summary = f.groupby("anchor_frame").d3.agg(["count", "median"])
    useful = summary[(summary["count"] >= 3) & (summary["median"] >= 1.5)]
    return int(useful.index.min()) if len(useful) else fallback

with st.sidebar:
    st.header("TactPipeline")
    split = st.radio("Split", ["valid", "test"], horizontal=True)
    seqs = available_seqs(split)
    if not seqs:
        st.warning(f"No baked {split} clips yet. Run dashboard/bake_all.py {split}.")
        st.stop()
    seq = st.selectbox("Clip", seqs)
    show_ids = st.checkbox("Track IDs", True)
    show_hull = st.checkbox("Team shape", True)
    show_fc = st.checkbox("Forecast", True)
    press_src = st.radio("Pressing", ["press_pred", "press_gt"], format_func=lambda s: "Predicted" if s=="press_pred" else "Ground truth")

df, fc = load(seq, split)
fmin, fmax = int(df.frame.min()), int(df.frame.max())
frames = list(range(fmin, fmax + 1))
forecast_anchors = set(fc.anchor_frame.astype(int))
initial_frame = useful_forecast_frame(fc, fmin)

def frame_label(value):
    return f"{value:03d}" + ("  • forecast available" if value in forecast_anchors else "")

frame = st.selectbox(
    "Frame",
    frames,
    index=frames.index(initial_frame),
    format_func=frame_label,
    key=f"frame_select_{split}_{seq}",
    help="Type a frame number to search. Entries marked • have forecast trajectories.",
)

st.markdown("**Pipeline:** Detection → Tracking → Calibration / Minimap → Tactics")
players = df[df.frame == frame]
# Forecast rows are anchored at each track's actual final observation frame.
# They begin at frame 50, after the model has a two-second (50-frame) history.
fc_now = fc[fc.anchor_frame == frame] if show_fc else None

if show_fc and fc_now is not None and len(fc_now):
    # Keep display controls separate from the model output. These filters never alter
    # positions, they only choose which already-predicted trajectories to draw.
    track_team = players.groupby("track_id").team.first().to_dict()
    fc_display = fc_now.copy()
    fc_display["team"] = fc_display.track_id.map(track_team)
    if {"x0", "y0"}.issubset(fc_display.columns):
        fc_display["d3"] = np.hypot(fc_display.h3x - fc_display.x0,
                                     fc_display.h3y - fc_display.y0)
    else:
        positions = players.groupby("track_id")[["x_pitch", "y_pitch"]].first()
        x0 = fc_display.track_id.map(positions.x_pitch)
        y0 = fc_display.track_id.map(positions.y_pitch)
        fc_display["d3"] = np.hypot(fc_display.h3x - x0, fc_display.h3y - y0)

    with st.sidebar:
        st.divider()
        st.subheader("Forecast view")
        forecast_team = st.selectbox("Team", ["Both", "left", "right"], key=f"forecast_team_{split}_{seq}")
        if forecast_team != "Both":
            fc_display = fc_display[fc_display.team == forecast_team]
        forecast_mode = st.radio("Display", ["Selected player", "Top movers", "All"],
                                 key=f"forecast_mode_{split}_{seq}")

        if forecast_mode == "Selected player" and len(fc_display):
            ordered = fc_display.sort_values("d3", ascending=False)
            track_ids = ordered.track_id.astype(int).tolist()
            chosen = st.selectbox("Player track ID", track_ids,
                                  format_func=lambda tid: f"Track {tid}",
                                  key=f"forecast_track_{split}_{seq}_{frame}")
            fc_display = fc_display[fc_display.track_id == chosen]
        elif forecast_mode == "Top movers":
            fc_display = fc_display.nlargest(3, "d3")

    fc_now = fc_display.drop(columns=["team", "d3"], errors="ignore")

c1, c2 = st.columns(2)
with c1:
    st.caption("Broadcast — detection + tracking")
    fp = frame_path(seq, split, frame)
    if os.path.exists(fp):
        st.pyplot(broadcast(np.array(Image.open(fp)), players, show_ids))
    else:
        st.warning(f"frame image missing: {fp}")
with c2:
    st.caption("2D minimap — reconstruction + tactics")
    if show_fc and not len(fc_now):
        st.caption("No forecast at this frame: move to an inference anchor (for example 50, 75, 100).")
    st.pyplot(minimap(players, fc_now, show_hull, show_fc, press_src))

# info + GT-vs-pred agreement
info = []
for team, g in players.groupby("team"):
    pr, gt = g.press_pred.iloc[0], g.press_gt.iloc[0]
    agree = "✓" if pr and pr == gt else ("✗" if gt else "–")
    info.append(f"{team}: pred={pr or '–'} gt={gt or '–'} {agree}")
st.write(f"Frame {frame}/{fmax} | players {len(players)} | " + " | ".join(info))
