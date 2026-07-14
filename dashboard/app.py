import glob
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.analysis import forecast_validation, player_trails, proximity_edges, team_shape_metrics
from dashboard.render import broadcast, minimap, occupancy_figure, shape_timeline_figure

st.set_page_config(page_title="TactPipeline", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
    [data-testid="stSidebar"] { background: #0f251c; }
    [data-testid="stSidebar"] * { color: #edf6f0; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label { color: #edf6f0 !important; }
    .hero { background: linear-gradient(120deg, #10291f 0%, #1f5440 100%); border-radius: 18px;
            padding: 1.25rem 1.5rem; color: #f5fbf7; margin-bottom: 1.1rem; }
    .hero h1 { font-size: 2rem; margin: 0; color: #ffffff; }
    .hero p { margin: .3rem 0 0; color: #cce0d4; }
    .eyebrow { color: #70d1a2; font-weight: 700; font-size: .76rem; letter-spacing: .11em; text-transform: uppercase; }
    .section-title { color: #163326; font-size: 1.1rem; font-weight: 700; margin: .75rem 0 .25rem; }
    [data-testid="stMetric"] { background: #f7faf8; border: 1px solid #e1ebe5; border-radius: 13px; padding: .65rem .8rem; }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] * { color: #17372a !important; }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] { font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.28rem; font-weight: 750; }
    [data-testid="stMetric"] [data-testid="stMetricDelta"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] * { color: #259e66 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] { background: #f2f6f3; border-radius: 9px; padding: .4rem .8rem; }
    @media (max-width: 700px) {
        .block-container { padding: .8rem .8rem 2rem; }
        .hero { padding: 1rem; border-radius: 14px; }
        .hero h1 { font-size: 1.55rem; }
        .hero p { font-size: .92rem; }
    }
</style>
""", unsafe_allow_html=True)

DEMO_MODE = not os.path.exists("dashboard/data/valid")
DATA_ROOT = "dashboard/demo/data" if DEMO_MODE else "dashboard/data"
VIDEO_ROOT = "dashboard/demo/videos" if DEMO_MODE else "dashboard/videos"


def clip_paths(seq, split):
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
    seqs = sorted(os.path.basename(path)[:-8] for path in glob.glob(pattern)
                  if not path.endswith("_forecast.parquet"))
    if split == "valid":
        seqs.extend(os.path.basename(path)[:-8] for path in glob.glob(os.path.join(DATA_ROOT, "*.parquet"))
                    if not path.endswith("_forecast.parquet"))
    return sorted(set(seqs))


@st.cache_data
def load(seq, split):
    players, forecast = clip_paths(seq, split)
    return pd.read_parquet(players), pd.read_parquet(forecast)


@st.cache_data
def shapes_for_clip(table):
    return team_shape_metrics(table)


def frame_path(seq, split, frame):
    return f"data/SoccerNetGS/{split}/SNGS-{seq}/img1/{frame:06d}.jpg"


def playback_path(seq, split):
    return f"{VIDEO_ROOT}/{split}/{seq}.mp4"


def composite_playback_path(seq, split):
    return f"{VIDEO_ROOT}/{split}/{seq}_composite.mp4"


def useful_forecast_frame(forecasts, fallback):
    required = {"anchor_frame", "x0", "y0", "h3x", "h3y"}
    if not required.issubset(forecasts.columns):
        return fallback
    table = forecasts.copy()
    table["d3"] = np.hypot(table.h3x - table.x0, table.h3y - table.y0)
    summary = table.groupby("anchor_frame").d3.agg(["count", "median"])
    useful = summary[(summary["count"] >= 3) & (summary["median"] >= 1.5)]
    return int(useful.index.min()) if len(useful) else fallback


with st.sidebar:
    st.markdown("## ⚽ TactPipeline")
    st.caption("Game video to team view")
    st.divider()
    split = st.radio("Data set", ["valid"] if DEMO_MODE else ["valid", "test"], horizontal=True)
    if DEMO_MODE:
        st.caption("Demo version")
    seqs = available_seqs(split)
    if not seqs:
        st.warning(f"No baked {split} clips found.")
        st.stop()
    seq = st.selectbox("Video", seqs, format_func=lambda value: f"SNGS-{value}")
    workspace = st.radio("View", ["Live view", "Video summary", "Watch video"])
    st.divider()
    press_src = st.radio("Pressing", ["press_pred", "press_gt"],
                        format_func=lambda value: "Model result" if value == "press_pred" else "Known label")
    show_ids = st.checkbox("Show player IDs", True)
    show_hull = st.checkbox("Show team shape", True)
    show_forecast = st.checkbox("Show future path", True)

df, fc = load(seq, split)
fmin, fmax = int(df.frame.min()), int(df.frame.max())
frames = list(range(fmin, fmax + 1))
forecast_anchors = set(fc.anchor_frame.astype(int))
initial_frame = useful_forecast_frame(fc, fmin)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Football team view</div>
  <h1>See the game, one frame at a time.</h1>
  <p>Follow players, team shape, pressing and future movement.</p>
</div>
""", unsafe_allow_html=True)

if workspace == "Watch video":
    st.markdown('<div class="section-title">Watch video</div>', unsafe_allow_html=True)
    playback_kind = st.radio("Video type", ["Player boxes", "Game video + pitch view"], horizontal=True)
    composite = playback_kind == "Game video + pitch view"
    video = composite_playback_path(seq, split) if composite else playback_path(seq, split)
    if os.path.exists(video):
        st.video(video)
        st.caption("The game video and pitch view move together.")
    else:
        st.info("This video is still being made.")
        suffix = " --composite" if composite else ""
        st.code(f".venv/bin/python dashboard/video.py {int(seq)} {split}{suffix}")
    st.stop()

if workspace == "Video summary":
    st.markdown('<div class="section-title">Video summary</div>', unsafe_allow_html=True)
    metric_table = shapes_for_clip(df)
    left, right = st.columns(2)
    with left:
        st.metric("Frames", f"{fmax - fmin + 1}", "30 seconds")
    with right:
        st.metric("Mean visible players / frame", f"{df.groupby('frame').size().mean():.1f}")
    analytics_view = st.radio("Choose a chart", ["Team movement", "Where teams played"], horizontal=True)
    if analytics_view == "Team movement":
        st.caption("See team width, depth, space and how far forward each team is.")
        st.pyplot(shape_timeline_figure(metric_table), use_container_width=True)
    else:
        st.caption("See where each team spent most of the video.")
        st.pyplot(occupancy_figure(df), use_container_width=True)
    st.stop()


def frame_label(value):
    return f"{value:03d}" + ("  • forecast available" if value in forecast_anchors else "")


st.markdown('<div class="section-title">Live view</div>', unsafe_allow_html=True)
frame = st.selectbox("Frame", frames, index=frames.index(initial_frame), format_func=frame_label,
                     help="Type a number to search. • means a future path is ready.")
players = df[df.frame == frame]
fc_anchor = fc[fc.anchor_frame == frame] if show_forecast else fc.iloc[0:0]

if len(fc_anchor):
    display = fc_anchor.copy()
    display["team"] = display.track_id.map(players.groupby("track_id").team.first())
    display["d3"] = np.hypot(display.h3x - display.x0, display.h3y - display.y0)
    with st.sidebar:
        st.divider()
        st.markdown("**Future path**")
        forecast_team = st.selectbox("Team", ["Both", "left", "right"])
        if forecast_team != "Both":
            display = display[display.team == forecast_team]
        forecast_mode = st.radio("Show", ["One player", "Top movers", "All"], horizontal=True)
        if forecast_mode == "One player" and len(display):
            ids = display.sort_values("d3", ascending=False).track_id.astype(int).tolist()
            selected_track = st.selectbox("Player track", ids, format_func=lambda value: f"Track {value}")
            display = display[display.track_id == selected_track]
        elif forecast_mode == "Top movers":
            display = display.nlargest(3, "d3")
    fc_display = display.drop(columns=["team", "d3"], errors="ignore")
else:
    fc_display = fc_anchor

metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
metric_a.metric("Frame", f"{frame} / {fmax}")
metric_b.metric("Visible players", len(players))
metric_c.metric("Future paths", len(fc_anchor))
pressing = {team: group[press_src].iloc[0] or "–" for team, group in players.groupby("team")}
metric_d.metric("Blue press", pressing.get("left", "–"))
metric_e.metric("Red press", pressing.get("right", "–"))

left, right = st.columns((1.05, 1), gap="large")
with left:
    st.caption("Game video — player boxes")
    image_file = frame_path(seq, split, frame)
    if os.path.exists(image_file):
        st.pyplot(broadcast(np.array(Image.open(image_file)), players, show_ids), use_container_width=True)
    else:
        st.info("Open Watch video to see the game video with the pitch view.")

with right:
    tactical_view = st.radio("Pitch view", ["Pitch", "Space map", "Player paths", "Player links", "Check future path"], horizontal=True)
    if tactical_view == "Pitch":
        st.caption("Team shape, pressing and future paths.")
        figure = minimap(players, fc_display, show_hull, show_forecast, press_src)
    elif tactical_view == "Space map":
        st.caption("Shows the player closest to each part of the pitch.")
        figure = minimap(players, show_hull=False, show_forecast=False, press_col=press_src, show_voronoi=True)
    elif tactical_view == "Player paths":
        seconds = st.slider("Past time (seconds)", 1, 5, 3)
        st.caption("Shows where players moved before this frame.")
        figure = minimap(players, show_hull=False, show_forecast=False, press_col=press_src,
                         trails=player_trails(df, frame, seconds))
    elif tactical_view == "Player links":
        neighbours = st.slider("Links per player", 1, 5, 3)
        st.caption("Shows which players are close to each other.")
        figure = minimap(players, show_hull=False, show_forecast=False, press_col=press_src,
                         edges=proximity_edges(players, neighbours))
    else:
        actual_future, errors = forecast_validation(df, fc_display)
        if not len(fc_display):
            st.info("Choose a frame marked • and turn on future paths to check a prediction.")
            figure = minimap(players, show_hull=show_hull, show_forecast=False, press_col=press_src)
        else:
            first, second, third = st.columns(3)
            first.metric("Average miss", f"{errors['ADE (available horizons)']:.2f} m" if np.isfinite(errors["ADE (available horizons)"]) else "–")
            second.metric("Miss after 3 sec", f"{errors['FDE (3 s)']:.2f} m" if np.isfinite(errors["FDE (3 s)"]) else "–")
            third.metric("Points found", f"{errors['available_horizons']} / 3")
            st.caption("Dashed line = future path. Gold line = where the player then moved.")
            figure = minimap(players, fc_display, show_hull=False, show_forecast=True,
                             press_col=press_src, actual_future=actual_future)
    st.pyplot(figure, use_container_width=True)
