"""Build annotated broadcast playback videos for the Streamlit dashboard."""
import sys
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Support both `python -m dashboard.video` and `python dashboard/video.py`.
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard.render import _player_label, minimap

TEAM_BGR = {"left": (225, 130, 35), "right": (45, 55, 225)}


def frame_path(seq, split, frame, dataset_root="data/SoccerNetGS"):
    return Path(dataset_root) / split / f"SNGS-{int(seq):03d}" / "img1" / f"{int(frame):06d}.jpg"


def render_broadcast_frame(image, players, frame_number):
    """Overlay tracker and tactical predictions on one BGR broadcast frame."""
    canvas = image.copy()
    for player in players.itertuples():
        color = TEAM_BGR.get(player.team, (230, 230, 230))
        x, y, w, h = (int(round(v)) for v in (player.bx, player.by, player.bw, player.bh))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, _player_label(player), (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    press = []
    for team, group in players.groupby("team"):
        value = group.press_pred.iloc[0] if "press_pred" in group and len(group) else ""
        if value:
            press.append(f"{team}: {value}")
    header = f"Frame {frame_number:04d}" + ("  |  " + "  |  ".join(press) if press else "")
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 560), 36), (18, 35, 28), -1)
    cv2.putText(canvas, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (245, 250, 246), 1, cv2.LINE_AA)
    return canvas


def render_composite_frame(image, players, forecasts, frame_number):
    """Place annotated broadcast and the matching tactical minimap side by side."""
    broadcast_frame = render_broadcast_frame(image, players, frame_number)
    figure = minimap(players, forecasts, show_hull=True, show_forecast=True, press_col="press_pred")
    figure.canvas.draw()
    map_rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    plt.close(figure)
    map_bgr = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
    height = min(broadcast_frame.shape[0], map_bgr.shape[0])
    broadcast_width = int(round(broadcast_frame.shape[1] * height / broadcast_frame.shape[0]))
    map_width = int(round(map_bgr.shape[1] * height / map_bgr.shape[0]))
    broadcast_frame = cv2.resize(broadcast_frame, (broadcast_width, height), interpolation=cv2.INTER_AREA)
    map_bgr = cv2.resize(map_bgr, (map_width, height), interpolation=cv2.INTER_AREA)
    gutter = np.full((height, 10, 3), (15, 37, 28), dtype=np.uint8)
    return np.hstack((broadcast_frame, gutter, map_bgr))


def build_video(seq, split="valid", output_path=None, data_root="dashboard/data", dataset_root="data/SoccerNetGS", fps=25):
    """Create an annotated MP4 for one baked dashboard clip."""
    seq = int(seq)
    table_path = Path(data_root) / split / f"{seq:03d}.parquet"
    if not table_path.exists() and split == "valid":
        table_path = Path(data_root) / f"{seq:03d}.parquet"  # legacy single-clip bake
    table = pd.read_parquet(table_path)
    output_path = Path(output_path or Path("dashboard/videos") / split / f"{seq:03d}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_image = cv2.imread(str(frame_path(seq, split, int(table.frame.min()), dataset_root)))
    if first_image is None:
        raise FileNotFoundError(f"No broadcast frames found for {split}/SNGS-{seq:03d}")
    height, width = first_image.shape[:2]
    intermediate_path = output_path.with_suffix(".intermediate.mp4")
    writer = cv2.VideoWriter(str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate_path}")

    try:
        for frame_number, players in table.groupby("frame", sort=True):
            image = cv2.imread(str(frame_path(seq, split, frame_number, dataset_root)))
            if image is not None:
                writer.write(render_broadcast_frame(image, players, int(frame_number)))
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode browser-compatible H.264 playback")
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(intermediate_path), "-c:v", "libx264",
                        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(output_path)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    finally:
        intermediate_path.unlink(missing_ok=True)
    return output_path


def build_composite_video(seq, split="valid", output_path=None, data_root="dashboard/data", dataset_root="data/SoccerNetGS", fps=25,
                          frame_step=2):
    """Build the thesis-demo video: broadcast plus synchronized minimap."""
    seq = int(seq)
    table_path = Path(data_root) / split / f"{seq:03d}.parquet"
    forecast_path = Path(data_root) / split / f"{seq:03d}_forecast.parquet"
    if not table_path.exists() and split == "valid":
        table_path = Path(data_root) / f"{seq:03d}.parquet"
        forecast_path = Path(data_root) / f"{seq:03d}_forecast.parquet"
    table, forecast = pd.read_parquet(table_path), pd.read_parquet(forecast_path)
    output_path = Path(output_path or Path("dashboard/videos") / split / f"{seq:03d}_composite.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = int(table.frame.min())
    first_image = cv2.imread(str(frame_path(seq, split, first_frame, dataset_root)))
    if first_image is None:
        raise FileNotFoundError(f"No broadcast frames found for {split}/SNGS-{seq:03d}")
    first_players = table[table.frame == first_frame]
    sample = render_composite_frame(first_image, first_players,
                                    forecast[forecast.anchor_frame == first_frame], first_frame)
    height, width = sample.shape[:2]
    intermediate_path = output_path.with_suffix(".intermediate.mp4")
    writer = cv2.VideoWriter(str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / frame_step, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate_path}")
    try:
        for frame_number, players in table.groupby("frame", sort=True):
            if (int(frame_number) - first_frame) % frame_step:
                continue
            image = cv2.imread(str(frame_path(seq, split, frame_number, dataset_root)))
            if image is not None:
                forecasts = forecast[forecast.anchor_frame == frame_number]
                writer.write(render_composite_frame(image, players, forecasts, int(frame_number)))
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode browser-compatible H.264 playback")
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(intermediate_path), "-c:v", "libx264",
                        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(output_path)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    finally:
        intermediate_path.unlink(missing_ok=True)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build an annotated dashboard playback video")
    parser.add_argument("seq", type=int)
    parser.add_argument("split", nargs="?", default="valid", choices=("valid", "test"))
    parser.add_argument("--composite", action="store_true", help="render broadcast + synchronized minimap")
    parser.add_argument("--frame-step", type=int, default=2, help="render every Nth frame for composite video")
    args = parser.parse_args()
    print(build_composite_video(args.seq, args.split, frame_step=args.frame_step)
          if args.composite else build_video(args.seq, args.split))
