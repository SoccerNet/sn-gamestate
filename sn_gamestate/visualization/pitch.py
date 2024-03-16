import logging
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sn_calibration_baseline.soccerpitch import SoccerPitch
from tracklab.utils.cv2 import draw_text

log = logging.getLogger(__name__)

pitch_file = Path(__file__).parent / "Radar.png"


def draw_pitch(patch, detections_pred, detections_gt,
               image_pred, image_gt,
               line_thickness=3,
               pitch_scale=3,
               pitch_image=None,
               draw_topview=True,
               ):

    # Draw the lines on the image pitch
    if "lines" in image_pred:
        image_height, image_width, _ = patch.shape
        for name, line in image_pred["lines"].items():
            if name == "Circle central" and len(line) > 4:
                points = np.array([(int(p["x"] * image_width), int(p["y"]*image_height)) for p in line])
                try:
                    ellipse = cv2.fitEllipse(points)
                    cv2.ellipse(patch, ellipse, color=SoccerPitch.palette[name],
                                thickness=line_thickness)
                except cv2.error:
                    log.warning("Could not draw ellipse")

            else:
                for j in np.arange(len(line)-1):
                    cv2.line(
                        patch,
                        (int(line[j]["x"] * image_width), int(line[j]["y"] * image_height)),
                        (int(line[j+1]["x"] * image_width), int(line[j+1]["y"] * image_height)),
                        color=SoccerPitch.palette[name],
                        thickness=line_thickness,  # TODO : make this a parameter
                    )

    # Draw the Top-view pitch
    if draw_topview:
        if detections_gt is not None and "bbox_pitch" in detections_gt:
            draw_radar_view(patch, detections_gt, scale=pitch_scale, group="ground truth")
        if detections_pred is not None and "bbox_pitch" in detections_pred:
            draw_radar_view(patch, detections_pred, scale=pitch_scale, group="predictions")


def draw_radar_view(patch, detections, scale, delta=32, group="ground truth"):
    pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
    pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
    sign = -1 if group == "ground truth" else +1
    radar_center_x = int(1920/2 - pitch_width * scale / 2 * sign - delta * sign)
    radar_center_y = int(1080 - pitch_height * scale / 2)
    radar_top_x = int(radar_center_x - pitch_width * scale / 2)
    radar_top_y = int(1080 - pitch_height * scale)
    radar_width = int(pitch_width * scale)
    radar_height = int(pitch_height * scale)
    if pitch_file is not None:
        radar_img = cv2.resize(cv2.imread(str(pitch_file)), (pitch_width * scale, pitch_height * scale))
        radar_img = cv2.bitwise_not(radar_img)
        cv2.line(radar_img, (0, 0), (0, radar_img.shape[0]), thickness=6, color=(0, 0, 255))
        cv2.line(radar_img, (radar_img.shape[1], 0), (radar_img.shape[1], radar_img.shape[0]), thickness=6,
                 color=(255, 0, 0))
    else:
        radar_img = np.ones((pitch_height * scale, pitch_width * scale, 3)) * 255

    alpha = 0.3
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :], 1-alpha, radar_img, alpha, 0.0)
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :], 1-alpha, radar_img, alpha, 0.0)
    draw_text(
        patch,
        group,
        (radar_center_x, radar_top_y-10),
        1, 3, 2,
        color_txt=(255, 255, 255),
        alignH="c",
        alignV="b",
    )
    for name, detection in detections.iterrows():
        if "role" in detection and detection.role == "ball":
            continue
        if "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
        bbox_name = "bbox_pitch"
        if not isinstance(detection[bbox_name], dict):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        cat = None
        if "jersey_number" in detection and detection.jersey_number is not None:
            if "role" in detection and detection.role == "player":
                if isinstance(detection.jersey_number, float) and np.isnan(detection.jersey_number):
                    cat = None
                else:
                    cat = f"{int(detection.jersey_number)}"

        if "role" in detection:
            if detection.role == "goalkeeper":
                cat = "GK"
            elif detection.role == "referee":
                cat = "RE"
                color = (238, 210, 2)
            elif detection.role == "other":
                cat = "OT"
                color = (0, 255, 0)
        if cat is not None:
            draw_text(
                patch,
                cat,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                1,
                0.2*scale,
                1,
                color_txt=color,
                alignH="c",
                alignV="c",
            )
        else:
            cv2.circle(
                patch,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                scale,
                color=color,
                thickness=-1
            )


def draw_radar_view_matplotlib(file_path, detections):
    pitch_file = Path(__file__).parent / "Football_field.png"
    scale = 1
    delta = 32
    pitch_width = 111  # 105 + 2 * 10
    pitch_height = 74  # 68 + 2 * 5

    radar_width = int(pitch_width)
    radar_height = int(pitch_height)
    radar_img = cv2.imread(str(pitch_file))
    # radar_img = cv2.bitwise_not(radar_img)
    radar_img[1][radar_img[1] == 255] = 0

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#2b9e00")
    ax.imshow(radar_img,
              extent=(-radar_width / 2, radar_width / 2, radar_height / 2, -radar_height / 2),
              interpolation="none", origin="lower")

    bbox_name = "bbox_pitch"
    x_middles = defaultdict(list)
    y_middles = defaultdict(list)
    color_dict = {"left": "#34409B", "right": "#ffffff", "referee": "yellow",
                  "other": "red"}
    all_xs = []
    all_ys = []
    all_notes = []
    all_colors = []
    notes = defaultdict(list)
    teams = []
    for name, detection in detections.iterrows():
        if detection[bbox_name] is None:
            continue
        team = detection["team"]
        role = detection["role"]
        if role == "ball" or pd.isnull(role):
            continue
        if pd.isnull(detection[bbox_name]):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = -np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        teams.append("<" if detection["team"] == "left" else ">")
        if role == "player":
            note = detection["jersey_number"]
            if pd.isnull(note):
                note = "?"  # team[:1].upper()
        elif role == "goalkeeper":
            note = "GK"
        elif role == "referee":
            note = "R"
        else:
            note = role[:2]

        all_xs.append(x_middle)
        all_ys.append(y_middle)
        all_notes.append(note)

        if team == "left" or team == "right":
            x_middles[team].append(x_middle)
            y_middles[team].append(y_middle)
            notes[team].append(note)
            color = color_dict[team]
        elif role is not None:
            x_middles[role].append(x_middle)
            y_middles[role].append(y_middle)
            notes[role].append(note)
            color = color_dict[role]
        else:
            continue
        all_colors.append(color)
    for (name, xs), (_, ys), (_, tnotes) in zip(x_middles.items(), y_middles.items(),
                                                notes.items()):
        marker = {"left": "o", "right": "o", "referee": "s", "other": "X"}[name]
        color = color_dict[name]
        alpha = 1
        for i, note in enumerate(tnotes):
            ax.scatter(xs[i], ys[i], marker=marker, alpha=1, s=20, linewidths=0.3,
                       facecolors=color, edgecolors='black')
            ax.annotate(note, (xs[i], ys[i]), ha="center",  # va="center",
                        alpha=alpha, xytext=(0, 4), fontsize=16, fontweight=500,
                        color=color, annotation_clip=True, clip_on=True,
                        textcoords='offset points')

    ax.set_xlim(-radar_width / 2 - 3, radar_width / 2 + 3)
    ax.set_ylim(-radar_height / 2 - 3, radar_height / 2 + 3)
    handles = [mpatches.Patch(color=color, label=name) for name, color in
               color_dict.items()]
    plt.legend(handles=handles, ncols=4, fontsize="x-small", framealpha=0, borderpad=0,
               labelcolor="white")
    plt.axis('off')
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
