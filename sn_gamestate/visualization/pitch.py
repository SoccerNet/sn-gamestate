import logging

import cv2
import numpy as np
from pathlib import Path

from sn_calibration_baseline.soccerpitch import SoccerPitch
from tracklab.utils.cv2 import draw_text

log = logging.getLogger(__name__)

pitch_file = Path(__file__).parent / "Radar.png"


def draw_pitch(patch, detections_pred, detections_gt,
               image_pred, image_gt,
               line_thickness=3,
               pitch_scale=3,
               pitch_image=None,
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
    if detections_gt is not None and "bbox_pitch" in detections_gt:
        draw_radar_view(patch, detections_gt, scale=pitch_scale, group="ground truth")
    if detections_pred is not None and "bbox_pitch" in detections_pred:
        draw_radar_view(patch, detections_pred, scale=pitch_scale, group="predictions")


# line_keypoints_match = {"Big rect. left bottom": [24, 25, 68],
#                         "Big rect. left main": [5, 25, 31, 34, 46, 64, 66],
#                         "Big rect. left top": [4, 5, 62],
#                         "Big rect. right bottom": [26, 27, 69],
#                         "Big rect. right main": [6, 26, 33, 36, 56, 65, 67],
#                         "Big rect. right top": [6, 7, 63],
#                         "Circle central": [32, 35, 38, 39, 42, 43, 48, 49, 50, 51, 52, 53, 54],
#                         "Circle left": [31, 34, 37, 41, 47],
#                         "Circle right": [33, 36, 40, 44, 55],
#                         "Goal left crossbar": [16, 12],
#                         "Goal left post left ": [16, 17],
#                         "Goal left post right": [12, 13],
#                         "Goal right crossbar": [15, 19],
#                         "Goal right post left": [15, 14],
#                         "Goal right post right": [19, 18],
#                         "Middle line": [2, 29, 32, 35, 51],
#                         "Side line bottom": [28, 29, 30, 70, 71, 72, 73],
#                         "Side line left": [1, 4, 8, 13, 17, 20, 24, 28],
#                         "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
#                         "Side line top": [1, 2, 3, 58, 59, 60, 61],
#                         "Small rect. left bottom": [20, 21],
#                         "Small rect. left main": [9, 21],
#                         "Small rect. left top": [8, 9],
#                         "Small rect. right bottom": [22, 23],
#                         "Small rect. right main": [10, 22],
#                         "Small rect. right top": [10, 11]}
#
#
# lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
#                 [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
#                 [[16.5, 13.84, 0.], [0., 13.84, 0.]],
#                 [[88.5, 54.16, 0.], [105., 54.16, 0.]],
#                 [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
#                 [[88.5, 13.84, 0.], [105., 13.84, 0.]],
#                 [[0., 37.66, -2.44], [0., 30.34, -2.44]],
#                 [[0., 37.66, 0.], [0., 37.66, -2.44]],
#                 [[0., 30.34, 0.], [0., 30.34, -2.44]],
#                 [[105., 37.66, -2.44], [105., 30.34, -2.44]],
#                 [[105., 30.34, 0.], [105., 30.34, -2.44]],
#                 [[105., 37.66, 0.], [105., 37.66, -2.44]],
#                 [[52.5, 0., 0.], [52.5, 68, 0.]],
#                 [[0., 68., 0.], [105., 68., 0.]],
#                 [[0., 0., 0.], [0., 68., 0.]],
#                 [[105., 0., 0.], [105., 68., 0.]],
#                 [[0., 0., 0.], [105., 0., 0.]],
#                 [[0., 43.16, 0.], [5.5, 43.16, 0.]],
#                 [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
#                 [[5.5, 24.84, 0.], [0., 24.84, 0.]],
#                 [[99.5, 43.16, 0.], [105., 43.16, 0.]],
#                 [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
#                 [[99.5, 24.84, 0.], [105., 24.84, 0.]]]
#
#
#
#
# def draw_pitch(patch, detections_pred, detections_gt,
#                image_pred, image_gt,
#                line_thickness=3,
#                pitch_scale=3,
#                pitch_image=None,
#                ):
#
#     #                 )
#     if "keypoints" in image_pred:
#         image_height, image_width, _ = patch.shape
#
#     for kp in image_pred["keypoints"].keys():
#         cv2.circle(patch, (int(image_pred["keypoints"][kp]["x"]),
#                            int(image_pred["keypoints"][kp]["y"])), radius=10,
#                            color=(0, 0, 255), thickness=-1)
#
#     # Draw the Top-view pitch
#     if detections_gt is not None and "bbox_pitch" in detections_gt:
#         draw_radar_view(patch, detections_gt, scale=pitch_scale, group="ground truth")
#     if detections_pred is not None and "bbox_pitch" in detections_pred:
#         draw_radar_view(patch, detections_pred, scale=pitch_scale, group="predictions")

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