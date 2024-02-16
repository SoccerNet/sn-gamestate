from typing import Any

import numpy as np
import pandas as pd

from sn_calibration_baseline.camera import Camera
from tracklab.pipeline import ImageLevelModule
from tracklab.utils.collate import Unbatchable, default_collate
import logging

log = logging.getLogger(__name__)


def collate_df(batch):
    idxs = [x[0] for x in batch]
    batch = [x[1] for x in batch]
    return idxs, batch


class Bbox2Pitch(ImageLevelModule):
    collate_fn = collate_df

    input_columns = dict(detection=["bbox_ltwh"],
                         image=["parameters"])
    output_columns = dict(detection=["bbox_pitch"],
                          image=[])

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)
        log.info(f"bbox2pitch: batch_size={batch_size}")

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        camera_parameters = metadata["parameters"]
        if isinstance(camera_parameters, dict):  # Camera parameters
            sn_cam = Camera(iwidth=image.shape[1], iheight=image.shape[0])
            sn_cam.from_json_parameters(camera_parameters)
            detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(sn_cam))
        elif isinstance(camera_parameters, (list, np.ndarray)):  # Homography
            detections["bbox_pitch"] = detections.bbox.ltrb().apply(
                get_bbox_pitch_homography(camera_parameters)
            )
        elif pd.isna(camera_parameters):
            log.warning(f"camera parameters were None/NA")
            return pd.DataFrame(columns=["bbox_pitch"])
        else:
            log.warning(f"camera parameters should be dict or list not {camera_parameters}")
            return pd.DataFrame(columns=["bbox_pitch"])

        return detections["bbox_pitch"]

    def process(self, batch, detections: pd.DataFrame, metadatas: pd.Series) -> Any:
        output_detections = []
        output_index = []
        for image_id, bbox_pitch in zip(metadatas.index, batch):
            image_detections = detections[detections.image_id == image_id]
            image_detections["bbox_pitch"] = bbox_pitch
            output_detections.extend(image_detections["bbox_pitch"])
            output_index.extend(image_detections.index)

        return pd.DataFrame({"bbox_pitch": output_detections}, index=output_index)


def get_bbox_pitch(cam):
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l+(r-l)/2, b, 1])

        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox

def get_bbox_pitch_homography(homography):
    try:
        hinv = np.linalg.inv(homography)
        def _get_bbox(bbox_ltrb):
            l, t, r, b = bbox_ltrb
            bl = np.array([l, b, 1])
            br = np.array([r, b, 1])
            bm = np.array([l + (r - l) / 2, b, 1])
            bird_lower_right = hinv @ br
            bird_lower_right /= bird_lower_right[2]
            bird_lower_left = hinv @ bl
            bird_lower_left /= bird_lower_left[2]
            bird_lower_middle = hinv @ bm
            bird_lower_middle /= bird_lower_middle[2]
            return {
                "x_bottom_left": bird_lower_left[0], "y_bottom_left": bird_lower_left[1],
                "x_bottom_right": bird_lower_right[0], "y_bottom_right": bird_lower_right[1],
                "x_bottom_middle": bird_lower_middle[0], "y_bottom_middle": bird_lower_middle[1],
            }
        return _get_bbox
    except np.linalg.LinAlgError:
        return lambda x: None