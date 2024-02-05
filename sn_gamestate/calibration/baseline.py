from typing import Any

import numpy as np
import pandas as pd

from sn_calibration_baseline.baseline_cameras import (normalization_transform,
                                                      estimate_homography_from_line_correspondences,
                                                      Camera)
from sn_calibration_baseline.camera import unproject_image_point
from sn_calibration_baseline.soccerpitch import SoccerPitch
from tracklab.pipeline import ImageLevelModule


class BaselineCalibration(ImageLevelModule):
    input_columns = []
    output_columns = []

    def __init__(self, batch_size, resolution_width, resolution_height, **kwargs):
        super().__init__(batch_size)
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        image = batch
        field = SoccerPitch()
        predictions = metadatas["lines"][0]
        camera_predictions = dict()

        line_matches = []
        potential_3d_2d_matches = {}
        src_pts = []
        success = False
        for k, v in predictions.items():
            if k == 'Circle central' or "unknown" in k:
                continue
            P3D1 = field.line_extremities_keys[k][0]
            P3D2 = field.line_extremities_keys[k][1]
            p1 = np.array(
                [v[0]['x'] * self.resolution_width, v[0]['y'] * self.resolution_height,
                 1.])
            p2 = np.array(
                [v[1]['x'] * self.resolution_width, v[1]['y'] * self.resolution_height,
                 1.])
            src_pts.extend([p1, p2])
            if P3D1 in potential_3d_2d_matches.keys():
                potential_3d_2d_matches[P3D1].extend([p1, p2])
            else:
                potential_3d_2d_matches[P3D1] = [p1, p2]
            if P3D2 in potential_3d_2d_matches.keys():
                potential_3d_2d_matches[P3D2].extend([p1, p2])
            else:
                potential_3d_2d_matches[P3D2] = [p1, p2]

            start = (int(p1[0]), int(p1[1]))
            end = (int(p2[0]), int(p2[1]))
            # cv.line(cv_image, start, end, (0, 0, 255), 1)

            line = np.cross(p1, p2)
            if np.isnan(np.sum(line)) or np.isinf(np.sum(line)):
                continue
            line_pitch = field.get_2d_homogeneous_line(k)
            if line_pitch is not None:
                line_matches.append((line_pitch, line))

        if len(line_matches) >= 4:
            target_pts = [field.point_dict[k][:2] for k in
                          potential_3d_2d_matches.keys()]
            T1 = normalization_transform(target_pts)
            T2 = normalization_transform(src_pts)
            success, homography = estimate_homography_from_line_correspondences(
                line_matches, T1, T2)
            if success:
                # cv_image = draw_pitch_homography(cv_image, homography)

                cam = Camera(self.resolution_width, self.resolution_height)
                success = cam.from_homography(homography)
                if success:
                    point_matches = []
                    added_pts = set()
                    for k, potential_matches in potential_3d_2d_matches.items():
                        p3D = field.point_dict[k]
                        projected = cam.project_point(p3D)

                        if 0 < projected[0] < self.resolution_width and 0 < projected[
                            1] < self.resolution_height:
                            dist = np.zeros(len(potential_matches))
                            for i, potential_match in enumerate(potential_matches):
                                dist[i] = np.sqrt(
                                    (projected[0] - potential_match[0]) ** 2 + (
                                            projected[1] - potential_match[1]) ** 2)
                            selected = np.argmin(dist)
                            if dist[selected] < 100:
                                point_matches.append(
                                    (p3D, potential_matches[selected][:2]))

                    if len(point_matches) > 3:
                        cam.solve_pnp(point_matches)
                        # cam.draw_colorful_pitch(cv_image, SoccerField.palette)
                        # print(image_path)
                        # cv.imshow("colorful pitch", cv_image)
                        # cv.waitKey(0)

        if success:
            camera_predictions = cam.to_json_parameters()
            # confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_projection()
            detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(cam))
        else:
            camera_predictions = {}
            detections["bbox_pitch"] = None
        return detections[["bbox_pitch"]], pd.DataFrame([
            pd.Series({"parameters": camera_predictions}, name=metadatas.iloc[0].name)
        ])

def get_bbox_pitch(cam):
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = [l, b]
        br = [r, b]
        bm = [l+(r-l)/2, b]
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