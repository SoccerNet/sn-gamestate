from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from sn_calibration_baseline.camera import Camera
from sn_calibration_baseline.detect_extremities import generate_class_synthesis, join_points
from tracklab.pipeline import ImageLevelModule
from tracklab.utils.download import download_file
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from tvcalib.inference import InferenceSegmentationModel, InferenceDatasetCalibration
from torchvision.models.segmentation import deeplabv3_resnet101
from tvcalib.module import TVCalibModule
from sn_calibration_baseline.soccerpitch import SoccerPitch
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, \
    SoccerPitchSNCircleCentralSplit


class TVCalib_Segmentation(ImageLevelModule):
    input_columns = {
        "image": [],
        "detection": [],
    }
    output_columns = {
        "image": ["lines"],
        "detection": []
    }

    def __init__(self, checkpoint, image_width, image_height, batch_size, device, **kwargs):
        super().__init__(batch_size)
        self.device = device
        self.model = deeplabv3_resnet101(
            num_classes=len(SoccerPitch.lines_classes) + 1, aux_loss=True
        )
        if Path(checkpoint).name == "train_59.pt":
            md5 = "c89ab863a12822b0e3a87cd6eebe7cae"
            download_file("https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt",
                          checkpoint, md5)
        self.model.load_state_dict(torch.load(checkpoint)["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.image_width = image_width
        self.image_height = image_height
        self.tfms = T.Compose(
            [
                T.Resize(256),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
        self.fn_get_line_extremities = partial(get_line_extremities, maxdist=30, width=455,
                                          height=256, num_points_lines=4,
                                          num_points_circles=8)

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        image = Image.fromarray(image).convert("RGB")
        image = self.tfms(image)
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        result = self.model(batch.to(self.device))
        sem_lines = result["out"].argmax(1).cpu().detach().numpy().astype(np.uint8)
        keypoints_raw = []
        for sem_line in sem_lines:
            skeleton = self.fn_generate_class_synthesis(sem_line)
            keypoints_raw.append(self.fn_get_line_extremities(skeleton))
        new_metadatas = pd.DataFrame(
            {
                "lines": keypoints_raw,
            },
            index=metadatas.index
        )
        return pd.DataFrame(), new_metadatas


class TVCalib(ImageLevelModule):
    input_columns = {
        "image": ["lines"],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        "image": ["parameters"],
        "detection": ["bbox_pitch"],
    }

    def __init__(self, image_width, image_height, lens_dist, optim_steps, batch_size, device, **kwargs):
        super().__init__(batch_size)
        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.object3d = SoccerPitchLineCircleSegments(
            device=device, base_field=SoccerPitchSNCircleCentralSplit()
        )
        self.model = TVCalibModule(
                        self.object3d,
                        get_cam_distr(1.96, batch_size, 1),
                        get_dist_distr(batch_size, 1) if lens_dist else None,
                        (image_height, image_width),
                        optim_steps,
                        device,
                        log_per_step=False,
                        tqdm_kwqargs={"disable": True},
                    )

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        keypoints_raw = metadata["lines"]
        per_sample_output = InferenceDatasetCalibration.prepare_per_sample(
            keypoints_raw,
            self.object3d,
            4, 8,
            self.image_width,
            self.image_height,
            0.0
        )
        for k in per_sample_output.keys():
            per_sample_output[k] = per_sample_output[k].unsqueeze(0)
        return per_sample_output

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        _batch_size = batch["lines__ndc_projected_selection_shuffled"].shape[0]
        per_sample_loss, cam, _ = self.model.self_optim_batch(batch)
        output_dict = detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss})
        for k in output_dict.keys():
            output_dict[k] = [x for x in output_dict[k].squeeze(1)]
        output_df = pd.DataFrame(output_dict)
        output_detections = []
        output_index = []
        camera_predictions = []
        for idx, params in output_df.iterrows():
            sn_cam = Camera(iwidth=self.image_width, iheight=self.image_height)
            # homography.append(params["homography"].numpy())
            sn_cam.from_json_parameters(params.to_dict())
            # sn_cam.set_camera(
            #     pan=params.pan_degrees, tilt=params.tilt_degrees, roll=params.roll_degrees,
            #     xfocal=params.x_focal_length, yfocal=params.y_focal_length,
            #     principal_point=params.principal_point,
            #     pos_x=params.position_meters[0], pos_y=params.position_meters[1],
            #     pos_z=params.position_meters[2]
            # )
            camera_predictions.append(sn_cam.to_json_parameters())
            image_detections = detections[detections.image_id == metadatas.iloc[idx].name]
            image_detections["bbox_pitch"] = image_detections.bbox.ltrb().apply(get_bbox_pitch(sn_cam))
            output_detections.extend(image_detections["bbox_pitch"])
            output_index.extend(image_detections.index)
        return pd.DataFrame({
            "bbox_pitch": output_detections
        },
            index=output_index
        ), pd.DataFrame({"parameters": camera_predictions}, index=metadatas.index)


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

def get_line_extremities(buckets, maxdist, width, height, num_points_lines,
                         num_points_circles):
    """
    Given the dictionary {lines_class: points}, finds plausible extremities of each line, i.e the extremities
    of the longest polyline that can be built on the class blobs,  and normalize its coordinates
    by the image size.
    :param buckets: The dictionary associating line classes to the set of circle centers that covers best the class
    prediction blobs in the segmentation mask
    :param maxdist: the maximal distance between two circle centers belonging to the same blob (heuristic)
    :param width: image width
    :param height: image height
    :return: a dictionary associating to each class its extremities
    """
    extremities = dict()
    for class_name, disks_list in buckets.items():
        polyline_list = join_points(disks_list, maxdist)
        max_len = 0
        longest_polyline = []
        for polyline in polyline_list:
            if len(polyline) > max_len:
                max_len = len(polyline)
                longest_polyline = polyline
        extremities[class_name] = [
            {'x': longest_polyline[0][1] / width, 'y': longest_polyline[0][0] / height},
            {'x': longest_polyline[-1][1] / width,
             'y': longest_polyline[-1][0] / height},

        ]
        num_points = num_points_lines
        if "Circle" in class_name:
            num_points = num_points_circles
        if num_points > 2:
            # equally spaced points along the longest polyline
            # skip first and last as they already exist
            for i in range(1, num_points - 1):
                extremities[class_name].insert(
                    len(extremities[class_name]) - 1,
                    {'x': longest_polyline[i * int(len(longest_polyline) / num_points)][
                              1] / width,
                     'y': longest_polyline[i * int(len(longest_polyline) / num_points)][
                              0] / height}
                )

    return extremities