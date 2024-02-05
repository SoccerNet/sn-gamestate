#%%
import random
from pathlib import Path
from typing import Union

import json
import numpy as np
import kornia
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm.auto import tqdm


from SoccerNet.Evaluation.utils_calibration import SoccerPitch
from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.sncalib_dataset import split_circle_central


def load_annotated_points(hparams, image_id, object3d):

    # load annotations from file
    file_points_raw = (
        Path(hparams["dataset"]["extremities_annotations"])
        / f'{hparams["dataset"]["extremities_prefix"]}{image_id}.json'
    )
    if not file_points_raw.exists():
        raise FileNotFoundError(file_points_raw)
    with open(file_points_raw) as fr:
        keypoints_raw = json.load(fr)

    # select according to pre-processing
    x_dict = InferenceDatasetCalibration.prepare_per_sample(
        keypoints_raw,
        object3d,
        hparams["dataset"]["num_points_on_line_segments"],
        hparams["dataset"]["num_points_on_circle_segments"],
        hparams["image_width"],
        hparams["image_height"],
    )
    points_line = x_dict["lines__px_projected_selection_shuffled"]
    points_circle = x_dict["circles__px_projected_selection_shuffled"]
    print(f"{points_line.shape=}, {points_circle.shape=}")
    return points_line, points_circle


def get_camera_from_per_sample_output(sample: dict, lens_dist: bool):
    phi_dict = {
        k: torch.Tensor(v).unsqueeze(0)
        for k, v in {
            "aov": [sample["aov_radian"]],
            "c_x": [sample["position_meters"][0]],
            "c_y": [sample["position_meters"][1]],
            "c_z": [sample["position_meters"][2]],
            "pan": np.deg2rad([sample["pan_degrees"]]),
            "tilt": np.deg2rad([sample["tilt_degrees"]]),
            "roll": np.deg2rad([sample["roll_degrees"]]),
        }.items()
    }

    psi = torch.tensor(sample["radial_distortion"][:2]) if lens_dist else None
    image_width = (
        int(sample["principal_point"][0] * 2)
        if "image_width" not in sample
        else sample["image_width"]
    )
    image_height = (
        int(sample["principal_point"][1] * 2)
        if "image_height" not in sample
        else sample["image_height"]
    )
    return SNProjectiveCamera(
        phi_dict,
        psi,
        sample["principal_point"],
        image_width,
        image_height,
        nan_check=False,
    )


class InferenceSegmentationModel:
    def __init__(self, checkpoint: Union[str, Path], device: str) -> None:
        self.device = device
        self.model = deeplabv3_resnet101(
            num_classes=len(SoccerPitch.lines_classes) + 1, aux_loss=True
        )
        self.model.load_state_dict(torch.load(checkpoint)["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def inference(self, img_batch):
        return self.model(img_batch)["out"].argmax(1)


class InferenceDatasetSegmentation(torch.utils.data.Dataset):
    def __init__(
        self,
        images_path,
        image_width,
        image_height,
        allowed_image_ext=set([".jpg", ".jpeg", ".png"]),
    ):
        super().__init__()
        self.tfms = T.Compose(
            [
                T.Resize(256),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.width = image_width
        self.height = image_height
        self.image_files = [
            f for f in Path(images_path).glob("**/*") if f.suffix in allowed_image_ext
        ]

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        image_raw_resized = T.functional.to_tensor(image.copy().resize((self.width, self.height)))
        image = self.tfms(image)

        return {
            "image_raw": image_raw_resized,
            "image": image,
            "image_id": self.image_files[idx].name,
        }

    def __len__(self):
        return len(self.image_files)


class InferenceDatasetCalibration(torch.utils.data.Dataset):
    def __init__(self, keypoints_raw, image_width_source, image_height_source, object3d) -> None:
        super().__init__()
        self.keypoints_raw = keypoints_raw
        self.w = image_width_source
        self.h = image_height_source
        self.object3d = object3d
        self.split_circle_central = True

    def __getitem__(self, idx):

        keypoints_dict = self.keypoints_raw[idx]
        if self.split_circle_central:
            keypoints_dict = split_circle_central(keypoints_dict)
        # add empty entries for non-visible segments
        for l in self.object3d.segment_names:
            if l not in keypoints_dict:
                keypoints_dict[l] = []

        per_sample_output = self.prepare_per_sample(
            keypoints_dict, self.object3d, 4, 8, self.w, self.h, pad_pixel_position_xy=0.0
        )
        for k in per_sample_output.keys():
            per_sample_output[k] = per_sample_output[k].unsqueeze(0)

        return per_sample_output

    def __len__(self):
        return len(self.keypoints_raw)

    @staticmethod
    def prepare_per_sample(
        keypoints_raw: dict,
        model3d,
        num_points_on_line_segments: int,
        num_points_on_circle_segments: int,
        image_width_source: int,
        image_height_source: int,
        pad_pixel_position_xy=0.0,
    ):
        r = {}
        pixel_stacked = {}
        for label, points in keypoints_raw.items():

            num_points_selection = num_points_on_line_segments
            if "Circle" in label:
                num_points_selection = num_points_on_circle_segments

            # rand select num_points_selection
            if num_points_selection > len(points):
                points_sel = points
            else:
                # random sample without replacement
                points_sel = random.sample(points, k=num_points_selection)

            if len(points_sel) > 0:
                xx = torch.tensor([a["x"] for a in points_sel])
                yy = torch.tensor([a["y"] for a in points_sel])
                pixel_stacked[label] = torch.stack([xx, yy], dim=-1)  # (?, 2)
                # scale pixel annotations from [0, 1] range to source image resolution
                # as this ranges from [1, {image_height, image_width}] shift pixel one left
                pixel_stacked[label][:, 0] = pixel_stacked[label][:, 0] * (image_width_source - 1)
                pixel_stacked[label][:, 1] = pixel_stacked[label][:, 1] * (image_height_source - 1)

        for segment_type, num_segments, segment_names in [
            ("lines", model3d.line_segments.shape[1], model3d.line_segments_names),
            ("circles", model3d.circle_segments.shape[1], model3d.circle_segments_names),
        ]:

            num_points_selection = num_points_on_line_segments
            if segment_type == "circles":
                num_points_selection = num_points_on_circle_segments
            px_projected_selection = (
                torch.zeros((num_segments, num_points_selection, 2)) + pad_pixel_position_xy
            )
            for segment_index, label in enumerate(segment_names):
                if label in pixel_stacked:
                    # set annotations to first positions
                    px_projected_selection[
                        segment_index, : pixel_stacked[label].shape[0], :
                    ] = pixel_stacked[label]

            randperm = torch.randperm(num_points_selection)
            px_projected_selection_shuffled = px_projected_selection.clone()
            px_projected_selection_shuffled[:, :, 0] = px_projected_selection_shuffled[
                :, randperm, 0
            ]
            px_projected_selection_shuffled[:, :, 1] = px_projected_selection_shuffled[
                :, randperm, 1
            ]

            is_keypoint_mask = (
                (0.0 <= px_projected_selection_shuffled[:, :, 0])
                & (px_projected_selection_shuffled[:, :, 0] < image_width_source)
            ) & (
                (0 < px_projected_selection_shuffled[:, :, 1])
                & (px_projected_selection_shuffled[:, :, 1] < image_height_source)
            )

            r[f"{segment_type}__is_keypoint_mask"] = is_keypoint_mask.unsqueeze(0)

            # reshape from (num_segments, num_points_selection, 2) to (3, num_segments, num_points_selection)
            px_projected_selection_shuffled = (
                kornia.geometry.conversions.convert_points_to_homogeneous(
                    px_projected_selection_shuffled
                )
            )
            px_projected_selection_shuffled = px_projected_selection_shuffled.view(
                num_segments * num_points_selection, 3
            )
            px_projected_selection_shuffled = px_projected_selection_shuffled.transpose(0, 1)
            px_projected_selection_shuffled = px_projected_selection_shuffled.view(
                3, num_segments, num_points_selection
            )
            # (3, num_segments, num_points_selection)
            r[f"{segment_type}__px_projected_selection_shuffled"] = px_projected_selection_shuffled

            ndc_projected_selection_shuffled = px_projected_selection_shuffled.clone()
            ndc_projected_selection_shuffled[0] = (
                ndc_projected_selection_shuffled[0] / image_width_source
            )
            ndc_projected_selection_shuffled[1] = (
                ndc_projected_selection_shuffled[1] / image_height_source
            )
            ndc_projected_selection_shuffled[1] = ndc_projected_selection_shuffled[1] * 2.0 - 1
            ndc_projected_selection_shuffled[0] = ndc_projected_selection_shuffled[0] * 2.0 - 1
            r[
                f"{segment_type}__ndc_projected_selection_shuffled"
            ] = ndc_projected_selection_shuffled

        return r
