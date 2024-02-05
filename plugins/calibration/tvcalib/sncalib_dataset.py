from pathlib import Path
import re
from typing import Union
import numpy as np
import pandas as pd
import json
import random
import torch
import kornia
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torch._six import string_classes
import collections
from operator import itemgetter


def split_circle_central(keypoints_dict):
    # split "circle central" in  "circle central left" and "circle central right"

    # assume main camera --> TODO behind the goal camera
    if "Circle central" in keypoints_dict:
        points_circle_central_left = []
        points_circle_central_right = []

        if "Middle line" in keypoints_dict:
            p_index_ymin, _ = min(
                enumerate([p["y"] for p in keypoints_dict["Middle line"]]),
                key=itemgetter(1),
            )
            p_index_ymax, _ = max(
                enumerate([p["y"] for p in keypoints_dict["Middle line"]]),
                key=itemgetter(1),
            )
            p_ymin = keypoints_dict["Middle line"][p_index_ymin]
            p_ymax = keypoints_dict["Middle line"][p_index_ymax]
            p_xmean = (p_ymin["x"] + p_ymax["x"]) / 2

            points_circle_central = keypoints_dict["Circle central"]
            for p in points_circle_central:
                if p["x"] < p_xmean:
                    points_circle_central_left.append(p)
                else:
                    points_circle_central_right.append(p)
        else:
            # circle is partly shown on the left or right side of the image
            # mean position is shown on the left part of the image --> label right
            circle_x = [p["x"] for p in keypoints_dict["Circle central"]]
            mean_x_circle = sum(circle_x) / len(circle_x)
            if mean_x_circle < 0.5:
                points_circle_central_right = keypoints_dict["Circle central"]
            else:
                points_circle_central_left = keypoints_dict["Circle central"]

        if len(points_circle_central_left) > 0:
            keypoints_dict["Circle central left"] = points_circle_central_left
        if len(points_circle_central_right) > 0:
            keypoints_dict["Circle central right"] = points_circle_central_right
        if len(points_circle_central_left) == 0 and len(points_circle_central_right) == 0:
            raise RuntimeError
        del keypoints_dict["Circle central"]
    return keypoints_dict


class BaseDataset(Dataset):
    def __init__(
        self,
        file_match_info: Path,
        extremities_annotations: Union[str, Path],
        model3d,
        filter_cam_type=None,
        extremities_prefix="",  # or extremities_ when using predicted
        constant_cam_position=1,
        split_circle_central=False,
        remove_invalid=False,
    ) -> None:
        super().__init__()

        self.dir_annotations = Path(extremities_annotations)
        self.dir_images = file_match_info.parent
        self.extremities_prefix = extremities_prefix
        if not file_match_info.exists():
            raise FileNotFoundError
        self.df_match_info = pd.read_json(file_match_info).T
        self.df_match_info["image_id"] = self.df_match_info.index

        if filter_cam_type and "camera" in self.df_match_info.columns:
            self.df_match_info_filter = self.df_match_info.loc[
                self.df_match_info["camera"] == filter_cam_type
            ]
            if len(self.df_match_info_filter) == 0:
                print(self.df_match_info.head(5))
                print(
                    "requiested cam type:",
                    filter_cam_type,
                    "given:",
                    self.df_match_info["camera"].unique().tolist(),
                )
                raise RuntimeError("No elements in dataset")
            self.df_match_info = self.df_match_info_filter
        if constant_cam_position > 1:
            self.df_match_info = (
                self.df_match_info.groupby(["league", "season", "match"]).agg(list).reset_index()
            )

            if not remove_invalid:
                if not (self.df_match_info["image_id"].agg(len) >= constant_cam_position).all():
                    print(self.df_match_info["image_id"].agg(len))
                    raise ValueError(
                        f"Tried to sample constant_cam_position={constant_cam_position} but this assumption does not hold for all samples"
                    )

            self.df_match_info["number_of_samples"] = self.df_match_info["image_id"].apply(
                lambda l: len(l)
            )
            self.df_match_info = self.df_match_info.loc[
                self.df_match_info["number_of_samples"] >= constant_cam_position
            ]

            self.df_match_info = self.df_match_info.apply(pd.Series.explode).reset_index()
            self.df_match_info = self.df_match_info.groupby(["league", "season", "match"]).sample(
                n=constant_cam_position, random_state=10
            )

            self.df_match_info = (
                self.df_match_info.groupby(["league", "season", "match"]).agg(list).reset_index()
            )

            self.df_match_info.drop(labels=["number_of_samples", "index"], inplace=True, axis=1)

        self.filter_cam_type = filter_cam_type
        self.constant_cam_position = constant_cam_position
        self.model3d = model3d
        self.split_circle_central = split_circle_central

    def __len__(self):
        return len(self.df_match_info)

    def __getitem__(self, idx):

        candidates_meta = self.df_match_info.iloc[idx].to_dict()

        if self.constant_cam_position == 1:
            candidates_meta = {k: [v] for k, v in candidates_meta.items()}

        candidates_meta["keypoints_raw"] = []
        for image_id in candidates_meta["image_id"]:
            file_annotation = self.dir_annotations / image_id
            file_annotation = (
                file_annotation.parent / f"{self.extremities_prefix}{file_annotation.stem}.json"
            )
            with open(file_annotation) as fr:
                keypoints_dict = json.load(fr)

            if self.split_circle_central:
                keypoints_dict = split_circle_central(keypoints_dict)

            # add empty entries for non-visible segments
            for l in self.model3d.segment_names:
                if l not in keypoints_dict:
                    keypoints_dict[l] = []
            candidates_meta["keypoints_raw"].append(keypoints_dict)

        # candidates_meta_out = {}
        # for k, v in candidates_meta.items():
        #     if isinstance(v, list):
        #         candidates_meta_out[k] = v[
        #             : self.constant_cam_position
        #         ]  # keep only constant_cam_position samples
        #     else:
        #         # samples with shared properties, here match, season, league
        #         candidates_meta_out[k] = v

        # return candidates_meta_out
        return {"meta": candidates_meta}


class FixedInputSizeDataset(BaseDataset):
    def __init__(
        self,
        file_match_info: Union[str, Path],
        model3d,
        image_width: int,
        image_height: int,
        num_points_on_circle_segments: int,
        num_points_on_line_segments: int,
        extremities_annotations,
        return_image=True,
        image_tfms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        filter_cam_type=None,
        extremities_prefix="",
        constant_cam_position=1,
        remove_invalid=False,
        split_circle_central=False,
    ) -> None:
        super().__init__(
            file_match_info=Path(file_match_info),
            extremities_annotations=extremities_annotations,
            model3d=model3d,
            filter_cam_type=filter_cam_type,
            extremities_prefix=extremities_prefix,
            constant_cam_position=constant_cam_position,
            remove_invalid=remove_invalid,
            split_circle_central=split_circle_central,
        )

        self.model3d = model3d
        self.num_points_on_circle_segments = num_points_on_circle_segments
        self.num_points_on_line_segments = num_points_on_line_segments
        self.image_width_source = image_width
        self.image_height_source = image_height
        self.pad_pixel_position_xy = 0.0

        self.return_image = return_image
        self.image_tfms = image_tfms

    def __len__(self):
        return len(self.df_match_info)

    def prepare_per_sample(self, keypoints_raw: dict, image_id: str):
        r = {}
        pixel_stacked = {}
        for label, points in keypoints_raw.items():

            num_points_selection = self.num_points_on_line_segments
            if "Circle" in label:
                num_points_selection = self.num_points_on_circle_segments

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
                pixel_stacked[label][:, 0] = pixel_stacked[label][:, 0] * (
                    self.image_width_source - 1
                )
                pixel_stacked[label][:, 1] = pixel_stacked[label][:, 1] * (
                    self.image_height_source - 1
                )

        for segment_type, num_segments, segment_names in [
            ("lines", self.model3d.line_segments.shape[1], self.model3d.line_segments_names),
            ("circles", self.model3d.circle_segments.shape[1], self.model3d.circle_segments_names),
        ]:
            # set non-visible pixels to (-1, -1) (pad_pixel_position_xy)

            num_points_selection = self.num_points_on_line_segments
            if segment_type == "circles":
                num_points_selection = self.num_points_on_circle_segments
            px_projected_selection = (
                torch.zeros((num_segments, num_points_selection, 2)) + self.pad_pixel_position_xy
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
                & (px_projected_selection_shuffled[:, :, 0] < self.image_width_source)
            ) & (
                (0 < px_projected_selection_shuffled[:, :, 1])
                & (px_projected_selection_shuffled[:, :, 1] < self.image_height_source)
            )

            r[f"{segment_type}__is_keypoint_mask"] = is_keypoint_mask.unsqueeze(
                0
            )  # (1, num_segments, num_points_selection)

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

            r[
                f"{segment_type}__px_projected_selection_shuffled"
            ] = px_projected_selection_shuffled  # (3, num_segments, num_points_selection)

            ndc_projected_selection_shuffled = px_projected_selection_shuffled.clone()
            ndc_projected_selection_shuffled[0] = (
                ndc_projected_selection_shuffled[0] / self.image_width_source
            )
            ndc_projected_selection_shuffled[1] = (
                ndc_projected_selection_shuffled[1] / self.image_height_source
            )
            ndc_projected_selection_shuffled[1] = ndc_projected_selection_shuffled[1] * 2.0 - 1
            ndc_projected_selection_shuffled[0] = ndc_projected_selection_shuffled[0] * 2.0 - 1
            r[
                f"{segment_type}__ndc_projected_selection_shuffled"
            ] = ndc_projected_selection_shuffled

        if self.return_image:
            file_image = self.dir_images / image_id
            image = Image.open(file_image).convert("RGB")
            if self.image_tfms:
                image = self.image_tfms(image)
            r["image"] = image
        return r

    def __getitem__(self, idx):
        meta_dict = super().__getitem__(idx)

        image_ids = meta_dict["meta"]["image_id"]
        keypoints_raw = meta_dict["meta"]["keypoints_raw"]

        per_sample_output = [
            self.prepare_per_sample(keypoints_raw[i], image_ids[i]) for i in range(len(image_ids))
        ]

        for k in per_sample_output[0].keys():
            meta_dict[k] = torch.stack([per_sample_output[i][k] for i in range(len(image_ids))])
        del meta_dict["meta"]["keypoints_raw"]

        meta_dict["image_id"] = image_ids
        del meta_dict["meta"]["image_id"]
        return meta_dict


def custom_list_collate(batch):
    r"""
    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
    Here is the general input type (based on the type of the element within the batch) to output type mapping:
    * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
    * NumPy Arrays -> :class:`torch.Tensor`
    * `float` -> :class:`torch.Tensor`
    * `int` -> :class:`torch.Tensor`
    * `str` -> `str` (unchanged)
    * `bytes` -> `bytes` (unchanged)
    * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
    * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    Args:
        batch: a single batch to be collated
    Examples:
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]

        >>> # modification
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1, 2], [2, 3, 4]])
        >>> [[0, 1, 2], [2, 3, 4]]
        >>> # original behavior
        >>> [[0, 2], [1, 3], [2, 4]]
    """

    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    default_collate_err_msg_format = "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return [torch.as_tensor(b) for b in batch]
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: custom_list_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(custom_list_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        # transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        return batch

        # if isinstance(elem, tuple):
        #     return [
        #         custom_list_collate(samples) for samples in transposed
        #     ]  # Backwards compatibility.
        # else:
        #     try:
        #         return elem_type([custom_list_collate(samples) for samples in transposed])
        #     except TypeError:
        #         # The sequence type may not support `__init__(iterable)` (e.g., `range`).
        #         return [custom_list_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class CamInferenceDataset(FixedInputSizeDataset):
    def __init__(
        self,
        file_per_sample_output: Union[str, Path],
        image_id_manual_selection: Union[list, None],
        file_match_info: Union[str, Path],
        model3d,
        image_width: int,
        image_height: int,
        num_points_on_circle_segments: int,
        num_points_on_line_segments: int,
        extremities_annotations,
        return_image=True,
        image_tfms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        filter_cam_type=None,
        extremities_prefix="",
        constant_cam_position=1,
        remove_invalid=False,
        split_circle_central=False,
    ) -> None:
        super().__init__(
            file_match_info=file_match_info,
            model3d=model3d,
            image_width=image_width,
            image_height=image_height,
            num_points_on_circle_segments=num_points_on_circle_segments,
            num_points_on_line_segments=num_points_on_line_segments,
            extremities_annotations=extremities_annotations,
            return_image=return_image,
            image_tfms=image_tfms,
            filter_cam_type=filter_cam_type,
            extremities_prefix=extremities_prefix,
            constant_cam_position=constant_cam_position,
            remove_invalid=remove_invalid,
            split_circle_central=split_circle_central,
        )

        self.df_per_sample_output_cams = pd.read_json(
            file_per_sample_output, orient="records", lines=True
        )
        self.df_per_sample_output_cams.set_index("image_ids", drop=True, inplace=True)

        self.image_id_manual_selection = image_id_manual_selection
        if self.image_id_manual_selection is not None and len(image_id_manual_selection) > 0:
            print(self.df_match_info)
            self.df_match_info = self.df_match_info.loc[image_id_manual_selection]

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx):
        meta_dict = super().__getitem__(idx)

        # add camera parameters

        # output selfCalib module
        cam_params = self.df_per_sample_output_cams.loc[meta_dict["image_id"][0]]

        # todo: assert T = 1
        phi_dict = {
            "aov": [cam_params["aov_radian"]],
            "c_x": [cam_params["position_meters"][0]],
            "c_y": [cam_params["position_meters"][1]],
            "c_z": [cam_params["position_meters"][2]],
            "pan": [np.deg2rad(cam_params["pan_degrees"])],
            "tilt": [np.deg2rad(cam_params["tilt_degrees"])],
            "roll": [np.deg2rad(cam_params["roll_degrees"])],
        }  # shape (1, 1) -> later (T, 1)
        meta_dict["phi_dict"] = phi_dict

        # according to kornia, tensor of max length 14
        # todo: more generic -> if k: all(zeros(v)) -> ignore

        meta_dict["psi"] = torch.tensor(cam_params["radial_distortion"][:2])

        return meta_dict


class HomographyInferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_per_sample_output: Union[str, Path],
        image_id_manual_selection: Union[list, None],
        dir_dataset: Path,
        tfms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    ) -> None:
        super().__init__()

        self.df_per_sample_output_cams = pd.read_json(
            file_per_sample_output, orient="records", lines=True
        )
        self.df_per_sample_output_cams["image_id"] = self.df_per_sample_output_cams["image_ids"]
        self.df_per_sample_output_cams.set_index("image_ids", drop=True, inplace=True)

        self.image_id_manual_selection = image_id_manual_selection
        if self.image_id_manual_selection is not None and len(image_id_manual_selection) > 0:
            self.df_per_sample_output_cams = self.df_per_sample_output_cams.loc[
                image_id_manual_selection
            ]

        self.dir_dataset = dir_dataset
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.df_per_sample_output_cams.index)

    def __getitem__(self, idx):

        sample = self.df_per_sample_output_cams.iloc[idx]

        image = Image.open(self.dir_dataset / sample["image_id"])
        image = self.tfms(image)

        return {
            "image_id": sample["image_id"],
            "h": torch.tensor(sample["homography"]),
            "image": image,
        }
