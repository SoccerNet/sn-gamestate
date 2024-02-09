from __future__ import absolute_import, division, print_function

import sys
import cv2
import torch
import numpy as np
import pandas as pd

from math import ceil
from pathlib import Path
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracklab.datastruct import EngineDatapipe
from tracklab.datastruct import TrackingDataset
# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import rescale_keypoints

from tracklab.utils.cv2 import overlay_heatmap
import tracklab

from prtreid.data import ImageDataset
from prtreid.utils.imagetools import (
    gkern,
    build_gaussian_heatmaps,
)
import logging

log = logging.getLogger(__name__)


class ReidDataset(ImageDataset):
    dataset_dir = "PoseTrack21"  # TODO
    annotations_dir = "posetrack_data"  # TODO
    img_ext = ".jpg"
    masks_ext = ".npy"
    reid_dir = "reid"
    reid_images_dir = "images"
    reid_masks_dir = "masks"
    reid_fig_dir = "figures"
    reid_anns_dir = "anns"
    images_anns_filename = "reid_crops_anns.json"
    masks_anns_filename = "reid_masks_anns.json"

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        "gaussian_joints": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "gaussian_keypoints": (
            17,
            False,
            ".npy",
            ["p{}".format(p) for p in range(1, 17)],
        ),
        "pose_on_img": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in ReidDataset.masks_dirs:
            return None
        else:
            return ReidDataset.masks_dirs[masks_dir]

    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """camid refers to video id: remove gallery samples from the different videos than query sample"""
        if self.eval_metric == "mot_inter_intra_video":
            return np.zeros_like(q_pid)
        elif self.eval_metric == "mot_inter_video":
            remove = g_camids == q_camid
            return remove
        elif self.eval_metric == "mot_intra_video":
            remove = g_camids != q_camid
            return remove
        else:
            raise ValueError

    def __init__(
        self,
        tracking_dataset: TrackingDataset,
        reid_config,
        role_mapping,
        pose_model=None,
        masks_dir="",
        **kwargs
    ):
        # Init
        self.tracking_dataset = tracking_dataset
        self.reid_config = reid_config
        self.pose_model = pose_model  #  can be used to generate pseudo labels for the reid dataset
        self.dataset_path = Path(self.tracking_dataset.dataset_path)
        self.role_mapping = role_mapping
        self.masks_dir = masks_dir
        self.pose_datapipe = EngineDatapipe(self.pose_model)
        self.column_mapping = {}
        self.pose_dl = DataLoader(
            dataset=self.pose_datapipe,
            batch_size=128,
            num_workers=0,  # FIXME issue with higher
            collate_fn=type(self.pose_model).collate_fn if self.pose_model else None,
            persistent_workers=False,
        )
        self.eval_metric = self.reid_config.eval_metric
        self.multi_video_queries_only = self.reid_config.multi_video_queries_only

        val_set = tracking_dataset.sets[self.reid_config.test.set_name]
        train_set = tracking_dataset.sets[self.reid_config.train.set_name]

        assert (
            self.reid_config.train.max_samples_per_id
            >= self.reid_config.train.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"
        assert (
            self.reid_config.test.max_samples_per_id
            >= self.reid_config.test.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"

        if self.masks_dir in self.masks_dirs:  # TODO
            (
                self.masks_parts_numbers,
                self.has_background,
                self.masks_suffix,
                self.masks_parts_names,
            ) = self.masks_dirs[self.masks_dir]
        else:
            (
                self.masks_parts_numbers,
                self.has_background,
                self.masks_suffix,
                self.masks_parts_names,
            ) = (None, None, None, None)

        # Build ReID dataset from MOT dataset
        self.build_reid_set(
            train_set,
            self.reid_config,
            "train",
            is_test_set=False,
        )

        self.build_reid_set(
            val_set,
            self.reid_config,
            "val",
            is_test_set=True,
        )

        train_gt_dets = train_set.detections_gt
        val_gt_dets = val_set.detections_gt

        # Get train/query/gallery sets as torchreid list format
        train_df = train_gt_dets[train_gt_dets["split"] == "train"]
        query_df = val_gt_dets[val_gt_dets["split"] == "query"]
        gallery_df = val_gt_dets[val_gt_dets["split"] == "gallery"]
        train, query, gallery = self.to_torchreid_dataset_format(
            [train_df, query_df, gallery_df]
        )

        super().__init__(train, query, gallery, **kwargs)

    def build_reid_set(self, tracking_set, reid_config, split, is_test_set):
        """
        Build ReID metadata for a given MOT dataset split.
        Only a subset of all MOT groundtruth detections is used for ReID.
        Detections to be used for ReID are selected according to the filtering criteria specified in the config 'reid_cfg'.
        Image crops and human parsing labels (masks) are generated for each selected detection only.
        If the config is changed and more detections are selected, the image crops and masks are generated only for
        these new detections.
        """
        image_metadatas = tracking_set.image_metadatas
        detections = tracking_set.detections_gt
        fig_size = reid_config.fig_size
        mask_size = reid_config.mask_size
        max_crop_size = reid_config.max_crop_size
        reid_set_cfg = reid_config.test if is_test_set else reid_config.train
        masks_mode = reid_config.masks_mode

        log.info("Loading {} set...".format(split))

        # Precompute all paths
        reid_path = Path(self.dataset_path, self.reid_dir, masks_mode) if self.reid_config.enable_human_parsing_labels else Path(self.dataset_path, self.reid_dir)
        reid_img_path = reid_path / self.reid_images_dir / split
        reid_mask_path = reid_path / self.reid_masks_dir / split
        reid_fig_path = reid_path / self.reid_fig_dir / split
        reid_anns_filepath = (
            reid_path
            / self.reid_images_dir
            / self.reid_anns_dir
            / (split + "_" + self.images_anns_filename)
        )
        masks_anns_filepath = (
            reid_path
            / self.reid_masks_dir
            / self.reid_anns_dir
            / (split + "_" + self.masks_anns_filename)
        )

        # Load reid crops metadata into existing ground truth detections dataframe
        self.load_reid_annotations(
            detections,
            reid_anns_filepath,
            ["reid_crop_path", "reid_crop_width", "reid_crop_height"],
        )

        # Load reid masks metadata into existing ground truth detections dataframe
        self.load_reid_annotations(detections, masks_anns_filepath, ["masks_path"])
        # Sampling of detections to be used to create the ReID dataset
        self.sample_detections_for_reid(detections, reid_set_cfg)

        # Save ReID detections crops and related metadata. Apply only on sampled detections
        self.save_reid_img_crops(
            detections,
            reid_img_path,
            split,
            reid_anns_filepath,
            image_metadatas,
            max_crop_size,
        )

        # Save human parsing pseudo ground truth and related metadata. Apply only on sampled detections
        if self.reid_config.enable_human_parsing_labels:
            self.save_reid_masks_crops(
                detections,
                reid_mask_path,
                reid_fig_path,
                split,
                masks_anns_filepath,
                image_metadatas,
                fig_size,
                mask_size,
                mode=masks_mode,
            )
        else:
            detections["masks_path"] = ''

        # Add 0-based pid column (for Torchreid compatibility) to sampled detections
        self.ad_pid_column(detections)

        # Flag sampled detection as a query or gallery if this is a test set
        if is_test_set:
            self.query_gallery_split(detections, reid_set_cfg.ratio_query_per_id)

    def load_reid_annotations(self, gt_dets, reid_anns_filepath, columns):
        if reid_anns_filepath.exists():
            reid_anns = pd.read_json(
                reid_anns_filepath, convert_dates=False, convert_axes=False
            )
            tmp_df = gt_dets.merge(
                reid_anns,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            gt_dets[columns] = tmp_df[columns]
        else:
            # no annotations yet, initialize empty columns
            for col in columns:
                gt_dets[col] = None

    def sample_detections_for_reid(self, dets_df, reid_cfg):
        dets_df["split"] = "none"

        # Filter detections by visibility
        dets_df_f1 = dets_df[dets_df.visibility >= reid_cfg.min_vis]

        # Filter detections by crop size
        keep = dets_df_f1.bbox_ltwh.apply(
            lambda x: x[2] > reid_cfg.min_w
        ) & dets_df_f1.bbox_ltwh.apply(lambda x: x[3] > reid_cfg.min_h)
        dets_df_f2 = dets_df_f1[keep]
        log.warning(
            "{} removed because too small samples (h<{} or w<{}) = {}".format(
                self.__class__.__name__,
                (reid_cfg.min_h),
                (reid_cfg.min_w),
                len(dets_df_f1) - len(dets_df_f2),
            )
        )

        # Filter detections by uniform sampling along each tracklet
        dets_df_f3 = (
            dets_df_f2.groupby("person_id")
            .apply(
                self.uniform_tracklet_sampling, reid_cfg.max_samples_per_id, "image_id"
            )
            .reset_index(drop=True)
            .copy()
        )
        log.warning(
            "{} removed for uniform tracklet sampling = {}".format(
                self.__class__.__name__, len(dets_df_f2) - len(dets_df_f3)
            )
        )

        # Keep only ids with at least MIN_SAMPLES appearances
        count_per_id = dets_df_f3.person_id.value_counts()
        ids_to_keep = count_per_id.index[count_per_id.ge((reid_cfg.min_samples_per_id))]
        dets_df_f4 = dets_df_f3[dets_df_f3.person_id.isin(ids_to_keep)]
        log.warning(
            "{} removed for not enough samples per id = {}".format(
                self.__class__.__name__, len(dets_df_f3) - len(dets_df_f4)
            )
        )

        # Keep only max_total_ids ids
        if reid_cfg.max_total_ids == -1 or reid_cfg.max_total_ids > len(
            dets_df_f4.person_id.unique()
        ):
            reid_cfg.max_total_ids = len(dets_df_f4.person_id.unique())
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        ids_to_keep = np.random.choice(
            dets_df_f4.person_id.unique(), replace=False, size=reid_cfg.max_total_ids
        )
        dets_df_f5 = dets_df_f4[dets_df_f4.person_id.isin(ids_to_keep)]

        dets_df.loc[dets_df.id.isin(dets_df_f5.id), "split"] = "train"
        log.info(
            "{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f5))
        )

    def save_reid_img_crops(
        self,
        gt_dets,
        save_path,
        set_name,
        reid_anns_filepath,
        metadatas_df,
        max_crop_size,
    ):
        """
        Save on disk all detections image crops from the ground truth dataset to build the reid dataset.
        Create a json annotation file with crops metadata.
        """
        max_h, max_w = max_crop_size
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.reid_crop_path.isnull()
        ]
        if len(gt_dets_for_reid) == 0:
            log.info(
                "All detections used for ReID already have their image crop saved on disk."
            )
            return
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} reid crops".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = metadatas_df[metadatas_df.id == image_id].iloc[0]
                img = cv2.imread(img_metadata.file_path)
                for index, det_metadata in dets_from_img.iterrows():
                    # crop and resize bbox from image
                    l, t, w, h = det_metadata.bbox.ltwh(
                        image_shape=(img.shape[1], img.shape[0]), rounded=True
                    )
                    pid = det_metadata.person_id
                    img_crop = img[t : t + h, l : l + w]
                    if h > max_h or w > max_w:
                        img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)

                    # save crop to disk
                    filename = "{}_{}_{}{}".format(
                        pid, video_id, img_metadata.id, self.img_ext
                    )
                    rel_filepath = Path(str(video_id), filename)
                    abs_filepath = Path(save_path, rel_filepath)
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(abs_filepath), img_crop)

                    # save image crop metadata
                    gt_dets.at[det_metadata.id, "reid_crop_path"] = str(abs_filepath)
                    gt_dets.at[det_metadata.id, "reid_crop_width"] = img_crop.shape[
                        0
                    ]
                    gt_dets.at[det_metadata.id, "reid_crop_height"] = img_crop.shape[
                        1
                    ]
                    pbar.update(1)

        log.info(
            'Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath)
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[
            ["id", "reid_crop_path", "reid_crop_width", "reid_crop_height"]
        ].to_json(reid_anns_filepath)

    def save_reid_masks_crops(
        self,
        gt_dets,
        masks_save_path,
        fig_save_path,
        set_name,
        reid_anns_filepath,
        metadatas_df,
        fig_size,
        masks_size,
        mode="gaussian_keypoints",
    ):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        fig_h, fig_w = fig_size
        mask_h, mask_w = masks_size
        g_scale = 10
        g_radius = int(mask_w / g_scale)
        gaussian = gkern(g_radius * 2 + 1)
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.masks_path.isnull()
        ]
        if len(gt_dets_for_reid) == 0:
            log.info("All reid crops already have human parsing masks labels.")
            return
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} human parsing labels".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = metadatas_df[metadatas_df.id == image_id].iloc[0]
                # load image once to get video frame size
                if mode == "pose_on_img":
                    fields_list = []
                    self.pose_datapipe.update(
                        metadatas_df[metadatas_df.id == image_id], None
                    )
                    for idxs, pose_batch in self.pose_dl:
                        batch_metadatas = metadatas_df.loc[idxs]
                        _, fields = self.pose_model.process(
                            pose_batch, batch_metadatas, return_fields=True
                        )
                        fields_list.extend(fields)

                    masks_gt_or = torch.concat(
                        (
                            fields_list[0][0][:, 1],
                            fields_list[0][1][:, 1],
                        )
                    )
                    img = cv2.imread(img_metadata.file_path)
                    masks_gt = resize(
                        masks_gt_or.numpy(),
                        (masks_gt_or.numpy().shape[0], img.shape[0], img.shape[1]),
                    )

                # loop on detections in frame
                for index, det_metadata in dets_from_img.iterrows():
                    if mode == "gaussian_keypoints":
                        # compute human parsing heatmaps as gaussian on each visible keypoint
                        img_crop = cv2.imread(det_metadata.reid_crop_path)
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        l, t, w, h = bbox_ltwh = det_metadata.bbox.ltwh(rounded=True)
                        keypoints_xyc = rescale_keypoints(
                            det_metadata.keypoints.in_bbox_coord(bbox_ltwh),
                            (w, h),
                            (mask_w, mask_h),
                        )
                        masks_gt_crop = build_gaussian_heatmaps(
                            keypoints_xyc, mask_w, mask_h, gaussian=gaussian
                        )
                    elif mode == "gaussian_joints":
                        # compute human parsing heatmaps as shapes around on each visible keypoint
                        img_crop = cv2.imread(det_metadata.reid_crop_path)
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        l, t, w, h = bbox_ltwh = det_metadata.bbox.ltwh(rounded=True)
                        keypoints_xyc = rescale_keypoints(
                            det_metadata.keypoints.in_bbox_coord(bbox_ltwh),
                            (w, h),
                            (mask_w, mask_h),
                        )
                        masks_gt_crop = build_gaussian_body_part_heatmaps(  # FIXME
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "pose_on_img_crops":
                        # compute human parsing heatmaps using output of pose model on cropped person image
                        img_crop = cv2.imread(det_metadata.reid_crop_path)
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        _, masks_gt_crop = self.pose_model.track_dataset()
                        masks_gt_crop = (
                            masks_gt_crop.squeeze().permute((1, 2, 0)).numpy()
                        )
                        masks_gt_crop = resize(
                            masks_gt_crop, (fig_h, fig_w, masks_gt_crop.shape[2])
                        )
                    elif mode == "pose_on_img":
                        # compute human parsing heatmaps using output of pose model on full image
                        l, t, w, h = det_metadata.bbox.ltwh(
                            image_shape=(img.shape[1], img.shape[0]), rounded=True
                        )
                        img_crop = img[t : t + h, l : l + w]
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        masks_gt_crop = masks_gt[:, t : t + h, l : l + w]
                        masks_gt_crop = resize(
                            masks_gt_crop, (masks_gt_crop.shape[0], fig_h, fig_w)
                        )
                    else:
                        raise ValueError("Invalid human parsing method")

                    # save human parsing heatmaps on disk
                    pid = det_metadata.person_id
                    filename = "{}_{}_{}".format(pid, video_id, image_id)
                    abs_filepath = Path(
                        masks_save_path, Path(video_id, filename + self.masks_ext)
                    )
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(abs_filepath), masks_gt_crop)

                    # save image crop with human parsing heatmaps overlayed on disk for visualization/debug purpose
                    img_with_heatmap = overlay_heatmap(
                        img_crop, masks_gt_crop.max(axis=0), weight=0.3
                    )
                    figure_filepath = Path(
                        fig_save_path, video_id, filename + self.img_ext
                    )
                    figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(figure_filepath), img_with_heatmap)

                    # record human parsing metadata for later json dump
                    gt_dets.at[det_metadata.id, "masks_path"] = str(abs_filepath)
                    pbar.update(1)

        log.info(
            'Saving reid human parsing annotations as json to "{}"'.format(
                reid_anns_filepath
            )
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[["id", "masks_path"]].to_json(reid_anns_filepath)

    def rescale_and_filter_keypoints(self, keypoints, bbox_ltwh, new_w, new_h):
        l, t, w, h = bbox_ltwh.astype(int)
        discarded_keypoints = 0
        rescaled_keypoints = {}
        for i, kp in enumerate(keypoints):
            # remove unvisible keypoints
            if kp[2] == 0:
                continue

            # put keypoints in bbox coord space
            kpx, kpy = kp[:2].astype(int) - np.array([l, t])

            # remove keypoints out of bbox
            if kpx < 0 or kpx >= w or kpy < 0 or kpy >= h:
                discarded_keypoints += 1
                continue

            # put keypoints in resized image coord space
            kpx, kpy = kpx * new_w / w, kpy * new_h / h

            rescaled_keypoints[i] = np.array([int(kpx), int(kpy), 1])
        return rescaled_keypoints, discarded_keypoints

    def query_gallery_split(self, gt_dets, ratio):
        def random_tracklet_sampling(_df):
            x = list(_df.index)
            size = ceil(len(x) * ratio)
            result = list(np.random.choice(x, size=size, replace=False))
            return _df.loc[result]

        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        queries_per_pid = gt_dets_for_reid.groupby("person_id").apply(
            random_tracklet_sampling
        )
        if self.eval_metric == "mot_inter_video" or self.multi_video_queries_only:
            # keep only queries that are in more than one video
            queries_per_pid = (
                queries_per_pid.droplevel(level=0)
                .groupby("person_id")["video_id"]
                .filter(lambda g: (g.nunique() > 1))
                .reset_index()
            )
            assert len(queries_per_pid) != 0, (
                "There were no identity with more than one videos to be used as queries. "
                "Try setting 'multi_video_queries_only' to False or not using "
                "eval_metric='mot_inter_video' or adjust the settings to sample a "
                "bigger ReID dataset."
            )
        gt_dets.loc[gt_dets.split != "none", "split"] = "gallery"
        gt_dets.loc[gt_dets.id.isin(queries_per_pid.id), "split"] = "query"

    def to_torchreid_dataset_format(self, dataframes):
        results = []
        column_mapping = {}
        column_mapping["role"] = self.role_mapping
        for col in self.reid_config.columns:
            if col not in column_mapping:
                unique_values = {element for df in dataframes for element in df[col].unique()}
                unique_values.discard(None)
                ordered_unique_values = list(unique_values)
                ordered_unique_values.sort()
                column_mapping[col] = {
                    v: i for i, v in enumerate(ordered_unique_values)
                }
                column_mapping[col][None] = -1

        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df["camid"] = df["video_id"]
            df["img_path"] = df["reid_crop_path"]
            # remove bbox_head as it is not available for each sample
            # df to list of dict
            sorted_df = df.sort_values(by=["pid"])
            # use only necessary annotations: using them all caused a
            # 'RuntimeError: torch.cat(): input types can't be cast to the desired output type Long' in collate.py
            # -> still has to be fixed
            data_list = sorted_df[
                ["pid", "camid", "img_path", "masks_path", "visibility", "image_id", "video_id"] + self.reid_config.columns
            ]
            # factorize all columns, i.e. replace string values with 0-based increasing ids
            for col in self.reid_config.columns:
                data_list[col] = data_list[col].map(column_mapping[col])
                self.column_mapping[col] = {value: key for key, value in column_mapping[col].items()}

            data_list = data_list.to_dict("records")
            results.append(data_list)
        return results

    def ad_pid_column(self, gt_dets):
        # create pids as 0-based increasing numbers
        gt_dets["pid"] = None
        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        gt_dets.loc[gt_dets_for_reid.index, "pid"] = pd.factorize(
            gt_dets_for_reid.person_id
        )[0]

    def uniform_tracklet_sampling(self, _df, max_samples_per_id, column):
        _df.sort_values(column)
        num_det = len(_df)
        if num_det > max_samples_per_id:
            # Select 'max_samples_per_id' evenly spaced indices, including first and last
            indices = np.round(np.linspace(0, num_det - 1, max_samples_per_id)).astype(
                int
            )
            assert len(indices) == max_samples_per_id
            return _df.iloc[indices]
        else:
            return _df
