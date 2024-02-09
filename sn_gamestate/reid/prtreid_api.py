import gdown
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

from tracklab.pipeline import DetectionLevelModule
# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import rescale_keypoints
from tracklab.utils.collate import default_collate
from sn_gamestate.reid.prtreid_dataset import ReidDataset
from prtreid.scripts.main import build_config, build_torchreid_model_engine
from prtreid.tools.feature_extractor import FeatureExtractor
from prtreid.utils.imagetools import (
    build_gaussian_heatmaps,
)
from tracklab.utils.collate import Unbatchable

import tracklab
from pathlib import Path


import prtreid
from torch.nn import functional as F
from prtreid.data.masks_transforms import (
    CocoToSixBodyMasks,
    masks_preprocess_transforms,
)
from prtreid.utils.tools import extract_test_embeddings
from prtreid.data.datasets import configure_dataset_class

from prtreid.scripts.default_config import engine_run_kwargs


class PRTReId(DetectionLevelModule):
    collate_fn = default_collate
    input_columns = ["bbox_ltwh"]
    output_columns = ["embeddings", "visibility_scores", "body_masks", "role_detection", "role_confidence"]
    role_mapping = {'ball': 0, 'goalkeeper': 1, 'other': 2, 'player': 3, 'referee': 4, None: -1}

    def __init__(
        self,
        cfg,
        tracking_dataset,
        dataset,
        device,
        save_path,
        job_id,
        use_keypoints_visibility_scores_for_reid,
        training_enabled,
        batch_size,
    ):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        tracking_dataset.name = dataset.name
        tracking_dataset.nickname = dataset.nickname
        self.dataset_cfg = dataset
        self.use_keypoints_visibility_scores_for_reid = (
            use_keypoints_visibility_scores_for_reid
        )
        tracking_dataset.name = self.dataset_cfg.name
        tracking_dataset.nickname = self.dataset_cfg.nickname
        additional_args = {
            "tracking_dataset": tracking_dataset,
            "reid_config": self.dataset_cfg,
            "role_mapping": self.role_mapping,
            "pose_model": None,
        }
        prtreid.data.register_image_dataset(
            tracking_dataset.name,
            configure_dataset_class(ReidDataset, **additional_args),
            tracking_dataset.nickname,
        )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.download_models(load_weights=self.cfg.model.load_weights,
                             pretrained_path=self.cfg.model.bpbreid.hrnet_pretrained_path,
                             backbone=self.cfg.model.bpbreid.backbone)
        self.inverse_role_mapping = {v: k for k, v in self.role_mapping.items()}
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None

    def download_models(self, load_weights, pretrained_path, backbone):
        if Path(load_weights).stem == "bpbreid_market1501_hrnet32_10642":
            md5 = "e79262f17e7486ece33eebe198c07841"
            gdown.cached_download(id="1m8FgfgQXf_i7zVEblvis1HLV6yHGdX7p", path=load_weights, md5=md5)
        if backbone == "hrnet32":
            md5 = "58ea12b0420aa3adaa2f74114c9f9721"
            path = Path(pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
            gdown.cached_download(id="1-gmeQ_n7NuyADiNK8EygHGDRV-HUesEZ", path=path,
                                  md5=md5)
    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
        if not self.cfg.model.bpbreid.learnable_attention_enabled and "keypoints_xyc" in detection:
            bbox_ltwh = detection.bbox.ltwh(
                image_shape=(image.shape[1], image.shape[0]), rounded=True
            )
            kp_xyc_bbox = detection.keypoints.in_bbox_coord(bbox_ltwh)
            kp_xyc_mask = rescale_keypoints(
                kp_xyc_bbox, (bbox_ltwh[2], bbox_ltwh[3]), (mask_w, mask_h)
            )
            if self.dataset_cfg.masks_mode == "gaussian_keypoints":
                pixels_parts_probabilities = build_gaussian_heatmaps(
                    kp_xyc_mask, mask_w, mask_h
                )
            else:
                raise NotImplementedError
            batch["masks"] = pixels_parts_probabilities

        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        im_crops = batch["img"]
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        if "masks" in batch:
            external_parts_masks = batch["masks"]
            external_parts_masks = external_parts_masks.cpu().detach().numpy()
        else:
            external_parts_masks = None
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                model=self.model,
                verbose=False,  # FIXME @Vladimir
            )
        reid_result = self.feature_extractor(
            im_crops, external_parts_masks=external_parts_masks
        )
        embeddings, visibility_scores, body_masks, _, role_cls_scores = extract_test_embeddings(
            reid_result, self.test_embeddings
        )
        
        role_scores_ = []
        role_scores_.append(role_cls_scores['globl'].cpu() if role_cls_scores is not None else None)
        role_scores_ = torch.cat(role_scores_, 0) if role_scores_[0] is not None else None
        roles = [torch.argmax(i).item() for i in role_scores_]
        roles = [self.inverse_role_mapping[index] for index in roles]
        role_confidence = [torch.max(i).item() for i in role_scores_]

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        body_masks = body_masks.cpu().detach().numpy()

        if self.use_keypoints_visibility_scores_for_reid:
            kp_visibility_scores = batch["visibility_scores"].numpy()
            if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
                kp_visibility_scores = np.concatenate(
                    [np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores],
                    axis=1,
                )
            visibility_scores = np.float32(kp_visibility_scores)

        reid_df = pd.DataFrame(
            {
                "embeddings": list(embeddings),
                "visibility_scores": list(visibility_scores),
                "body_masks": list(body_masks),
                "role_detection": roles,
                "role_confidence": role_confidence,
            },
            index=detections.index,
        )
        return reid_df

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))
