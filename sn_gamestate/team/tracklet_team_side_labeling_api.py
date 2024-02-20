import pandas as pd
import torch
import numpy as np
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging


log = logging.getLogger(__name__)


class TrackletTeamSideLabeling(VideoLevelModule):
    """
    This module labels the team side (left, right) of the detections with role = {'player', 'goalkeeper'} based on the team clustering.
    Team 'left'/'right' means the team with its goal on the left/right side of the image (from the camera perspective).
    """

    input_columns = ["track_id", "team_cluster", "bbox_pitch", "role"]
    output_columns = ["team"]
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        if "track_id" not in detections.columns:
            return detections
        
        team_a = detections[detections.team_cluster == 0]
        team_b = detections[detections.team_cluster == 1]
        xa_coordinates = [bbox["x_bottom_middle"] if isinstance(bbox, dict) else np.nan for bbox in team_a.bbox_pitch]  # (x, y) are the center of a bbox
        xb_coordinates = [bbox["x_bottom_middle"] if isinstance(bbox, dict) else np.nan for bbox in team_b.bbox_pitch]  # (x, y) are the center of a bbox
        
        avg_a = np.nanmean(xa_coordinates)
        avg_b = np.nanmean(xb_coordinates)
        
        if avg_a > avg_b:
            detections.loc[team_a.index, "team"] = ['right'] * len(team_a)
            detections.loc[team_b.index, "team"] = ['left'] * len(team_b)
        else:
            detections.loc[team_a.index, "team"] = ['left'] * len(team_a)
            detections.loc[team_b.index, "team"] = ['right'] * len(team_b)

        # Goalkeeper labeling
        goalkeepers = detections[detections.role == "goalkeeper"].dropna(subset=["bbox_pitch"])
        gk_team = goalkeepers.bbox_pitch.apply(lambda bbox: "right" if (bbox["x_bottom_middle"] > 0) else "left")
        detections.loc[goalkeepers.index, "team"] = gk_team

        return detections
