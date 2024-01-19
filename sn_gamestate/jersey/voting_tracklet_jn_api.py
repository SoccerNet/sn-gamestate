import pandas as pd
import torch
import numpy as np
from tracklab.utils.attribute_voting import select_highest_voted_att

from tracklab.pipeline.videolevel_module import VideoLevelModule

import logging

log = logging.getLogger(__name__)


class VotingTrackletJerseyNumber(VideoLevelModule):
    input_columns = ["track_id", "jersey_number", "jn_confidence"]
    output_columns = ["jn_tracklet"]

    def __init__(self, cfg, device, tracking_dataset=None):
        pass

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        detections["jn_tracklet"] = [np.nan] * len(detections)
        if "track_id" not in detections.columns:
            return detections
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            jersey_numbers = tracklet.jersey_number
            jn_confidences = tracklet.jn_confidence
            tracklet_jn = [select_highest_voted_att(jersey_numbers,
                                                    jn_confidences)] * len(tracklet)
            detections.loc[tracklet.index, "jn_tracklet"] = tracklet_jn

        return detections
