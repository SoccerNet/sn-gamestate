import pandas as pd
import torch
import numpy as np

from tracklab.pipeline.videolevel_module import VideoLevelModule

import logging


log = logging.getLogger(__name__)


class VotingTrackletJerseyNumber(VideoLevelModule):
    
    input_columns = ["jersey_number", "jn_confidence"]
    output_columns = ["jn_tracklet"]
    
    def __init__(self, cfg, device, tracking_dataset=None):
        pass

    def select_best_jersey_number(self, jursey_numbers, jn_confidences):
        
        confidence_sum = {}
        
        # Iterate through the predictions to calculate the total confidence for each jersey number
        for jn, conf in zip(jursey_numbers, jn_confidences):
            if jn not in confidence_sum:
                confidence_sum[jn] = 0
            confidence_sum[jn] += conf
        
        # Find the jersey number with the maximum total confidence
        if len(confidence_sum) == 0:
            return None
        max_confidence_jn = max(confidence_sum, key=confidence_sum.get)
        return max_confidence_jn
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections["jn_tracklet"] = [np.nan] * len(detections)
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            jursey_numbers = tracklet.jursey_number
            jn_confidences = tracklet.jn_confidence
            tracklet_jn = [self.select_best_jersey_number(jursey_numbers, jn_confidences)] * len(tracklet)            
            detections.loc[tracklet.index, "jn_tracklet"] = tracklet_jn
            
        return detections


def bbox_easyocr_to_image_ltwh(easy_ocr_bbox, bbox_tlwh):
    easy_ocr_bbox = np.array(easy_ocr_bbox)  # only need the top left point
    tl = easy_ocr_bbox[0]
    tr = easy_ocr_bbox[1]
    br = easy_ocr_bbox[2]
    bl = easy_ocr_bbox[3]

    width = tr[1] - tl[1]
    height = bl[0] - tl[0]

    jn_bbox = bbox_tlwh.clone()
    jn_bbox[0] = jn_bbox[0] + tl[0]
    jn_bbox[1] = jn_bbox[1] + tl[1]
    jn_bbox[2] = width
    jn_bbox[3] = height
    return jn_bbox
