import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from tracklab.utils.attribute_voting import select_highest_voted_att
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


log = logging.getLogger(__name__)


class TrackletTeamLabeling(VideoLevelModule):
    
    input_columns = ["track_id", "embeddings", "role"]
    output_columns = ["team_cluster"]
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections["team_cluster"] = [np.nan] * len(detections)
        if "track_id" not in detections.columns:
            return detections
        
        embedding_tracklet = pd.DataFrame(columns=['track_id', 'embeddings'])
        player_detections = detections[detections.role == "player"]
        for track_id in player_detections.track_id.unique():
            if np.isnan(track_id):
                continue
            tracklet = player_detections[player_detections.track_id == track_id]
            embeddings = np.array([i for i in tracklet.embeddings])
            embeddings = np.mean(np.mean(embeddings, axis=1), axis=0)  # avg over all parts of body
            embedding_tracklet.loc[len(embedding_tracklet.index)] = [track_id, embeddings]

        ############## do clustering on the tracklet level embeddings ##########
        embeddings = np.asarray([i for i in embedding_tracklet.embeddings])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
        embedding_tracklet.insert(2, "team", kmeans.labels_, True)
        
        for i, track_id in enumerate(embedding_tracklet.track_id.unique()):
            tracklet = detections[detections.track_id == track_id]
            tracklet_team = [embedding_tracklet.iloc[i]["team"]] * len(tracklet) 
            detections.loc[tracklet.index, "team_cluster"] = tracklet_team
            
        return detections