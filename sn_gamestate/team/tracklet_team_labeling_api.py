import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
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

        player_detections = detections[detections.role == "player"]

        # Compute mean embeddings for each track_id
        embeddings_list = []
        for track_id, group in player_detections.groupby("track_id"):
            if np.isnan(track_id): continue
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            embeddings_list.append({'track_id': track_id, 'embeddings': embeddings})

        embedding_tracklet = pd.DataFrame(embeddings_list)

        # Perform KMeans clustering on the embeddings
        embeddings = np.vstack(embedding_tracklet.embeddings.values)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)

        # Map the team cluster back to the original detections DataFrame
        embedding_tracklet['team_cluster'] = kmeans.labels_
        detections = detections.merge(embedding_tracklet[['track_id', 'team_cluster']], on='track_id', how='left')

        return detections
