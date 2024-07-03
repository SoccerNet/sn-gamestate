import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
warnings.filterwarnings("ignore")
from sklearn.cluster import HDBSCAN


log = logging.getLogger(__name__)


class TrackletTeamHDBClustering(VideoLevelModule):
    """
    This module performs KMeans clustering on the embeddings of the tracklets to cluster the detections with role "player" into two teams.
    Teams are labeled as 0 and 1, and transformer into 'left' and 'right' in a separate module.
    """
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
            if np.isnan(track_id):
                continue
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            embeddings_list.append({'track_id': track_id, 'embeddings': embeddings})

        if not embeddings_list:  # Check if embeddings_list is empty
            detections['team_cluster'] = np.nan  # Initialize 'team_cluster' with a default value
            return detections

        embedding_tracklet = pd.DataFrame(embeddings_list)

        if len(embedding_tracklet) == 1:  # Only one track_id and embedding
            embedding_tracklet['team_cluster'] = 0
        else:
            # Perform KMeans clustering on the embeddings
            embeddings = np.vstack(embedding_tracklet.embeddings.values)
            hdbscan = HDBSCAN(min_cluster_size=5).fit(embeddings)
            labels = hdbscan.labels_.astype(float)
            log.info(f"labels from clustering : {[f'{u}: {c}' for u, c in zip(*np.unique(labels, return_counts=True))]}")
            unique, inverse, count = np.unique(labels, return_inverse=True, return_counts=True)
            count = count[unique >= 0]
            unique = unique[unique >= 0]
            argcount = np.argsort(-count)
            labels += 100

            if len(argcount) > 0:
                labels[labels == (unique[argcount[0]] + 100)] = 0
            if len(argcount) > 1:
                labels[labels == (unique[argcount[1]] + 100)] = 1
            labels[labels > 1] = np.nan

            embedding_tracklet['team_cluster'] = labels


        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(embedding_tracklet[['track_id', 'team_cluster']], on='track_id', how='left', sort=False)

        return detections
