from pathlib import Path
from typing import Any

import cv2
import gdown
import numpy as np
import pandas as pd

from sn_calibration_baseline.detect_extremities import SegmentationNetwork, \
    generate_class_synthesis, get_line_extremities
from tracklab.pipeline import ImageLevelModule


class BaselinePitch(ImageLevelModule):
    input_columns = []
    output_columns = []

    def __init__(self, mean_file, std_file, model_weights,
                 resolution_width, resolution_height,
                 batch_size, **kwargs):
        super().__init__(batch_size=batch_size)
        self.download_model(model_weights)
        self.model_weights = model_weights
        self.model = SegmentationNetwork(model_weights, mean_file, std_file)
        self.model.model.eval()  # the model of my model is my model !
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height

    def download_model(self, model_weights):
        if Path(model_weights).stem == "soccer_pitch_segmentation":
            md5 = "229023fbbd7e51c1abc56825d3492874"
            gdown.cached_download(id="1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM", path=model_weights, md5=md5)

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (self.model.width, self.model.height), interpolation=cv2.INTER_LINEAR)
        image = np.asarray(image, np.float32) / 255.
        image = (image - self.model.mean) / self.model.std
        image = image.transpose((2, 0, 1))
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        results = self.model.model.forward(batch.float())
        results = results["out"].cpu().detach().numpy()
        results = results.transpose(0, 2, 3, 1)
        output_pred = []
        for result, idx in zip(results, metadatas.index):
            semlines = np.asarray(np.argmax(result, axis=2), dtype=np.uint8)
            skeletons = generate_class_synthesis(semlines, radius=6)
            extremities = get_line_extremities(skeletons, maxdist=40,
                                               width=self.resolution_width,
                                               height=self.resolution_height)
            output_pred.append(pd.Series(
                {"lines": extremities},
                name=idx,
            ))
        return pd.DataFrame(), pd.DataFrame(output_pred)
