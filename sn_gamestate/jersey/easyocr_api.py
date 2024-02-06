import pandas as pd
import torch
import easyocr
import logging

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

log = logging.getLogger(__name__)


class EasyOCR(DetectionLevelModule):
    input_columns = []
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.cfg = cfg

    def no_jersey_number(self):
        return [None, None, 0]

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }

        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []
        images_np = [img.cpu().numpy() for img in batch['img']]
        self.reader = easyocr.Reader(['en'], gpu=True)
        if self.batch_size == 1:
            for img in batch['img']:
                img_np = img.cpu().numpy()
                result = self.reader.readtext(img_np, **self.cfg)
                if result == []:
                    jn = self.no_jersey_number()
                else:
                    result = result[
                        0]  # only take the first result (highest confidence)
                    try:
                        # see if the result is a number
                        int(result[1])
                    except ValueError:
                        jn = self.no_jersey_number()
                    else:
                        jn = [result[0], result[1], result[2]]

                jersey_number_detection.append(jn[1])
                jersey_number_confidence.append(jn[2])
        else:
            results = self.reader.readtext_batched(images_np, n_width=64, n_height=128,
                                                   workers=8, **self.cfg)
            for result in results:
                if result == []:
                    jn = self.no_jersey_number()
                else:
                    result = result[
                        0]  # only take the first result (highest confidence)
                    try:
                        # see if the result is a number
                        int(result[1])
                    except ValueError:
                        jn = self.no_jersey_number()
                    else:
                        jn = [result[0], result[1], result[2]]

                jersey_number_detection.append(jn[1])
                jersey_number_confidence.append(jn[2])

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence

        return detections
