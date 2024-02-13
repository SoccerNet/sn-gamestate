import pandas as pd
import torch
import numpy as np
from mmocr.apis import MMOCRInferencer
# from mmengine.infer.infer import BaseInferencer
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
import logging

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from multiprocessing import Pool

log = logging.getLogger(__name__)


class MMOCR(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.ocr = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015', rec='SAR')
        self.batch_size = batch_size

        self.textdetinferencer = TextDetInferencer(
            'dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        self.textrecinferencer = TextRecInferencer('SAR', device=device)

    def no_jersey_number(self):
        return None, 0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        # print('crop shape', crop.shape)
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        # print('crop shape', crop.shape)
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }

        return batch

    def extract_numbers(self, text):
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def choose_best_jersey_number(self, jersey_numbers, jn_confidences):
        if len(jersey_numbers) == 0:
            return self.no_jersey_number()
        else:
            jn_confidences = np.array(jn_confidences)
            idx_sort = np.argsort(jn_confidences)
            return jersey_numbers[idx_sort[-1]], jn_confidences[
                idx_sort[-1]]  # return the highest confidence jersey number

    def extract_jersey_numbers_from_ocr(self, prediction):
        jersey_numbers = []
        jn_confidences = []
        for txt, conf in zip(prediction['rec_texts'], prediction['rec_scores']):
            jn = self.extract_numbers(txt)
            if jn is not None:
                jersey_numbers.append(jn)
                jn_confidences.append(conf)
        jersey_number, jn_confidence = self.choose_best_jersey_number(jersey_numbers,
                                                                      jn_confidences)
        if jersey_number is not None:
            jersey_number = jersey_number[:2]  # Only two-digit numbers are possible
        return jersey_number, jn_confidence

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']
        # for img in batch['img']:
        #     img = img.cpu().numpy()
        # images_np = batch['img']
        # images_np = [img for img in batch['img']]
        # import pickle
        # with open('images_np.pkl', 'wb') as f:
        #     pickle.dump(images_np, f)
        # predictions = self.ocr(images_np, **self.cfg)['predictions']

        predictions = self.run_mmocr_inference(images_np)
        for prediction in predictions:
            jn, conf = self.extract_jersey_numbers_from_ocr(prediction)
            jersey_number_detection.append(jn)
            jersey_number_confidence.append(conf)

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence

        return detections

    def run_mmocr_inference(self, images_np):
        # print('run detection inference')
        result = {}
        result['det'] = self.textdetinferencer(
            images_np,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )['predictions']

        # print('run recognition inference')
        result['rec'] = []
        for img, det_data_sample in zip(
                images_np, result['det']):
            det_pred = det_data_sample.pred_instances
            rec_inputs = []
            for polygon in det_pred['polygons']:
                # Roughly convert the polygon to a quadangle with
                # 4 points
                quad = bbox2poly(poly2bbox(polygon)).tolist()
                rec_input = crop_img(img, quad)
                if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                    # rec_input = np.zeros((1, 1, 3), dtype=np.uint8)
                    continue
                rec_inputs.append(rec_input)
            result['rec'].append(
                self.textrecinferencer(
                    rec_inputs,
                    return_datasamples=True,
                    batch_size=self.batch_size,
                    progress_bar=False)['predictions'])

        pred_results = [{} for _ in range(len(result['rec']))]
        for i, rec_pred in enumerate(result['rec']):
            result_out = dict(rec_texts=[], rec_scores=[])
            for rec_pred_instance in rec_pred:
                rec_dict_res = self.textrecinferencer.pred2dict(
                    rec_pred_instance)
                result_out['rec_texts'].append(rec_dict_res['text'])
                result_out['rec_scores'].append(rec_dict_res['scores'])
            pred_results[i].update(result_out)

        return pred_results
