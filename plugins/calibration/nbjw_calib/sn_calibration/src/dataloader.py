"""
DataLoader used to train the segmentation network used for the prediction of extremities.
"""

import json
import os
import time
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.soccerpitch import SoccerPitch


class SoccerNetDataset(Dataset):
    def __init__(self,
                 datasetpath,
                 split="test",
                 width=640,
                 height=360,
                 mean="../resources/mean.npy",
                 std="../resources/std.npy"):
        self.mean = np.load(mean)
        self.std = np.load(std)
        self.width = width
        self.height = height

        dataset_dir = os.path.join(datasetpath, split)
        if not os.path.exists(dataset_dir):
            print("Invalid dataset path !")
            exit(-1)

        frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]

        self.data = []
        self.n_samples = 0
        for frame in frames:

            frame_index = frame.split(".")[0]
            annotation_file = os.path.join(dataset_dir, f"{frame_index}.json")
            if not os.path.exists(annotation_file):
                continue
            with open(annotation_file, "r") as f:
                groundtruth_lines = json.load(f)
            img_path = os.path.join(dataset_dir, frame)
            if groundtruth_lines:
                self.data.append({
                    "image_path": img_path,
                    "annotations": groundtruth_lines,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        img = cv.imread(item["image_path"])
        img = cv.resize(img, (self.width, self.height), interpolation=cv.INTER_LINEAR)

        mask = np.zeros(img.shape[:-1], dtype=np.uint8)
        img = np.asarray(img, np.float32) / 255.
        img -= self.mean
        img /= self.std
        img = img.transpose((2, 0, 1))
        for class_number, class_ in enumerate(SoccerPitch.lines_classes):
            if class_ in item["annotations"].keys():
                key = class_
                line = item["annotations"][key]
                prev_point = line[0]
                for i in range(1, len(line)):
                    next_point = line[i]
                    cv.line(mask,
                            (int(prev_point["x"] * mask.shape[1]), int(prev_point["y"] * mask.shape[0])),
                            (int(next_point["x"] * mask.shape[1]), int(next_point["y"] * mask.shape[0])),
                            class_number + 1,
                            2)
                    prev_point = next_point
        return img, mask


if __name__ == "__main__":

    # Load the arguments
    parser = ArgumentParser(description='dataloader')

    parser.add_argument('--SoccerNet_path', default="./annotations/", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('--tiny', required=False, type=int, default=None, help='Select a subset of x games')
    parser.add_argument('--split', required=False, type=str, default="test", help='Select the split of data')
    parser.add_argument('--num_workers', required=False, type=int, default=4,
                        help='number of workers for the dataloader')
    parser.add_argument('--resolution_width', required=False, type=int, default=1920,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=1080,
                        help='height resolution of the images')
    parser.add_argument('--preload_images', action='store_true',
                        help="Preload the images when constructing the dataset")
    parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

    args = parser.parse_args()

    start_time = time.time()
    soccernet = SoccerNetDataset(args.SoccerNet_path, split=args.split)
    with tqdm(enumerate(soccernet), total=len(soccernet), ncols=160) as t:
        for i, data in t:
            img = soccernet[i][0].astype(np.uint8).transpose((1, 2, 0))
            print(img.shape)
            print(img.dtype)
            cv.imshow("Normalized image", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print(data[1].shape)
            cv.imshow("Mask", soccernet[i][1].astype(np.uint8))
            cv.waitKey(0)
            cv.destroyAllWindows()
            continue
    end_time = time.time()
    print(end_time - start_time)
