import sys
import cv2
import math
import copy
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from ellipse import LsqEllipse
from itertools import product
from functools import reduce

from nbjw_calib.utils.utils_field import _draw_field
from nbjw_calib.utils.utils_heatmap import generate_gaussian_array_vectorized


class KeypointsWCDB(object):
    def __init__(self, image, homography, size_out=(960,540)):

        self.keypoint_world_coords_2D = [[0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84],
                                         [105., 13.84], [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84],
                                         [0., 30.34], [0., 30.34], [105., 30.34], [105., 30.34], [0., 37.66],
                                         [0., 37.66], [105., 37.66], [105., 37.66], [0., 43.16], [5.5, 43.16],
                                         [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16], [88.5, 54.16],
                                         [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
                                         [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31],
                                         [19.99, 32.29], [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7],
                                         [43.68, 36.46], [61.31, 36.46], [85., 35.7], [11., 34.], [16.5, 34.],
                                         [20.15, 34.], [46.03, 27.53], [58.97, 27.53], [43.35, 34.], [52.5, 34.],
                                         [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.], [88.5, 34.],
                                         [94., 34.]]  # 57

        self.keypoint_aux_world_coords_2D = [[5.5, 0], [16.5, 0], [88.5, 0], [99.5, 0], [5.5, 13.84], [99.5, 13.84],
                                             [16.5, 24.84], [88.5, 24.84], [16.5, 43.16], [88.5, 43.16], [5.5, 54.16],
                                             [99.5, 54.16], [5.5, 68], [16.5, 68], [88.5, 68], [99.5, 68]]

        self.lines_retrieval = [[24, 25], [5, 25], [4, 5], [26, 27], [6, 26], [12, 16], [16, 17], [12, 13], [15, 19],
                                [14, 15], [18, 19], [2, 29], [28, 29, 30], [1, 4, 8, 13, 17, 20, 24, 28],
                                [3, 7, 11, 14, 18, 23, 27, 30], [1, 2, 3], [20, 21], [9, 21], [8, 9], [22, 23],
                                [10, 22], [10, 11]]

        # self.homography = self.get_homography(homography)
        self.homography = homography
        self.image = image

        self.w, self.h = size_out
        self.size = (self.w, self.h)
        self.h_extra = self.h * 0.5
        self.w_extra = self.w * 0.5

        self.keypoints_final = {}

        self.num_channels = len(self.keypoint_world_coords_2D) + 1
        self.mask_array = np.ones(self.num_channels).astype(int)


    def get_tensor_w_mask(self):

        self.get_kp_from_homography()
        for kp in [12,15,16,19]:
            self.mask_array[kp-1] = 0
        heatmap_tensor = generate_gaussian_array_vectorized(self.num_channels, self.keypoints_final, self.size,
                                                            down_ratio=2, sigma=2)
        return heatmap_tensor, self.mask_array


    def kpmeters2yards(self, kp):
        wp = self.keypoint_world_coords_2D[kp - 1]
        wp_arr = np.array([wp[0] * 1.09361, wp[1] * 1.09361, 1.])
        return wp_arr


    # def get_homography(self, homography_file):
    #     with open(homography_file, 'r') as file:
    #         lines = file.readlines()
    #         matrix_elements = []
    #         for line in lines:
    #             matrix_elements.extend([float(element) for element in line.split()])
    #     homography = np.array(matrix_elements).reshape((3, 3))
    #     homography = homography / homography[2:3, 2:3]
    #     return homography


    def get_kp_from_homography(self):
        for kp in range(1, len(self.keypoint_world_coords_2D)+1):
            if kp not in [12, 15, 16, 19]:
                #wp_arr = self.kpmeters2yards(kp)
                wp = self.keypoint_world_coords_2D[kp-1]
                img_pt = np.linalg.inv(self.homography) @ np.array([wp[0], wp[1], 1.])
                img_pt /= img_pt[-1]
                img_pt[0] *= self.w / self.image.size[0]
                img_pt[1] *= self.h / self.image.size[1]

                self.keypoints_final[kp] = {'x': img_pt[0],
                                            'y': img_pt[1],
                                            'in_frame': True if 0 <= img_pt[0] <= self.w and 0 <= img_pt[1] <= self.w else False,
                                            'close_to_frame': True if -self.w_extra <= img_pt[0] <= self.w + self.w_extra and \
                                                                      -self.h_extra <= img_pt[1] <= self.h + self.h_extra else False}


    def get_lines_from_keypoints(self):
        if len(self.keypoints_final) == 0:
            self.get_kp_from_homography()

            ...

    def draw_keypoints(self, scale=1):

        if len(self.keypoints_final) == 0:
            self.get_kp_from_homography()

        fig, ax = plt.subplots(figsize=(scale*15, scale*7.5))
        ax.imshow(self.image)
        for kp in self.keypoints_final.keys():
            if kp <= 30:
                if self.keypoints_final[kp]['close_to_frame']:
                    x, y = self.keypoints_final[kp]['x'], self.keypoints_final[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    ax.scatter(x, y, c='r', s=scale*10, zorder=10)


            elif 30 < kp <= 36:
                if self.keypoints_final[kp]['close_to_frame']:
                    x, y = self.keypoints_final[kp]['x'], self.keypoints_final[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    ax.scatter(x, y, c='b', s=scale*10, zorder=10)


            elif 36 < kp <= 44:
                if self.keypoints_final[kp]['close_to_frame']:
                    x, y = self.keypoints_final[kp]['x'], self.keypoints_final[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    ax.scatter(x, y, c='pink', s=scale*10, zorder=10)


            elif 44 < kp <= 57:
                if self.keypoints_final[kp]['close_to_frame']:
                    x, y = self.keypoints_final[kp]['x'], self.keypoints_final[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    ax.scatter(x, y, c='green', s=scale*10, zorder=10)

        plt.show()


