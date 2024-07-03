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
from nbjw_calib.utils.utils_heatmap import generate_gaussian_array_vectorized_l


class LineKeypointsWCDB(object):
    def __init__(self, image, homography, size_out):

        self.lines_list = ["Big rect. left bottom",
                           "Big rect. left main",
                           "Big rect. left top",
                           "Big rect. right bottom",
                           "Big rect. right main",
                           "Big rect. right top",
                           "Goal left crossbar",
                           "Goal left post left ",
                           "Goal left post right",
                           "Goal right crossbar",
                           "Goal right post left",
                           "Goal right post right",
                           "Middle line",
                           "Side line bottom",
                           "Side line left",
                           "Side line right",
                           "Side line top",
                           "Small rect. left bottom",
                           "Small rect. left main",
                           "Small rect. left top",
                           "Small rect. right bottom",
                           "Small rect. right main",
                           "Small rect. right top"]

        self.line_extremities = [[[0., 54.16], [16.5, 54.16]],
                                 [[16.5, 13.84], [16.5, 54.16]],
                                 [[0., 13.84], [16.5, 13.84]],
                                 [[88.5, 54.16], [105., 54.16]],
                                 [[88.5, 13.84], [88.5, 54.16]],
                                 [[88.5, 13.84], [105., 13.84]],
                                 [[], []],
                                 [[], []],
                                 [[], []],
                                 [[], []],
                                 [[], []],
                                 [[], []],
                                 [[52.5, 0.], [52.5, 68.]],
                                 [[0., 68.], [105., 68.]],
                                 [[0., 0.], [0., 68.]],
                                 [[105., 0.], [105., 68.]],
                                 [[0., 0.], [105., 0.]],
                                 [[0., 43.16], [5.5, 43.16]],
                                 [[5.5, 24.84], [5.5, 43.16]],
                                 [[0., 24.84], [5.5, 24.84]],
                                 [[99.5, 43.16], [105., 43.16]],
                                 [[99.5, 24.84], [99.5, 43.16]],
                                 [[99.5, 24.84], [105., 24.84]],]


        self.homography = homography
        self.image = image

        self.w, self.h = size_out
        self.size = (self.w, self.h)
        self.h_extra = self.h * 0.5
        self.w_extra = self.w * 0.5

        self.lines = {}
        self.num_channels = len(self.lines_list)
        self.mask_array = np.ones(self.num_channels + 1).astype(int)


    def get_tensor_w_mask(self):
        self.get_lines()
        for line in [7, 8, 9, 10, 11, 12]:
            self.mask_array[line-1] = 0

        heatmap_tensor = generate_gaussian_array_vectorized_l(self.num_channels, self.lines, self.size, down_ratio=2,
                                                              sigma=2)
        return heatmap_tensor, self.mask_array


    def get_lines(self):
        for count, line in enumerate(self.line_extremities):
            if len(line[0]) > 0:
                # p1 = self.kpmeters2yards(line[0])
                # p2 = self.kpmeters2yards(line[1])

                p1_img = np.linalg.inv(self.homography) @ np.array([line[0][0], line[0][1], 1.])
                p1_img /= p1_img[-1]
                p2_img = np.linalg.inv(self.homography) @ np.array([line[1][0], line[1][1], 1.])
                p2_img /= p2_img[-1]

                p1_img[0] *= self.w / self.image.size[0]
                p1_img[1] *= self.h / self.image.size[1]
                p2_img[0] *= self.w / self.image.size[0]
                p2_img[1] *= self.h / self.image.size[1]

                flag, pt1, pt2 = cv2.clipLine((0, 0, self.w, self.h), (int(p1_img[0]), int(p1_img[1])), (int(p2_img[0]), int(p2_img[1])))

                if flag:
                    self.lines[count+1] = {'x_1': pt1[0], 'y_1': pt1[1], 'x_2': pt2[0], 'y_2': pt2[1]}


    def kpmeters2yards(self, p):
        wp_arr = np.array([p[0] * 1.09361, p[1] * 1.09361, 1.])
        return wp_arr

    def get_homography(self, homography_file):
        with open(homography_file, 'r') as file:
            lines = file.readlines()
            matrix_elements = []
            for line in lines:
                matrix_elements.extend([float(element) for element in line.split()])
        homography = np.array(matrix_elements).reshape((3, 3))
        homography = homography / homography[2:3, 2:3]
        return homography




