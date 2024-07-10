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
from nbjw_calib.utils.utils_geometry import line_intersection, ellipse_intersection, find_tangent_points, are_points_collinear


class KeypointsDB(object):

    def __init__(self, data, image):

        self.keypoint_pair_list = [['Side line top', 'Side line left'],
                                   ['Side line top', 'Middle line'],
                                   ['Side line right', 'Side line top'],
                                   ['Side line left', 'Big rect. left top'],
                                   ['Big rect. left top', 'Big rect. left main'],
                                   ['Big rect. right top', 'Big rect. right main'],
                                   ['Side line right', 'Big rect. right top'],
                                   ['Side line left', 'Small rect. left top'],
                                   ['Small rect. left top', 'Small rect. left main'],
                                   ['Small rect. right top', 'Small rect. right main'],
                                   ['Side line right', 'Small rect. right top'],
                                   ['Goal left crossbar', 'Goal left post right'],
                                   ['Side line left', 'Goal left post right'],
                                   ['Side line right', 'Goal right post left'],
                                   ['Goal right crossbar', 'Goal right post left'],
                                   ['Goal left crossbar', 'Goal left post left '],
                                   ['Side line left', 'Goal left post left '],
                                   ['Side line right', 'Goal right post right'],
                                   ['Goal right crossbar', 'Goal right post right'],
                                   ['Side line left', 'Small rect. left bottom'],
                                   ['Small rect. left bottom', 'Small rect. left main'],
                                   ['Small rect. right bottom', 'Small rect. right main'],
                                   ['Side line right', 'Small rect. right bottom'],
                                   ['Side line left', 'Big rect. left bottom'],
                                   ['Big rect. left bottom', 'Big rect. left main'],
                                   ['Big rect. right main', 'Big rect. right bottom'],
                                   ['Side line right', 'Big rect. right bottom'],
                                   ['Side line left', 'Side line bottom'],
                                   ['Side line bottom', 'Middle line'],
                                   ['Side line bottom', 'Side line right']]

        self.keypoint_aux_pair_list = [['Small rect. left main', 'Side line top'],
                                       ['Big rect. left main', 'Side line top'],
                                       ['Big rect. right main', 'Side line top'],
                                       ['Small rect. right main', 'Side line top'],
                                       ['Small rect. left main', 'Big rect. left top'],
                                       ['Big rect. right top', 'Small rect. right main'],
                                       ['Small rect. left top', 'Big rect. left main'],
                                       ['Small rect. right top', 'Big rect. right main'],
                                       ['Small rect. left bottom', 'Big rect. left main'],
                                       ['Small rect. right bottom', 'Big rect. right main'],
                                       ['Small rect. left main', 'Big rect. left bottom'],
                                       ['Small rect. right main', 'Big rect. right bottom'],
                                       ['Small rect. left main', 'Side line bottom'],
                                       ['Big rect. left main', 'Side line bottom'],
                                       ['Big rect. right main', 'Side line bottom'],
                                       ['Small rect. right main', 'Side line bottom']]  # 20


        self.keypoint1_pair_list = [['Circle left', 'Big rect. left main'],
                                    ['Circle central', 'Middle line'],
                                    ['Circle right', 'Big rect. right main']]

        self.keypoint2_triplet_list = [['Circle left', 5, [37]],
                                       ['Circle central', 2, [38, 39]],
                                       ['Circle right', 6, [40]],
                                       ['Circle left', 25, [41]],
                                       ['Circle central', 29, [42, 43]],
                                       ['Circle right', 26, [44]]]

        self.keypoints3_check_dict = {'Big rect. left main': [46],
                                      'Circle left': [47],
                                      'Middle line': [51],
                                      'Circle central': list(range(48, 51))+list(range(52, 55)),
                                      'Circle right': [55],
                                      'Big rect. right main': [56]}

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


        self.data = data
        self.image = image
        _, self.h, self.w = self.image.size()
        #self.h, self.w, _ = self.image.shape
        #self.w, self.h = size_out
        self.size = (self.w, self.h)

        self.h_extra = self.h * 0.5
        self.w_extra = self.w * 0.5

        self.keypoints = {}
        self.keypoints1 = {}
        self.keypoints_aux = {}
        self.keypoints2 = {}
        self.keypoints3 = {}
        self.keypoints_final = {}
        self.mask_array = np.ones(58).astype(int)

        self.proj_err_th = 5.
        self.num_channels = len(self.keypoint_world_coords_2D) + 1

    def get_full_keypoints(self):

        self.upsample_ellipse()
        self.get_main_keypoints()
        self.get_keypoints12strat()
        self.retrieve_missing_keypoints()
        self.get_keypoints3_from_homography()
        self.refine_sanity_check([self.keypoints, self.keypoints_aux], [self.keypoints3])
        self.merge_keypoints()

    def get_tensor_w_mask(self):

        self.get_full_keypoints()
        heatmap_tensor = generate_gaussian_array_vectorized(self.num_channels, self.keypoints_final, self.size,
                                                            down_ratio=2, sigma=2, proj_err_th=self.proj_err_th)
        return heatmap_tensor, self.mask_array

    def draw_keypoints(self, show_heatmap=False, scale=1):

        if len(self.keypoints) == 0:
            self.get_full_keypoints()

        if show_heatmap:
            heatmap_tensor = generate_gaussian_array_vectorized(self.num_channels, self.keypoints_final, self.size,
                                                                down_ratio=2, sigma=2)
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(scale*15, scale*7.5))

            s = ax2.matshow(heatmap_tensor[-1])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(s, ax=ax2, cax=cax)

        else:
            fig, ax = plt.subplots(figsize=(scale*15, scale*7.5))

        ax.imshow(self.image)

        for kp in self.keypoints.keys():
            if self.keypoints[kp]:
                if self.keypoints[kp]['close_to_frame']:
                    x, y = self.keypoints[kp]['x'], self.keypoints[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    ax.scatter(x, y, c='orange' if self.keypoints[kp]['retrieved'] else 'r', s=scale*10, zorder=10)

        for kp in self.keypoints_aux.keys():
            if self.keypoints_aux[kp]:
                if self.keypoints_aux[kp]['close_to_frame']:
                    x, y = self.keypoints_aux[kp]['x'], self.keypoints_aux[kp]['y']
                    ax.text(x, y, s=f'{kp}', zorder=11)
                    ax.scatter(x, y, c='yellow', s=scale*10, zorder=5)

        for kp in self.keypoints1.keys():
            if self.keypoints1[kp] and kp in list(range(31, 37)):
                if self.keypoints1[kp]['close_to_frame']:
                    x, y = self.keypoints1[kp]['x'], self.keypoints1[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    if 'proj_err' in self.keypoints1[kp].keys():
                        if self.keypoints1[kp]['proj_err'] > 5.:
                            ax.scatter(x, y, c='black', s=scale*10, zorder=10)
                        else:
                            ax.scatter(x, y, c='b', s=scale*10, zorder=10)
                    else:
                        ax.scatter(x, y, c='b', s=scale*10, zorder=10)

        for kp in self.keypoints2.keys():
            if self.keypoints2[kp] and kp in list(range(37, 45)):
                if self.keypoints2[kp]['close_to_frame']:
                    x, y = self.keypoints2[kp]['x'], self.keypoints2[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    if 'proj_err' in self.keypoints2[kp].keys():
                        if self.keypoints2[kp]['proj_err'] > 5.:
                            ax.scatter(x, y, c='black', s=scale*10, zorder=10)
                        else:
                            ax.scatter(x, y, c='pink', s=scale*10, zorder=10)
                    else:
                        ax.scatter(x, y, c='pink', s=scale*10, zorder=10)

        for kp in self.keypoints3.keys():
            if self.keypoints3[kp] and kp in list(range(45, 58)):
                if self.keypoints3[kp]['close_to_frame']:
                    x, y = self.keypoints3[kp]['x'], self.keypoints3[kp]['y']
                    ax.text(x, y, s=kp, zorder=11)
                    ax.scatter(x, y, c='green', s=scale*10, zorder=10)

        plt.show()

    def draw_field(self, fig_size=8):
        def get_color(kp):
            if kp < 30:
                return 'red'
            elif 30 <= kp < 36:
                return 'blue'
            elif 36 <= kp < 44:
                return 'pink'
            else:
                return 'green'

        f, ax = _draw_field(fig_size)

        for count, kp in enumerate(self.keypoint_world_coords_2D):
            x, y = kp[0], 68 - kp[1]
            if count + 1 in [12, 16]:
                x -= 2
            elif count + 1 in [15, 19]:
                x += 2
            ax.text(x, y + .5, s=count + 1, zorder=11)
            ax.scatter(x, y, c=get_color(count), s=10, zorder=10)

        for count, kp in enumerate(self.keypoint_aux_world_coords_2D):
            x, y = kp[0], 68 - kp[1]
            ax.text(x, y + .5, s=f'{count + 1 + 57}', zorder=11)
            ax.scatter(x, y, c='yellow', s=10, zorder=10)

        plt.show()

    def upsample_ellipse(self):
        num_upsample_points = 5

        for line in ['Circle left', 'Circle central', 'Circle right']:
            if line in self.data.keys():
                if len(self.data[line]) == 4:
                    set_list = self.data[line]
                    x, y = [], []
                    for point in self.data[line]:
                        x.append(point['x'] * self.w)
                        y.append(point['y'] * self.h)

                    coefficients = np.polyfit(x, y, 2)
                    poly_func = np.poly1d(coefficients)

                    num_upsample_points = 5
                    x_upsample = np.linspace(min(x), max(x), num_upsample_points)
                    y_upsample = poly_func(x_upsample) + np.random.normal(scale=0.5, size=num_upsample_points)

                    points = [{'x': p[0] / self.w, 'y': p[1] / self.h} for p in list(zip(x_upsample, y_upsample))]

                    self.data[line] = points


    def get_correspondences(self, keypoints=True, keypoints_aux=True, keypoints1=False, keypoints2=False, keypoints3=False, only_ground_plane=False):
        world_points_p1, world_points_p2, world_points_p3 = [], [], []
        img_points_p1, img_points_p2, img_points_p3 = [], [], []

        if keypoints:
            for kp in self.keypoints.keys():
                if self.keypoints[kp]['close_to_frame'] and not self.keypoints[kp]['retrieved']:
                    wp = self.keypoint_world_coords_2D[kp - 1]
                    if kp in [12, 16]:
                        world_points_p2.append([2.44, wp[1], 0.])
                        img_points_p2.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])
                    elif kp in [1, 4, 8, 13, 17, 20, 24, 28]:
                        world_points_p1.append([wp[0], wp[1], 0.])
                        world_points_p2.append([0., wp[1], 0.])
                        img_points_p1.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])
                        img_points_p2.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])
                    elif kp in [3, 7, 11, 14, 18, 23, 27, 30]:
                        world_points_p1.append([wp[0], wp[1], 0.])
                        world_points_p3.append([0., wp[1], 0.])
                        img_points_p1.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])
                        img_points_p3.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])
                    elif kp in [15, 19]:
                        world_points_p3.append([2.44, wp[1], 0.])
                        img_points_p3.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])
                    else:
                        world_points_p1.append([wp[0], wp[1], 0.])
                        img_points_p1.append([self.keypoints[kp]['x'], self.keypoints[kp]['y']])

        if keypoints_aux:
            for kp in self.keypoints_aux.keys():
                if self.keypoints_aux[kp]['close_to_frame']:
                    wp = self.keypoint_aux_world_coords_2D[kp - 57 - 1]
                    world_points_p1.append([wp[0], wp[1], 0.])
                    img_points_p1.append([self.keypoints_aux[kp]['x'], self.keypoints_aux[kp]['y']])

        if keypoints1:
            for kp in self.keypoints1.keys():
                if self.keypoints1[kp]['close_to_frame']:
                    wp = self.keypoint_world_coords_2D[kp - 1]
                    world_points_p1.append([wp[0], wp[1], 0.])
                    img_points_p1.append([self.keypoints1[kp]['x'], self.keypoints1[kp]['y']])

        if keypoints2:
            for kp in self.keypoints2.keys():
                if self.keypoints2[kp]['close_to_frame']:
                    wp = self.keypoint_world_coords_2D[kp - 1]
                    world_points_p1.append([wp[0], wp[1], 0.])
                    img_points_p1.append([self.keypoints2[kp]['x'], self.keypoints2[kp]['y']])

        if keypoints3:
            for kp in self.keypoints3.keys():
                if self.keypoints3[kp]['close_to_frame']:
                    wp = self.keypoint_world_coords_2D[kp - 1]
                    world_points_p1.append([wp[0], wp[1], 0.])
                    img_points_p1.append([self.keypoints3[kp]['x'], self.keypoints3[kp]['y']])


        if only_ground_plane:
            return [world_points_p1], [img_points_p1]
        else:
            return [world_points_p1, world_points_p2, world_points_p3], [img_points_p1, img_points_p2, img_points_p3]


    def get_frame_projection(self, obj_list, img_list, use_ransac=25.0):

        obj_points, img_points, ord_points = [], [], []

        for i in range(len(obj_list)):
            if len(obj_list[i]) >= 4 and not all(item[0] == obj_list[i][0][0] for item in obj_list[i]):
                obj_points.append(np.array(obj_list[i], dtype=np.float32))
                img_points.append(np.array(img_list[i], dtype=np.float32))
                ord_points.append(i)

        if len(obj_points) == 0:
            return None

        elif ord_points[0] != 0:
            return None

        if use_ransac > 0:
            H, mask = cv2.findHomography(np.array(obj_points[0], dtype=np.float32), np.array(img_points[0], dtype=np.float32), cv2.RANSAC, use_ransac)
        else:
            H, mask = cv2.findHomography(np.array(obj_points[0], dtype=np.float32), np.array(img_points[0], dtype=np.float32))

        return H

    def refine_sanity_check(self, set_reference, set_to_check):

        def find_first_close_to_frame(dictionary):
            for key, nested_dict in dictionary.items():
                if nested_dict.get('close_to_frame') == True and key not in [12, 15, 16, 19]:
                    return key
            return None

        def find_columns_with_minus_one(matrix):
            return [col for col in range(len(matrix[0])) if any(row[col] == -1 for row in matrix)]

        dict_ref = reduce(lambda x, y: {**x, **y}, set_reference)
        dict_check = reduce(lambda x, y: {**x, **y}, set_to_check)

        ref_point1 = find_first_close_to_frame(self.keypoints)

        if ref_point1:

            ipr = [self.keypoints[ref_point1]['x'], self.keypoints[ref_point1]['y']]
            wpr = self.keypoint_world_coords_2D[ref_point1 - 1]

            world_coord_matrix = np.zeros((len(dict_ref), len(dict_check)))
            img_coord_matrix = np.zeros((len(dict_ref), len(dict_check)))

            for count1, kp1 in enumerate(dict_ref.keys()):
                if kp1 not in [12, 15, 16, 19]:
                    wp1 = self.keypoint_world_coords_2D[kp1 - 1] if kp1 < 57 else self.keypoint_aux_world_coords_2D[kp1 - 57 - 1]
                    vw1 = np.array([wp1[0], wp1[1]]) -  np.array([wpr[0], wpr[1]])
                    vi1 = np.array([dict_ref[kp1]['x'], dict_ref[kp1]['y']]) - np.array([ipr[0], ipr[1]])
                    for count2, kp2 in enumerate(dict_check.keys()):
                        wp2 = self.keypoint_world_coords_2D[kp2 - 1]
                        is_collinear = are_points_collinear(wpr, wp1, wp2, 5.)

                        if is_collinear:
                            world_coord_matrix[count1, count2] = 0
                            img_coord_matrix[count1, count2] = 0

                        else:
                            vw2 = np.array([wp2[0], wp2[1]]) -  np.array([wpr[0], wpr[1]])
                            vi2 = np.array([dict_check[kp2]['x'], dict_check[kp2]['y']]) - np.array([ipr[0], ipr[1]])

                            crossw = np.cross(vw1, vw2)
                            crossi = np.cross(vi1, vi2)

                            world_coord_matrix[count1, count2] = crossw
                            img_coord_matrix[count1, count2] = crossi

            cols_to_remove = find_columns_with_minus_one(np.sign(np.multiply(world_coord_matrix, img_coord_matrix)))
            kp_to_remove = [list(dict_check.keys())[i] for i in cols_to_remove]

            for kp in kp_to_remove:
                if kp <= 30:
                    del self.keypoints[kp]
                elif 30 < kp <= 36:
                    del self.keypoints1[kp]
                elif 36 < kp <= 44:
                    del self.keypoints2[kp]
                elif 44 < kp <= 57:
                    del self.keypoints3[kp]
                else:
                    del self.keypoints_aux[kp]

    def get_main_keypoints(self):

        for count, pair in enumerate(self.keypoint_pair_list):
            if all(x in self.data.keys() for x in pair):
                x, y = line_intersection(self.data, pair, self.w, self.h)
                if not np.isnan(x):
                    if (0 <= x < self.w and 0 <= y < self.h):
                        self.keypoints[count + 1] = {'x': x, 'y': y, 'in_frame': True, 'close_to_frame': True,
                                                     'retrieved': False}
                    elif (0 - self.w_extra <= x < self.w + self.w_extra and 0 - self.h_extra <= y < self.h + self.h_extra):
                        self.keypoints[count + 1] = {'x': x, 'y': y, 'in_frame': False, 'close_to_frame': True,
                                                     'retrieved': False}
                    else:
                        self.keypoints[count + 1] = {'x': x, 'y': y, 'in_frame': False, 'close_to_frame': False,
                                                     'retrieved': False}

        for count, pair in enumerate(self.keypoint_aux_pair_list):
            if all(x in self.data.keys() for x in pair):

                x, y = line_intersection(self.data, pair, self.w, self.h)
                if not np.isnan(x):
                    if (0 <= x < self.w and 0 <= y < self.h):
                        self.keypoints_aux[count + 57 + 1] = {'x': x, 'y': y, 'in_frame': True, 'close_to_frame': True,
                                                              'retrieved': False}
                    elif (0 - self.w_extra <= x < self.w + self.w_extra and 0 - self.h_extra <= y < self.h + self.h_extra):
                        self.keypoints_aux[count + 57 + 1] = {'x': x, 'y': y, 'in_frame': False, 'close_to_frame': True,
                                                              'retrieved': False}
                    else:
                        self.keypoints_aux[count + 57 + 1] = {'x': x, 'y': y, 'in_frame': False, 'close_to_frame': False,
                                                              'retrieved': False}

    def retrieve_missing_keypoints(self):

        obj_list, img_list = self.get_correspondences(keypoints1=True, keypoints2=True, only_ground_plane=True)
        H = self.get_frame_projection(obj_list, img_list)

        if H is not None:
            for kp in range(1, 31):
                if kp not in self.keypoints.keys() and kp not in [12, 15, 16, 19]:

                    min_dist = np.inf
                    world_coord = self.keypoint_world_coords_2D[kp - 1]
                    for kp2 in self.keypoints.keys():
                        kp_dist = np.linalg.norm(
                            np.array(self.keypoint_world_coords_2D[kp2 - 1]) - np.array(world_coord))
                        if kp_dist < min_dist:
                            min_dist = kp_dist

                    if min_dist < 5.:

                        p_proj = H @ np.array([world_coord[0], world_coord[1], 1])
                        p_proj /= p_proj[-1]

                        if (0 <= p_proj[0] < self.w and 0 <= p_proj[1] < self.h):
                            self.keypoints[kp] = {'x': p_proj[0],
                                                  'y': p_proj[1],
                                                  'in_frame': True,
                                                  'close_to_frame': True,
                                                  'retrieved': True
                                                  }

        self.keypoints = dict(sorted(self.keypoints.items()))


    def get_kp1and2proposals(self):

        num_proposals = 0
        keypoints1_proposals = {}
        for count, pair in enumerate(self.keypoint1_pair_list):
            if all(x in self.data.keys() for x in pair):
                intersections = ellipse_intersection(self.data, pair, self.w, self.h)

                if len(intersections) == 2:
                    num_proposals += 2
                    p1, p2 = intersections
                    keypoints1_proposals[count + 1 + 30] = [p1, p2]
                    keypoints1_proposals[count + 1 + 30 + 3] = [p1, p2]

                elif len(intersections) == 1:
                    num_proposals += 1
                    p1 = intersections[0]
                    keypoints1_proposals[count + 1 + 30] = [p1, None]
                    keypoints1_proposals[count + 1 + 30 + 3] = [p1, None]

                else:
                    self.mask_array[count + 30] = 0
                    self.mask_array[count + 30 + 3] = 0

        keypoints2_proposals = {}
        for triplet in self.keypoint2_triplet_list:
            if triplet[0] in self.data.keys():
                if triplet[1] in self.keypoints.keys():
                    if self.keypoints[triplet[1]]['close_to_frame']:

                        x1, y1 = [], []
                        for count2, point in enumerate(self.data[triplet[0]]): # Ellipse should be first one of the triplet
                            x1.append(point['x'] * self.w)
                            y1.append(point['y'] * self.h)

                        if len(x1) > 4:
                            X = np.array(list(zip(x1, y1)))
                            reg = LsqEllipse().fit(X)

                            try:
                                center, width, height, theta = reg.as_parameters()
                            except:
                                for kp in triplet[2]:
                                    self.mask_array[kp - 1] = 0
                                continue

                            if isinstance(theta, complex):
                                for kp in triplet[2]:
                                    self.mask_array[kp - 1] = 0
                                    continue

                            ext_p = np.array([self.keypoints[triplet[1]]['x'], self.keypoints[triplet[1]]['y']])
                            inter_list = find_tangent_points(center, width, height, theta, ext_p)

                            if len(inter_list) == 2:
                                p1, p2 = inter_list
                            else:
                                for kp in triplet[2]:
                                    self.mask_array[kp - 1] = 0
                                continue

                            if len(triplet[2]) > 1:
                                num_proposals += 2
                                keypoints2_proposals[triplet[2][0]] = [p1, p2]
                                keypoints2_proposals[triplet[2][1]] = [p1, p2]

                            else:
                                num_proposals += 1
                                keypoints2_proposals[triplet[2][0]] = [p1, p2]

                        else:
                            for kp in triplet[2]:
                                self.mask_array[kp - 1] = 0
                    else:
                        for kp in triplet[2]:
                            self.mask_array[kp - 1] = 0
                else:
                    for kp in triplet[2]:
                        self.mask_array[kp - 1] = 0

        keypoints1_proposals = dict(sorted(keypoints1_proposals.items()))
        keypoints2_proposals = dict(sorted(keypoints2_proposals.items()))

        return keypoints1_proposals, keypoints2_proposals, num_proposals


    def get_keypoints1and2(self):

        def rep_err(H, obj_list_aux, img_list_aux):
            if H is not None:
                rep_err = 0
                for count, point in enumerate(obj_list_aux[0]):
                    p = np.dot(H, np.array([point[0], point[1], 1]))
                    if p [-1] != 0:
                        p /= p[-1]
                    else:
                        p[-1] += 1e-16
                        p /= p[-1]
                    #print(point, p, img_list_aux[0][count])
                    rep_err += np.sum((np.array([img_list_aux[0][count][0], img_list_aux[0][count][1]]) - p[:2]) ** 2)

                rep_err = np.sqrt(rep_err) / len(obj_list_aux[0])
                return rep_err
            return np.inf

        def find_first_in_frame(dictionary):
            for key, nested_dict in dictionary.items():
                if nested_dict.get('close_to_frame') == True:
                    return key
            return None

        def sanity_check(candidates_set, combination):
            reference_point = find_first_in_frame(self.keypoints)

            if not reference_point:
                return False

            wpr = self.keypoint_world_coords_2D[reference_point - 1]
            imr = [self.keypoints[reference_point]['x'], self.keypoints[reference_point]['y']]

            world_coord_matrix = np.zeros((len(candidates_set), len(candidates_set)))
            img_coord_matrix = np.zeros((len(candidates_set), len(candidates_set)))

            for count1, kp1 in enumerate(candidates_set):
                if combination[count1]:
                    wp1 = self.keypoint_world_coords_2D[kp1 - 1]
                    vw1 = np.array([wp1[0], wp1[1]]) -  np.array([wpr[0], wpr[1]])
                    vi1 = np.array([combination[count1][0], combination[count1][1]]) - np.array([imr[0], imr[1]])
                    for count2, kp2 in enumerate(candidates_set):
                        if combination[count2] and count2 != count1:
                            wp2 = self.keypoint_world_coords_2D[kp2 - 1]
                            vw2 = np.array([wp2[0], wp2[1]]) -  np.array([wpr[0], wpr[1]])
                            vi2 = np.array([combination[count2][0], combination[count2][1]]) - np.array([imr[0], imr[1]])

                            crossw = np.cross(vw1, vw2)
                            crossi = np.cross(vi1, vi2)

                            world_coord_matrix[count1, count2] = np.sign(crossw)
                            img_coord_matrix[count1, count2] = np.sign(crossi)

            return (np.multiply(world_coord_matrix, img_coord_matrix) >= 0).all()

        forbidden_combinations = [(31, 34), (32, 35), (33, 36), (38, 39), (42, 43)]
        locations_set1, locations_set2, num_proposals = self.get_kp1and2proposals()

        if num_proposals != 0:
            obj_list, img_list = self.get_correspondences(only_ground_plane=True)
            if num_proposals + len(obj_list[0]) >= 4:

                candidates_set1 = list(locations_set1.keys())
                candidates_set2 = list(locations_set2.keys())
                candidates_set = candidates_set1 + candidates_set2

                forbidden_combinations_pos = [(candidates_set.index(p1), candidates_set.index(p2)) for p1, p2 in forbidden_combinations if p1 in candidates_set and p2 in                candidates_set]


                combinations = list(product(*[locations_set1.get(point, []) if point in candidates_set1 else locations_set2.get(point, []) for point in candidates_set1 + candidates_set2]))
                filtered_combinations = [
                    comb for comb in combinations
                    if all(
                        (index1, index2) not in forbidden_combinations_pos
                        or
                        (comb[index1] is None and comb[index2] is None)
                        or
                        (comb[index1] != comb[index2] if comb[index1] is not None and comb[index2] is not None else True)
                        for index1, index2 in forbidden_combinations_pos
                    )
                ]

                min_rep_err = np.inf
                min_rep_err_sub = np.inf
                min_rep_err_comb = None
                min_rep_err_sub_comb = None
                for comb in filtered_combinations:
                    if sanity_check(candidates_set, comb):
                        obj_list_aux, img_list_aux = copy.deepcopy(obj_list), copy.deepcopy(img_list)
                        for count, kp in enumerate(candidates_set):
                            if comb[count]:
                                wp = self.keypoint_world_coords_2D[kp - 1]
                                obj_list_aux[0].append([wp[0], wp[1], 0.])
                                img_list_aux[0].append([comb[count][0], comb[count][1]])

                        H = self.get_frame_projection(obj_list_aux, img_list_aux, use_ransac=0)
                        comb_rep_err = rep_err(H, obj_list_aux, img_list_aux)

                        if comb_rep_err < min_rep_err:
                            min_rep_err = np.mean(np.array(comb_rep_err))
                            min_rep_err_comb = comb

                        if comb_rep_err < min_rep_err_sub and any(x is not None for x in comb):
                            min_rep_err_sub = comb_rep_err
                            min_rep_err_sub_comb = comb


                final_comb = min_rep_err_comb if min_rep_err_comb else min_rep_err_sub_comb if min_rep_err_sub_comb else None

                if final_comb:
                    count1f = -1
                    for count1, kp1 in enumerate(candidates_set1):
                        count1f += 1
                        if final_comb[count1]:
                            x, y = final_comb[count1]
                            self.keypoints1[kp1] = {'x': x,
                                                    'y': y,
                                                    'in_frame': 0 <= x <= self.w and 0 <= y <= self.h,
                                                    'close_to_frame': -self.w_extra <= x <= self.w + self.w_extra and \
                                                                          -self.h_extra <= y <= self.h + self.h_extra}

                    for count2, kp2 in enumerate(candidates_set2):
                        if final_comb[count1f+count2+1]:
                            x, y = final_comb[count1f+count2+1]
                            self.keypoints2[kp2] = {'x': x,
                                                    'y': y,
                                                    'in_frame': 0 <= x <= self.w and 0 <= y <= self.h,
                                                    'close_to_frame': -self.w_extra <= x <= self.w + self.w_extra and \
                                                                      -self.h_extra <= y <= self.h + self.h_extra}

                else: #mask keypoints if we can't find reliable combination
                    for kp in candidates_set:
                        self.mask_array[kp - 1] = 0

            else:
                for kp in list(locations_set1.keys()) + list(locations_set2.keys()):
                    self.mask_array[kp - 1] = 0


    def get_keypoints1_from_ellipse(self):

        mask_ellipse_intersection = {'left': [['Circle left', 'Big rect. left main'], [31, 34]],
                                     'central': [['Circle central', 'Middle line'], [32, 35]],
                                     'right': [['Circle right', 'Big rect. right main'], [33, 36]]}

        if not all(value is None for value in self.keypoints.values()):

            obj_list, img_list = self.get_correspondences(only_ground_plane=True)
            H = self.get_frame_projection(obj_list, img_list)
            if H is not None:

                for count, pair in enumerate(self.keypoint1_pair_list):
                    if all(x in self.data.keys() for x in pair):
                        intersections = ellipse_intersection(self.data, pair, self.w, self.h)
                        if len(intersections) == 2:

                            p1, p2 = intersections
                            inter_list = [p1, p2]
                            p_world1 = self.keypoint_world_coords_2D[count + 30]
                            p_world2 = self.keypoint_world_coords_2D[count + 30 + 3]

                            p1_proj = np.linalg.inv(H) @ np.array([p1[0], p1[1], 1])
                            p1_proj /= p1_proj[-1]
                            p2_proj = np.linalg.inv(H) @ np.array([p2[0], p2[1], 1])
                            p2_proj /= p2_proj[-1]

                            cost_matrix = np.array([[np.linalg.norm(p_world1 - np.array([p1_proj[0], p1_proj[1]])),
                                                     np.linalg.norm(p_world2 - np.array([p1_proj[0], p1_proj[1]]))],
                                                    [np.linalg.norm(p_world1 - np.array([p2_proj[0], p2_proj[1]])),
                                                     np.linalg.norm(p_world2 - np.array([p2_proj[0], p2_proj[1]]))]])


                            try:
                                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                            except:
                                self.mask_array[count + 30] = 0
                                self.mask_array[count + 30 + 3] = 0
                                continue

                            matching = list(zip(row_ind, col_ind))
                            matching_sort = sorted(matching, key=lambda x: x[1])

                            self.keypoints1[count + 30 + 1] = {
                                'x': inter_list[matching_sort[0][0]][0],
                                'y': inter_list[matching_sort[0][0]][1],
                                'proj_err': cost_matrix[matching_sort[0]],
                                'in_frame': (0 <= intersections[matching_sort[0][0]][0] < self.w and 0 <=
                                             intersections[matching_sort[0][0]][1] < self.h),
                                'close_to_frame': (0 - self.w_extra <= intersections[matching_sort[0][0]][
                                    0] < self.w + self.w_extra and 0 - self.h_extra <= intersections[matching_sort[0][0]][
                                                       1] < self.h + self.h_extra)
                            }

                            self.keypoints1[count + 30 + 4] = {
                                'x': inter_list[matching_sort[1][0]][0],
                                'y': inter_list[matching_sort[1][0]][1],
                                'proj_err': cost_matrix[matching_sort[1]],
                                'in_frame': (0 <= intersections[matching_sort[1][0]][0] < self.w and 0 <=
                                             intersections[matching_sort[1][0]][1] < self.h),
                                'close_to_frame': (0 - self.w_extra <= intersections[matching_sort[1][0]][
                                    0] < self.w + self.w_extra and 0 - self.h_extra <= intersections[matching_sort[1][0]][
                                                       1] < self.h + self.h_extra)
                            }

                            for kp in [count + 30 + 1, count + 30 + 4]:
                                if self.keypoints1[kp]['proj_err'] > self.proj_err_th:
                                    self.mask_array[kp - 1] = 0

                        elif len(intersections) == 1:

                            p1 = intersections[0]
                            p_world1 = self.keypoint_world_coords_2D[count + 30]
                            p_world2 = self.keypoint_world_coords_2D[count + 33]

                            p1_proj = np.linalg.inv(H) @ np.array([p1[0], p1[1], 1])
                            p1_proj /= p1_proj[-1]

                            cost_matrix = np.array([[np.linalg.norm(p_world1 - np.array([p1_proj[0], p1_proj[1]])),
                                                     np.linalg.norm(p_world2 - np.array([p1_proj[0], p1_proj[1]]))]])
                            row_ind, col_ind = linear_sum_assignment(cost_matrix)
                            matching = list(zip(row_ind, col_ind))

                            if matching[0][1] == 0:
                                self.keypoints1[count + 30 + 1] = {'x': p1[0],
                                                                   'y': p1[1],
                                                                   'proj_err': cost_matrix[row_ind[0], col_ind[0]],
                                                                   'in_frame': (0 <= p1[0] < self.w and 0 <= p1[
                                                                       1] < self.h),
                                                                   'close_to_frame': (0 - self.w_extra <= p1[
                                                                       0] < self.w + self.w_extra and 0 - self.h_extra <= p1[
                                                                                          1] < self.h + self.h_extra)}
                                if self.keypoints1[count + 30 + 1]['proj_err'] > self.proj_err_th:
                                    self.mask_array[count + 30] = 0

                            else:
                                self.keypoints1[count + 30 + 4] = {'x': p1[0],
                                                                   'y': p1[1],
                                                                   'proj_err': cost_matrix[row_ind[0], col_ind[0]],
                                                                   'in_frame': (0 <= p1[0] < self.w and 0 <= p1[
                                                                       1] < self.h),
                                                                   'close_to_frame': (0 - self.w_extra <= p1[
                                                                       0] < self.w + self.w_extra and 0 - self.h_extra <= p1[
                                                                                          1] < self.h + self.h_extra)}
                                if self.keypoints1[count + 30 + 4]['proj_err'] > self.proj_err_th:
                                    self.mask_array[count + 30 + 3] = 0
                        else:
                            self.mask_array[count + 30] = 0
                            self.mask_array[count + 33] = 0

            else:
                for side in mask_ellipse_intersection.keys():
                    if all(x in self.data.keys() for x in mask_ellipse_intersection[side][0]):
                        for kp in mask_ellipse_intersection[side][1]:
                            self.mask_array[kp - 1] = 0


            self.keypoints1 = dict(sorted(self.keypoints1.items()))

    def get_keypoints2_from_tangents(self):

        mask_tangents = {'Circle left': {5: [37], 25: [1]},
                         'Circle central': {2: [38, 39], 29: [42, 43]},
                         'Circle right': {6: [40], 26: [44]}}

        if not all(value is None for value in self.keypoints.values()):

            obj_list, img_list = self.get_correspondences(only_ground_plane=True, keypoints1=True)
            H = self.get_frame_projection(obj_list, img_list)
            if H is not None:

                for triplet in self.keypoint2_triplet_list:
                    if triplet[0] in self.data.keys():
                        if triplet[1] in self.keypoints.keys():
                            if self.keypoints[triplet[1]]['close_to_frame']:

                                x1, y1 = [], []

                                # Ellipse should be first one of the triplet
                                for count2, point in enumerate(self.data[triplet[0]]):
                                    x1.append(point['x'] * self.w)
                                    y1.append(point['y'] * self.h)

                                if len(x1) > 4:
                                    X = np.array(list(zip(x1, y1)))
                                    reg = LsqEllipse().fit(X)

                                    try:
                                        center, width, height, theta = reg.as_parameters()
                                    except:
                                        continue

                                    if isinstance(theta, complex):
                                        continue
                                else:
                                    continue

                                ext_p = np.array([self.keypoints[triplet[1]]['x'], self.keypoints[triplet[1]]['y']])
                                inter_list = find_tangent_points(center, width, height, theta, ext_p)

                                if len(inter_list) == 2:
                                    p1, p2 = inter_list
                                else:
                                    continue

                                if len(triplet[2]) > 1:

                                    p_world1 = self.keypoint_world_coords_2D[triplet[2][0] - 1]
                                    p_world2 = self.keypoint_world_coords_2D[triplet[2][1] - 1]

                                    p1_proj = np.linalg.inv(H) @ np.array([p1[0], p1[1], 1])
                                    p1_proj /= p1_proj[-1]
                                    p2_proj = np.linalg.inv(H) @ np.array([p2[0], p2[1], 1])
                                    p2_proj /= p2_proj[-1]

                                    cost_matrix = np.array([[np.linalg.norm(p_world1 - np.array([p1_proj[0], p1_proj[1]])),
                                                             np.linalg.norm(p_world2 - np.array([p1_proj[0], p1_proj[1]]))],
                                                            [np.linalg.norm(p_world1 - np.array([p2_proj[0], p2_proj[1]])),
                                                             np.linalg.norm(
                                                                 p_world2 - np.array([p2_proj[0], p2_proj[1]]))]])
                                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                                    matching = list(zip(row_ind, col_ind))
                                    matching_sort = sorted(matching, key=lambda x: x[1])

                                    self.keypoints2[triplet[2][0]] = {'x': inter_list[matching_sort[0][0]][0],
                                                                      'y': inter_list[matching_sort[0][0]][1],
                                                                      'proj_err': cost_matrix[matching_sort[0]],
                                                                      'in_frame': (0 <= inter_list[matching_sort[0][0]][
                                                                          0] < self.w and 0 <=
                                                                                   inter_list[matching_sort[0][0]][
                                                                                       1] < self.h),
                                                                      'close_to_frame': (0 - self.w_extra <=
                                                                                         inter_list[matching_sort[0][0]][
                                                                                             0] < self.w + self.w_extra and 0 - self.h_extra <=
                                                                                         inter_list[matching_sort[0][0]][
                                                                                             1] < self.h + self.h_extra)
                                                                      }

                                    self.keypoints2[triplet[2][1]] = {'x': inter_list[matching_sort[1][0]][0],
                                                                      'y': inter_list[matching_sort[1][0]][1],
                                                                      'proj_err': cost_matrix[matching_sort[1]],
                                                                      'in_frame': (0 <= inter_list[matching_sort[1][0]][
                                                                          0] < self.w and 0 <=
                                                                                   inter_list[matching_sort[1][0]][
                                                                                       1] < self.h),
                                                                      'close_to_frame': (0 - self.w_extra <=
                                                                                         inter_list[matching_sort[1][0]][
                                                                                             0] < self.w + self.w_extra and 0 - self.h_extra <=
                                                                                         inter_list[matching_sort[1][0]][
                                                                                             1] < self.h + self.h_extra)
                                                                      }

                                    for kp in [triplet[2][0], triplet[2][1]]:
                                        if self.keypoints2[kp]['proj_err'] > self.proj_err_th:
                                            self.mask_array[kp - 1] = 0

                                else:

                                    p_world1 = self.keypoint_world_coords_2D[triplet[2][0] - 1]

                                    p1_proj = np.linalg.inv(H) @ np.array([p1[0], p1[1], 1])
                                    p1_proj /= p1_proj[-1]
                                    p2_proj = np.linalg.inv(H) @ np.array([p2[0], p2[1], 1])
                                    p2_proj /= p2_proj[-1]

                                    cost_matrix = np.array([[np.linalg.norm(p_world1 - np.array([p1_proj[0], p1_proj[1]]))],
                                                            [np.linalg.norm(
                                                                p_world1 - np.array([p2_proj[0], p2_proj[1]]))]])

                                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                                    matching = list(zip(row_ind, col_ind))
                                    matching_sort = sorted(matching, key=lambda x: x[1])

                                    self.keypoints2[triplet[2][0]] = {'x': inter_list[matching_sort[0][0]][0],
                                                                      'y': inter_list[matching_sort[0][0]][1],
                                                                      'proj_err': cost_matrix[matching_sort[0]],
                                                                      'in_frame': (0 <= inter_list[matching_sort[0][0]][
                                                                          0] < self.w and 0 <=
                                                                                   inter_list[matching_sort[0][0]][
                                                                                       1] < self.h),
                                                                      'close_to_frame': (0 - self.w_extra <=
                                                                                         inter_list[matching_sort[0][0]][
                                                                                             0] < self.w + self.w_extra and 0 - self.h_extra <=
                                                                                         inter_list[matching_sort[0][0]][
                                                                                             1] < self.h + self.h_extra)}

                                    if self.keypoints2[triplet[2][0]]['proj_err'] > self.proj_err_th:
                                        self.mask_array[triplet[2][0] - 1] = 0
                            else:
                                for kp in triplet[2]:
                                    self.mask_array[kp - 1] = 0
                        else:
                            for kp in triplet[2]:
                                self.mask_array[kp - 1] = 0

            else:
                for line in mask_tangents.keys():
                    if line in self.data.keys():
                        if len(self.data[line]) < 4:
                            for kp in mask_tangents[line].keys():
                                for kp_tangent in mask_tangents[line][kp]:
                                    self.mask_array[kp_tangent - 1] = 0
                        else:
                            for kp in mask_tangents[line].keys():
                                for kp_tangent in mask_tangents[line][kp]:
                                    if kp not in self.keypoints.keys():
                                        self.mask_array[kp_tangent - 1] = 0


            self.keypoints2 = dict(sorted(self.keypoints2.items()))


    def get_keypoints12strat(self):
        n = 0
        for key, nested_dict in self.keypoints.items():
            if nested_dict.get('close_to_frame') == True:
                n += 1
        for key, nested_dict in self.keypoints_aux.items():
            if nested_dict.get('close_to_frame') == True:
                n += 1

        if n < 4:
            self.get_keypoints1and2()
        else:
            self.get_keypoints1_from_ellipse()
            self.get_keypoints2_from_tangents()
            self.refine_sanity_check([self.keypoints, self.keypoints_aux], [self.keypoints1, self.keypoints2])


    def get_keypoints3_from_homography(self):

        def check_num_lines(world_points):
            if len(world_points) > 0:
                world_points = np.array(world_points)
                projected_points_xy = world_points[:, :2]
                sorted_points_x = projected_points_xy[np.argsort(projected_points_xy[:, 0])]
                sorted_points_y = projected_points_xy[np.argsort(projected_points_xy[:, 1])]

                unique_x_coordinates = np.unique(sorted_points_x[:, 0])
                unique_y_coordinates = np.unique(sorted_points_y[:, 1])

                num_lines_x = len(unique_x_coordinates)
                num_lines_y = len(unique_y_coordinates)

                if num_lines_x < 3 or num_lines_y < 3:
                    return False
                else:
                    return True
            return False

        mask_homography_based = {'Big rect. left main': [46],
                                 'Circle left': [47],
                                 'Circle central': list(range(48, 55)),
                                 'Circle right': [55],
                                 'Big rect. right main': [56]}

        penalty_dict = {45: [[5, 25],[68, 62]], #Well paired!
                        57: [[6, 26],[69, 63]]}

        for kp in [45, 57]:
            if all(keypoint in self.keypoints.keys() for keypoint in penalty_dict[kp][0]) and \
                    all(keypoint in self.keypoints_aux.keys() for keypoint in penalty_dict[kp][1]):
                if all(self.keypoints[keypoint]['close_to_frame']==True for keypoint in penalty_dict[kp][0]) and \
                        all(self.keypoints_aux[keypoint]['close_to_frame']==True for keypoint in penalty_dict[kp][1]):

                    p11 = np.array([self.keypoints[penalty_dict[kp][0][0]]['x'], self.keypoints[penalty_dict[kp][0][0]]['y']])
                    p12 = np.array([self.keypoints[penalty_dict[kp][0][1]]['x'], self.keypoints[penalty_dict[kp][0][1]]['y']])
                    p21 = np.array([self.keypoints_aux[penalty_dict[kp][1][0]]['x'], self.keypoints_aux[penalty_dict[kp][1][0]]['y']])
                    p22 = np.array([self.keypoints_aux[penalty_dict[kp][1][1]]['x'], self.keypoints_aux[penalty_dict[kp][1][1]]['y']])

                    x1, x2 = np.array([p11[0], p21[0]]), np.array([p12[0], p22[0]])
                    y1, y2 = np.array([p11[1], p21[1]]), np.array([p12[1], p22[1]])

                    slope1, intercept1, r1, p1, se1 = linregress(x1, y1)
                    slope2, intercept2, r2, p2, se2 = linregress(x2, y2)

                    x_intersection = (intercept2 - intercept1) / (slope1 - slope2)
                    y_intersection = slope1 * x_intersection + intercept1

                    self.keypoints3[kp] = {'x': x_intersection,
                                           'y': y_intersection,
                                           'in_frame': (0 <= x_intersection < self.w and 0 <= y_intersection < self.h),
                                           'close_to_frame': (0 - self.w_extra <= x_intersection < self.w + self.w_extra and \
                                                              0 - self.h_extra <= y_intersection < self.h + self.h_extra),
                                           'geometric': True}

        if len(self.keypoints3) != 0:
            self.refine_sanity_check([self.keypoints, self.keypoints_aux], [self.keypoints3])

        obj_list, img_list = self.get_correspondences(only_ground_plane=True, keypoints1=True, keypoints2=True, keypoints3=True)
        if check_num_lines(obj_list[0]):
            H = self.get_frame_projection(obj_list, img_list)
            if H is not None:

                points_to_compute = []
                for key in self.keypoints3_check_dict.keys():
                    if key in self.data.keys():
                        for point in self.keypoints3_check_dict[key]:
                            points_to_compute.append(point)

                for kp in points_to_compute:
                    wp = self.keypoint_world_coords_2D[kp - 1]
                    p = H @ np.array([wp[0], wp[1], 1.])
                    p /= p[-1]

                    self.keypoints3[kp] = {'x': p[0],
                                           'y': p[1],
                                           'in_frame': (0 <= p[0] < self.w and 0 <= p[1] < self.h),
                                           'close_to_frame': (0 - self.w_extra <= p[0] < self.w + self.w_extra and \
                                                              0 - self.h_extra <= p[1] < self.h + self.h_extra)}


                for kp in [45, 57]:
                    if kp not in self.keypoints3.keys():
                        w_coord = self.keypoint_world_coords_2D[kp - 1]
                        p = H @ np.array([w_coord[0], w_coord[1], 1])
                        p /= p[-1]

                        self.keypoints3[kp] = {'x': p[0],
                                               'y': p[1],
                                               'in_frame': (0 <= p[0] < self.w and 0 <= p[1] < self.h),
                                               'close_to_frame': (0 - self.w_extra <= p[0] < self.w + self.w_extra and \
                                                                  0 - self.h_extra <= p[1] < self.h + self.h_extra),
                                               'geometric': False}

            else:
                for key in mask_homography_based.keys():
                    if key in self.data.keys():
                        for kp in mask_homography_based[key]:
                            self.mask_array[kp - 1] = 0

                for kp in [45, 57]:
                    if kp not in self.keypoints3.keys():
                        self.mask_array[kp - 1] = 0
        else:
            for key in mask_homography_based.keys():
                if key in self.data.keys():
                    for kp in mask_homography_based[key]:
                        self.mask_array[kp - 1] = 0

            for kp in [45, 57]:
                if kp not in self.keypoints3.keys():
                    self.mask_array[kp - 1] = 0



        self.keypoints3 = dict(sorted(self.keypoints3.items()))

    def merge_keypoints(self):

        # Update the result_dict with each individual dictionary
        for kp in [self.keypoints, self.keypoints1, self.keypoints2, self.keypoints3]:
            self.keypoints_final.update(kp)

        self.keypoints_final = dict(sorted(self.keypoints_final.items()))
























































