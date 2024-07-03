import sys
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nbjw_calib.utils.utils_geometry import line_intersection
from nbjw_calib.utils.utils_heatmap import generate_gaussian_array_vectorized_l


class LineKeypointsDB(object):
    def __init__(self, data, image):

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

        self.line_keypoints_dict = {1: [24, 25],
                                    2: [5, 25],
                                    3: [4, 5],
                                    4: [26, 27],
                                    5: [6, 26],
                                    6: [6, 7],
                                    7: [12, 16],
                                    8: [16, 17],
                                    9: [12, 13],
                                    10: [15, 19],
                                    11: [14, 15],
                                    12: [18, 19],
                                    13: [2, 29],
                                    14: [28, 30],
                                    15: [1, 28],
                                    16: [3, 30],
                                    17: [1, 3],
                                    18: [20, 21],
                                    19: [9, 21],
                                    20: [8, 9],
                                    21: [22, 23],
                                    22: [10, 22],
                                    23: [10, 11]}

        self.data = data
        self.image = image
        _, self.h, self.w = self.image.size()
        # self.h, self.w, _ = self.image.shape
        self.size = (self.w, self.h)

        self.num_channels = len(self.lines_list)
        self.lines = {}
        self.keypoints = {}

    def get_tensor(self):
        self.get_lines()
        self.refine_point_lines()
        heatmap_tensor = generate_gaussian_array_vectorized_l(self.num_channels, self.lines, self.size, down_ratio=2,
                                                              sigma=2)
        return heatmap_tensor

    def draw_keypoints(self, show_heatmap=False):

        if len(self.lines) == 0:
            self.get_lines()
            self.refine_point_lines()

        if show_heatmap:
            heatmap = generate_gaussian_array_vectorized_l(self.num_channels, self.lines, self.size, down_ratio=2,
                                                           sigma=2)
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

            s = ax2.matshow(heatmap.sum)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(s, ax=ax2, cax=cax)

        else:
            fig, ax = plt.subplots()

        ax.imshow(self.image)

        for kp in self.lines.keys():
            for suf in ['_1', '_2']:
                x, y = self.lines[kp]['x' + suf], self.lines[kp]['y' + suf]
                ax.text(x, y, s=kp, zorder=11)
                ax.scatter(x, y, c='red', s=10, zorder=10)

        plt.show()

    def find_most_distanced_points(self, segment):
        # Generate all pairs of points in the segment
        point_pairs = list(itertools.combinations(segment, 2))

        # Calculate the Euclidean distance for each pair of points
        distances = [math.dist((pair[0]['x'], pair[0]['y']), (pair[1]['x'], pair[1]['y'])) for pair in point_pairs]

        # Find the indices of the pair with the maximum distance
        max_distance_index = distances.index(max(distances))

        # Extract the points from the pair with the maximum distance
        most_distanced_points = list(point_pairs[max_distance_index])

        return most_distanced_points

    def get_main_keypoints(self):
        for count, pair in enumerate(self.keypoint_pair_list):
            if all(x in self.data.keys() for x in pair):
                x, y = line_intersection(self.data, pair, self.w, self.h)
                if not np.isnan(x):
                    if (0 <= x < self.w and 0 <= y < self.h):
                        self.keypoints[count + 1] = {'x': x, 'y': y, 'in_frame': True}
                    else:
                        self.keypoints[count + 1] = {'x': x, 'y': y, 'in_frame': False}
                else:
                    self.keypoints[count + 1] = {'in_frame': False}
            else:
                self.keypoints[count + 1] = {'in_frame': False}

    def get_lines(self):
        for line in self.data.keys():
            if line in self.lines_list:
                if len(self.data[line]) > 1:
                    points = self.find_most_distanced_points(self.data[line])
                    x1, y1, x2, y2 = points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']
                    self.lines[self.lines_list.index(line) + 1] = {'x_1': x1 * self.w, 'y_1': y1 * self.h,
                                                                   'x_2': x2 * self.w, 'y_2': y2 * self.h}

    def refine_point_lines(self):
        self.get_main_keypoints()
        for line in self.lines.keys():
            p1 = np.array([self.lines[line]['x_1'], self.lines[line]['y_1']])
            p2 = np.array([self.lines[line]['x_2'], self.lines[line]['y_2']])

            kp_to_refine = self.line_keypoints_dict[line]
            kp1 = np.array([self.keypoints[kp_to_refine[0]]['x'], self.keypoints[kp_to_refine[0]]['y']]) if \
                self.keypoints[kp_to_refine[0]]['in_frame'] else \
                np.array([np.nan, np.nan])
            kp2 = np.array([self.keypoints[kp_to_refine[1]]['x'], self.keypoints[kp_to_refine[1]]['y']]) if \
                self.keypoints[kp_to_refine[1]]['in_frame'] else \
                np.array([np.nan, np.nan])


            if all(np.isnan(kp1)) and all(np.isnan(kp2)):
                continue

            elif not all(np.isnan(kp1)) and all(np.isnan(kp2)):
                dist1 = np.linalg.norm(p1 - kp1)
                dist2 = np.linalg.norm(p2 - kp1)

                if dist1 < dist2:
                    self.lines[line]['x_1'] = kp1[0]
                    self.lines[line]['y_1'] = kp1[1]
                else:
                    self.lines[line]['x_2'] = kp1[0]
                    self.lines[line]['y_2'] = kp1[1]

            elif all(np.isnan(kp1)) and not all(np.isnan(kp2)):
                dist1 = np.linalg.norm(p1 - kp2)
                dist2 = np.linalg.norm(p2 - kp2)

                if dist1 < dist2:
                    self.lines[line]['x_1'] = kp2[0]
                    self.lines[line]['y_1'] = kp2[1]
                else:
                    self.lines[line]['x_2'] = kp2[0]
                    self.lines[line]['y_2'] = kp2[1]
            else:
                dis11 = np.linalg.norm(p1 - kp1)
                dis12 = np.linalg.norm(p1 - kp2)
                dis21 = np.linalg.norm(p2 - kp1)
                dis22 = np.linalg.norm(p2 - kp2)

                min_distance_p1 = min(dis11, dis12)
                min_distance_p2 = min(dis21, dis22)

                if min_distance_p1 < min_distance_p2:
                    if dis11 < dis12:
                        self.lines[line]['x_1'] = kp1[0]
                        self.lines[line]['y_1'] = kp1[1]
                        self.lines[line]['x_2'] = kp2[0]
                        self.lines[line]['y_2'] = kp2[1]
                    else:
                        self.lines[line]['x_1'] = kp2[0]
                        self.lines[line]['y_1'] = kp2[1]
                        self.lines[line]['x_2'] = kp1[0]
                        self.lines[line]['y_2'] = kp1[1]
                else:
                    if dis11 < dis21:
                        self.lines[line]['x_1'] = kp1[0]
                        self.lines[line]['y_1'] = kp1[1]
                        self.lines[line]['x_2'] = kp2[0]
                        self.lines[line]['y_2'] = kp2[1]
                    else:
                        self.lines[line]['x_1'] = kp2[0]
                        self.lines[line]['y_1'] = kp2[1]
                        self.lines[line]['x_2'] = kp1[0]
                        self.lines[line]['y_2'] = kp1[1]