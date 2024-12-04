import sys
import cv2
import copy
import itertools
import numpy as np

from itertools import chain
from scipy.optimize import least_squares

from pnlcalib.utils.utils_optimize import vector_to_mtx, point_to_line_distance, get_opt_vector, line_plane_intersection, \
    plane_from_P, plane_from_H

keypoint_world_coords_2D = [[0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
                            [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
                            [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
                            [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
                            [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
                            [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
                            [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
                            [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
                            [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
                            [88.5, 34.], [94., 34.]]  # 57

keypoint_aux_world_coords_2D = [[5.5, 0], [16.5, 0], [88.5, 0], [99.5, 0], [5.5, 13.84], [99.5, 13.84], [16.5, 24.84],
                                [88.5, 24.84], [16.5, 43.16], [88.5, 43.16], [5.5, 54.16], [99.5, 54.16], [5.5, 68],
                                [16.5, 68], [88.5, 68], [99.5, 68]]

line_world_coords_3D = [[[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                        [[16.5, 13.84, 0.], [0., 13.84, 0.]], [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                        [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                        [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]],
                        [[0., 30.34, 0.], [0., 30.34, -2.44]], [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                        [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
                        [[52.5, 0., 0.], [52.5, 68, 0.]], [[0., 68., 0.], [105., 68., 0.]],
                        [[0., 0., 0.], [0., 68., 0.]],
                        [[105., 0., 0.], [105., 68., 0.]], [[0., 0., 0.], [105., 0., 0.]],
                        [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                        [[5.5, 43.16, 0.], [5.5, 24.84, 0.]], [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                        [[99.5, 43.16, 0.], [105., 43.16, 0.]], [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                        [[99.5, 24.84, 0.], [105., 24.84, 0.]]]

keypoint_world_coords_2D = [[x - 52.5, y - 34] for x, y in keypoint_world_coords_2D]
keypoint_aux_world_coords_2D = [[x - 52.5, y - 34] for x, y in keypoint_aux_world_coords_2D]
line_world_coords_3D = [[[x1 - 52.5, y1 - 34, z1], [x2 - 52.5, y2 - 34, z2]] for [[x1, y1, z1], [x2, y2, z2]] in
                        line_world_coords_3D]


def rotation_matrix_to_pan_tilt_roll(rotation):
    """
    Decomposes the rotation matrix into pan, tilt and roll angles. There are two solutions, but as we know that cameramen
    try to minimize roll, we take the solution with the smallest roll.
    :param rotation: rotation matrix
    :return: pan, tilt and roll in radians
    """
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = - first_tilt

    sign_first_tilt = 1. if np.sin(first_tilt) > 0. else -1.
    sign_second_tilt = 1. if np.sin(second_tilt) > 0. else -1.

    first_pan = np.arctan2(sign_first_tilt * orientation[0, 2], sign_first_tilt * - orientation[1, 2])
    second_pan = np.arctan2(sign_second_tilt * orientation[0, 2], sign_second_tilt * - orientation[1, 2])
    first_roll = np.arctan2(sign_first_tilt * orientation[2, 0], sign_first_tilt * orientation[2, 1])
    second_roll = np.arctan2(sign_second_tilt * orientation[2, 0], sign_second_tilt * orientation[2, 1])

    # print(f"first solution {first_pan*180./np.pi}, {first_tilt*180./np.pi}, {first_roll*180./np.pi}")
    # print(f"second solution {second_pan*180./np.pi}, {second_tilt*180./np.pi}, {second_roll*180./np.pi}")
    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Conversion from euler angles to orientation matrix.
    :param pan:
    :param tilt:
    :param roll:
    :return: orientation matrix
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]])
    rotMat = np.dot(Rpan, np.dot(Rtilt, Rroll))
    return rotMat


class FramebyFrameCalib:
    def __init__(self, iwidth=960, iheight=540, denormalize=False):
        self.image_width = iwidth
        self.image_height = iheight
        self.denormalize = denormalize
        self.calibration = None
        self.principal_point = np.array([iwidth / 2, iheight / 2])
        self.position = None
        self.rotation = None
        self.homography = None

    def update(self, kp_dict, lines_dict):
        self.keypoints_dict = kp_dict
        self.lines_dict = lines_dict

        if self.denormalize:
            self.denormalize_keypoints()

        self.subsets = self.get_keypoints_subsets()

    def denormalize_keypoints(self):
        for kp in self.keypoints_dict.keys():
            self.keypoints_dict[kp]['x'] *= self.image_width
            self.keypoints_dict[kp]['y'] *= self.image_height
        for line in self.lines_dict.keys():
            self.lines_dict[line]['x_1'] *= self.image_width
            self.lines_dict[line]['y_1'] *= self.image_height
            self.lines_dict[line]['x_2'] *= self.image_width
            self.lines_dict[line]['y_2'] *= self.image_height

    def get_keypoints_subsets(self):
        full, main, ground_plane = {}, {}, {}

        for kp in self.keypoints_dict.keys():
            wp = keypoint_world_coords_2D[kp - 1] if kp <= 57 else keypoint_aux_world_coords_2D[kp - 1 - 57]

            full[kp] = {'xi': self.keypoints_dict[kp]['x'], 'yi': self.keypoints_dict[kp]['y'],
                        'xw': wp[0], 'yw': wp[1], 'zw': -2.44 if kp in [12, 15, 16, 19] else 0.}
            if kp <= 30:
                main[kp] = {'xi': self.keypoints_dict[kp]['x'], 'yi': self.keypoints_dict[kp]['y'],
                            'xw': wp[0], 'yw': wp[1], 'zw': -2.44 if kp in [12, 15, 16, 19] else 0.}
            if kp not in [12, 15, 16, 19]:
                ground_plane[kp] = {'xi': self.keypoints_dict[kp]['x'], 'yi': self.keypoints_dict[kp]['y'],
                                    'xw': wp[0], 'yw': wp[1], 'zw': -2.44 if kp in [12, 15, 16, 19] else 0.}

        return {'full': full, 'main': main, 'ground_plane': ground_plane}

    def get_per_plane_correspondences(self, mode, use_ransac):
        self.obj_pts, self.img_pts, self.ord_pts = None, None, None

        if mode not in ['full', 'main', 'ground_plane']:
            sys.exit("Wrong mode. Select mode between 'full', 'main_keypoints', 'ground_plane'")

        world_points_p1, world_points_p2, world_points_p3 = [], [], []
        img_points_p1, img_points_p2, img_points_p3 = [], [], []
        keys_p1, keys_p2, keys_p3 = [], [], []

        keypoints = self.subsets[mode]
        for kp in keypoints.keys():
            if kp in [12, 16]:
                keys_p2.append(kp)
                world_points_p2.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p2.append([keypoints[kp]['xi'], keypoints[kp]['yi']])

            elif kp in [1, 4, 8, 13, 17, 20, 24, 28]:
                keys_p1.append(kp)
                keys_p2.append(kp)
                world_points_p1.append([keypoints[kp]['xw'], keypoints[kp]['yw'], keypoints[kp]['zw']])
                world_points_p2.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p1.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
                img_points_p2.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
            elif kp in [3, 7, 11, 14, 18, 23, 27, 30]:
                keys_p1.append(kp)
                keys_p3.append(kp)
                world_points_p1.append([keypoints[kp]['xw'], keypoints[kp]['yw'], keypoints[kp]['zw']])
                world_points_p3.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p1.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
                img_points_p3.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
            elif kp in [15, 19]:
                keys_p3.append(kp)
                world_points_p3.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p3.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
            else:
                keys_p1.append(kp)
                world_points_p1.append([keypoints[kp]['xw'], keypoints[kp]['yw'], keypoints[kp]['zw']])
                img_points_p1.append([keypoints[kp]['xi'], keypoints[kp]['yi']])

        obj_points, img_points, key_points, ord_points = [], [], [], []

        if mode == 'ground_plane':
            obj_list = [world_points_p1]
            img_list = [img_points_p1]
            key_list = [keys_p1]
        else:
            obj_list = [world_points_p1, world_points_p2, world_points_p3]
            img_list = [img_points_p1, img_points_p2, img_points_p3]
            key_list = [keys_p1, keys_p2, keys_p3]

        if use_ransac > 0.:
            for i in range(len(obj_list)):
                if len(obj_list[i]) >= 4 and not all(item[0] == obj_list[i][0][0] for item in obj_list[i]) \
                        and not all(item[1] == obj_list[i][0][1] for item in obj_list[i]):
                    if i == 0:
                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC,
                                                       use_ransac)
                        obj_list[i] = [obj for count, obj in enumerate(obj_list[i]) if status[count] == 1]
                        img_list[i] = [obj for count, obj in enumerate(img_list[i]) if status[count] == 1]
                        key_list[i] = [obj for count, obj in enumerate(key_list[i]) if status[count] == 1]

        for i in range(len(obj_list)):
            if len(obj_list[i]) >= 4 and not all(item[0] == obj_list[i][0][0] for item in obj_list[i]) \
                    and not all(item[1] == obj_list[i][0][1] for item in obj_list[i]):
                obj_points.append(np.array(obj_list[i], dtype=np.float32))
                img_points.append(np.array(img_list[i], dtype=np.float32))
                key_points.append(key_list[i])
                ord_points.append(i)

        self.obj_pts = obj_points
        self.img_pts = img_points
        self.key_pts = key_points
        self.ord_pts = ord_points

    def get_correspondences(self, mode):
        obj_pts, img_pts, prob_pts = [], [], []
        keypoints = list(set(list(itertools.chain(*self.key_pts))))
        for kp in keypoints:
            obj_pts.append([self.subsets[mode][kp]['xw'], self.subsets[mode][kp]['yw'], self.subsets[mode][kp]['zw']])
            img_pts.append([self.subsets[mode][kp]['xi'], self.subsets[mode][kp]['yi']])

        return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)

    def change_plane_coords(self, w=105, h=68):
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.rotation = self.rotation @ R
        if self.ord_pts[0] == 1:
            self.position = np.linalg.inv(R) @ self.position + np.array([-w / 2, 0, 0])
        elif self.ord_pts[0] == 2:
            self.position = np.linalg.inv(R) @ self.position + np.array([w / 2, 0, 0])

    def reproj_err(self, obj_pts, img_pts):
        if self.calibration is not None:
            x_focal_length = self.calibration[0, 0]
            y_focal_length = self.calibration[1, 1]
            x_principal_point = self.calibration[0, 2]
            y_principal_point = self.calibration[1, 2]
            position_meters = self.position
            rotation = self.rotation

            It = np.eye(4)[:-1]
            It[:, -1] = -position_meters
            Q = np.array([[x_focal_length, 0, x_principal_point],
                          [0, y_focal_length, y_principal_point],
                          [0, 0, 1]])
            P = Q @ (rotation @ It)

            err, n = 0, 0
            for i in range(len(obj_pts)):
                proj_point = P @ np.array([obj_pts[i][0], obj_pts[i][1], obj_pts[i][2], 1.])
                proj_point /= proj_point[-1]
                err_point = (img_pts[i] - proj_point[:2])
                err += np.sum(err_point ** 2)
                n += 1

            return np.sqrt(err / n)
        else:
            return None

    def reproj_err_ground(self, obj_pts, img_pts):
        if self.homography is not None:
            err, n = 0, 0
            for i in range(len(obj_pts)):
                proj_point = self.homography @ np.array([obj_pts[i][0], obj_pts[i][1], 1.])
                proj_point /= proj_point[-1]
                err_point = (img_pts[i] - proj_point[:2])
                err += np.sum(err_point ** 2)
                n += 1

            return np.sqrt(err / n)
        else:
            return None

    def vector_to_params(self, vector):
        position = vector[:3]
        rot_vector = np.array(vector[3:])
        rotation, _ = cv2.Rodrigues(rot_vector)
        self.position = position
        self.rotation = rotation

    def projection_from_cam(self):
        It = np.eye(4)[:-1]
        It[:, -1] = -self.position
        P = self.calibration @ (self.rotation @ It)
        return P

    def lines_consensus(self, threshold=50):
        P = self.projection_from_cam()
        plane_normal, plane_point = plane_from_P(P, self.position, self.principal_point)

        self.lines_dict_cons = {}
        if plane_normal is not None:
            for key, value in self.lines_dict.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)
                if len(p) == 2:
                    proj1 = P @ np.array([p[0][0], p[0][1], p[0][2], 1.])
                    proj2 = P @ np.array([p[1][0], p[1][1], p[1][2], 1.])
                else:
                    proj1 = P @ np.array([wp1[0], wp1[1], wp1[2], 1.])
                    proj2 = P @ np.array([wp2[0], wp2[1], wp2[2], 1.])

                proj1 /= proj1[-1]
                proj2 /= proj2[-1]
                distance1 = point_to_line_distance(proj1, proj2, np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1, proj2, np.array([x2, y2]))

                if distance2 <= threshold and distance1 <= threshold:
                    self.lines_dict_cons[key] = value

    def line_optimizer(self, vector, img_pts, obj_pts):
        P = vector_to_mtx(vector, self.calibration)
        if not any(np.isnan(P.flatten())):

            plane_normal, plane_point = plane_from_P(P, self.position, self.principal_point)

            points, proj_points = [], []
            for i in range(len(img_pts)):
                points.append(img_pts[i])
                proj_point = P @ np.array([obj_pts[i][0], obj_pts[i][1], obj_pts[i][2], 1.])
                scale = proj_point[-1]
                proj_point /= scale
                proj_points.append(proj_point[:2])

            err1 = (np.array(points) - np.array(proj_points)).ravel()

            err2 = []
            for key, value in self.lines_dict_cons.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)

                if len(p) == 2:
                    proj1 = P @ np.array([p[0][0], p[0][1], p[0][2], 1.])
                    proj2 = P @ np.array([p[1][0], p[1][1], p[1][2], 1.])
                else:
                    proj1 = P @ np.array([wp1[0], wp1[1], wp1[2], 1.])
                    proj2 = P @ np.array([wp2[0], wp2[1], wp2[2], 1.])

                proj1 /= proj1[-1]
                proj2 /= proj2[-1]
                distance1 = point_to_line_distance(proj1, proj2, np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1, proj2, np.array([x2, y2]))
                err2.append([distance1, distance2])

            return np.concatenate((err1, np.array(err2).ravel()))

        else:
            err = []
            for i in range(len(img_pts) + len(self.lines_dict_cons)):
                err.append([np.inf, np.inf])
            return np.array(err).ravel()

    def get_cam_params(self, mode='full', use_ransac=0, refine=False, refine_w_lines=False):
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        flags = flags | cv2.CALIB_FIX_TANGENT_DIST | \
                cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY
        flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | \
                cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | \
                cv2.CALIB_FIX_K6

        self.get_per_plane_correspondences(mode=mode, use_ransac=use_ransac)

        if len(self.obj_pts) == 0:
            return None, None

        obj_pts, img_pts = self.get_correspondences(mode)
        if len(obj_pts) < 6:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_pts,
                self.img_pts,
                (self.image_width, self.image_height),
                None,
                None,
                flags=flags,
            )

        else:
            mtx = cv2.initCameraMatrix2D(
                self.obj_pts,
                self.img_pts,
                (self.image_width, self.image_height),
                aspectRatio=1.0,
            )
            if not np.isnan(np.min(mtx)):
                flags2 = flags | cv2.CALIB_USE_INTRINSIC_GUESS

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    [obj_pts],
                    [img_pts],
                    (self.image_width, self.image_height),
                    mtx,
                    None,
                    flags=flags2,
                )
            else:
                ret = False

        if ret:
            self.calibration = mtx
            self.principal_point = np.array([mtx[0, 2], mtx[1, 2]])
            R, _ = cv2.Rodrigues(rvecs[0])
            self.rotation = R
            self.position = (-np.transpose(self.rotation) @ tvecs[0]).T[0]

            if self.ord_pts[0] != 0:
                self.change_plane_coords()

            obj_pts, img_pts = self.get_correspondences(mode)
            rep_err = self.reproj_err(obj_pts, img_pts)

            if refine:
                if not np.isnan(rep_err):
                    rvec, _ = cv2.Rodrigues(self.rotation)
                    tvec = -self.rotation @ self.position

                    rvecs, tvecs = cv2.solvePnPRefineLM(obj_pts, img_pts, self.calibration, dist, rvec, tvec,
                                                        (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                                                         20000, 1e-5))

                    self.rotation, _ = cv2.Rodrigues(rvecs)
                    self.position = - np.transpose(self.rotation) @ tvecs
                    rep_err = self.reproj_err(obj_pts, img_pts)

            if refine_w_lines:
                if not np.isnan(rep_err):
                    self.lines_consensus()
                    vector = get_opt_vector(self.position, self.rotation)
                    res = least_squares(self.line_optimizer, vector, verbose=0, ftol=1e-4, x_scale="jac", method='trf',
                                        args=(img_pts, obj_pts))

                    vector_opt = res['x']
                    if not any(np.isnan(vector_opt)):
                        self.vector_to_params(vector_opt)
                        rep_err = self.reproj_err(obj_pts, img_pts)

            pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)

            pan = np.rad2deg(pan)
            tilt = np.rad2deg(tilt)
            roll = np.rad2deg(roll)

            cam_params = {"pan_degrees": pan,
                          "tilt_degrees": tilt,
                          "roll_degrees": roll,
                          "x_focal_length": self.calibration[0, 0],
                          "y_focal_length": self.calibration[1, 1],
                          "principal_point": [self.principal_point[0], self.principal_point[1]],
                          "position_meters": [self.position[0], self.position[1], self.position[2]],
                          "rotation_matrix": [[self.rotation[0, 0], self.rotation[0, 1], self.rotation[0, 2]],
                                              [self.rotation[1, 0], self.rotation[1, 1], self.rotation[1, 2]],
                                              [self.rotation[2, 0], self.rotation[2, 1], self.rotation[2, 2]]],
                          "radial_distortion": [0., 0., 0., 0., 0., 0.],
                          "tangential_distortion": [0., 0.],
                          "thin_prism_distortion": [0., 0., 0., 0.]}

            return cam_params, rep_err
        else:
            return None, None

    def estimate_calibration_matrix_from_plane_homography(self, homography):
        """
        This method initializes the calibration matrix from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225). The extraction is sensitive to noise, which is why we keep the
        principal point in the middle of the image rather than using the one extracted by this method.
        :param homography: homography between the world plane of the pitch and the image
        """
        H = np.reshape(homography, (9,))
        A = np.zeros((5, 6))
        A[0, 1] = 1.
        A[1, 0] = 1.
        A[1, 2] = -1.
        A[2, 3] = self.principal_point[1] / self.principal_point[0]
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        u, s, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        try:
            Ktinv = np.linalg.cholesky(W)
        except np.linalg.LinAlgError:
            K = np.eye(3)
            return False, K

        K = np.linalg.inv(np.transpose(Ktinv))
        K /= K[2, 2]

        self.xfocal_length = K[0, 0]
        self.yfocal_length = K[1, 1]
        # the principal point estimated by this method is very noisy, better keep it in the center of the image
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        # self.principal_point = (K[0,2], K[1,2])
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')
        return True, K

    def from_homography(self):
        """
        This method initializes the essential camera parameters from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
        same homography, we extract rough rotation and position estimates (Example 8.1 of Multiple View Geometry in
        computer vision, p196).
        :param homography: The homography that captures the transformation between the 3D flat model of the soccer pitch
         and its image.
        """
        success, _ = self.estimate_calibration_matrix_from_plane_homography(self.homography)
        if not success:
            return False

        hprim = np.linalg.inv(self.calibration) @ self.homography
        lambda1 = 1 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] * lambda1
        r1 = hprim[:, 1] * lambda2
        r2 = np.cross(r0, r1)

        R = np.column_stack((r0, r1, r2))
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh
        self.rotation = R
        t = hprim[:, 2] * lambda3
        self.position = - np.transpose(R) @ t
        return True

    def lines_consensus_ground(self, threshold=100):
        H = self.homography
        plane_normal, plane_point = plane_from_H(H, self.position, self.principal_point)

        self.lines_dict_cons = {}
        if plane_normal is not None:
            for key, value in self.lines_dict.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)
                if len(p) == 2:
                    proj1 = H @ np.array([p[0][0], p[0][1], 1.])
                    proj2 = H @ np.array([p[1][0], p[1][1], 1.])
                else:
                    proj1 = H @ np.array([wp1[0], wp1[1], 1.])
                    proj2 = H @ np.array([wp2[0], wp2[1], 1.])

                proj1 /= proj1[-1]
                proj2 /= proj2[-1]
                distance1 = point_to_line_distance(proj1, proj2, np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1, proj2, np.array([x2, y2]))

                if distance2 <= threshold and distance1 <= threshold:
                    self.lines_dict_cons[key] = value

    def line_optimizer_ground(self, vector, img_pts, obj_pts):
        H = np.append(vector, 1).reshape(3, 3)
        if not any(np.isnan(H.flatten())):

            plane_normal, plane_point = plane_from_H(H, self.position, self.principal_point)

            points, proj_points = [], []
            for i in range(len(img_pts)):
                # if pts[0][i] <= 57:
                points.append(img_pts[i])
                proj_point = H @ np.array([obj_pts[i][0], obj_pts[i][1], 1.])
                scale = proj_point[-1]
                proj_point /= scale
                proj_points.append(proj_point[:2])

            err1 = (np.array(points) - np.array(proj_points)).ravel()

            err2 = []
            for key, value in self.lines_dict_cons.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)

                if len(p) == 2:
                    proj1 = H @ np.array([p[0][0], p[0][1], 1.])
                    proj2 = H @ np.array([p[1][0], p[1][1], 1.])
                else:
                    proj1 = H @ np.array([wp1[0], wp1[1], 1.])
                    proj2 = H @ np.array([wp2[0], wp2[1], 1.])

                proj1 /= proj1[-1]
                proj2 /= proj2[-1]
                distance1 = point_to_line_distance(proj1, proj2, np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1, proj2, np.array([x2, y2]))
                err2.append([distance1, distance2])

            return np.concatenate((0.01 * err1, np.array(err2).ravel()))
        else:
            err = []
            for i in range(len(img_pts) + len(self.lines_dict_cons)):
                err.append([np.inf, np.inf])
            return np.array(err).ravel()

    def get_homography_from_ground_plane(self, use_ransac=5., inverse=False, refine_lines=False):
        self.get_per_plane_correspondences(mode='ground_plane', use_ransac=use_ransac)
        obj_pts, img_pts = self.get_correspondences('ground_plane')

        if len(obj_pts) >= 4:
            if use_ransac > 0:
                H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)
            else:
                H, mask = cv2.findHomography(obj_pts, img_pts)

            if H is not None:
                self.homography = H
                self.from_homography()
                rep_err = self.reproj_err_ground(obj_pts, img_pts)
                if self.position is not None:
                    if refine_lines:
                        self.lines_consensus_ground()
                        vector = H.flatten()[:-1]
                        res = least_squares(self.line_optimizer_ground, vector, verbose=0, ftol=1e-4, x_scale="jac",
                                            method='lm', args=(img_pts, obj_pts))

                        vector_opt = res['x']
                        if not any(np.isnan(vector_opt)):
                            H = np.append(vector_opt, 1).reshape(3, 3)
                            self.homography = H
                            rep_err = self.reproj_err_ground(obj_pts, img_pts)
                if inverse:
                    H_inv = np.linalg.inv(H)
                    return H_inv / H_inv[-1, -1], rep_err
                else:
                    return H, rep_err
            else:
                return None, None
        else:
            return None, None

    def heuristic_voting(self, refine=False, refine_lines=False):
        final_results = []
        for mode in ['full', 'ground_plane', 'main']:
            for use_ransac in [0, 5, 10, 15, 25, 50]:
                cam_params, ret = self.get_cam_params(mode=mode, use_ransac=use_ransac,
                                                      refine=refine, refine_w_lines=refine_lines)
                if ret:
                    result_dict = {'mode': mode, 'use_ransac': use_ransac, 'rep_err': ret,
                                   'cam_params': cam_params, 'calib_plane': self.ord_pts[0]}
                    final_results.append(result_dict)

        if final_results:
            final_results.sort(key=lambda x: (x['rep_err'], x['mode']))
            for res in final_results:
                if res['mode'] == 'full' and res['use_ransac'] == 0 and res['rep_err'] <= 5.:
                    return res
            # Return the first element in the sorted list (if it's not empty)
            return final_results[0]
        else:
            return None

    def heuristic_voting_ground(self, refine_lines=False):
        final_results = []
        for use_ransac in [0, 5, 10, 15, 25, 50]:
            H, ret = self.get_homography_from_ground_plane(use_ransac=use_ransac, inverse=True,
                                                           refine_lines=refine_lines)
            if H is not None:
                result_dict = {'use_ransac': use_ransac, 'rep_err': ret, 'homography': H}
                final_results.append(result_dict)

        if final_results:
            final_results.sort(key=lambda x: (x['rep_err']))
            for res in final_results:
                if res['use_ransac'] == 0 and res['rep_err'] <= 5.:
                    return res
            # Return the first element in the sorted list (if it's not empty)
            return final_results[0]
        else:
            return None


