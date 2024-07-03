import sys
import cv2
import copy
import itertools
import numpy as np

from itertools import chain

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

keypoint_world_coords_2D = [[x - 52.5, y - 34] for x, y in keypoint_world_coords_2D]
keypoint_aux_world_coords_2D = [[x - 52.5, y - 34] for x, y in keypoint_aux_world_coords_2D]

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

    def update(self, keypoints_dict):
        self.keypoints_dict = keypoints_dict

        if self.denormalize:
            self.denormalize_keypoints()

        self.subsets = self.get_keypoints_subsets()

    def denormalize_keypoints(self):
        for kp in self.keypoints_dict.keys():
            self.keypoints_dict[kp]['x'] *= self.image_width
            self.keypoints_dict[kp]['y'] *= self.image_height

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
                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC, use_ransac)
                        obj_list[i] = [obj for count, obj in enumerate(obj_list[i]) if status[count]==1]
                        img_list[i] = [obj for count, obj in enumerate(img_list[i]) if status[count]==1]
                        key_list[i] = [obj for count, obj in enumerate(key_list[i]) if status[count]==1]

        for i in range(len(obj_list)):
            if len(obj_list[i]) >= 4 and not all(item[0] == obj_list[i][0][0] for item in obj_list[i])\
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
        obj_pts, img_pts = [], []
        keypoints = list(set(list(itertools.chain(*self.key_pts))))
        for kp in keypoints:
            obj_pts.append([self.subsets[mode][kp]['xw'], self.subsets[mode][kp]['yw'], self.subsets[mode][kp]['zw']])
            img_pts.append([self.subsets[mode][kp]['xi'], self.subsets[mode][kp]['yi']])

        return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)

    def change_plane_coords(self, w=105, h=68):
        R = np.array([[0,0,-1], [0,1,0], [1,0,0]])
        self.rotation = self.rotation @ R
        if self.ord_pts[0] == 1:
            self.position = np.linalg.inv(R) @ self.position + np.array([-w/2, 0, 0])
        elif self.ord_pts[0] == 2:
            self.position = np.linalg.inv(R) @ self.position + np.array([w/2, 0, 0])

    def get_cam_params(self, mode='full', use_ransac=0, refine=False):
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        flags = flags | cv2.CALIB_FIX_TANGENT_DIST | \
                cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY
        flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | \
                cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | \
                cv2.CALIB_FIX_K6

        self.get_per_plane_correspondences(mode=mode, use_ransac=use_ransac)

        if len(self.obj_pts) == 0:
            return None, None

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts,
                                                          (self.image_width, self.image_height),
                                                           None, None, flags=flags)
        if ret:
            self.calibration = mtx
            R, _ = cv2.Rodrigues(rvecs[0])
            self.rotation = R
            self.position = (-np.transpose(self.rotation) @ tvecs[0]).T[0]

            if self.ord_pts[0] != 0:
                self.change_plane_coords()

            if refine:
                obj_pts, img_pts = self.get_correspondences(mode)
                rvec, _ = cv2.Rodrigues(self.rotation)
                tvec = -self.rotation @ self.position

                rvecs, tvecs = cv2.solvePnPRefineLM(obj_pts, img_pts, self.calibration, dist, rvec, tvec,
                                              (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20000, 0.00001))

                self.rotation, _ = cv2.Rodrigues(rvecs)
                self.position = - np.transpose(self.rotation) @ tvecs

            pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)
            pan = np.rad2deg(pan)
            tilt = np.rad2deg(tilt)
            roll = np.rad2deg(roll)

            cam_params = {"pan_degrees": pan,
                          "tilt_degrees": tilt,
                          "roll_degrees": roll,
                          "x_focal_length": mtx[0,0],
                          "y_focal_length": mtx[1,1],
                          "principal_point": [self.image_width/2., self.image_height/2.],
                          "position_meters": [self.position[0], self.position[1], self.position[2]],
                          "rotation_matrix": [[self.rotation[0,0], self.rotation[0,1], self.rotation[0,2]],
                                          [self.rotation[1,0], self.rotation[1,1], self.rotation[1,2]],
                                          [self.rotation[2,0], self.rotation[2,1], self.rotation[2,2]]],
                          # "radial_distortion": [dist[0][0], dist[0][1], dist[0][-1], 0., 0., 0.],
                          # "tangential_distortion": [dist[0][2], dist[0][3]],
                          "radial_distortion": [0., 0., 0., 0., 0., 0.],
                          "tangential_distortion": [0., 0.],
                          "thin_prism_distortion": [0., 0., 0., 0.]}
            return cam_params, ret
        else:
            return None, None


    def get_homography_from_ground_plane(self, use_ransac=5., inverse=False):
        self.get_per_plane_correspondences(mode='ground_plane', use_ransac=use_ransac)
        obj_pts, img_pts = self.get_correspondences('ground_plane')

        if len(obj_pts) >= 4:
            H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)

            if H is None:
                return None
            if inverse:
                det = np.linalg.det(H)
                if np.isclose(det, 0):
                    return None
                H_inv = np.linalg.inv(H)
                return H_inv / H_inv[-1, -1]
            else:
                return H
        else:
            return None


    def get_homography_from_3D_projection(self, use_ransac=5., inverse=False):
        cam_params, ret = self.get_cam_params(mode='full', use_ransac=use_ransac)

        # Extract relevant camera parameters from the dictionary
        #pan_degrees = cam_params['pan_degrees']
        #tilt_degrees = cam_params['tilt_degrees']
        #roll_degrees = cam_params['roll_degrees']
        x_focal_length = cam_params['x_focal_length']
        y_focal_length = cam_params['y_focal_length']
        principal_point = np.array(cam_params['principal_point'])
        position_meters = np.array(cam_params['position_meters'])
        rotation = np.array(cam_params['rotation_matrix'])

        #pan = pan_degrees * np.pi / 180.
        #tilt = tilt_degrees * np.pi / 180.
        #roll = roll_degrees * np.pi / 180.

        #rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))


        # Compute translation matrix
        It = np.eye(4)[:-1]
        It[:, -1] = -position_meters

        # Compute projection matrix
        Q = np.array([[x_focal_length, 0, principal_point[0]],
                      [0, y_focal_length, principal_point[1]],
                      [0, 0, 1]])

        P = Q @ (rotation @ It)

        H = P[:, [0, 1, 3]]  # (3, 3)
        H = H / H[-1, -1]  # normalize homography

        if inverse:
            H_inv = np.linalg.inv(H)
            return H_inv / H_inv[-1, -1]
        else:
            return H

    def heuristic_voting(self):
        final_results = []
        for mode in ['full', 'ground_plane', 'main']:
            for use_ransac in [0, 5, 10, 15, 25, 50]:
                cam_params, ret = self.get_cam_params(mode=mode, use_ransac=use_ransac)
                if ret:
                    result_dict = {'mode': mode, 'use_ransac': use_ransac, 'rep_err': ret,
                                   'cam_params': cam_params, 'calib_plane': self.ord_pts[0]}
                    final_results.append(result_dict)

        if final_results:
            final_results.sort(key=lambda x: (x['rep_err'], x['mode']))
            for res in final_results:
                if res['mode'] == 'full' and res['use_ransac'] == 0 and res['rep_err'] <= 10.:
                    return res
            # Return the first element in the sorted list (if it's not empty)
            return final_results[0]
        else:
            return None


