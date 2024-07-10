import sys
import cv2
import copy
import itertools
import numpy as np

from itertools import chain
from scipy.optimize import least_squares

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

keypoint_world_coords_2D += keypoint_aux_world_coords_2D

class SequentialCalib:
    def __init__(self, iwidth=960, iheight=540, temporal_ord=3, denormalize=False):

        self.image_width = iwidth
        self.image_height = iheight
        self.denormalize = denormalize

        if temporal_ord <= 3:
            self.temporal_ord = temporal_ord
        else:
            sys.exit("Temporal order must be between 0 and 3")

        self.key_pts = None
        self.subsets = None

        self.estimate = None
        self.estimate_1 = None
        self.estimate_2 = None
        self.estimate_3 = None


    def update(self, keypoints_dict):
        self.keypoints_dict = copy.deepcopy(keypoints_dict)

        if self.denormalize:
            self.denormalize_keypoints()

        self.subsets = self.get_keypoints_subsets()

        self.estimate_3 = copy.deepcopy(self.estimate_2)
        self.estimate_2 = copy.deepcopy(self.estimate_1)
        self.estimate_1 = copy.deepcopy(self.estimate)


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
        obj_pts, img_pts, key_pts = [], [], []
        keypoints = self.subsets[mode].keys()
        for kp in keypoints:
            obj_pts.append([self.subsets[mode][kp]['xw'], self.subsets[mode][kp]['yw'], self.subsets[mode][kp]['zw']])
            img_pts.append([self.subsets[mode][kp]['xi'], self.subsets[mode][kp]['yi']])
            key_pts.append(kp)
        return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32), np.array(key_pts, dtype=np.float32)


    def change_plane_coords(self, w=105, h=68):
        R = np.array([[0,0,-1], [0,1,0], [1,0,0]])
        self.rotation = self.rotation @ R
        if self.ord_pts[0] == 1:
            self.position = np.linalg.inv(R) @ self.position + np.array([-w/2, 0, 0])
        elif self.ord_pts[0] == 2:
            self.position = np.linalg.inv(R) @ self.position + np.array([w/2, 0, 0])


    def get_cam_params(self, mode='full', use_ransac=0, refine=False):

        def params_to_vector(cam_params):
            x_focal_length = np.array([cam_params['x_focal_length']])
            y_focal_length = np.array([cam_params['y_focal_length']])
            rotation = np.array(cam_params["rotation_matrix"])
            position_meters = np.array(cam_params['position_meters'])
            rotation = rotation.flatten()

            return np.concatenate((x_focal_length, y_focal_length, position_meters, rotation))

        def vector_to_mtx(vector):
            x_focal_length = vector[0]
            y_focal_length = vector[1]
            position_meters = vector[2:5]
            rotation = vector[5:].reshape(3, 3)

            It = np.eye(4)[:-1]
            It[:, -1] = -position_meters
            Q = np.array([[x_focal_length, 0, self.image_width / 2],
                          [0, y_focal_length, self.image_height / 2],
                          [0, 0, 1]])
            P = Q @ (rotation @ It)

            return P

        def reproj_err(vector, img_pts, obj_pts, key_pts):

            P = vector_to_mtx(vector)

            proj_points = []
            for idx in key_pts:
                wp = keypoint_world_coords_2D[int(idx)-1]
                proj_point = P @ np.array([wp[0], wp[1], 0 if idx not in [12,15,16,19] else -2.44, 1.])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            reproj_err = np.sqrt(np.sum(np.square(np.array(proj_points) - np.array(img_pts)), axis=1))

            return np.mean(reproj_err)

        def fun_1(params, points, points_idx, estimate_1):
            proj_points = []
            for idx in points_idx:
                P = vector_to_mtx(params)
                wp = keypoint_world_coords_2D[int(idx) - 1]
                proj_point = P @ np.array([wp[0], wp[1], 0 if idx not in [12, 15, 16, 19] else -2.44, 1.])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            err1 = (points - np.array(proj_points)).ravel()
            err2 = (params - estimate_1).ravel()

            return np.concatenate((err1, err2))

        def fun_2(params, points, points_idx, estimate_1, estimate_2):
            proj_points = []
            for idx in points_idx:
                P = vector_to_mtx(params)
                wp = keypoint_world_coords_2D[int(idx) - 1]
                proj_point = P @ np.array([wp[0], wp[1], 0 if idx not in [12, 15, 16, 19] else -2.44, 1.])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            err1 = (points - np.array(proj_points)).ravel()
            err2 = (params - 2*estimate_1 + estimate_2).ravel()

            return np.concatenate((err1, err2))

        def fun_3(params, points, points_idx, estimate_1, estimate_2, estimate_3):
            proj_points = []
            for idx in points_idx:
                P = vector_to_mtx(params)
                wp = keypoint_world_coords_2D[int(idx) - 1]
                proj_point = P @ np.array([wp[0], wp[1], 0 if idx not in [12, 15, 16, 19] else -2.44, 1.])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            err1 = (points - np.array(proj_points)).ravel()
            err2 = (params - 3*estimate_1 + 3*estimate_2 - estimate_3).ravel()

            return np.concatenate((err1, err2))


        def check_success(res):
            if res['success']:
                vector = res['x']
                x_focal_length = vector[0]
                y_focal_length = vector[1]
                self.position = vector[2:5]
                self.rotation = vector[5:].reshape(3, 3)

                self.calibration = np.array([[x_focal_length, 0, self.image_width/2],
                                             [0, y_focal_length, self.image_height/2],
                                             [0, 0, 1]])

                return reproj_err(vector, img_pts, obj_pts, key_pts), False
            else:
                return None, True

        def check_for_nan(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    if not check_for_nan(value):
                        return False
                elif isinstance(value, float) and value != value:
                    return False
            return True

        self.get_per_plane_correspondences(mode=mode, use_ransac=use_ransac)

        if len(self.obj_pts) == 0:
            return None, None

        from_zero = self.temporal_ord == 0

        obj_pts, img_pts, key_pts = self.get_correspondences(mode=mode)
        ransac_key_pts = list(set([item for sublist in self.key_pts for item in sublist]))

        obj_pts = [pt for count, pt in enumerate(obj_pts) if key_pts[count] in ransac_key_pts]
        img_pts = [pt for count, pt in enumerate(img_pts) if key_pts[count] in ransac_key_pts]
        key_pts = [pt for pt in key_pts if pt in ransac_key_pts]

        if self.temporal_ord >= 3 and self.estimate_1 is not None and self.estimate_2 is not None and \
            self.estimate_3 is not None:
            res = least_squares(fun_3, self.estimate_1, verbose=0, x_scale='jac', ftol=1e-4,
                                method='lm',
                                args=(img_pts, key_pts, self.estimate_1, self.estimate_2, self.estimate_3))
            ret, from_zero = check_success(res)

        elif self.temporal_ord >= 2 and self.estimate_1 is not None and self.estimate_2 is not None:
            res = least_squares(fun_2, self.estimate_1, verbose=0, x_scale='jac', ftol=1e-4,
                                    method='lm', args=(img_pts, key_pts, self.estimate_1, self.estimate_2))
            ret, from_zero = check_success(res)

        elif self.temporal_ord >= 1 and self.estimate_1 is not None:
            res = least_squares(fun_1, self.estimate_1, verbose=0, x_scale='jac', ftol=1e-4,
                                method='lm', args=(img_pts, key_pts, self.estimate_1))
            ret, from_zero = check_success(res)
        else:
            from_zero = True

        if from_zero:

            flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
            flags = flags | cv2.CALIB_FIX_TANGENT_DIST | \
                    cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY
            flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | \
                    cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | \
                    cv2.CALIB_FIX_K6


            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts,
                                                                   (self.image_width, self.image_height),
                                                                   None, None, flags=flags)
            if ret:
                self.calibration = mtx
                R, _ = cv2.Rodrigues(rvecs[0])
                self.rotation = R
                self.position = (-np.transpose(self.rotation) @ tvecs[0]).T[0]

                if refine:
                    obj_pts, img_pts, key_pts = self.get_correspondences(mode=mode)
                    ransac_key_pts = list(set([item for sublist in self.key_pts for item in sublist]))

                    obj_pts = [pt for count, pt in enumerate(obj_pts) if key_pts[count] in ransac_key_pts]
                    img_pts = [pt for count, pt in enumerate(img_pts) if key_pts[count] in ransac_key_pts]
                    key_pts = [pt for pt in key_pts if pt in ransac_key_pts]

                    rvec, _ = cv2.Rodrigues(self.rotation)
                    tvec = -self.rotation @ self.position

                    rvecs, tvecs = cv2.solvePnPRefineLM(obj_pts, img_pts, self.calibration, dist, rvec, tvec,
                                                  (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20000, 0.00001))

                    self.rotation, _ = cv2.Rodrigues(rvecs)
                    self.position = - np.transpose(self.rotation) @ tvecs

            else:
                return None, None

        cam_params = {"x_focal_length": self.calibration[0,0],
                      "y_focal_length": self.calibration[1,1],
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

        if check_for_nan(cam_params):
            return cam_params, ret
        else:
            return None, None



    def get_homography_from_ground_plane(self, use_ransac=5., inverse=False):

        def fun_1(params, points, points_idx, estimate_1):

            proj_points = []
            h = np.reshape(params, (3,3))
            for kp in points_idx:
                wp = keypoint_world_coords_2D[int(kp)-1]
                proj_point = h @ np.array([wp[0], wp[1], 1])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            err1 = (points - np.array(proj_points)).ravel()
            err2 = (params - estimate_1).ravel()

            return np.concatenate((err1, err2))

        def fun_2(params, points, points_idx, estimate_1, estimate_2):

            proj_points = []
            h = np.reshape(params, (3,3))
            for kp in points_idx:
                wp = keypoint_world_coords_2D[int(kp)-1]
                proj_point = h @ np.array([wp[0], wp[1], 1])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            err1 = (points - np.array(proj_points)).ravel()
            err2 = (params - 2*estimate_1 + estimate_2).ravel()

            return np.concatenate((err1, err2))

        def fun_3(params, points, points_idx, estimate_1, estimate_2, estimate_3):

            proj_points = []
            h = np.reshape(params, (3,3))
            for kp in points_idx:
                wp = keypoint_world_coords_2D[int(kp)-1]
                proj_point = h @ np.array([wp[0], wp[1], 1])
                proj_point /= proj_point[-1]
                proj_points.append(proj_point[:2])

            err1 = (points - np.array(proj_points)).ravel()
            err2 = (params - 3*estimate_1 + 3*estimate_2 - estimate_3).ravel()

            return np.concatenate((err1, err2))


        obj_pts, img_pts, key_pts = self.get_correspondences('ground_plane')

        if len(obj_pts) >= 4:
            H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)

            if H is None:
                self.estimate = None
                return None

            img_pts = [pt for count, pt in enumerate(img_pts) if mask[count]==1]
            key_pts = [pt for count, pt in enumerate(key_pts) if mask[count]==1]

            if self.temporal_ord >= 3 and self.estimate_1 is not None and self.estimate_2 is not None and \
                    self.estimate_3 is not None:
                res = least_squares(fun_3, self.estimate_1.reshape(9, ), verbose=0, x_scale='jac', ftol=1e-8,
                                    method='lm',
                                    args=(img_pts, key_pts, self.estimate_1.reshape(9, ),
                                          self.estimate_2.reshape(9, ), self.estimate_3.reshape(9,)))
                print(res["success"], res["cost"])
                if res['success']:
                    H = np.reshape(res['x'], (3, 3))
                    H /= H[-1, -1]


            elif self.temporal_ord >= 2 and self.estimate_1 is not None and self.estimate_2 is not None:
                res = least_squares(fun_2, self.estimate_1.reshape(9,), verbose=0, x_scale='jac', ftol=1e-8,
                                    method='lm',
                                    args=(img_pts, key_pts, self.estimate_1.reshape(9,), self.estimate_2.reshape(9,)))
                if res['success']:
                    H = np.reshape(res['x'], (3, 3))
                    H /= H[-1, -1]


            elif self.temporal_ord >= 1 and self.estimate_1 is not None:
                res = least_squares(fun_1, self.estimate_1.reshape(9, ), verbose=0, x_scale='jac', ftol=1e-8,
                                    method='lm', args=(img_pts, key_pts, self.estimate_1.reshape(9, )))
                if res['success']:
                    H = np.reshape(res['x'], (3, 3))
                    H /= H[-1, -1]

            self.estimate = H
            if inverse:
                H = np.linalg.inv(H)
                H /= H[-1, -1]

            return H

        else:
            self.estimate = None
            return None


    def calibrate(self, mode="full", use_ransac=5):
        cam_params, ret = self.get_cam_params(mode=mode, use_ransac=use_ransac)
        if ret:
            result_dict = {'mode': mode, 'use_ransac': use_ransac, 'rep_err': ret,
                           'cam_params': cam_params, 'calib_plane': self.ord_pts[0]}
            return result_dict
        else:
            None


    def heuristic_voting(self):

        def params_to_vector(cam_params):
            x_focal_length = np.array([cam_params['x_focal_length']])
            y_focal_length = np.array([cam_params['y_focal_length']])
            rotation = np.array(cam_params['rotation_matrix'])
            position_meters = np.array(cam_params['position_meters'])

            rotation = rotation.flatten()

            return np.concatenate((x_focal_length, y_focal_length, position_meters, rotation))

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
                    self.estimate = params_to_vector(res["cam_params"])
                    return res

            # Return the first element in the sorted list (if it's not empty)
            self.estimate = params_to_vector(final_results[0]["cam_params"])
            return final_results[0]

        else:
            self.estimate = None
            return None


