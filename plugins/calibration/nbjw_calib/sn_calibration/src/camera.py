import cv2 as cv
import numpy as np



from soccerpitch import SoccerPitch


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


def unproject_image_point(homography, point2D):
    """
    Given the homography from the world plane of the pitch and the image and a point localized on the pitch plane in the
    image, returns the coordinates of the point in the 3D pitch plane.
    /!\ Only works for correspondences on the pitch (Z = 0).
    :param homography: the homography
    :param point2D: the image point whose relative coordinates on the world plane of the pitch are to be found
    :return: A 2D point on the world pitch plane in homogenous coordinates (X,Y,1) with X and Y being the world
    coordinates of the point.
    """
    hinv = np.linalg.inv(homography)
    pitchpoint = hinv @ point2D
    pitchpoint = pitchpoint / pitchpoint[2]
    return pitchpoint


class Camera:

    def __init__(self, iwidth=960, iheight=540):
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.calibration = np.eye(3)
        self.radial_distortion = np.zeros(6)
        self.thin_prism_disto = np.zeros(4)
        self.tangential_disto = np.zeros(2)
        self.image_width = iwidth
        self.image_height = iheight
        self.xfocal_length = 1
        self.yfocal_length = 1
        self.principal_point = (self.image_width / 2, self.image_height / 2)

    def solve_pnp(self, point_matches):
        """
        With a known calibration matrix, this method can be used in order to retrieve rotation and translation camera
        parameters.
        :param point_matches: A list of pairs of 3D-2D point matches .
        """
        target_pts = np.array([pt[0] for pt in point_matches])
        src_pts = np.array([pt[1] for pt in point_matches])
        _, rvec, t, inliers = cv.solvePnPRansac(target_pts, src_pts, self.calibration, None)
        self.rotation, _ = cv.Rodrigues(rvec)
        self.position = - np.transpose(self.rotation) @ t.flatten()

    def refine_camera(self, pointMatches):
        """
        Once that there is a minimal set of initial camera parameters (calibration, rotation and position roughly known),
        this method can be used to refine the solution using a non-linear optimization procedure.
        :param pointMatches:  A list of pairs of 3D-2D point matches .

        """
        rvec, _ = cv.Rodrigues(self.rotation)
        target_pts = np.array([pt[0] for pt in pointMatches])
        src_pts = np.array([pt[1] for pt in pointMatches])

        rvec, t = cv.solvePnPRefineLM(target_pts, src_pts, self.calibration, None, rvec, -self.rotation @ self.position,
                                      (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20000, 0.00001))
        self.rotation, _ = cv.Rodrigues(rvec)
        self.position = - np.transpose(self.rotation) @ t

    def from_homography(self, homography):
        """
        This method initializes the essential camera parameters from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
        same homography, we extract rough rotation and position estimates (Example 8.1 of Multiple View Geometry in
        computer vision, p196).
        :param homography: The homography that captures the transformation between the 3D flat model of the soccer pitch
         and its image.
        """
        success, _ = self.estimate_calibration_matrix_from_plane_homography(homography)
        if not success:
            return False

        hprim = np.linalg.inv(self.calibration) @ homography
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

    def to_json_parameters(self):
        """
        Saves camera to a JSON serializable dictionary.
        :return: The dictionary
        """
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)
        camera_dict = {
            "pan_degrees": pan * 180. / np.pi,
            "tilt_degrees": tilt * 180. / np.pi,
            "roll_degrees": roll * 180. / np.pi,
            "position_meters": self.position.tolist(),
            "x_focal_length": self.xfocal_length,
            "y_focal_length": self.yfocal_length,
            "principal_point": [self.principal_point[0], self.principal_point[1]],
            "radial_distortion": self.radial_distortion.tolist(),
            "tangential_distortion": self.tangential_disto.tolist(),
            "thin_prism_distortion": self.thin_prism_disto.tolist()

        }
        return camera_dict

    def from_json_parameters(self, calib_json_object):
        """
        Loads camera parameters from dictionary.
        :param calib_json_object: the dictionary containing camera parameters.
        """
        self.principal_point = calib_json_object["principal_point"]
        self.image_width = 2 * self.principal_point[0]
        self.image_height = 2 * self.principal_point[1]
        self.xfocal_length = calib_json_object["x_focal_length"]
        self.yfocal_length = calib_json_object["y_focal_length"]

        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

        pan = calib_json_object['pan_degrees'] * np.pi / 180.
        tilt = calib_json_object['tilt_degrees'] * np.pi / 180.
        roll = calib_json_object['roll_degrees'] * np.pi / 180.

        self.rotation = np.array([
            [-np.sin(pan) * np.sin(roll) * np.cos(tilt) + np.cos(pan) * np.cos(roll),
             np.sin(pan) * np.cos(roll) + np.sin(roll) * np.cos(pan) * np.cos(tilt), np.sin(roll) * np.sin(tilt)],
            [-np.sin(pan) * np.cos(roll) * np.cos(tilt) - np.sin(roll) * np.cos(pan),
             -np.sin(pan) * np.sin(roll) + np.cos(pan) * np.cos(roll) * np.cos(tilt), np.sin(tilt) * np.cos(roll)],
            [np.sin(pan) * np.sin(tilt), -np.sin(tilt) * np.cos(pan), np.cos(tilt)]
        ], dtype='float')

        self.rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))

        self.position = np.array(calib_json_object['position_meters'], dtype='float')

        self.radial_distortion = np.array(calib_json_object['radial_distortion'], dtype='float')
        self.tangential_disto = np.array(calib_json_object['tangential_distortion'], dtype='float')
        self.thin_prism_disto = np.array(calib_json_object['thin_prism_distortion'], dtype='float')

    def distort(self, point):
        """
        Given a point in the normalized image plane, apply distortion
        :param point: 2D point on the normalized image plane
        :return: 2D distorted point
        """
        numerator = 1
        denominator = 1
        radius = np.sqrt(point[0] * point[0] + point[1] * point[1])

        for i in range(3):
            k = self.radial_distortion[i]
            numerator += k * radius ** (2 * (i + 1))
            k2n = self.radial_distortion[i + 3]
            denominator += k2n * radius ** (2 * (i + 1))

        radial_distortion_factor = numerator / denominator
        xpp = point[0] * radial_distortion_factor + \
              2 * self.tangential_disto[0] * point[0] * point[1] + self.tangential_disto[1] * (
                      radius ** 2 + 2 * point[0] ** 2) + \
              self.thin_prism_disto[0] * radius ** 2 + self.thin_prism_disto[1] * radius ** 4
        ypp = point[1] * radial_distortion_factor + \
              2 * self.tangential_disto[1] * point[0] * point[1] + self.tangential_disto[0] * (
                      radius ** 2 + 2 * point[1] ** 2) + \
              self.thin_prism_disto[2] * radius ** 2 + self.thin_prism_disto[3] * radius ** 4
        return np.array([xpp, ypp], dtype=np.float32)

    def project_point(self, point3D, distort=True):
        """
        Uses current camera parameters to predict where a 3D point is seen by the camera.
        :param point3D: The 3D point in world coordinates.
        :param distort: optional parameter to allow projection without distortion.
        :return: The 2D coordinates of the imaged point
        """
        point = point3D - self.position
        rotated_point = self.rotation @ np.transpose(point)
        if rotated_point[2] <= 1e-3 :
            return np.zeros(3)
        rotated_point = rotated_point / rotated_point[2]
        if distort:
            distorted_point = self.distort(rotated_point)
        else:
            distorted_point = rotated_point
        x = distorted_point[0] * self.xfocal_length + self.principal_point[0]
        y = distorted_point[1] * self.yfocal_length + self.principal_point[1]
        return np.array([x, y, 1])

    def scale_resolution(self, factor):
        """
        Adapts the internal parameters for image resolution changes
        :param factor: scaling factor
        """
        self.xfocal_length = self.xfocal_length * factor
        self.yfocal_length = self.yfocal_length * factor
        self.image_width = self.image_width * factor
        self.image_height = self.image_height * factor

        self.principal_point = (self.image_width / 2, self.image_height / 2)

        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

    def draw_corners(self, image, color=(0, 255, 0)):
        """
        Draw the corners of a standard soccer pitch in the image.
        :param image: cv image
        :param color
        :return: the image mat modified.
        """
        field = SoccerPitch()
        for pt3D in field.point_dict.values():
            projected = self.project_point(pt3D)
            if projected[2] == 0.:
                continue
            projected /= projected[2]
            if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                cv.circle(image, (int(projected[0]), int(projected[1])), 3, color, 2)
        return image

    def draw_pitch(self, image, color=(0, 255, 0)):
        """
        Draws all the lines of the pitch on the image.
        :param image
        :param color
        :return: modified image
        """
        field = SoccerPitch()

        polylines = field.sample_field_points()
        for line in polylines.values():
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    cv.line(image, (int(prev_point[0]), int(prev_point[1])), (int(projected[0]), int(projected[1])),
                            color, 1)
                prev_point = projected
        return image

    def draw_colorful_pitch(self, image, palette):
        """
        Draws all the lines of the pitch on the image, each line color is specified by the palette argument.

        :param image:
        :param palette: dictionary associating line classes names with their BGR color.
        :return: modified image
        """
        field = SoccerPitch()

        polylines = field.sample_field_points()
        for key, line in polylines.items():
            if key not in palette.keys():
                print(f"Can't draw {key}")
                continue
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    # BGR color
                    cv.line(image, (int(prev_point[0]), int(prev_point[1])), (int(projected[0]), int(projected[1])),
                            palette[key][::-1], 1)
                prev_point = projected
        return image

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

