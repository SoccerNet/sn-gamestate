import argparse
import json
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from src.camera import Camera
from src.soccerpitch import SoccerPitch


def normalization_transform(points):
    """
    Computes the similarity transform such that the list of points is centered around (0,0) and that its distance to the
    center is sqrt(2).
    :param points: point cloud that we wish to normalize
    :return: the affine transformation matrix
    """
    center = np.mean(points, axis=0)

    d = 0.
    nelems = 0
    for p in points:
        nelems += 1
        x = p[0] - center[0]
        y = p[1] - center[1]
        di = np.sqrt(x ** 2 + y ** 2)
        d += (di - d) / nelems

    if d <= 0.:
        s = 1.
    else:
        s = np.sqrt(2) / d
    T = np.zeros((3, 3))
    T[0, 0] = s
    T[0, 2] = -s * center[0]
    T[1, 1] = s
    T[1, 2] = -s * center[1]
    T[2, 2] = 1
    return T


def estimate_homography_from_line_correspondences(lines, T1=np.eye(3), T2=np.eye(3)):
    """
    Given lines correspondences, computes the homography that maps best the two set of lines.
    :param lines: list of pair of 2D lines matches.
    :param T1: Similarity transform to normalize the elements of the source reference system
    :param T2: Similarity transform to normalize the elements of the target reference system
    :return: boolean to indicate success or failure of the estimation, homography
    """
    homography = np.eye(3)
    A = np.zeros((len(lines) * 2, 9))

    for i, line_pair in enumerate(lines):
        src_line = np.transpose(np.linalg.inv(T1)) @ line_pair[0]
        target_line = np.transpose(np.linalg.inv(T2)) @ line_pair[1]
        u = src_line[0]
        v = src_line[1]
        w = src_line[2]

        x = target_line[0]
        y = target_line[1]
        z = target_line[2]

        A[2 * i, 0] = 0
        A[2 * i, 1] = x * w
        A[2 * i, 2] = -x * v
        A[2 * i, 3] = 0
        A[2 * i, 4] = y * w
        A[2 * i, 5] = -v * y
        A[2 * i, 6] = 0
        A[2 * i, 7] = z * w
        A[2 * i, 8] = -v * z

        A[2 * i + 1, 0] = x * w
        A[2 * i + 1, 1] = 0
        A[2 * i + 1, 2] = -x * u
        A[2 * i + 1, 3] = y * w
        A[2 * i + 1, 4] = 0
        A[2 * i + 1, 5] = -u * y
        A[2 * i + 1, 6] = z * w
        A[2 * i + 1, 7] = 0
        A[2 * i + 1, 8] = -u * z

    try:
        u, s, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return False, homography
    v = np.eye(3)
    has_positive_singular_value = False
    for i in range(s.shape[0] - 1, -2, -1):
        v = np.reshape(vh[i], (3, 3))

        if s[i] > 0:
            has_positive_singular_value = True
            break

    if not has_positive_singular_value:
        return False, homography

    homography = np.reshape(v, (3, 3))
    homography = np.linalg.inv(T2) @ homography @ T1
    homography /= homography[2, 2]

    return True, homography


def draw_pitch_homography(image, homography):
    """
    Draws points along the soccer pitch markings elements in the image based on the homography projection.
    /!\ This function assumes that the resolution of the image is 540p.
    :param image
    :param homography: homography that captures the relation between the world pitch plane and the image
    :return: modified image
    """
    field = SoccerPitch()
    polylines = field.sample_field_points()
    for line in polylines.values():

        for point in line:
            if point[2] == 0.:
                hp = np.array((point[0], point[1], 1.))
                projected = homography @ hp
                if projected[2] == 0.:
                    continue
                projected /= projected[2]
                if 0 < projected[0] < 960 and 0 < projected[1] < 540:
                    cv.circle(image, (int(projected[0]), int(projected[1])), 1, (255, 0, 0), 1)

    return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Baseline for camera parameters extraction')

    parser.add_argument('-s', '--soccernet', default="/home/fmg/data/SN23/calibration-2023-bis/", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('-p', '--prediction', default="/home/fmg/results/SN23-tests/",
                        required=False, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('--split', required=False, type=str, default="valid", help='Select the split of data')
    parser.add_argument('--resolution_width', required=False, type=int, default=960,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=540,
                        help='height resolution of the images')
    args = parser.parse_args()

    field = SoccerPitch()

    dataset_dir = os.path.join(args.soccernet, args.split)
    if not os.path.exists(dataset_dir):
        print("Invalid dataset path !")
        exit(-1)

    with open(os.path.join(dataset_dir, "per_match_info.json"), 'r') as f:
        match_info = json.load(f)

    with tqdm(enumerate(match_info.keys()), total=len(match_info.keys()), ncols=160) as t:
        for i, match in t:
            frame_list = match_info[match].keys()

            for frame in frame_list:
                frame_index = frame.split(".")[0]
                prediction_file = os.path.join(args.prediction, args.split, f"extremities_{frame_index}.json")

                if not os.path.exists(prediction_file):
                    continue

                with open(prediction_file, 'r') as f:
                    predictions = json.load(f)

                camera_predictions = dict()
                image_path = os.path.join(dataset_dir, frame)
                # cv_image = cv.imread(image_path)
                # cv_image = cv.resize(cv_image, (args.resolution_width, args.resolution_height))

                line_matches = []
                potential_3d_2d_matches = {}
                src_pts = []
                success = False
                for k, v in predictions.items():
                    if k == 'Circle central' or "unknown" in k:
                        continue
                    P3D1 = field.line_extremities_keys[k][0]
                    P3D2 = field.line_extremities_keys[k][1]
                    p1 = np.array([v[0]['x'] * args.resolution_width, v[0]['y'] * args.resolution_height, 1.])
                    p2 = np.array([v[1]['x'] * args.resolution_width, v[1]['y'] * args.resolution_height, 1.])
                    src_pts.extend([p1, p2])
                    if P3D1 in potential_3d_2d_matches.keys():
                        potential_3d_2d_matches[P3D1].extend([p1, p2])
                    else:
                        potential_3d_2d_matches[P3D1] = [p1, p2]
                    if P3D2 in potential_3d_2d_matches.keys():
                        potential_3d_2d_matches[P3D2].extend([p1, p2])
                    else:
                        potential_3d_2d_matches[P3D2] = [p1, p2]

                    start = (int(p1[0]), int(p1[1]))
                    end = (int(p2[0]), int(p2[1]))
                    # cv.line(cv_image, start, end, (0, 0, 255), 1)

                    line = np.cross(p1, p2)
                    if np.isnan(np.sum(line)) or np.isinf(np.sum(line)):
                        continue
                    line_pitch = field.get_2d_homogeneous_line(k)
                    if line_pitch is not None:
                        line_matches.append((line_pitch, line))

                if len(line_matches) >= 4:
                    target_pts = [field.point_dict[k][:2] for k in potential_3d_2d_matches.keys()]
                    T1 = normalization_transform(target_pts)
                    T2 = normalization_transform(src_pts)
                    success, homography = estimate_homography_from_line_correspondences(line_matches, T1, T2)
                    if success:
                        # cv_image = draw_pitch_homography(cv_image, homography)

                        cam = Camera(args.resolution_width, args.resolution_height)
                        success = cam.from_homography(homography)
                        if success:
                            point_matches = []
                            added_pts = set()
                            for k, potential_matches in potential_3d_2d_matches.items():
                                p3D = field.point_dict[k]
                                projected = cam.project_point(p3D)

                                if 0 < projected[0] < args.resolution_width and 0 < projected[
                                    1] < args.resolution_height:
                                    dist = np.zeros(len(potential_matches))
                                    for i, potential_match in enumerate(potential_matches):
                                        dist[i] = np.sqrt((projected[0] - potential_match[0]) ** 2 + (
                                                projected[1] - potential_match[1]) ** 2)
                                    selected = np.argmin(dist)
                                    if dist[selected] < 100:
                                        point_matches.append((p3D, potential_matches[selected][:2]))

                            if len(point_matches) > 3:
                                cam.refine_camera(point_matches)
                                # cam.draw_colorful_pitch(cv_image, SoccerField.palette)
                                # print(image_path)
                                # cv.imshow("colorful pitch", cv_image)
                                # cv.waitKey(0)

                if success:
                    camera_predictions = cam.to_json_parameters()

                task2_prediction_file = os.path.join(args.prediction, args.split, f"camera_{frame_index}.json")
                if camera_predictions:
                    with open(task2_prediction_file, "w") as f:
                        json.dump(camera_predictions, f, indent=4)
