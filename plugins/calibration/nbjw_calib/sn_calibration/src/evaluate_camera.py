import argparse
import json
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from camera import Camera
from evaluate_extremities import scale_points, distance, mirror_labels
from soccerpitch import SoccerPitch


def get_polylines(camera_annotation, width, height, sampling_factor=0.2):
    """
    Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
    projects the 3D points belonging to the terrain model in order to give a dictionary associating the classes
    observed and the points projected in the image.

    :param camera_annotation: camera parameters in their json/dictionary format
    :param width: image width for evaluation
    :param height: image height for evaluation
    :return: a dictionary with keys corresponding to a class observed in the image ( a line of the 3D model whose
    projection falls in the image) and values are then the list of 2D projected points.
    """

    cam = Camera(width, height)
    cam.from_json_parameters(camera_annotation)
    field = SoccerPitch()
    projections = dict()
    sides = [
        np.array([1, 0, 0]),
        np.array([1, 0, -width + 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, -height + 1])
    ]
    for key, points in field.sample_field_points(sampling_factor).items():
        projections_list = []
        in_img = False
        prev_proj = np.zeros(3)
        for i, point in enumerate(points):
            ext = cam.project_point(point)
            if ext[2] < 1e-5:
                # point at infinity or behind camera
                continue
            if 0 <= ext[0] < width and 0 <= ext[1] < height:

                if not in_img and i > 0:

                    line = np.cross(ext, prev_proj)
                    in_img_intersections = []
                    dist_to_ext = []
                    for side in sides:
                        intersection = np.cross(line, side)
                        intersection /= intersection[2]
                        if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                            in_img_intersections.append(intersection)
                            dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                    if in_img_intersections:
                        intersection = in_img_intersections[np.argmin(dist_to_ext)]

                        projections_list.append(
                            {
                                "x": intersection[0],
                                "y": intersection[1]
                            }
                        )

                projections_list.append(
                    {
                        "x": ext[0],
                        "y": ext[1]
                    }
                )
                in_img = True
            elif in_img:
                # first point out
                line = np.cross(ext, prev_proj)

                in_img_intersections = []
                dist_to_ext = []
                for side in sides:
                    intersection = np.cross(line, side)
                    intersection /= intersection[2]
                    if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                        in_img_intersections.append(intersection)
                        dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                if in_img_intersections:
                    intersection = in_img_intersections[np.argmin(dist_to_ext)]

                    projections_list.append(
                        {
                            "x": intersection[0],
                            "y": intersection[1]
                        }
                    )
                in_img = False
            prev_proj = ext
        if len(projections_list):
            projections[key] = projections_list
    return projections


def distance_to_polyline(point, polyline):
    """
    Computes euclidian distance between a point and a polyline.
    :param point: 2D point
    :param polyline: a list of 2D point
    :return: the distance value
    """
    if 0 < len(polyline) < 2:
        dist = distance(point, polyline[0])
        return dist
    else:
        dist_to_segments = []
        point_np = np.array([point["x"], point["y"], 1])

        for i in range(len(polyline) - 1):
            origin_segment = np.array([
                polyline[i]["x"],
                polyline[i]["y"],
                1
            ])
            end_segment = np.array([
                polyline[i + 1]["x"],
                polyline[i + 1]["y"],
                1
            ])
            line = np.cross(origin_segment, end_segment)
            line /= np.sqrt(np.square(line[0]) + np.square(line[1]))

            # project point on line l
            projected = np.cross((np.cross(np.array([line[0], line[1], 0]), point_np)), line)
            projected = projected / projected[2]

            v1 = projected - origin_segment
            v2 = end_segment - origin_segment
            k = np.dot(v1, v2) / np.dot(v2, v2)
            if 0 < k < 1:

                segment_distance = np.sqrt(np.sum(np.square(projected - point_np)))
            else:
                d1 = distance(point, polyline[i])
                d2 = distance(point, polyline[i + 1])
                segment_distance = np.min([d1, d2])

            dist_to_segments.append(segment_distance)
        return np.min(dist_to_segments)


def evaluate_camera_prediction(projected_lines, groundtruth_lines, threshold):
    """
    Computes confusion matrices for a level of precision specified by the threshold.
    A groundtruth line is correctly classified if it lies at less than threshold pixels from a line of the prediction
    of the same class.
    Computes also the reprojection error of each groundtruth point : the reprojection error is the L2 distance between
    the point and the projection of the line.
    :param projected_lines: dictionary of detected lines classes as keys and associated predicted points as values
    :param groundtruth_lines: dictionary of annotated lines classes as keys and associated annotated points as values
    :param threshold: distance in pixels that distinguishes good matches from bad ones
    :return: confusion matrix, per class confusion matrix & per class reprojection errors
    """
    global_confusion_mat = np.zeros((2, 2), dtype=np.float32)
    per_class_confusion = {}
    dict_errors = {}
    detected_classes = set(projected_lines.keys())
    groundtruth_classes = set(groundtruth_lines.keys())

    false_positives_classes = detected_classes - groundtruth_classes
    for false_positive_class in false_positives_classes:
        # false_positives = len(projected_lines[false_positive_class])
        if "Circle" not in false_positive_class:
            # Count only extremities for lines, independently of soccer pitch sampling
            false_positives = 2.
        else:
            false_positives = 9.
        per_class_confusion[false_positive_class] = np.array([[0., false_positives], [0., 0.]])
        global_confusion_mat[0, 1] += 1

    false_negatives_classes = groundtruth_classes - detected_classes
    for false_negatives_class in false_negatives_classes:
        false_negatives = len(groundtruth_lines[false_negatives_class])
        per_class_confusion[false_negatives_class] = np.array([[0., 0.], [false_negatives, 0.]])
        global_confusion_mat[1, 0] += 1

    common_classes = detected_classes - false_positives_classes

    for detected_class in common_classes:

        detected_points = projected_lines[detected_class]
        groundtruth_points = groundtruth_lines[detected_class]

        per_class_confusion[detected_class] = np.zeros((2, 2))

        all_below_dist = 1
        for point in groundtruth_points:

            dist_to_poly = distance_to_polyline(point, detected_points)
            if dist_to_poly < threshold:
                per_class_confusion[detected_class][0, 0] += 1
            else:
                per_class_confusion[detected_class][0, 1] += 1
                all_below_dist *= 0

            if detected_class in dict_errors.keys():
                dict_errors[detected_class].append(dist_to_poly)
            else:
                dict_errors[detected_class] = [dist_to_poly]

        if all_below_dist:
            global_confusion_mat[0, 0] += 1
        else:
            global_confusion_mat[0, 1] += 1

    return global_confusion_mat, per_class_confusion, dict_errors


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation camera calibration task')

    parser.add_argument('-s', '--soccernet', default="/home/fmg/data/SN23/calibration-2023-bis/", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('-p', '--prediction', default="/home/fmg/results/SN23-tests/",
                        required=False, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('-t', '--threshold', default=5, required=False, type=int,
                        help="Accuracy threshold in pixels")
    parser.add_argument('--split', required=False, type=str, default="valid", help='Select the split of data')
    parser.add_argument('--resolution_width', required=False, type=int, default=960,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=540,
                        help='height resolution of the images')
    args = parser.parse_args()

    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}

    dataset_dir = os.path.join(args.soccernet, args.split)
    if not os.path.exists(dataset_dir):
        print("Invalid dataset path !")
        exit(-1)

    annotation_files = [f for f in os.listdir(dataset_dir) if ".json" in f]

    missed, total_frames = 0, 0
    with tqdm(enumerate(annotation_files), total=len(annotation_files), ncols=160) as t:
        for i, annotation_file in t:
            frame_index = annotation_file.split(".")[0]
            annotation_file = os.path.join(args.soccernet, args.split, annotation_file)
            prediction_file = os.path.join(args.prediction, args.split, f"camera_{frame_index}.json")

            total_frames += 1

            if not os.path.exists(prediction_file):
                missed += 1

                continue

            with open(annotation_file, 'r') as f:
                line_annotations = json.load(f)

            with open(prediction_file, 'r') as f:
                predictions = json.load(f)

            line_annotations = scale_points(line_annotations, args.resolution_width, args.resolution_height)

            image_path = os.path.join(args.soccernet, args.split, f"{frame_index}.jpg")

            img_groundtruth = line_annotations

            img_prediction = get_polylines(predictions, args.resolution_width, args.resolution_height,
                                           sampling_factor=0.9)

            confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(img_prediction,
                                                                                     img_groundtruth,
                                                                                     args.threshold)

            confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(img_prediction,
                                                                                     mirror_labels(img_groundtruth),
                                                                                     args.threshold)

            accuracy1, accuracy2 = 0., 0.
            if confusion1.sum() > 0:
                accuracy1 = confusion1[0, 0] / confusion1.sum()

            if confusion2.sum() > 0:
                accuracy2 = confusion2[0, 0] / confusion2.sum()

            if accuracy1 > accuracy2:
                accuracy = accuracy1
                confusion = confusion1
                per_class_conf = per_class_conf1
                reproj_errors = reproj_errors1
            else:
                accuracy = accuracy2
                confusion = confusion2
                per_class_conf = per_class_conf2
                reproj_errors = reproj_errors2

            accuracies.append(accuracy)
            if confusion[0, :].sum() > 0:
                precision = confusion[0, 0] / (confusion[0, :].sum())
                precisions.append(precision)
            if (confusion[0, 0] + confusion[1, 0]) > 0:
                recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
                recalls.append(recall)

            for line_class, errors in reproj_errors.items():
                if line_class in dict_errors.keys():
                    dict_errors[line_class].extend(errors)
                else:
                    dict_errors[line_class] = errors

            for line_class, confusion_mat in per_class_conf.items():
                if line_class in per_class_confusion_dict.keys():
                    per_class_confusion_dict[line_class] += confusion_mat
                else:
                    per_class_confusion_dict[line_class] = confusion_mat

    completeness_score = (total_frames - missed) / total_frames
    mAccuracy = np.mean(accuracies)

    final_score = completeness_score * mAccuracy
    print(f" On SoccerNet {args.split} set, final score of : {final_score}")
    print(f" On SoccerNet {args.split} set, completeness rate of : {completeness_score}")

    mRecall = np.mean(recalls)
    sRecall = np.std(recalls)
    medianRecall = np.median(recalls)
    print(
        f" On SoccerNet {args.split} set, recall mean value : {mRecall * 100:2.2f}% with standard deviation of {sRecall * 100:2.2f}% and median of {medianRecall * 100:2.2f}%")

    mPrecision = np.mean(precisions)
    sPrecision = np.std(precisions)
    medianPrecision = np.median(precisions)
    print(
        f" On SoccerNet {args.split} set, precision mean value : {mPrecision * 100:2.2f}% with standard deviation of {sPrecision * 100:2.2f}% and median of {medianPrecision * 100:2.2f}%")

    sAccuracy = np.std(accuracies)
    medianAccuracy = np.median(accuracies)
    print(
        f" On SoccerNet {args.split} set, accuracy mean value :  {mAccuracy * 100:2.2f}% with standard deviation of {sAccuracy * 100:2.2f}% and median of {medianAccuracy * 100:2.2f}%")

    print()

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        print(
            f"For class {line_class}, accuracy of {class_accuracy * 100:2.2f}%, precision of {class_precision * 100:2.2f}%  and recall of {class_recall * 100:2.2f}%")

        for k, v in dict_errors.items():
            fig, ax1 = plt.subplots(figsize=(11, 8))
            ax1.hist(v, bins=30, range=(0, 60))
            ax1.set_title(k)
            ax1.set_xlabel("Errors in pixel")
            os.makedirs(f"./results/", exist_ok=True)
            plt.savefig(f"./results/{k}_reprojection_error.png")
            plt.close(fig)
