import cv2
import torch
import scipy
import random
import numpy as np

from shapely.geometry import Point, Polygon, MultiPoint

def calculate_metrics(gt, pred, mask, conf_th=0.1, dist_th=5):
    geometry_mask = (mask[:, :-1] > 0).cpu()

    pred_mask = torch.all((pred[:, :, :, -1] > conf_th), dim=-1)
    gt_mask = torch.all((gt[:, :, :, -1] > conf_th), dim=-1)

    pred_pos = pred[geometry_mask][:, 0, :]
    pred_mask = pred_mask[geometry_mask]
    gt_pos = gt[geometry_mask][:, 0, :]
    gt_mask = gt_mask[geometry_mask]

    distances = torch.norm(pred_pos - gt_pos, dim=1)

    # Count true positives, false positives, and false negatives based on distance threshold
    true_positives = ((distances < dist_th) & pred_mask & gt_mask).sum().item()
    true_negatives = (~pred_mask & ~gt_mask).sum().item()
    false_positives = ((pred_mask & ~gt_mask) | ((distances >= dist_th) & pred_mask & gt_mask)).sum().item()
    false_negatives = (~pred_mask & gt_mask).sum().item()

    # Calculate precision, recall, and F1 score
    accuracy = (true_positives + true_negatives) / geometry_mask.sum().item()
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1


def calculate_metrics_l(gt, pred, conf_th=0.1, dist_th=5):

    pred_pos = pred[:, :, :, :-1]
    gt_pos = gt[:, :, :, :-1]

    pred_mask = torch.all((pred[:, :, :, -1] > conf_th), dim=-1)
    gt_mask = torch.all((gt[:, :, :, -1] > conf_th), dim=-1)

    gt_flip = torch.flip(gt_pos, dims=[2])

    distances1 = torch.norm(pred_pos - gt_pos, dim=-1)
    distances2 = torch.norm(pred_pos - gt_flip, dim=-1)

    distances1_bool = torch.all((distances1 < dist_th), dim=-1)
    distances2_bool = torch.all((distances2 < dist_th), dim=-1)

    # Count true positives, false positives, and false negatives based on distance threshold
    true_positives = ((distances1_bool | distances2_bool) & pred_mask & gt_mask).sum().item()
    true_negatives = (~pred_mask & ~gt_mask).sum().item()
    false_positives = (
            (pred_mask & ~gt_mask) | ((~distances1_bool & ~distances2_bool) & pred_mask & gt_mask)).sum().item()
    false_negatives = (~pred_mask & gt_mask).sum().item()

    # Calculate precision, recall, and F1 score
    accuracy = (true_positives + true_negatives) / (gt.size()[1] * gt.size()[0])
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1


def calculate_metrics_l_with_mask(gt, pred, mask, conf_th=0.1, dist_th=5):

    #only works with batch 1. Should be adapted to batch > 1 in an organic way or just do a loop over batch

    geometry_mask = (mask[:, :-1] > 0).cpu()

    pred = pred[geometry_mask]
    gt = gt[geometry_mask]

    pred_pos = pred[:, :, :-1]
    gt_pos = gt[:, :, :-1]

    pred_mask = torch.all((pred[:, :, -1] > conf_th), dim=-1)
    gt_mask = torch.all((gt[:, :, -1] > conf_th), dim=-1)

    gt_flip = torch.flip(gt_pos, dims=[1])

    distances1 = torch.norm(pred_pos - gt_pos, dim=-1)
    distances2 = torch.norm(pred_pos - gt_flip, dim=-1)

    distances1_bool = torch.all((distances1 < dist_th), dim=-1)
    distances2_bool = torch.all((distances2 < dist_th), dim=-1)

    # Count true positives, false positives, and false negatives based on distance threshold
    true_positives = ((distances1_bool | distances2_bool) & pred_mask & gt_mask).sum().item()
    true_negatives = (~pred_mask & ~gt_mask).sum().item()
    false_positives = (
            (pred_mask & ~gt_mask) | ((~distances1_bool & ~distances2_bool) & pred_mask & gt_mask)).sum().item()
    false_negatives = (~pred_mask & gt_mask).sum().item()

    # Calculate precision, recall, and F1 score
    accuracy = (true_positives + true_negatives) / geometry_mask.sum().item()
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1


def calc_iou_whole_with_poly(pred_h, gt_h, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    corners = np.array([[0, 0],
                        [frame_w - 1, 0],
                        [frame_w - 1, frame_h - 1],
                        [0, frame_h - 1]], dtype=np.float64)

    mapping_mat = np.linalg.inv(gt_h)
    mapping_mat /= mapping_mat[2, 2]

    gt_corners = cv2.perspectiveTransform(
        corners[:, None, :], gt_h)  # inv_gt_mat * (gt_mat * [x, y, 1])
    gt_corners = cv2.perspectiveTransform(
        gt_corners, np.linalg.inv(gt_h))
    gt_corners = gt_corners[:, 0, :]

    pred_corners = cv2.perspectiveTransform(
        corners[:, None, :], gt_h)  # inv_pred_mat * (gt_mat * [x, y, 1])
    pred_corners = cv2.perspectiveTransform(
        pred_corners, np.linalg.inv(pred_h))
    pred_corners = pred_corners[:, 0, :]

    gt_poly = Polygon(gt_corners.tolist())
    pred_poly = Polygon(pred_corners.tolist())

    # f, axarr = plt.subplots(1, 2, figsize=(16, 12))
    # axarr[0].plot(*gt_poly.exterior.coords.xy)
    # axarr[1].plot(*pred_poly.exterior.coords.xy)
    # plt.show()

    if pred_poly.is_valid is False:
        return 0., None, None

    if not gt_poly.intersects(pred_poly):
        print('not intersects')
        iou = 0.
    else:
        intersection = gt_poly.intersection(pred_poly).area
        union = gt_poly.area + pred_poly.area - intersection
        if union <= 0.:
            print('whole union', union)
            iou = 0.
        else:
            iou = intersection / union

    return iou, None, None

def calc_iou_part(pred_h, gt_h, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    # field template binary mask
    field_mask = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    gt_mask = cv2.warpPerspective(field_mask, gt_h, (template_w, template_h),
                                  cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    pred_mask = cv2.warpPerspective(field_mask, pred_h, (template_w, template_h),
                                    cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    gt_mask[gt_mask > 0] = 255
    pred_mask[pred_mask > 0] = 255

    intersection = ((gt_mask > 0) * (pred_mask > 0)).sum()
    union = (gt_mask > 0).sum() + (pred_mask > 0).sum() - intersection

    if union <= 0:
        print('part union', union)
        # iou = float('nan')
        iou = 0.
    else:
        iou = float(intersection) / float(union)

    # === blending ===
    gt_white_area = (gt_mask[:, :, 0] == 255) & (
            gt_mask[:, :, 1] == 255) & (gt_mask[:, :, 2] == 255)
    gt_fill = gt_mask.copy()
    gt_fill[gt_white_area, 0] = 255
    gt_fill[gt_white_area, 1] = 0
    gt_fill[gt_white_area, 2] = 0
    pred_white_area = (pred_mask[:, :, 0] == 255) & (
            pred_mask[:, :, 1] == 255) & (pred_mask[:, :, 2] == 255)
    pred_fill = pred_mask.copy()
    pred_fill[pred_white_area, 0] = 0
    pred_fill[pred_white_area, 1] = 255
    pred_fill[pred_white_area, 2] = 0
    gt_maskf = gt_fill.astype(float) / 255
    pred_maskf = pred_fill.astype(float) / 255
    fill_resultf = cv2.addWeighted(gt_maskf, 0.5,
                                   pred_maskf, 0.5, 0.0)
    fill_result = np.uint8(fill_resultf * 255)

    return iou

def calc_proj_error(pred_h, gt_h, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    field_mask = np.ones((template_h, template_w, 3), dtype=np.uint8) * 255
    gt_mask = cv2.warpPerspective(field_mask, np.linalg.inv(
        gt_h), (frame_w, frame_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    gt_gray = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        gt_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.squeeze(contours[0])
    poly = Polygon(contour)
    sample_pts = []
    num_pts = 2500
    while len(sample_pts) <= num_pts:
        x = random.sample(range(0, frame_w), 1)
        y = random.sample(range(0, frame_h), 1)
        p = Point(x[0], y[0])
        if p.within(poly):
            sample_pts.append([x[0], y[0]])
    sample_pts = np.array(sample_pts, dtype=np.float32)

    field_dim_x, field_dim_y = 100, 60
    x_scale = field_dim_x / template_w
    y_scale = field_dim_y / template_h
    scaling_mat = np.eye(3)
    scaling_mat[0, 0] = x_scale
    scaling_mat[1, 1] = y_scale
    gt_temp_grid = cv2.perspectiveTransform(
        sample_pts.reshape(-1, 1, 2), scaling_mat @ gt_h)
    gt_temp_grid = gt_temp_grid.reshape(-1, 2)
    pred_temp_grid = cv2.perspectiveTransform(
        sample_pts.reshape(-1, 1, 2), scaling_mat @ pred_h)
    pred_temp_grid = pred_temp_grid.reshape(-1, 2)

    # TODO compute distance in top view
    gt_grid_list = []
    pred_grid_list = []
    for gt_pts, pred_pts in zip(gt_temp_grid, pred_temp_grid):
        if 0 <= gt_pts[0] < field_dim_x and 0 <= gt_pts[1] < field_dim_y and \
                0 <= pred_pts[0] < field_dim_x and 0 <= pred_pts[1] < field_dim_y:
            gt_grid_list.append(gt_pts)
            pred_grid_list.append(pred_pts)
    gt_grid_list = np.array(gt_grid_list)
    pred_grid_list = np.array(pred_grid_list)

    if gt_grid_list.shape != pred_grid_list.shape:
        print('proj error:', gt_grid_list.shape, pred_grid_list.shape)
    assert gt_grid_list.shape == pred_grid_list.shape, 'shape mismatch'

    if gt_grid_list.size != 0 and pred_grid_list.size != 0:
        distance_list = calc_euclidean_distance(
            gt_grid_list, pred_grid_list, axis=1)
        return distance_list.mean()  # average all keypoints
    else:
        print(gt_grid_list)
        print(pred_grid_list)
        return float('nan')

def calc_euclidean_distance(a, b, _norm=np.linalg.norm, axis=None):
    return _norm(a - b, axis=axis)

def gen_template_grid():
    # === set uniform grid ===
    # field_dim_x, field_dim_y = 105.000552, 68.003928 # in meter
    field_dim_x, field_dim_y = 114.83, 74.37  # in yard
    # field_dim_x, field_dim_y = 115, 74 # in yard
    nx, ny = (13, 7)
    x = np.linspace(0, field_dim_x, nx)
    y = np.linspace(0, field_dim_y, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uniform_grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
    uniform_grid = np.concatenate((uniform_grid, np.ones(
        (uniform_grid.shape[0], 1))), axis=1)  # top2bottom, left2right
    # TODO: class label in template, each keypoints is (x, y, c), c is label that starts from 1
    for idx, pts in enumerate(uniform_grid):
        pts[2] = idx + 1  # keypoints label
    return uniform_grid

def calc_reproj_error(pred_h, gt_h, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    uniform_grid = gen_template_grid()  # grid shape (91, 3), (x, y, label)
    template_grid = uniform_grid[:, :2].copy()
    template_grid = template_grid.reshape(-1, 1, 2)

    gt_warp_grid = cv2.perspectiveTransform(template_grid, np.linalg.inv(gt_h))
    gt_warp_grid = gt_warp_grid.reshape(-1, 2)
    pred_warp_grid = cv2.perspectiveTransform(
        template_grid, np.linalg.inv(pred_h))
    pred_warp_grid = pred_warp_grid.reshape(-1, 2)

    # TODO compute distance in camera view
    gt_grid_list = []
    pred_grid_list = []
    for gt_pts, pred_pts in zip(gt_warp_grid, pred_warp_grid):
        if 0 <= gt_pts[0] < frame_w and 0 <= gt_pts[1] < frame_h and \
                0 <= pred_pts[0] < frame_w and 0 <= pred_pts[1] < frame_h:
            gt_grid_list.append(gt_pts)
            pred_grid_list.append(pred_pts)
    gt_grid_list = np.array(gt_grid_list)
    pred_grid_list = np.array(pred_grid_list)

    if gt_grid_list.shape != pred_grid_list.shape:
        print('reproj error:', gt_grid_list.shape, pred_grid_list.shape)
    assert gt_grid_list.shape == pred_grid_list.shape, 'shape mismatch'

    if gt_grid_list.size != 0 and pred_grid_list.size != 0:
        distance_list = calc_euclidean_distance(
            gt_grid_list, pred_grid_list, axis=1)
        distance_list /= frame_h  # normalize by image height
        return distance_list.mean()  # average all keypoints
    else:
        print(gt_grid_list)
        print(pred_grid_list)
        return float('nan')

