import copy
import torch
import numpy as np

from scipy.stats import linregress
from typing import List, Optional, Tuple


def generate_gaussian_matrix_vectorized(h, w, px, py, sigma=2):
    # Create a grid of indices
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    # Calculate Gaussian values for the entire grid
    matrix = np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma**2))

    return matrix

def resize_keypoints(keypoints, original_size, new_size):

    ratio_h = new_size[0] / original_size[0]
    ratio_w = new_size[1] / original_size[1]

    resized_keypoints = {}
    for kp, values in keypoints.items():
        x_resized = int(values['x'] * ratio_w)
        y_resized = int(values['y'] * ratio_h)
        resized_keypoints[kp] = {'x': x_resized, 'y': y_resized, 'in_frame': values['in_frame']}
        if 'proj_err' in values.keys():
            resized_keypoints[kp]['proj_err'] = values['proj_err']

    return resized_keypoints


def generate_gaussian_array_vectorized(num_matrices, keypoints, original_size, down_ratio=2, sigma=2, proj_err_th=5.):

    new_size = tuple(ti/down_ratio for ti in original_size)
    resized_keypoints = resize_keypoints(keypoints, original_size, new_size)

    # Create an array of center points based on resized keypoints
    center_points = []

    for kp in range(1, num_matrices):
        if kp in resized_keypoints.keys():
            if (resized_keypoints[kp]['in_frame']):
                if 'proj_err' in resized_keypoints[kp].keys():
                    if resized_keypoints[kp]['proj_err'] <= proj_err_th:
                        center_points.append([resized_keypoints[kp]['x'], resized_keypoints[kp]['y']])
                    else:
                        center_points.append([np.inf, np.inf])
                else:
                    center_points.append([resized_keypoints[kp]['x'], resized_keypoints[kp]['y']])
            else:
                center_points.append([np.inf, np.inf])
        else:
            center_points.append([np.inf, np.inf])

    center_points = np.array(center_points)

    # Generate Gaussian matrices for all center points
    matrices = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma) for px, py in center_points]
    matrices = np.array(matrices)
    matrices = np.concatenate((matrices, 1-matrices.sum(axis=0, keepdims=True)), axis=0)
    matrices = np.clip(matrices, 0, 1)

    return matrices


def resize_keypoints_l(keypoints, original_size, new_size):
    ratio_h = new_size[0] / original_size[0]
    ratio_w = new_size[1] / original_size[1]

    resized_keypoints = {}
    for kp, values in keypoints.items():
        x1_resized = int(values['x_1'] * ratio_w)
        y1_resized = int(values['y_1'] * ratio_h)
        x2_resized = int(values['x_2'] * ratio_w)
        y2_resized = int(values['y_2'] * ratio_h)

        resized_keypoints[kp] = {'x_1': x1_resized, 'y_1': y1_resized, 'x_2': x2_resized, 'y_2': y2_resized}

    return resized_keypoints

def generate_gaussian_array_vectorized_l(num_matrices, keypoints, original_size, down_ratio=2, sigma=2, sigma_mult=1):

    def sigma_f(px, py, size, sigma):
        #multiply sigma if point in image border
        if (px < 5 or px > size[0] - 5) | (py < 5 or py > size[1] - 5):
            return sigma_mult*sigma
        else:
            return sigma

    new_size = tuple(int(ti / down_ratio) for ti in original_size)
    resized_keypoints = resize_keypoints_l(keypoints, original_size, new_size)

    # Create an array of center points based on resized keypoints for both points
    center_points = []

    for kp in range(1, num_matrices+1):
        if kp in resized_keypoints.keys():
            center_points.append([resized_keypoints[kp]['x_1'], resized_keypoints[kp]['y_1']])
            center_points.append([resized_keypoints[kp]['x_2'], resized_keypoints[kp]['y_2']])

        else:
            center_points.append([np.inf, np.inf])
            center_points.append([np.inf, np.inf])

    center_points = np.array(center_points)

    # Generate Gaussian matrices for both points and sum them
    matrices1 = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma_f(px, py, new_size, sigma)) for px, py in
                 center_points[::2]]
    matrices2 = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma_f(px, py, new_size, sigma)) for px, py in
                 center_points[1::2]]
    matrices = np.array(matrices1) + np.array(matrices2)

    matrices_border = np.zeros((1, new_size[1], new_size[0]))

    for kp in range(1, num_matrices+1):
        if kp in resized_keypoints.keys():
            x1, y1 = resized_keypoints[kp]['x_1'], resized_keypoints[kp]['y_1']
            x2, y2 = resized_keypoints[kp]['x_2'], resized_keypoints[kp]['y_2']

            pixel_dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            num_gaussians = int(pixel_dist / (sigma))

            if num_gaussians != 1:
                for i in range(num_gaussians):
                    alpha = i / (num_gaussians - 1)
                    x = int(x1 + alpha * (x2 - x1))
                    y = int(y1 + alpha * (y2 - y1))
                    matrices_border[0, :, :] += generate_gaussian_matrix_vectorized(new_size[0], new_size[1], x, y, sigma)
            else:
                x, y = abs(x2 - x1) / 2, abs(y2 - y1) / 2
                matrices_border[0, :, :] += generate_gaussian_matrix_vectorized(new_size[0], new_size[1], x, y, sigma)

    matrices_border = np.clip(matrices_border, 0, 1)

    matrices_combined = np.concatenate((matrices, matrices_border), axis=0)

    return matrices_combined


def get_keypoints_from_heatmap_batch_maxpool(
        heatmap: torch.Tensor,
        scale: int = 2,
        max_keypoints: int = 1,
        min_keypoint_pixel_distance: int = 15,
        return_scores: bool = True,
) -> List[List[List[Tuple[int, int]]]]:
    """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Inspired by mmdetection and CenterNet:
      https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

    Args:
        heatmap (torch.Tensor): NxCxHxW heatmap batch
        max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
        min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

        Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
        abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
        rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

    Returns:
        The extracted keypoints for each batch, channel and heatmap; and their scores
    """
    batch_size, n_channels, _, width = heatmap.shape

    # obtain max_keypoints local maxima for each channel (w/ maxpool)

    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance
    # exclude border keypoints by padding with highest possible value
    # bc the borders are more susceptible to noise and could result in false positives
    padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
    max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
    # if the value equals the original value, it is the local maximum
    local_maxima = max_pooled_heatmap == heatmap
    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima

    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
    # at this point either score > 0.0, in which case the index is a local maximum
    # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

    #  remove top-k that are not local maxima and threshold (if required)
    # thresholding shouldn't be done during training

    #  moving them to CPU now to avoid multiple GPU-mem accesses!
    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

    # have to do this manually as the number of maxima for each channel can be different
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            locs = []
            for candidate_idx in range(candidates.shape[0]):
                # convert to (u,v)
                loc = candidates[candidate_idx][::-1] * scale
                loc = loc.tolist()
                if return_scores:
                    loc.append(scores[batch_idx, channel_idx, candidate_idx])
                locs.append(loc)
            filtered_indices[batch_idx][channel_idx] = locs

    return torch.tensor(filtered_indices)


def get_keypoints_from_heatmap_batch_maxpool_l(
        heatmap: torch.Tensor,
        scale: int = 2,
        max_keypoints: int = 2,
        min_keypoint_pixel_distance: int = 10,
        return_scores: bool = True,
) -> List[List[List[Tuple[int, int]]]]:
    """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Inspired by mmdetection and CenterNet:
      https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

    Args:
        heatmap (torch.Tensor): NxCxHxW heatmap batch
        max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
        min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

        Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
        abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
        rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

    Returns:
        The extracted keypoints for each batch, channel and heatmap; and their scores
    """
    batch_size, n_channels, _, width = heatmap.shape
    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = int((kernel-1)/2)

    max_pooled_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    # if the value equals the original value, it is the local maximum
    local_maxima = max_pooled_heatmap == heatmap

    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima

    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
    # at this point either score > 0.0, in which case the index is a local maximum
    # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

    #  remove top-k that are not local maxima and threshold (if required)
    # thresholding shouldn't be done during training

    #  moving them to CPU now to avoid multiple GPU-mem accesses!
    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

    # have to do this manually as the number of maxima for each channel can be different
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            locs = []
            for candidate_idx in range(candidates.shape[0]):
                # convert to (u,v)
                loc = candidates[candidate_idx][::-1] * scale
                loc = loc.tolist()
                if return_scores:
                    loc.append(scores[batch_idx, channel_idx, candidate_idx])
                locs.append(loc)
            filtered_indices[batch_idx][channel_idx] = locs

    return torch.tensor(filtered_indices)


def coords_to_dict(coords, threshold=0.05, ground_plane_only=False):
    kp_list = []
    for batch in range(coords.size()[0]):
        keypoints = {}
        for count, c in enumerate(range(coords.size(1))):
            if coords.size(2) == 1:
                if ground_plane_only and c+1 in [12,15,16,19]:
                    continue
                if coords[batch, c, 0, -1] > threshold:
                    keypoints[count+1] = {'x': coords[batch, c, 0, 0].item(),
                                          'y': coords[batch, c, 0, 1].item(),
                                          'p': coords[batch, c, 0, 2].item()}
            else:
                if ground_plane_only and c+1 in [7,8,9,10,11,12]:
                    continue
                if coords[batch, c, 0, -1] > threshold and coords[batch, c, 1, -1] > threshold:
                    keypoints[count+1] = {'x_1': coords[batch, c, 0, 0].item(),
                                          'y_1': coords[batch, c, 0, 1].item(),
                                          'p_1': coords[batch, c, 0, 2].item(),
                                          'x_2': coords[batch, c, 1, 0].item(),
                                          'y_2': coords[batch, c, 1, 1].item(),
                                          'p_2': coords[batch, c, 1, 2].item()}

        kp_list.append(keypoints)
    return kp_list


def complete_keypoints(kp_dict, lines_dict, w, h, normalize=False):

    def line_intersection(x1, y1, x2, y2):
        #1e-7 sum in case there are two identical coordinate values
        x1[-1] += 1e-7
        x2[-1] += 1e-7
        slope1, intercept1, r1, p1, se1 = linregress(x1, y1)
        slope2, intercept2, r2, p2, se2 = linregress(x2, y2)

        x_intersection = (intercept2 - intercept1) / (slope1 - slope2 + 1e-7)
        y_intersection = slope1 * x_intersection + intercept1

        return x_intersection, y_intersection

    lines_list = ["Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Big rect. right bottom",
                  "Big rect. right main", "Big rect. right top", "Goal left crossbar", "Goal left post left ",
                  "Goal left post right", "Goal right crossbar", "Goal right post left", "Goal right post right",
                  "Middle line", "Side line bottom", "Side line left", "Side line right", "Side line top",
                  "Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
                  "Small rect. right main", "Small rect. right top"]


    keypoints_line_list = [['Side line top', 'Side line left'], ['Side line top', 'Middle line'],
                           ['Side line right', 'Side line top'], ['Side line left', 'Big rect. left top'],
                           ['Big rect. left top', 'Big rect. left main'], ['Big rect. right top', 'Big rect. right main'],
                           ['Side line right', 'Big rect. right top'], ['Side line left', 'Small rect. left top'],
                           ['Small rect. left top', 'Small rect. left main'], ['Small rect. right top', 'Small rect. right main'],
                           ['Side line right', 'Small rect. right top'], ['Goal left crossbar', 'Goal left post right'],
                           ['Side line left', 'Goal left post right'], ['Side line right', 'Goal right post left'],
                           ['Goal right crossbar', 'Goal right post left'], ['Goal left crossbar', 'Goal left post left '],
                           ['Side line left', 'Goal left post left '], ['Side line right', 'Goal right post right'],
                           ['Goal right crossbar', 'Goal right post right'], ['Side line left', 'Small rect. left bottom'],
                           ['Small rect. left bottom', 'Small rect. left main'], ['Small rect. right bottom', 'Small rect. right main'],
                           ['Side line right', 'Small rect. right bottom'], ['Side line left', 'Big rect. left bottom'],
                           ['Big rect. left bottom', 'Big rect. left main'], ['Big rect. right main', 'Big rect. right bottom'],
                           ['Side line right', 'Big rect. right bottom'], ['Side line left', 'Side line bottom'],
                           ['Side line bottom', 'Middle line'], ['Side line bottom', 'Side line right']]


    keypoint_aux_pair_list = [['Small rect. left main', 'Side line top'], ['Big rect. left main', 'Side line top'],
                              ['Big rect. right main', 'Side line top'], ['Small rect. right main', 'Side line top'],
                              ['Small rect. left main', 'Big rect. left top'], ['Big rect. right top', 'Small rect. right main'],
                              ['Small rect. left top', 'Big rect. left main'], ['Small rect. right top', 'Big rect. right main'],
                              ['Small rect. left bottom', 'Big rect. left main'], ['Small rect. right bottom', 'Big rect. right main'],
                              ['Small rect. left main', 'Big rect. left bottom'], ['Small rect. right main', 'Big rect. right bottom'],
                              ['Small rect. left main', 'Side line bottom'], ['Big rect. left main', 'Side line bottom'],
                              ['Big rect. right main', 'Side line bottom'], ['Small rect. right main', 'Side line bottom']]

    w_extra = 0.5 * w
    h_extra = 0.5 * h

    complete_list = []
    for batch in range(len(kp_dict)):
        complete_dict = copy.deepcopy(kp_dict[batch])
        for key in range(1, 31):
            if key not in kp_dict[batch].keys():
                line_keys = keypoints_line_list[key-1]
                line_key1, line_key2 = lines_list.index(line_keys[0]) + 1, lines_list.index(line_keys[1]) + 1
                if all(line_key in lines_dict[batch].keys() for line_key in [line_key1, line_key2]):
                    x1 = [lines_dict[batch][line_key1]['x_1'], lines_dict[batch][line_key1]['x_2']]
                    y1 = [lines_dict[batch][line_key1]['y_1'], lines_dict[batch][line_key1]['y_2']]
                    x2 = [lines_dict[batch][line_key2]['x_1'], lines_dict[batch][line_key2]['x_2']]
                    y2 = [lines_dict[batch][line_key2]['y_1'], lines_dict[batch][line_key2]['y_2']]
                    new_kp = line_intersection(x1, y1, x2, y2)
                    if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                        complete_dict[key] = {'x': round(new_kp[0], 0), 'y': round(new_kp[1], 0), 'p': 1.}

        for key in range(1, len(keypoint_aux_pair_list)):
            line_keys = keypoint_aux_pair_list[key-1]
            line_key1, line_key2 = lines_list.index(line_keys[0]) + 1, lines_list.index(line_keys[1]) + 1
            if all(line_key in lines_dict[batch].keys() for line_key in [line_key1, line_key2]):
                x1 = [lines_dict[batch][line_key1]['x_1'], lines_dict[batch][line_key1]['x_2']]
                y1 = [lines_dict[batch][line_key1]['y_1'], lines_dict[batch][line_key1]['y_2']]
                x2 = [lines_dict[batch][line_key2]['x_1'], lines_dict[batch][line_key2]['x_2']]
                y2 = [lines_dict[batch][line_key2]['y_1'], lines_dict[batch][line_key2]['y_2']]
                new_kp = line_intersection(x1, y1, x2, y2)
                if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                    complete_dict[key+57] = {'x': round(new_kp[0], 0), 'y': round(new_kp[1], 0), 'p': 1.}

        if normalize:
            for kp in complete_dict.keys():
                complete_dict[kp]['x'] /= w
                complete_dict[kp]['y'] /= h

        complete_dict = dict(sorted(complete_dict.items()))
        complete_list.append(complete_dict)

    return complete_list






