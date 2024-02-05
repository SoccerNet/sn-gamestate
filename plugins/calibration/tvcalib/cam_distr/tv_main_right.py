from math import pi
from tvcalib.utils.data_distr import mean_std_with_confidence_interval


def get_cam_distr(sigma_scale: float, batch_dim: int, temporal_dim: int):
    cam_distr = {
        "pan": {
            "minmax": (-pi / 4, pi / 4),  # in deg -45°, 45°
            "dimension": (
                batch_dim,
                temporal_dim,
            ),
        },
        "tilt": {
            "minmax": (pi / 4, pi / 2),  # in deg 45°, 90°
            "dimension": (
                batch_dim,
                temporal_dim,
            ),
        },
        "roll": {
            "minmax": (-pi / 18, pi / 18),  # in deg -10°, 10°
            "dimension": (
                batch_dim,
                temporal_dim,
            ),
        },
        "aov": {
            "minmax": (pi / 22, pi / 2),  # (8.2°, 90°)
            "dimension": (
                batch_dim,
                temporal_dim,
            ),
        },
        "c_x": {
            "minmax": (36 - 16.5, 36 + 16.5),
            "dimension": (
                batch_dim,
                1,
            ),
        },
        "c_y": {
            "minmax": (40.0, 110.0),
            "dimension": (
                batch_dim,
                1,
            ),
        },
        "c_z": {
            "minmax": (-40.0, -5.0),
            "dimension": (
                batch_dim,
                1,
            ),
        },
    }

    for k, params in cam_distr.items():
        cam_distr[k]["mean_std"] = mean_std_with_confidence_interval(
            *params["minmax"], sigma_scale=sigma_scale
        )
    return cam_distr


def get_dist_distr(batch_dim: int, temporal_dim: int, _sigma_scale: float = 2.57):
    return {
        "k1": {
            "minmax": [0.0, 0.5],  # we clip min(0.0, x)
            "mean_std": (0.0, _sigma_scale * 0.5),
            "dimension": (batch_dim, temporal_dim),
        },
        "k2": {
            "minmax": [-0.1, 0.1],
            "mean_std": (0.0, _sigma_scale * 0.1),
            "dimension": (batch_dim, temporal_dim),
        },
    }
