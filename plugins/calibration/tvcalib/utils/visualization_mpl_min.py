import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle


def init_figure(img_width, img_height, img_delta_w=0.2, img_delta_h=0.1):
    figsize = (16 + (2 * 16 * img_delta_w), 9 + (2 * 9 * img_delta_h))
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(hspace=0.0, wspace=0.00001, left=0, bottom=0, right=1, top=1)
    ax.set_xlim([0 - img_width * img_delta_w, img_width + img_width * img_delta_w])
    ax.set_ylim([0 - img_height * img_delta_h, img_height + img_height * img_delta_h])
    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
    return fig, ax


def draw_reprojection(ax, object3d, cam, dist_circles=0.25, kwargs={"alpha": 0.5, "linewidth": 5}):

    lines3d = object3d.line_segments.transpose(0, 1).transpose(-1, -2)
    for lidx, line_name in enumerate(object3d.line_segments_names):
        with torch.no_grad():
            line3d = lines3d[lidx]
            points_px = cam.project_point2pixel(line3d, lens_distortion=False).cpu()[0, 0]
            line2d = Line2D(
                points_px[:, 0],
                points_px[:, 1],
                color=object3d.cmap_01[line_name],
                **kwargs,
            )
            ax.add_line(line2d)

    points3d_circle = {
        k: torch.from_numpy(np.stack(v, axis=0)).float()
        for k, v in object3d._field_sncalib.sample_field_points(
            dist=1.0, dist_circles=dist_circles
        ).items()
        if k in object3d.circle_segments_names
    }
    for circle_name, circle3d in points3d_circle.items():
        with torch.no_grad():
            points_px = cam.project_point2pixel(circle3d, lens_distortion=False).cpu()[0, 0]
            # print(circle_name, circle3d.shape, points_px.shape)
            circle2d = Polygon(
                points_px[:, :2],
                closed=True,
                color=object3d.cmap_01[circle_name],
                fill=False,
                **kwargs,
            )
            ax.add_patch(circle2d)

    return ax


def draw_selected_points(
    ax,
    object3d,
    points_line,
    points_circle,
    kwargs_outer={
        "zorder": 1000,
        "rasterized": False,
        "s": 500,
        "alpha": 0.3,
        "facecolor": "none",
        "linewidths": 4.0,
    },
    kwargs_inner={
        "zorder": 1000,
        "rasterized": False,
        "s": 50,
        "marker": ".",
        "color": "k",
        "linewidths": 4.0,
    },
):
    # outer circle
    for s_l in range(points_line.shape[-2]):
        m_l = ~((points_line[0, s_l] == 0.0) & (points_line[1, s_l] == 0.0))
        color = [c / 255 for c in object3d.line_palette[s_l]]
        ax.scatter(points_line[0, s_l, m_l], points_line[1, s_l, m_l], color=color, **kwargs_outer)
    for s_c in range(points_circle.shape[-2]):
        m_c = ~((points_circle[0, s_c] == 0.0) & (points_circle[1, s_c] == 0.0))
        color = [c / 255 for c in object3d.circle_palette[s_c]]
        ax.scatter(
            points_circle[0, s_c, m_c], points_circle[1, s_c, m_c], color=color, **kwargs_outer
        )

    # inner circle
    for s_l in range(points_line.shape[-2]):
        m_l = ~((points_line[0, s_l] == 0.0) & (points_line[1, s_l] == 0.0))
        ax.scatter(points_line[0, s_l, m_l], points_line[1, s_l, m_l], **kwargs_inner)
    for s_c in range(points_circle.shape[-2]):
        m_c = ~((points_circle[0, s_c] == 0.0) & (points_circle[1, s_c] == 0.0))
        ax.scatter(points_circle[0, s_c, m_c], points_circle[1, s_c, m_c], **kwargs_inner)
    return ax


def draw_image(
    ax,
    img: torch.tensor,
    imshow_kwargs={"alpha": 1.0},
):
    img_pil = to_pil_image((img * 255.0).to(torch.uint8))
    ax.imshow(img_pil, resample=False, **imshow_kwargs)
    return ax


def frame_image(
    ax,
    image_width,
    image_height,
    frame_kwargs={"linewidth": 1, "edgecolor": "k", "facecolor": "none", "alpha": 0.5},
):
    ax.add_patch(Rectangle((0, 0), image_width, image_height, **frame_kwargs))
    return ax
