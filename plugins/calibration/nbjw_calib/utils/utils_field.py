import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _point_to_meters_inverse(p, w=105, h=68):
    return np.array([p[0] * h, p[1] * w])

def _point_to_meters(p, w=105, h=68):
    return np.array([p[0] * w, p[1] * h])

def _draw_field(width=105, height=68, fig_size=8, lines_color="#bcbcbc", background_color="white"):
    ratio = width / float(height)
    f, ax = plt.subplots(1, 1, figsize=(fig_size * ratio, fig_size), dpi=100)

    if background_color: 
        ax.add_patch(patches.Rectangle((0, 0), width, height, color=background_color))

    line_pts = [
        [_point_to_meters([0, 0]), _point_to_meters([0, 1])],
        [_point_to_meters([1, 0]), _point_to_meters([1, 1])],
        [_point_to_meters([0, 1]), _point_to_meters([1, 1])],  
        [_point_to_meters([0, 0]), _point_to_meters([1, 0])],
    ]

    for line_pt in line_pts:
        ax.plot([line_pt[0][0], line_pt[1][0]], [line_pt[0][1], line_pt[1][1]], '-', alpha=0.8,
                lw=1.5, zorder=2, color=lines_color)

    line_pts = [
        [_point_to_meters([0.5, 0]), _point_to_meters([0.5, 1])],

        [[0, 24.85], [0, 2.85]],
        [[0, 13.85], [16.5, 13.85]],
        [[0, 54.15], [16.5, 54.15]],
        [[16.5, 13.85], [16.5, 54.15]],

        [[0, 24.85], [5.5, 24.85]],
        [[0, 43.15], [5.5, 43.15]],
        [[5.5, 24.85], [5.5, 43.15]],

        [[105, 24.85], [105, 2.85]],
        [[105, 13.85], [88.5, 13.85]],
        [[105, 54.15], [88.5, 54.15]],
        [[88.5, 13.85], [88.5, 54.15]],

        [[105, 24.85], [99.5, 24.85]],
        [[105, 43.15], [99.5, 43.15]],
        [[99.5, 24.85], [99.5, 43.14]]
    ]

    for line_pt in line_pts:
        ax.plot([line_pt[0][0], line_pt[1][0]], [line_pt[0][1], line_pt[1][1]], '-',
                alpha=0.8, lw=1.5, zorder=2, color=lines_color)
    
    # Circles
    ax.add_patch(patches.Wedge((94.0, 34.0), 9, 128, 232, fill=False, edgecolor=lines_color,
                               facecolor=lines_color, zorder=4, width=0.02))

    ax.add_patch(patches.Wedge((11.0, 34.0), 9, 308, 52, fill=False, edgecolor=lines_color,
                               facecolor=lines_color, zorder=4, width=0.02))

    ax.add_patch(patches.Wedge((52.5, 34), 9.5, 0, 360, fill=False, edgecolor=lines_color,
                               facecolor=lines_color, zorder=4, width=0.02))

    plt.axis('off')
    
    return f, ax