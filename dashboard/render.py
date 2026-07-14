import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mplsoccer import Pitch
from scipy.spatial import Voronoi

TEAM_COLOR = {"left": "tab:blue", "right": "tab:red"}
PRESS_TINT = {"low": "#2c7fb8", "mid": "#fec44f", "high": "#de2d26"}
PITCH_COLOR = "#17372A"
LINE_COLOR = "#EEF5F0"

def _pitch_xy(points):
    """Convert centred metric coordinates to mplsoccer's 0..105 × 0..68 pitch."""
    return np.asarray(points, dtype=float) + np.array([52.5, 34.0])

def _player_label(player):
    """Prefer a recognised shirt number; never render a missing OCR value as 'nan'."""
    jersey = str(player.jersey).strip()
    if jersey and jersey.lower() not in {"nan", "none", "<na>"}:
        return _format_identifier(jersey)
    return _format_identifier(player.track_id)

def _format_identifier(value):
    """Render integer-like pandas values (for example 4.0) as a clean identifier."""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text

def broadcast(frame_img, players, show_ids=True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(frame_img)
    for p in players.itertuples():
        ax.add_patch(Rectangle((p.bx, p.by), p.bw, p.bh, fill=False,
                               ec=TEAM_COLOR.get(p.team, "white"), lw=1.5))
        if show_ids:
            ax.text(p.bx, p.by - 4, str(p.track_id), color="white", fontsize=7,
                    bbox=dict(fc=TEAM_COLOR.get(p.team, "grey"), ec="none", pad=0.5))
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig

def _hull(ax, P, color):
    if len(P) >= 3:
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(P); pts = P[np.append(h.vertices, h.vertices[0])]
            ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.12)
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1, alpha=0.5)
        except Exception:
            pass

def _clip_polygon_to_pitch(polygon):
    """Clip a polygon to the 105 x 68 pitch rectangle."""
    polygon = np.asarray(polygon, dtype=float)
    for axis, boundary, keep_greater in ((0, 0.0, True), (0, 105.0, False),
                                         (1, 0.0, True), (1, 68.0, False)):
        if not len(polygon):
            break
        clipped = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in polygon:
            current_inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if current_inside != previous_inside:
                delta = current[axis] - previous[axis]
                if delta:
                    t = (boundary - previous[axis]) / delta
                    clipped.append(previous + t * (current - previous))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        polygon = np.asarray(clipped, dtype=float)
    return polygon

def _voronoi_cells(points):
    """Return finite, pitch-clipped Voronoi cells keyed by point index."""
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return []
    vor = Voronoi(points)
    center = points.mean(axis=0)
    radius = np.ptp(points, axis=0).max() * 3
    ridges = {}
    for (a, b), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(a, []).append((b, v1, v2))
        ridges.setdefault(b, []).append((a, v1, v2))

    cells = []
    vertices = vor.vertices.tolist()
    for point_index, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region:
            continue
        finite = [vertex for vertex in region if vertex >= 0]
        if len(finite) != len(region):
            for other_index, v1, v2 in ridges[point_index]:
                if v1 >= 0 and v2 >= 0:
                    continue
                vertex = v1 if v1 >= 0 else v2
                tangent = points[other_index] - points[point_index]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = (points[point_index] + points[other_index]) / 2
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                vertices.append((vor.vertices[vertex] + direction * radius).tolist())
                finite.append(len(vertices) - 1)
        polygon = np.asarray([vertices[vertex] for vertex in finite])
        polygon = polygon[np.argsort(np.arctan2(polygon[:, 1] - polygon[:, 1].mean(),
                                                polygon[:, 0] - polygon[:, 0].mean()))]
        polygon = _clip_polygon_to_pitch(polygon)
        if len(polygon) >= 3:
            cells.append((point_index, polygon))
    return cells

def _draw_voronoi(ax, players):
    """Draw nearest-player territory, deliberately distinct from pitch control."""
    points = _pitch_xy(players[["x_pitch", "y_pitch"]].values)
    # Qhull does not accept duplicate nodes. Preserve one team label per unique point.
    unique_points, original_indices = np.unique(points, axis=0, return_index=True)
    try:
        cells = _voronoi_cells(unique_points)
    except Exception:
        return
    teams = players.team.to_numpy()
    for point_index, polygon in cells:
        team = teams[original_indices[point_index]]
        color = TEAM_COLOR.get(team, "grey")
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.10, lw=0, zorder=1.5)
        ax.plot(np.r_[polygon[:, 0], polygon[0, 0]], np.r_[polygon[:, 1], polygon[0, 1]],
                color=color, alpha=0.16, lw=0.5, zorder=1.6)

def _draw_trails(ax, trails, frame):
    if trails is None or not len(trails):
        return
    for track_id, group in trails.groupby("track_id"):
        group = group.sort_values("frame")
        if len(group) < 2:
            continue
        points = _pitch_xy(group[["x_pitch", "y_pitch"]].values)
        team = group.team.iloc[-1]
        ax.plot(points[:, 0], points[:, 1], color=TEAM_COLOR.get(team, "grey"),
                alpha=0.45, lw=1.4, zorder=2.2)

def _draw_proximity_edges(ax, edges):
    if edges is None or not len(edges):
        return
    for edge in edges.itertuples():
        points = _pitch_xy([[edge.x1, edge.y1], [edge.x2, edge.y2]])
        ax.plot(points[:, 0], points[:, 1], color=TEAM_COLOR.get(edge.team, "grey"),
                alpha=0.44, lw=1.0, zorder=2.8)

def _draw_actual_future(ax, actual_future):
    if actual_future is None or not len(actual_future):
        return
    actual = actual_future.iloc[0]
    points = np.array([[actual.h1x, actual.h1y], [actual.h2x, actual.h2y], [actual.h3x, actual.h3y]], dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return
    points = _pitch_xy(points)
    ax.plot(points[:, 0], points[:, 1], color="#F6D365", lw=2.4, alpha=0.95, zorder=5.5)
    ax.scatter(points[:, 0], points[:, 1], color="#F6D365", s=25, edgecolors="#1B1B1B", linewidths=0.5, zorder=5.6)

def minimap(players, forecast=None, show_hull=True, show_forecast=True, press_col="press_pred", show_voronoi=False,
            trails=None, trail_frame=None, edges=None, actual_future=None):
    fig, ax = plt.subplots(figsize=(8.2, 5.3), facecolor="#F7F9F8")
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                  pitch_color=PITCH_COLOR, line_color=LINE_COLOR, linewidth=1.35,
                  stripe=True, stripe_color="#1D422F", corner_arcs=True,
                  pad_top=3, pad_bottom=1)
    pitch.draw(ax=ax)
    # SoccerNet's pitch-y convention is opposite to the broadcast-facing minimap.
    # Inverting the display axis flips every overlay without altering stored coordinates.
    ax.invert_yaxis()
    if show_voronoi:
        _draw_voronoi(ax, players)
    _draw_trails(ax, trails, trail_frame)
    _draw_proximity_edges(ax, edges)
    for team, g in players.groupby("team"):
        c = TEAM_COLOR.get(team, "grey")
        P = _pitch_xy(g[["x_pitch", "y_pitch"]].values)
        ax.scatter(P[:, 0], P[:, 1], c=c, s=70, edgecolors="white", zorder=4)
        for p in g.itertuples():
            x, y = _pitch_xy([[p.x_pitch, p.y_pitch]])[0]
            ax.text(x, y, _player_label(p), color="white", fontsize=6,
                    ha="center", va="center", zorder=5)
        if show_hull:
            _hull(ax, P, c)
        # pressing badge per team
        press = g[press_col].iloc[0] if press_col in g and len(g) else ""
        if press:
            xside = 13 if team == "left" else 92
            team_name = "Blue" if team == "left" else "Red"
            ax.text(xside, 64, f"{team_name}: {press}", color="white", fontsize=8, ha="center",
                    bbox=dict(fc=PRESS_TINT.get(press, "grey"), ec="none", pad=1.5), zorder=6)
    if show_forecast and forecast is not None and len(forecast):
        teams = players.groupby("track_id").team.first().to_dict()
        positions = players.groupby("track_id")[["x_pitch", "y_pitch"]].first()
        for f in forecast.itertuples():
            # Full 0/1/2/3-second path. The start point comes from the same smoothed
            # coordinate system used to create the GRU input, avoiding a visible jump.
            # Older baked files do not contain x0/y0; fall back to the displayed track.
            if hasattr(f, "x0") and hasattr(f, "y0"):
                start = (f.x0, f.y0)
            elif f.track_id in positions.index:
                start = positions.loc[f.track_id].to_numpy()
            else:
                continue
            path = _pitch_xy([start, [f.h1x, f.h1y],
                               [f.h2x, f.h2y], [f.h3x, f.h3y]])
            color = TEAM_COLOR.get(teams.get(f.track_id), "0.15")
            ax.plot(path[:, 0], path[:, 1], color=color, lw=2.1, ls="--",
                    alpha=0.9, zorder=3)
            ax.scatter(path[1:3, 0], path[1:3, 1], color=color, s=16,
                       alpha=0.9, zorder=4)
            ax.scatter(path[3, 0], path[3, 1], color=color, marker="D", s=34,
                       edgecolors="white", linewidths=0.6, zorder=5)
            ax.annotate("", xy=path[3], xytext=path[2],
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.1), zorder=5)
    _draw_actual_future(ax, actual_future)
    # Keep anomalous forecast positions from expanding the viewport beyond the pitch.
    ax.set_xlim(0, 105)
    ax.set_ylim(68, 0)
    fig.tight_layout()
    return fig

def occupancy_figure(table):
    """Render a side-by-side team occupancy map for the selected clip/window."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), facecolor="#F7F9F8")
    for ax, team in zip(axes, ("left", "right")):
        pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                      pitch_color=PITCH_COLOR, line_color=LINE_COLOR, linewidth=1.2,
                      stripe=True, stripe_color="#1D422F", corner_arcs=True, pad_top=2, pad_bottom=1)
        pitch.draw(ax=ax)
        ax.invert_yaxis()
        group = table[table.team == team]
        if len(group):
            points = _pitch_xy(group[["x_pitch", "y_pitch"]].values)
            ax.hexbin(points[:, 0], points[:, 1], gridsize=(13, 9), extent=(0, 105, 0, 68),
                      mincnt=1, cmap="Blues" if team == "left" else "Reds", alpha=0.72, zorder=2)
        team_name = "Blue" if team == "left" else "Red"
        ax.set_title(f"{team_name} team space", fontsize=10, color="#20332A", pad=8)
        ax.set_xlim(0, 105); ax.set_ylim(68, 0)
    fig.tight_layout()
    return fig

def shape_timeline_figure(metrics):
    """Compact team-shape time series for the Clip Analytics workspace."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.5), sharex=True, facecolor="#F7F9F8")
    fields = [("width_m", "Team width (m)"), ("depth_m", "Team depth (m)"),
              ("hull_area_m2", "Team space (m²)"), ("centroid_from_own_goal_m", "How far forward (m)")]
    for ax, (field, title) in zip(axes.ravel(), fields):
        for team, group in metrics.groupby("team"):
            ax.plot(group.frame, group[field], color=TEAM_COLOR.get(team, "grey"), label=team.title(), lw=1.5)
        ax.set_title(title, loc="left", fontsize=9, color="#20332A")
        ax.grid(alpha=0.17)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    fig.tight_layout()
    return fig
