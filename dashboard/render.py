import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

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

def minimap(players, forecast=None, show_hull=True, show_forecast=True, press_col="press_pred"):
    fig, ax = plt.subplots(figsize=(8.2, 5.3), facecolor="#F7F9F8")
    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68,
                  pitch_color=PITCH_COLOR, line_color=LINE_COLOR, linewidth=1.35,
                  stripe=True, stripe_color="#1D422F", corner_arcs=True,
                  pad_top=3, pad_bottom=1)
    pitch.draw(ax=ax)
    # SoccerNet's pitch-y convention is opposite to the broadcast-facing minimap.
    # Inverting the display axis flips every overlay without altering stored coordinates.
    ax.invert_yaxis()
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
            ax.text(xside, 64, f"{team}: {press}", color="white", fontsize=8, ha="center",
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
    # Keep anomalous forecast positions from expanding the viewport beyond the pitch.
    ax.set_xlim(0, 105)
    ax.set_ylim(68, 0)
    fig.tight_layout()
    return fig
