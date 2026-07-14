#!/usr/bin/env python3
"""Visualise the 3A forecaster on a real pitch: past / true-future / GRU / const-vel.

Draws each sample window on a full 105x68 m pitch (origin = centre) and picks
representative (typical-movement) windows. Saves phase3a/forecast_viz.png.
"""
import sys, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase3a.windows import build_windows
from phase3a.models import GRUForecaster
from phase3a.physics import const_velocity, interp_const_accel

CSV = sys.argv[1] if len(sys.argv) > 1 else "thesis_results/phase2/features_test_smoothed.csv"

def draw_pitch(ax, c="0.55"):
    ax.add_patch(plt.Rectangle((-52.5, -34), 105, 68, fill=False, ec=c, lw=1))
    ax.plot([0, 0], [-34, 34], c=c, lw=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec=c, lw=1))
    ax.add_patch(plt.Rectangle((-52.5, -20.16), 16.5, 40.32, fill=False, ec=c, lw=1))
    ax.add_patch(plt.Rectangle((52.5 - 16.5, -20.16), 16.5, 40.32, fill=False, ec=c, lw=1))
    ax.set_xlim(-56, 56); ax.set_ylim(-38, 38); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

if __name__ == "__main__":
    w = build_windows(CSV)
    m = GRUForecaster(); m.load_state_dict(torch.load("phase3a/gru.pt", map_location="cpu")); m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(w["x"])).numpy()
    gru = interp_const_accel(pred, w["x"][:, -1, 2:4])
    cv  = const_velocity(w["x"])

    p0 = w["p0"][:, None, :]
    past, gt = p0 + w["x"][:, :, :2], p0 + w["gt"]
    gru_abs, cv_abs = p0 + gru, p0 + cv
    pred_pt = w["p0"][:, None, :] + pred

    # representative windows: real but typical motion (8-22 m of true 3s displacement)
    disp = np.linalg.norm(w["gt"][:, -1], axis=1)
    band = np.where((disp > 8) & (disp < 22))[0]
    sel = band[np.linspace(0, len(band) - 1, 6).astype(int)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, i in zip(axes.ravel(), sel):
        draw_pitch(ax)
        ax.plot(past[i, :, 0], past[i, :, 1], color="0.6", lw=1.6, label="past (2s)")
        ax.plot(gt[i, :, 0], gt[i, :, 1], color="green", lw=2.4, label="true future (3s)")
        ax.plot(gru_abs[i, :, 0], gru_abs[i, :, 1], color="tab:blue", lw=2, label="GRU")
        ax.plot(cv_abs[i, :, 0], cv_abs[i, :, 1], color="tab:orange", lw=1.5, ls="--", label="const-vel")
        ax.scatter(pred_pt[i, :, 0], pred_pt[i, :, 1], color="tab:blue", s=26, zorder=5)
        ax.scatter([past[i, -1, 0]], [past[i, -1, 1]], color="black", s=34, zorder=6)
        ax.set_title(f"GRU FDE@3s: {np.linalg.norm(gru_abs[i,-1]-gt[i,-1]):.1f} m", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Phase 3A forecaster on the pitch (black=now, dots=GRU 1/2/3s predictions)")
    fig.tight_layout()
    fig.savefig("phase3a/forecast_viz.png", dpi=120)
    print(f"windows after outlier guard: {len(w['x'])}  |  saved phase3a/forecast_viz.png")
