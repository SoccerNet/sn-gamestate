import numpy as np

HORIZONS = (25, 50, 75)

def const_velocity(x, fps=25):
    v = x[:, -1, 2:4]                       # (B,2) m/s at last input frame
    t = (np.arange(1, 76) / fps)[None, :, None]   # (1,75,1) seconds
    return (v[:, None, :] * t).astype(np.float32)  # (B,75,2)

def const_acceleration(x, fps=25):
    v_last = x[:, -1, 2:4]
    v_prev = x[:, -6, 2:4]                   # 5 frames earlier
    a = (v_last - v_prev) / (5.0 / fps)      # (B,2) m/s^2
    t = (np.arange(1, 76) / fps)[None, :, None]
    d = v_last[:, None, :] * t + 0.5 * a[:, None, :] * t**2
    return d.astype(np.float32)

def interp_const_accel(pts3, v0, fps=25):
    """Piecewise constant-acceleration through knots at t=0 (disp 0, vel v0)
    and the 3 predicted displacements at frames 25/50/75."""
    B = pts3.shape[0]
    knot_f = np.array([0, 25, 50, 75])
    knot_p = np.concatenate([np.zeros((B, 1, 2), np.float32), pts3], axis=1)  # (B,4,2)
    out = np.zeros((B, 75, 2), np.float32)
    v = v0.astype(np.float32).copy()          # per-second velocity, updated per segment
    for k in range(3):
        f0, f1 = knot_f[k], knot_f[k + 1]
        dt = (f1 - f0) / fps
        p_start = knot_p[:, k, :]
        p_end = knot_p[:, k + 1, :]
        a = 2.0 * (p_end - p_start - v * dt) / dt**2       # (B,2)
        for f in range(f0 + 1, f1 + 1):
            tt = (f - f0) / fps
            out[:, f - 1, :] = p_start + v * tt + 0.5 * a * tt**2
        v = v + a * dt                                     # velocity continuity
    return out