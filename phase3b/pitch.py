import numpy as np

LENGTH, GOAL = 105.0, 52.5

def dist_to_own_goal(x, team):
    x = np.asarray(x, dtype=float)
    left = np.asarray(team) == "left"
    return np.where(left, x + GOAL, GOAL - x)   # 0 at own goal, 105 at opponent goal

def pressing_third(x, team):
    h = dist_to_own_goal(x, team) / LENGTH
    out = np.full(h.shape, "mid", dtype=object)
    out[h < 1/3] = "low"
    out[h >= 2/3] = "high"
    return out