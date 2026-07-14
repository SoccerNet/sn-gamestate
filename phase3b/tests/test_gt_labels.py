import numpy as np
from phase3b.pitch import dist_to_own_goal, pressing_third

def test_block_height_and_thirds():
    team = np.array(["left", "left", "right", "right"])
    x    = np.array([-45.0,  10.0,  45.0,  -10.0])   # deep, pushed-up, deep, pushed-up
    d = dist_to_own_goal(x, team)
    np.testing.assert_allclose(d, [7.5, 62.5, 7.5, 62.5], atol=1e-6)
    assert list(pressing_third(x, team)) == ["low", "mid", "low", "mid"]

def test_high_press():
    # left team centroid well into opponent half -> high
    assert pressing_third(np.array([40.0]), np.array(["left"]))[0] == "high"