import numpy as np
from phase3a.physics import const_velocity, const_acceleration, interp_const_accel

def test_const_velocity_straight_line():
    # v = 2.5 m/s in x, zero relative start; frame t -> 2.5*(t/25) metres
    x = np.zeros((1, 50, 4), np.float32); x[0, :, 2] = 2.5
    d = const_velocity(x)
    assert d.shape == (1, 75, 2)
    np.testing.assert_allclose(d[0, 24], [2.5, 0.0], atol=1e-4)   # +1s
    np.testing.assert_allclose(d[0, 74], [7.5, 0.0], atol=1e-4)   # +3s

def test_interp_reproduces_constant_velocity():
    v0 = np.array([[2.5, 0.0]], np.float32)
    pts = np.array([[[2.5, 0], [5.0, 0], [7.5, 0]]], np.float32)  # 1/2/3s of 2.5 m/s
    d = interp_const_accel(pts, v0)
    assert d.shape == (1, 75, 2)
    np.testing.assert_allclose(d[0, 24], [2.5, 0.0], atol=1e-3)
    np.testing.assert_allclose(d[0, 74], [7.5, 0.0], atol=1e-3)