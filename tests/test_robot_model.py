# Including libraries
import math

import numpy as np
import pytest

from config import CONFIG
from robot_model import (
    analyse_model, controllability_matrix, default_landmark, innovation,
    is_controllable, is_observable, jacobian_F, jacobian_G, measurement_jacobian,
    measurement_model, motion_model, observability_matrix, odometry_increment,
    pose_error, tracking_error_model, wrap_angle,
)

LANDMARK = np.array([1.9, 1.0])
POSE = np.array([0.0, 0.0, 0.3])


def numeric_jacobian(func, x, eps=1e-6):
    """Central finite-difference Jacobian of func at x."""
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        cols.append((func(x + d) - func(x - d)) / (2 * eps))
    return np.stack(cols, axis=1)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert abs(wrap_angle(10.0)) <= math.pi


def test_default_landmark_matches_config():
    np.testing.assert_allclose(default_landmark(), CONFIG.env.dock_pose[:2])


# ------------------------------------------------------------------ odometry
def test_odometry_straight():
    ds, dth = odometry_increment([0.0, 0.0], [3.0, 3.0])
    assert ds == pytest.approx(CONFIG.robot.wheel_radius * 3.0)
    assert dth == pytest.approx(0.0)


def test_odometry_rotation_in_place():
    ds, dth = odometry_increment([0.0, 0.0], [-1.0, 1.0])
    assert ds == pytest.approx(0.0)
    assert dth == pytest.approx(2 * CONFIG.robot.wheel_radius / CONFIG.robot.wheel_base)


def test_odometry_uses_difference_not_absolute_value():
    a = odometry_increment([5.0, 5.0], [6.0, 7.0])
    b = odometry_increment([0.0, 0.0], [1.0, 2.0])
    assert a == pytest.approx(b)


# -------------------------------------------------------------- motion model
def test_motion_straight():
    x = motion_model([0.0, 0.0, 0.0], 0.5, 0.0)
    np.testing.assert_allclose(x, [0.5, 0.0, 0.0], atol=1e-12)


def test_motion_heading_north():
    x = motion_model([1.0, 1.0, math.pi / 2], 0.5, 0.0)
    np.testing.assert_allclose(x, [1.0, 1.5, math.pi / 2], atol=1e-12)


def test_motion_rotation_wraps_heading():
    x = motion_model([0.0, 0.0, 3.0], 0.0, 0.5)
    assert -math.pi <= x[2] < math.pi
    assert x[2] == pytest.approx(wrap_angle(3.5))


def test_motion_full_circle_returns_to_start():
    v, w, dt = 0.2, 0.5, 0.01
    x = np.array([0.0, 0.0, 0.0])
    for _ in range(int(round(2 * math.pi / w / dt))):
        x = motion_model(x, v * dt, w * dt)
    np.testing.assert_allclose(x[:2], [0.0, 0.0], atol=1e-2)


def test_jacobian_F_matches_numeric():
    ds, dth = 0.05, 0.1
    expected = numeric_jacobian(lambda s: motion_model(s, ds, dth), POSE)
    np.testing.assert_allclose(jacobian_F(POSE, ds, dth), expected, atol=1e-6)


def test_jacobian_G_matches_numeric():
    ds, dth = 0.05, 0.1
    expected = numeric_jacobian(lambda u: motion_model(POSE, u[0], u[1]), [ds, dth])
    np.testing.assert_allclose(jacobian_G(POSE, ds, dth), expected, atol=1e-6)


# --------------------------------------------------------- measurement model
def test_measurement_landmark_straight_ahead():
    z = measurement_model([0.0, 0.0, 0.0], [2.0, 0.0])
    np.testing.assert_allclose(z, [2.0, 0.0], atol=1e-12)


def test_measurement_landmark_on_the_left():
    z = measurement_model([0.0, 0.0, 0.0], [0.0, 1.0])
    np.testing.assert_allclose(z, [1.0, math.pi / 2], atol=1e-12)


def test_measurement_bearing_is_relative_to_heading():
    z = measurement_model([0.0, 0.0, math.pi / 2], [0.0, 1.0])
    assert z[1] == pytest.approx(0.0)


def test_measurement_jacobian_matches_numeric():
    expected = numeric_jacobian(lambda s: measurement_model(s, LANDMARK), POSE)
    np.testing.assert_allclose(measurement_jacobian(POSE, LANDMARK), expected, atol=1e-5)


def test_measurement_jacobian_rejects_robot_on_landmark():
    with pytest.raises(ValueError):
        measurement_jacobian([1.9, 1.0, 0.0], LANDMARK)


def test_innovation_wraps_bearing():
    nu = innovation([1.0, -3.1], [1.0, 3.1])
    assert nu[1] == pytest.approx(wrap_angle(-6.2))
    assert abs(nu[1]) < 0.1


# ------------------------------------------------------------ tracking model
def test_pose_error_zero_at_reference():
    np.testing.assert_allclose(pose_error(POSE, POSE), [0.0, 0.0, 0.0], atol=1e-12)


def test_pose_error_is_in_robot_frame():
    # Robot faces +y, reference is 1 m further along +y: error is "ahead" (ex = 1).
    e = pose_error([0.0, 0.0, math.pi / 2], [0.0, 1.0, math.pi / 2])
    np.testing.assert_allclose(e, [1.0, 0.0, 0.0], atol=1e-12)


def test_pose_error_left_offset_is_positive_ey():
    e = pose_error([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(e, [0.0, 1.0, 0.0], atol=1e-12)


def test_tracking_error_model_shapes_and_values():
    Ad, Bd = tracking_error_model(0.2, 0.1, 0.02)
    assert Ad.shape == (3, 3) and Bd.shape == (3, 2)
    assert Ad[1, 2] == pytest.approx(0.2 * 0.02)
    assert Bd[0, 0] == pytest.approx(-0.02)


def test_tracking_error_model_matches_nonlinear_dynamics():
    # Small error, small input change: linear model should agree with the real kinematics.
    v_ref, w_ref, dt = 0.2, 0.1, 0.02
    Ad, Bd = tracking_error_model(v_ref, w_ref, dt)
    e0 = np.array([0.01, -0.01, 0.02])
    du = np.array([0.01, -0.02])

    x_ref = np.array([1.0, 2.0, 0.5])
    # Build a robot pose that gives error e0 relative to the reference.
    # The error is expressed in the ROBOT frame, so rotate by the robot heading.
    th = x_ref[2] - e0[2]
    c, s = math.cos(th), math.sin(th)
    x = np.array([x_ref[0] - (c * e0[0] - s * e0[1]),
                  x_ref[1] - (s * e0[0] + c * e0[1]),
                th])
    # Sanity check: the constructed pose really has the intended error.
    np.testing.assert_allclose(pose_error(x, x_ref), e0, atol=1e-12)


# ----------------------------------------- controllability / observability
def test_controllability_matrix_shape():
    Ad, Bd = tracking_error_model(0.2, 0.0, 0.02)
    assert controllability_matrix(Ad, Bd).shape == (3, 6)


def test_controllable_when_moving_forward():
    assert is_controllable(*tracking_error_model(0.2, 0.0, 0.02))


def test_controllable_when_only_rotating_reference():
    assert is_controllable(*tracking_error_model(0.0, 0.5, 0.02))


def test_not_controllable_when_reference_is_still():
    # Non-holonomic robot: the lateral error cannot be corrected without motion.
    Ad, Bd = tracking_error_model(0.0, 0.0, 0.02)
    assert np.linalg.matrix_rank(controllability_matrix(Ad, Bd)) == 2


def test_observability_matrix_shape():
    F = jacobian_F(POSE, 0.004, 0.0)
    H = measurement_jacobian(POSE, LANDMARK)
    assert observability_matrix(F, H).shape == (6, 3)


def test_observable_when_moving_with_marker_visible():
    F = jacobian_F(POSE, 0.2 * 0.02, 0.0)
    H = measurement_jacobian(POSE, LANDMARK)
    assert is_observable(F, H)


def test_not_observable_when_stationary():
    F = jacobian_F(POSE, 0.0, 0.0)
    H = measurement_jacobian(POSE, LANDMARK)
    assert np.linalg.matrix_rank(observability_matrix(F, H)) == 2


def test_analyse_model_moving_and_stopped():
    moving = analyse_model(POSE, 0.2, 0.0)
    assert moving["controllability_rank"] == 3
    assert moving["observability_rank"] == 3
    stopped = analyse_model(POSE, 0.0, 0.0)
    assert stopped["controllability_rank"] < 3
    assert stopped["observability_rank"] < 3