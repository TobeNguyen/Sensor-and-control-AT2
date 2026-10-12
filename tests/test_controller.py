# Including libraries
import math

import numpy as np
import pytest

from config import CONFIG
from controller import (
    DEFAULT_CRUISE_SPEED, MIN_TRACK_SPEED, LQRTracker, Reference, limit_command,
    lqr_gain, safe_command, segment_reference,
)
from robot_model import motion_model, tracking_error_model

MAX_V = CONFIG.robot.max_linear_vel
MAX_W = CONFIG.robot.max_angular_vel


@pytest.fixture(scope="module")
def tracker():
    return LQRTracker()


# ------------------------------------------------------------------------ LQR
def test_gain_shape(tracker):
    assert tracker.K.shape == (2, 3)


def test_closed_loop_is_stable(tracker):
    assert np.all(np.abs(tracker.closed_loop_poles()) < 1.0)


def test_lqr_gain_minimises_cost_better_than_zero_gain():
    Ad, Bd = tracking_error_model(0.15, 0.0, 0.02)
    K = lqr_gain(Ad, Bd, CONFIG.controller.Q(), CONFIG.controller.R())
    assert np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))) < 1.0


def test_larger_state_weight_gives_larger_gain():
    from dataclasses import replace
    heavy = replace(CONFIG, controller=replace(CONFIG.controller, q_diag=(100.0, 100.0, 10.0)))
    assert np.abs(LQRTracker(heavy).K).max() > np.abs(LQRTracker().K).max()


def test_rejects_uncontrollable_design_point():
    with pytest.raises(ValueError):
        LQRTracker(v_design=0.0, w_design=0.0)


# -------------------------------------------------------------------- compute
def test_zero_error_returns_feedforward(tracker):
    ref = Reference(1.0, 2.0, 0.5, v=0.1, w=0.05)
    v, w = tracker.compute([1.0, 2.0, 0.5], ref)
    assert v == pytest.approx(0.1)
    assert w == pytest.approx(0.05)


def test_reference_ahead_speeds_up(tracker):
    v, _ = tracker.compute([0.0, 0.0, 0.0], Reference(0.1, 0.0, 0.0, v=0.1))
    assert v > 0.1


def test_reference_behind_slows_down(tracker):
    v, _ = tracker.compute([0.0, 0.0, 0.0], Reference(-0.05, 0.0, 0.0, v=0.1))
    assert v < 0.1


def test_reference_on_the_left_turns_left(tracker):
    _, w = tracker.compute([0.0, 0.0, 0.0], Reference(0.0, 0.2, 0.0, v=0.1))
    assert w > 0


def test_reference_on_the_right_turns_right(tracker):
    _, w = tracker.compute([0.0, 0.0, 0.0], Reference(0.0, -0.2, 0.0, v=0.1))
    assert w < 0


def test_heading_error_turns_towards_reference_heading(tracker):
    _, w = tracker.compute([0.0, 0.0, 0.0], Reference(0.0, 0.0, 0.3, v=0.1))
    assert w > 0


def test_command_is_always_within_limits(tracker):
    v, w = tracker.compute([0.0, 0.0, 0.0], Reference(100.0, 50.0, 3.0, v=5.0, w=5.0))
    assert 0.0 <= v <= MAX_V
    assert abs(w) <= MAX_W


@pytest.mark.parametrize("bad_state", [
    [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0], None,
])
def test_invalid_estimate_gives_stop(tracker, bad_state):
    if bad_state is None:
        bad_state = [float("nan")] * 3
    assert tracker.compute(bad_state, Reference(1.0, 0.0, 0.0, v=0.1)) == (0.0, 0.0)


def test_invalid_reference_gives_stop(tracker):
    assert tracker.compute([0, 0, 0], Reference(float("nan"), 0.0, 0.0)) == (0.0, 0.0)


def test_front_obstacle_stops_translation_but_not_rotation(tracker):
    ref = Reference(1.0, 0.5, 0.5, v=0.1)
    v, w = tracker.compute([0.0, 0.0, 0.0], ref, front_range=0.2)
    assert v == 0.0
    assert w != 0.0


def test_free_front_does_not_change_command(tracker):
    ref = Reference(0.3, 0.1, 0.1, v=0.1)
    free = tracker.compute([0.0, 0.0, 0.0], ref, front_range=CONFIG.lidar.range_max)
    none = tracker.compute([0.0, 0.0, 0.0], ref)
    assert free == pytest.approx(none)


# ---------------------------------------------------------- limit / safe command
def test_limit_command_clips_and_blocks_reverse():
    assert limit_command(5.0, 5.0) == (MAX_V, MAX_W)
    assert limit_command(-1.0, -5.0) == (0.0, -MAX_W)
    assert limit_command(-1.0, 0.0, allow_reverse=True)[0] == -MAX_V


def test_safe_command_is_monotonic_in_front_range():
    stop = CONFIG.robot.body_radius + 0.05
    speeds = [safe_command(0.2, 0.0, r)[0] for r in np.linspace(stop - 0.1, 3.0, 40)]
    assert speeds[0] == 0.0
    assert speeds[-1] == pytest.approx(0.2)
    assert all(b >= a - 1e-12 for a, b in zip(speeds, speeds[1:]))


def test_safe_command_keeps_angular_velocity():
    assert safe_command(0.2, 0.7, 0.0)[1] == 0.7


# --------------------------------------------------------- segment reference
P0, P1 = (0.0, 0.0), (2.0, 0.0)


def test_segment_reference_mid_segment():
    ref = segment_reference([0.5, 0.3, 0.0], P0, P1)
    assert (ref.x, ref.y) == pytest.approx((0.5, 0.0))
    assert ref.theta == pytest.approx(0.0)
    assert ref.v == pytest.approx(DEFAULT_CRUISE_SPEED)


def test_segment_reference_heading_follows_segment_direction():
    ref = segment_reference([0.0, 0.5, 0.0], (0.0, 0.0), (0.0, 2.0))
    assert ref.theta == pytest.approx(math.pi / 2)


def test_segment_reference_slows_down_near_the_end():
    near = segment_reference([1.9, 0.0, 0.0], P0, P1)
    assert MIN_TRACK_SPEED <= near.v < DEFAULT_CRUISE_SPEED
    assert near.v == pytest.approx(DEFAULT_CRUISE_SPEED * 0.1 / 0.4)


def test_segment_reference_never_below_minimum_speed():
    assert segment_reference([1.99, 0.0, 0.0], P0, P1).v == pytest.approx(MIN_TRACK_SPEED)


def test_segment_reference_past_the_end_is_clamped():
    ref = segment_reference([2.5, 0.0, 0.0], P0, P1)
    assert (ref.x, ref.y) == pytest.approx((2.0, 0.0))


def test_segment_reference_degenerate_segment():
    ref = segment_reference([0.0, 0.0, 0.4], (1.0, 1.0), (1.0, 1.0))
    assert (ref.x, ref.y, ref.v) == pytest.approx((1.0, 1.0, 0.0))


# ------------------------------------------------------------------ closed loop
def run_closed_loop(tracker, x0, p0, p1, max_steps=3000):
    dt = CONFIG.sim.control_dt
    x = np.array(x0, dtype=float)
    speeds = []
    for _ in range(max_steps):
        v, w = tracker.compute(x, segment_reference(x, p0, p1))
        speeds.append((v, w))
        x = motion_model(x, v * dt, w * dt)
        if np.hypot(*(np.asarray(p1) - x[:2])) < 0.05:
            break
    return x, speeds


def test_closed_loop_converges_to_the_line_and_the_end(tracker):
    x, _ = run_closed_loop(tracker, [0.0, 0.2, 0.0], P0, P1)
    assert np.hypot(x[0] - 2.0, x[1] - 0.0) < 0.1
    assert abs(x[1]) < 0.05          # cross-track error
    assert abs(x[2]) < 0.15          # heading error


def test_closed_loop_recovers_from_heading_error(tracker):
    x, _ = run_closed_loop(tracker, [0.0, 0.0, 0.8], P0, P1)
    assert np.hypot(x[0] - 2.0, x[1]) < 0.1
    assert abs(x[1]) < 0.05


def test_closed_loop_commands_respect_limits(tracker):
    _, speeds = run_closed_loop(tracker, [0.0, 0.5, 1.0], P0, P1)
    assert all(0.0 <= v <= MAX_V and abs(w) <= MAX_W for v, w in speeds)