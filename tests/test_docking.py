# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

import docking
from config import CONFIG
from docking import (
    ALIGN_TIMEOUT, BETA_MAX, DOCK_MAX_SPEED, DOCK_MIN_SPEED, K_AXIS, K_BEARING, K_CENTER,
    K_RANGE, SEARCH_OMEGA, DockCommand, DockPhase, DockingController, docking_command,
    is_docked, min_visible_range, propagate_marker,
)
from robot_model import measurement_model, motion_model, wrap_angle
from sensing import (
    MarkerMeasurement, OdometryIncrement, Sensing, camera_forward_offset, camera_matrix,
)

D = CONFIG.docking
DOCK_X, DOCK_Y, DOCK_YAW = CONFIG.env.dock_pose
LANDMARK = np.array([DOCK_X, DOCK_Y])
HEAD_ON = wrap_angle(DOCK_YAW - math.pi)      # world heading that looks straight at the marker
DT = CONFIG.sim.control_dt
K = camera_matrix()
HALF_FOV = math.atan(K[0, 2] / K[0, 0])       # horizontal half field of view
MIN_RANGE = min_visible_range(CONFIG, margin=1.0)


def marker(rng, bearing, view, t):
    return MarkerMeasurement(rng, bearing, view, CONFIG.camera.marker_id, t)


def odom(t, ds=0.0, dtheta=0.0):
    return OdometryIncrement(ds, dtheta, t)


def lateral_error(x):
    """Distance of the robot from the line through the marker normal."""
    return abs(-math.sin(DOCK_YAW) * (x[0] - DOCK_X) + math.cos(DOCK_YAW) * (x[1] - DOCK_Y))


# ------------------------------------------------------------- pure functions
def test_does_not_import_the_simulator():
    assert not hasattr(docking, "Simulation")


def test_min_visible_range_formula():
    expected = (1.2 * (CONFIG.camera.marker_size / 2)
                / math.tan(math.radians(CONFIG.camera.fovy_deg) / 2)
                + camera_forward_offset() + 0.005)
    assert min_visible_range(CONFIG) == pytest.approx(expected)


def test_larger_margin_needs_larger_range():
    assert min_visible_range(CONFIG, 1.5) > min_visible_range(CONFIG, 1.0)


def test_target_distance_keeps_marker_in_view():
    assert D.target_distance >= min_visible_range(CONFIG), (
        "Marker leaves the camera view before docking: increase "
        "DockingConfig.target_distance (0.40 is recommended).")


def test_propagate_marker_moving_forward():
    r, b, psi = propagate_marker(0.8, 0.0, 0.0, 0.1, 0.0)
    assert r == pytest.approx(0.7) and b == pytest.approx(0.0) and psi == pytest.approx(0.0)


def test_propagate_marker_turning_left_moves_marker_to_the_right():
    r, b, psi = propagate_marker(1.0, 0.0, 0.3, 0.0, 0.1)
    assert r == pytest.approx(1.0)
    assert b == pytest.approx(-0.1)
    assert psi == pytest.approx(0.2)


def test_propagate_marker_matches_motion_model():
    pose0 = np.array([0.0, 0.0, 0.2])
    ds, dth = 0.1, 0.15
    pose1 = motion_model(pose0, ds, dth)
    r0, b0 = measurement_model(pose0, np.array([3.0, 1.0]))
    r1, b1 = measurement_model(pose1, np.array([3.0, 1.0]))
    r, b, _ = propagate_marker(r0, b0, 0.0, ds, dth)
    assert r == pytest.approx(r1, abs=1e-9)
    assert b == pytest.approx(b1, abs=1e-9)


def test_docking_command_far_and_centred_drives_straight_at_max_speed():
    v, w = docking_command(D.target_distance + 0.5, 0.0, 0.0)
    assert v == pytest.approx(DOCK_MAX_SPEED)
    assert w == pytest.approx(0.0)


def test_docking_command_slows_down_near_the_target():
    v, _ = docking_command(D.target_distance + D.distance_tolerance + 0.02, 0.0, 0.0)
    assert DOCK_MIN_SPEED <= v < DOCK_MAX_SPEED
    assert v == pytest.approx(K_RANGE * (D.distance_tolerance + 0.02))


def test_docking_command_has_a_minimum_speed():
    tight = replace(CONFIG, docking=replace(D, distance_tolerance=0.005))
    v, _ = docking_command(D.target_distance + 0.006, 0.0, 0.0, tight)
    assert v == pytest.approx(DOCK_MIN_SPEED)


def test_docking_command_stops_translation_when_arrived():
    v, _ = docking_command(D.target_distance, 0.0, 0.0)
    assert v == 0.0


def test_docking_command_marker_on_the_left_turns_left():
    assert docking_command(1.0, 0.2, 0.0)[1] > 0


def test_docking_command_marker_on_the_right_turns_right():
    assert docking_command(1.0, -0.2, 0.0)[1] < 0


def test_docking_command_turns_first_when_marker_is_far_off_centre():
    v, w = docking_command(1.0, BETA_MAX + 0.1, 0.0)
    assert v == 0.0 and w > 0


def test_docking_command_view_angle_term():
    assert docking_command(1.0, 0.0, +0.2)[1] < 0
    assert docking_command(1.0, 0.0, -0.2)[1] > 0


def test_docking_command_matches_the_control_law():
    _, w = docking_command(1.0, 0.1, 0.05)
    assert w == pytest.approx((K_BEARING + K_AXIS) * 0.1 - K_AXIS * 0.05)


def test_docking_command_centres_the_marker_when_arrived():
    v, w = docking_command(D.target_distance, 0.1, 0.1)
    assert v == 0.0 and w == pytest.approx(K_CENTER * 0.1)


def test_docking_command_respects_velocity_limits():
    v, w = docking_command(5.0, 0.55, -1.0)
    assert 0.0 <= v <= CONFIG.robot.max_linear_vel
    assert abs(w) <= CONFIG.robot.max_angular_vel


def test_is_docked_requires_range_bearing_and_axis():
    assert is_docked(D.target_distance, 0.0, 0.0)
    assert not is_docked(D.target_distance + D.distance_tolerance + 0.05, 0.0, 0.0)
    assert not is_docked(D.target_distance, D.bearing_tolerance + 0.05, 0.0)
    assert not is_docked(D.target_distance, 0.0, -0.5)     # robot far off the marker axis


# ------------------------------------------------------------ state machine
def test_search_without_marker_rotates_in_place():
    cmd = DockingController().update(None, odom(0.0))
    assert cmd.phase is DockPhase.SEARCH and not cmd.finished
    assert cmd.v == 0.0 and cmd.w == pytest.approx(SEARCH_OMEGA)
    assert math.isnan(cmd.marker_range)


def test_begin_chooses_search_direction_towards_the_marker():
    dock = DockingController()
    dock.begin((DOCK_X - 0.7, DOCK_Y, math.pi / 2))        # marker is on the right
    assert dock.update(None, odom(0.0)).w == pytest.approx(-SEARCH_OMEGA)
    dock.begin((DOCK_X - 0.7, DOCK_Y, -math.pi / 2))       # marker is on the left
    assert dock.update(None, odom(0.0)).w == pytest.approx(+SEARCH_OMEGA)


def test_begin_ignores_an_invalid_pose_hint():
    dock = DockingController()
    dock.begin([float("nan"), 0.0, 0.0])
    assert dock.update(None, odom(0.0)).w == pytest.approx(SEARCH_OMEGA)


def test_search_times_out():
    dock = DockingController()
    cmd = None
    for k in range(int((D.max_search_time + 2.0) / DT)):
        cmd = dock.update(None, odom(k * DT))
    assert cmd.phase is DockPhase.FAILED and cmd.finished
    assert (cmd.v, cmd.w) == (0.0, 0.0)


def test_approach_when_the_marker_is_visible():
    cmd = DockingController().update(marker(0.8, 0.0, 0.0, 0.0), odom(0.0))
    assert cmd.phase is DockPhase.APPROACH and not cmd.finished
    assert cmd.v == pytest.approx(DOCK_MAX_SPEED)
    assert cmd.w == pytest.approx(0.0, abs=1e-9)
    assert cmd.marker_range == pytest.approx(0.8)
    assert cmd.axis_angle == pytest.approx(0.0)


def test_last_measurement_is_propagated_with_odometry():
    dock = DockingController()
    dock.update(marker(0.8, 0.0, 0.0, 0.0), odom(0.0))
    cmd = dock.update(None, odom(0.02, ds=0.1))            # no new camera frame
    assert cmd.phase is DockPhase.APPROACH
    assert cmd.marker_range == pytest.approx(0.7)


def test_lost_marker_returns_to_search_towards_its_last_side():
    dock = DockingController()
    dock.update(marker(0.8, 0.3, 0.0, 0.0), odom(0.0))
    for k in range(1, 31):                                  # 0.6 s without a measurement
        cmd = dock.update(None, odom(k * DT))
        if k == 10:
            assert cmd.phase is DockPhase.APPROACH          # still within the timeout
    assert cmd.phase is DockPhase.SEARCH
    assert cmd.w == pytest.approx(+SEARCH_OMEGA)            # marker was on the left


def test_docked_is_latched():
    dock = DockingController()
    cmd = dock.update(marker(D.target_distance, 0.0, 0.0, 0.0), odom(0.0))
    assert cmd.phase is DockPhase.DOCKED and cmd.finished and cmd.docked
    assert (cmd.v, cmd.w) == (0.0, 0.0)
    cmd = dock.update(marker(2.0, 0.5, 0.0, DT), odom(DT))
    assert cmd.phase is DockPhase.DOCKED


def test_arrived_but_off_axis_aligns_then_fails():
    dock = DockingController()
    cmd = dock.update(marker(D.target_distance, 0.0, -0.5, 0.0), odom(0.0))
    assert cmd.phase is DockPhase.ALIGN and cmd.v == 0.0
    for k in range(1, int((ALIGN_TIMEOUT + 1.0) / DT)):
        cmd = dock.update(marker(D.target_distance, 0.0, -0.5, k * DT), odom(k * DT))
    assert cmd.phase is DockPhase.FAILED and cmd.finished


def test_align_centres_the_marker_without_translating():
    cmd = DockingController().update(marker(D.target_distance, 0.2, 0.2, 0.0), odom(0.0))
    assert cmd.phase is DockPhase.ALIGN
    assert cmd.v == 0.0 and cmd.w == pytest.approx(K_CENTER * 0.2)


def test_invalid_marker_is_treated_as_not_visible():
    bad = MarkerMeasurement(float("nan"), 0.0, 0.0, 0, 0.0)
    assert DockingController().update(bad, odom(0.0)).phase is DockPhase.SEARCH


def test_reset_restores_initial_state():
    dock = DockingController()
    dock.update(marker(D.target_distance, 0.0, 0.0, 0.0), odom(0.0))
    assert dock.finished
    dock.reset()
    assert not dock.finished and dock.phase is DockPhase.SEARCH


def test_dock_command_docked_property():
    cmd = DockCommand(0.0, 0.0, DockPhase.DOCKED, 0.4, 0.0, 0.0, True)
    assert cmd.docked


# ------------------------------------------------- closed loop (no MuJoCo needed)
def true_marker(pose, t, rng=None):
    """Marker measurement computed from geometry, as the camera would deliver it."""
    r, b = measurement_model(pose, LANDMARK)
    view = wrap_angle(DOCK_YAW - pose[2] - math.pi)
    if abs(b) >= HALF_FOV or r <= MIN_RANGE or abs(view) >= 1.2:
        return None                                        # outside the field of view
    if rng is not None:                                    # measurement noise
        r += rng.normal(0.0, 0.01)
        b += rng.normal(0.0, 0.005)
        view += rng.normal(0.0, 0.03)
    return marker(r, b, view, t)


def run_dock(x0, noise_seed=None, max_time=60.0):
    """Simulate docking with a 10 Hz camera and exact kinematics."""
    rng = None if noise_seed is None else np.random.default_rng(noise_seed)
    dock = DockingController()
    dock.begin(x0)
    x = np.array(x0, dtype=float)
    ds = dth = 0.0
    last_cam = -math.inf
    cmd, t = None, 0.0
    for k in range(int(max_time / DT)):
        t = k * DT
        m = None
        if t - last_cam >= 0.1 - 1e-9:
            last_cam = t
            m = true_marker(x, t, rng)
        cmd = dock.update(m, odom(t, ds, dth))
        if cmd.finished:
            break
        ds, dth = cmd.v * DT, cmd.w * DT
        x = motion_model(x, ds, dth)
    return x, cmd, t


def check_final_pose(x, lateral_tol, heading_tol):
    rng = math.hypot(DOCK_X - x[0], DOCK_Y - x[1])
    assert D.target_distance - D.distance_tolerance <= rng <= (
        D.target_distance + D.distance_tolerance + 0.04)
    assert lateral_error(x) < lateral_tol
    assert abs(wrap_angle(x[2] - HEAD_ON)) < heading_tol


def test_closed_loop_docks_from_the_navigation_handoff_pose():
    wx, wy = CONFIG.env.waypoints[-1]
    x, cmd, t = run_dock((wx, wy, math.pi / 2))             # marker starts out of view
    assert cmd.phase is DockPhase.DOCKED
    assert t < 40.0
    check_final_pose(x, lateral_tol=0.05, heading_tol=0.2)


def test_closed_loop_docks_from_a_small_lateral_offset():
    wx, wy = CONFIG.env.waypoints[-1]
    x, cmd, _ = run_dock((wx, wy - 0.08, HEAD_ON))
    assert cmd.phase is DockPhase.DOCKED
    check_final_pose(x, lateral_tol=0.05, heading_tol=0.2)


def test_closed_loop_docks_with_measurement_noise():
    wx, wy = CONFIG.env.waypoints[-1]
    x, cmd, _ = run_dock((wx, wy, math.pi / 2), noise_seed=1)
    assert cmd.phase is DockPhase.DOCKED
    check_final_pose(x, lateral_tol=0.08, heading_tol=0.25)


def test_closed_loop_fails_when_the_marker_is_never_seen():
    far_away = (DOCK_X - 0.7, DOCK_Y, math.pi)              # facing away from the marker
    dock = DockingController()
    cmd = None
    for k in range(int((D.max_search_time + 2.0) / DT)):
        cmd = dock.update(None, odom(k * DT))
    assert far_away is not None and cmd.phase is DockPhase.FAILED


# --------------------------------------------------------- real MuJoCo (optional)
@pytest.fixture
def real_sim(tmp_path):
    pytest.importorskip("mujoco")
    from simulation import Simulation
    sim = Simulation(scene_dir=tmp_path, seed=0)
    yield sim
    sim.close()


def read_or_skip(sensing):
    try:
        return sensing.read()
    except Exception as exc:  # no OpenGL context available
        pytest.skip(f"Offscreen rendering not available: {exc}")


@pytest.mark.parametrize("rng", [D.target_distance, D.target_distance + D.distance_tolerance])
def test_real_marker_is_visible_at_docking_range(real_sim, rng):
    real_sim.reset((DOCK_X - rng, DOCK_Y, 0.0))
    frame = read_or_skip(Sensing(real_sim, camera_period=0.0))
    assert frame.marker is not None
    assert frame.marker.range == pytest.approx(rng, abs=0.05)


@pytest.mark.parametrize("theta", [0.2, -0.2])
def test_real_view_angle_matches_geometry(real_sim, theta):
    real_sim.reset((DOCK_X - 0.7, DOCK_Y, theta))
    m = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    assert m is not None
    assert m.view_angle == pytest.approx(wrap_angle(DOCK_YAW - theta - math.pi), abs=0.1)


def test_real_docking_end_to_end(real_sim):
    wx, wy = CONFIG.env.waypoints[-1]
    start = (wx, wy, math.pi / 2)
    real_sim.reset(start)
    sensing = Sensing(real_sim)                             # camera at 10 Hz
    dock = DockingController()
    dock.begin(start)

    cmd = None
    for _ in range(int(60.0 / DT)):
        frame = read_or_skip(sensing)
        cmd = dock.update(frame.marker, frame.odometry)
        if cmd.finished:
            break
        real_sim.step(cmd.v, cmd.w)

    assert cmd is not None and cmd.phase is DockPhase.DOCKED
    # Ground truth is used here for EVALUATION only.
    x = real_sim.ground_truth()
    assert not real_sim.collided
    check_final_pose(x, lateral_tol=0.06, heading_tol=0.2)