# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

import docking
from dataclasses import replace
from config import CONFIG, RealismConfig
IDEAL = replace(CONFIG, realism=RealismConfig(enabled=False))
from docking import (
    ALIGN_TIMEOUT, BETA_MAX, BLIND_TIMEOUT, DOCK_CONFIRM_FRAMES, DOCK_MAX_SPEED, DOCK_MIN_SPEED,
    K_AXIS, K_BEARING, K_CENTER, K_RANGE, MAX_DOCK_TIME, SEARCH_OMEGA, DockCommand, DockPhase,
    DockingController, docking_command, is_docked, marker_from_pose, min_visible_range,
    propagate_marker,
)
from estimator import EKFEstimator
from robot_model import measurement_model, motion_model, view_angle_model, wrap_angle
from sensing import (
    MarkerMeasurement, OdometryIncrement, Sensing, camera_forward_offset, camera_matrix,
)

D = CONFIG.docking
DOCK_X, DOCK_Y, DOCK_YAW = CONFIG.env.dock_pose
HANDOFF = (1.05, 1.0)
LANDMARK = np.array([DOCK_X, DOCK_Y])
HEAD_ON = wrap_angle(DOCK_YAW - math.pi)      # world heading that looks straight at the marker
DT = CONFIG.sim.control_dt
K = camera_matrix()
HALF_FOV = math.atan(K[0, 2] / K[0, 0])       # horizontal half field of view
MIN_RANGE = min_visible_range(CONFIG, margin=1.0)

class ConfigTagMap:
    # Test helper: pretends the config poses were "learned" at run time
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def table(self):
        return {int(i): (np.array(p[:2], dtype=float), float(p[2]))
                for i, p in self.cfg.tag_table().items()}
    
def make_dock(x0=None):
    # Controller that already knows the dock (as if learned during SURVEY)
    dock = DockingController()
    dock.begin(x0, landmark=LANDMARK, dock_yaw=DOCK_YAW)
    return dock

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


def test_docking_command_never_lets_the_view_term_cancel_the_turn_towards_the_marker():
    # Far off-centre the robot does not translate; the view term must not stop it from turning.
    v, w = docking_command(1.0, -(BETA_MAX + 0.1), -1.0)
    assert v == 0.0 and w < 0

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
    dock.begin((DOCK_X - 0.7, DOCK_Y, math.pi / 2), landmark=LANDMARK, dock_yaw=DOCK_YAW)   # marker on the right
    assert dock.update(None, odom(0.0)).w == pytest.approx(-SEARCH_OMEGA)
    dock.begin((DOCK_X - 0.7, DOCK_Y, -math.pi / 2), landmark=LANDMARK, dock_yaw=DOCK_YAW)  # marker on the left
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
        r += rng.normal(0.0, CONFIG.camera.range_noise_std)
        b += rng.normal(0.0, CONFIG.camera.bearing_noise_std)
        view += rng.normal(0.0, getattr(CONFIG.camera, "view_noise_std", 0.03))
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
    wx, wy = HANDOFF
    x, cmd, t = run_dock((wx, wy, math.pi / 2))             # marker starts out of view
    assert cmd.phase is DockPhase.DOCKED
    assert t < 40.0
    check_final_pose(x, lateral_tol=0.05, heading_tol=0.2)


def test_closed_loop_docks_from_a_small_lateral_offset():
    wx, wy = HANDOFF
    x, cmd, _ = run_dock((wx, wy - 0.08, HEAD_ON))
    assert cmd.phase is DockPhase.DOCKED
    check_final_pose(x, lateral_tol=0.05, heading_tol=0.2)


def test_closed_loop_docks_with_measurement_noise():
    wx, wy = HANDOFF
    x, cmd, _ = run_dock((wx, wy, math.pi / 2), noise_seed=1)
    assert cmd.phase is DockPhase.DOCKED
    check_final_pose(x, lateral_tol=0.08, heading_tol=0.25)


def test_closed_loop_fails_when_the_marker_is_never_seen():
    far_away = (DOCK_X - 0.7, DOCK_Y, math.pi)              # facing away from the marker
    dock = DockingController()
    dock.begin(far_away)
    x, cmd = np.array(far_away, dtype=float), None
    for k in range(int((D.max_search_time + 2.0) / DT)):
        cmd = dock.update(None, odom(k * DT))               # the camera never delivers a detection
        if cmd.finished:
            break
        x = motion_model(x, cmd.v * DT, cmd.w * DT)
    assert cmd.phase is DockPhase.FAILED
    np.testing.assert_allclose(x[:2], far_away[:2])         # it only rotated, it never drove blind


def test_closed_loop_search_finds_a_marker_behind_the_robot():
    x, cmd, _ = run_dock((DOCK_X - 0.7, DOCK_Y, math.pi))
    assert cmd.phase is DockPhase.DOCKED
    check_final_pose(x, lateral_tol=0.05, heading_tol=0.2)


def test_docking_fails_after_the_total_time_limit():
    dock = DockingController()
    cmd = None
    for k in range(int((MAX_DOCK_TIME + 1.0) / DT)):         # robot never moves, marker stays far
        cmd = dock.update(marker(1.5, 0.0, 0.0, k * DT), odom(k * DT))
    assert cmd.phase is DockPhase.FAILED and cmd.finished


# ----------------------------------------- docking driven by the EKF estimate (x_est)
def test_marker_from_pose_matches_the_models():
    pose = np.array([DOCK_X - 0.8, DOCK_Y - 0.1, 0.15])
    r, b, v = marker_from_pose(pose, CONFIG, LANDMARK, DOCK_YAW)
    r_ref, b_ref = measurement_model(pose, LANDMARK)
    assert r == pytest.approx(r_ref) and b == pytest.approx(b_ref)
    assert v == pytest.approx(view_angle_model(pose, DOCK_YAW))


def test_marker_from_pose_rejects_unusable_poses():
    assert marker_from_pose([float("nan"), 0.0, 0.0], CONFIG, LANDMARK, DOCK_YAW) is None
    assert marker_from_pose([DOCK_X, DOCK_Y, 0.0], CONFIG, LANDMARK, DOCK_YAW) is None   # on top of the marker
    assert marker_from_pose([1.0, 2.0], CONFIG, LANDMARK, DOCK_YAW) is None              # wrong shape


def test_marker_from_pose_needs_a_learned_dock():
    assert marker_from_pose([DOCK_X - 0.8, DOCK_Y, 0.0]) is None


def test_estimate_mode_keeps_approaching_without_a_marker():
    pose = (DOCK_X - 1.0, DOCK_Y, HEAD_ON)
    dock = make_dock()
    dock.update(marker(1.0, 0.0, 0.0, 0.0), odom(0.0), pose)
    cmd = None
    for k in range(1, int(1.5 / DT)):                        # 1.5 s with no detection
        cmd = dock.update(None, odom(k * DT), pose)
    assert cmd.phase is DockPhase.APPROACH and cmd.v > 0.0  # raw mode would be searching by now
    assert cmd.marker_range == pytest.approx(1.0)


def test_estimate_mode_turns_towards_the_marker_when_facing_away():
    cmd = make_dock().update(None, odom(0.0), (DOCK_X - 0.7, DOCK_Y, math.pi / 2))
    assert cmd.v == 0.0 and cmd.w < 0.0                     # marker is on the right


def test_estimate_mode_stops_trusting_the_estimate_when_blind_for_too_long():
    pose = (DOCK_X - 1.0, DOCK_Y, HEAD_ON)
    dock = make_dock()
    cmd = None
    n_blind = int((BLIND_TIMEOUT + 1.0) / DT)
    for k in range(int((BLIND_TIMEOUT + D.max_search_time + 2.0) / DT)):
        cmd = dock.update(None, odom(k * DT), pose)
        if k == n_blind:
            assert cmd.phase is DockPhase.SEARCH and cmd.v == 0.0
    assert cmd.phase is DockPhase.FAILED


def test_estimate_mode_needs_a_real_detection_to_declare_docked():
    pose = (DOCK_X - D.target_distance, DOCK_Y, HEAD_ON)    # the estimate says "docked"
    dock = make_dock()
    cmd = dock.update(None, odom(0.0), pose)
    assert cmd.phase is DockPhase.ALIGN and not cmd.docked
    for k in range(1, DOCK_CONFIRM_FRAMES + 1):              # several consecutive real detections
        cmd = dock.update(marker(D.target_distance, 0.0, 0.0, k * DT), odom(k * DT), pose)
    assert cmd.docked and cmd.finished


def test_estimate_mode_falls_back_to_the_raw_measurement_for_an_unusable_pose():
    cmd = DockingController().update(marker(0.8, 0.0, 0.0, 0.0), odom(0.0), [float("nan"), 0.0, 0.0])
    assert cmd.phase is DockPhase.APPROACH
    assert cmd.marker_range == pytest.approx(0.8)


def run_dock_ekf(x0, seed=0, bias=0.0, blackout=None, use_estimate=True, max_time=60.0):
    """Docking with the EKF in the loop: noisy odometry with a heading bias [rad/step], noisy
    10 Hz camera and an optional camera blackout window (t0, t1) [s]. The controller never sees
    the true pose; the EKF starts from a perturbed pose drawn from its own P0."""
    rng = np.random.default_rng(seed)
    rob = CONFIG.robot
    s_ds = rob.wheel_radius * rob.encoder_noise_std / math.sqrt(2.0)
    s_dth = rob.wheel_radius * rob.encoder_noise_std * math.sqrt(2.0) / rob.wheel_base
    x = np.array(x0, dtype=float)
    ekf = EKFEstimator(CONFIG, gate=9.21, use_lidar=False)
    ekf.set_tag_map(ConfigTagMap())          # dock known, as if learned
    ekf.reset(x + rng.normal(0.0, CONFIG.ekf.init_pose_std))
    dock = DockingController()
    dock.begin(ekf.x, landmark=LANDMARK, dock_yaw=DOCK_YAW)
    ds = dth = 0.0
    last_cam = -math.inf
    cmd, t = None, 0.0
    for k in range(int(max_time / DT)):
        t = k * DT
        ds_m = ds + rng.normal(0.0, s_ds)
        dth_m = dth + bias + rng.normal(0.0, s_dth)
        m = None
        if t - last_cam >= 0.1 - 1e-9:
            last_cam = t
            if blackout is None or not (blackout[0] <= t < blackout[1]):
                m = true_marker(x, t, rng)
        ekf.predict(ds_m, dth_m)
        ekf.update(m)
        cmd = dock.update(m, odom(t, ds_m, dth_m), ekf.x.copy() if use_estimate else None)
        if cmd.finished:
            break
        ds, dth = cmd.v * DT, cmd.w * DT
        x = motion_model(x, ds, dth)
    return x, cmd, t


def final_pose_ok(x, lateral_tol=0.08, heading_tol=0.25):
    """Ground-truth check of the final pose (evaluation only)."""
    rng = math.hypot(DOCK_X - x[0], DOCK_Y - x[1])
    return (D.target_distance - D.distance_tolerance <= rng <= D.target_distance + D.distance_tolerance + 0.04
            and lateral_error(x) < lateral_tol
            and abs(wrap_angle(x[2] - HEAD_ON)) < heading_tol)


def test_closed_loop_with_ekf_docks_from_the_handoff_pose():
    wx, wy = HANDOFF
    x, cmd, t = run_dock_ekf((wx, wy, math.pi / 2), seed=1)
    assert cmd.phase is DockPhase.DOCKED and t < 40.0
    assert final_pose_ok(x)


def test_closed_loop_with_ekf_survives_odometry_bias_and_a_marker_blackout():
    wx, wy = HANDOFF
    x, cmd, _ = run_dock_ekf((wx, wy, math.pi / 2), seed=2, bias=0.0004, blackout=(2.0, 4.0))
    assert cmd.phase is DockPhase.DOCKED
    assert final_pose_ok(x, lateral_tol=0.10, heading_tol=0.30)


def test_estimate_mode_docks_faster_than_raw_mode_through_a_marker_blackout():
    # The marker disappears for 2 s during the approach. Raw mode has to stop and search,
    # estimate mode keeps driving on the EKF prediction.
    wx, wy = HANDOFF
    times = {True: [], False: []}
    ok = {True: 0, False: 0}
    for seed in range(10):
        for use_estimate in (True, False):
            x, cmd, t = run_dock_ekf((wx, wy, math.pi / 2), seed=seed, bias=0.0004,
                                    blackout=(2.0, 4.0), use_estimate=use_estimate)
            times[use_estimate].append(t)
            ok[use_estimate] += int(cmd.phase is DockPhase.DOCKED and final_pose_ok(x))
    print("blackout 2 s, mean docking time [s] (EKF | raw):",
        round(float(np.mean(times[True])), 1), round(float(np.mean(times[False])), 1))
    assert ok[True] >= 9 and ok[True] >= ok[False]
    assert np.mean(times[True]) < np.mean(times[False])


def success_rate(pos_span, heading_span, n_trials, use_estimate):
    """Fraction of random handoff poses that end DOCKED AND pass the ground-truth pose check."""
    wx, wy = HANDOFF
    ok = 0
    for seed in range(n_trials):
        r = np.random.default_rng(100 + seed)
        x0 = (wx + r.uniform(-pos_span, pos_span), wy + r.uniform(-pos_span, pos_span),
            math.pi / 2 + r.uniform(-heading_span, heading_span))
        x, cmd, _ = run_dock_ekf(x0, seed=seed, bias=0.0004, use_estimate=use_estimate)
        ok += int(cmd.phase is DockPhase.DOCKED and final_pose_ok(x))
    return ok / n_trials


def test_repeated_trials_nominal_envelope():
    # Handoff within +-0.10 m and +-0.3 rad of the nominal pose: the system should always dock.
    rate = success_rate(0.10, 0.3, 20, use_estimate=True)
    print("nominal envelope, success rate (EKF estimate):", rate)
    assert rate >= 0.95


def test_repeated_trials_stress_envelope():
    # +-0.15 m / +-0.5 rad includes starts very close to the marker and far off its axis. There the
    # control law can stall: robot arrives at the target range with a view-angle error above
    # AXIS_TOLERANCE and ALIGN only rotates in place, so it times out. Known limit, so only a floor.
    rate = success_rate(0.15, 0.5, 20, use_estimate=True)
    print("stress envelope, success rate (EKF estimate):", rate)
    assert rate >= 0.85

# --------------------------------------------------------- real MuJoCo (optional)
@pytest.fixture
def real_sim(tmp_path):
    pytest.importorskip("mujoco")
    from simulation import Simulation
    sim = Simulation(IDEAL, scene_dir=tmp_path, seed=0)
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
    wx, wy = HANDOFF
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