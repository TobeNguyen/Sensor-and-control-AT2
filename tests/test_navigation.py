# Including libraries
import math

import numpy as np
import pytest

import navigation
from config import CONFIG
from controller import DEFAULT_CRUISE_SPEED
from navigation import (
    NavPhase, Navigator, choose_heading, free_distance, heading_candidates,
)
from robot_model import motion_model
from sensing import make_lidar_scan

N_BEAMS = CONFIG.lidar.num_beams
ANGLES = np.linspace(0.0, 2.0 * math.pi, N_BEAMS, endpoint=False)
BODY = CONFIG.robot.body_radius
FREE_SCAN = make_lidar_scan(np.full(N_BEAMS, CONFIG.lidar.range_max), ANGLES)
BIG_ROOM = (10.0, 10.0)

# Noise-free 2D LiDAR
def fake_scan(pose, obstacles=CONFIG.env.obstacles, room_half=CONFIG.env.room_half_size):
    x, y, th = pose
    dirs = np.column_stack([np.cos(th + ANGLES), np.sin(th + ANGLES)])
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dirs[:, 0] > 0, (room_half[0] - x) / dirs[:, 0],
                    np.where(dirs[:, 0] < 0, (-room_half[0] - x) / dirs[:, 0], np.inf))
        ty = np.where(dirs[:, 1] > 0, (room_half[1] - y) / dirs[:, 1],
                    np.where(dirs[:, 1] < 0, (-room_half[1] - y) / dirs[:, 1], np.inf))
    t = np.minimum(tx, ty)
    for ox, oy, r in obstacles:
        oc = np.array([ox - x, oy - y])
        t_ca = dirs @ oc
        d2 = oc @ oc - t_ca ** 2
        hit = (t_ca > 0) & (d2 < r * r)
        t_hit = t_ca - np.sqrt(np.maximum(r * r - d2, 0.0))
        t = np.where(hit & (t_hit > 0), np.minimum(t, t_hit), t)
    return make_lidar_scan(np.minimum(t, CONFIG.lidar.range_max), ANGLES)


# ---------------------------------------------------------------- free_distance
def test_free_distance_without_points_returns_look():
    assert free_distance(np.empty((0, 2)), 0.0, 0.8, 0.3) == 0.8


def test_free_distance_obstacle_ahead():
    assert free_distance(np.array([[0.5, 0.0]]), 0.0, 0.8, 0.3) == pytest.approx(0.5)


def test_free_distance_ignores_lateral_far_and_behind_points():
    pts = np.array([[0.5, 0.6], [-0.5, 0.0], [1.5, 0.0]])
    assert free_distance(pts, 0.0, 0.8, 0.3) == 0.8


def test_free_distance_depends_on_heading():
    pts = np.array([[0.0, 0.5]])           # obstacle on the left
    assert free_distance(pts, 0.0, 0.8, 0.3) == 0.8
    assert free_distance(pts, math.pi / 2, 0.8, 0.3) == pytest.approx(0.5)


# -------------------------------------------------------------- choose_heading
def test_heading_candidates_include_goal():
    assert np.any(np.isclose(heading_candidates(0.123), 0.123))


def test_choose_heading_returns_goal_when_free():
    h, d, found = choose_heading(np.empty((0, 2)), 0.37, 0.3, 0.8)
    assert found and h == pytest.approx(0.37) and d == pytest.approx(0.8)


def test_choose_heading_detours_around_blocked_goal():
    pts = np.array([[0.5, 0.0]])
    h, d, found = choose_heading(pts, 0.0, 0.3, 0.8)
    assert found and abs(h) > 0.3 and d >= 0.8 - 1e-9


def test_choose_heading_hysteresis_keeps_previous_side():
    pts = np.array([[0.5, 0.0]])           # symmetric obstacle, both sides equally good
    left, _, _ = choose_heading(pts, 0.0, 0.3, 0.8, prev_rel=+0.8)
    right, _, _ = choose_heading(pts, 0.0, 0.3, 0.8, prev_rel=-0.8)
    assert left > 0 > right


def test_choose_heading_all_blocked():
    pts = 0.4 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])
    _, d, found = choose_heading(pts, 0.0, 0.3, 0.8)
    assert not found and d < 0.8


# ------------------------------------------------------------------- Navigator
def test_defaults_come_from_config():
    assert len(Navigator().waypoints) == len(CONFIG.env.waypoints)


def test_does_not_import_the_simulator():
    assert not hasattr(navigation, "Simulation")


def test_track_phase_on_free_map():
    nav = Navigator(waypoints=[(3.0, 0.0)])
    cmd = nav.update([0.0, 0.0, 0.0], FREE_SCAN)
    assert cmd.phase is NavPhase.TRACK and not cmd.finished
    assert cmd.v == pytest.approx(DEFAULT_CRUISE_SPEED)
    assert cmd.w == pytest.approx(0.0, abs=1e-6)
    assert cmd.goal_distance == pytest.approx(3.0)


def test_turns_towards_a_waypoint_on_the_left():
    nav = Navigator(waypoints=[(0.0, 3.0)])
    assert nav.update([0.0, 0.0, 0.0], FREE_SCAN).w > 0


def test_advances_to_next_waypoint_when_reached():
    nav = Navigator(waypoints=[(1.0, 0.0), (2.0, 0.0)])
    cmd = nav.update([0.95, 0.0, 0.0], FREE_SCAN)
    assert cmd.waypoint_index == 1 and not cmd.finished
    np.testing.assert_allclose(nav.target, [2.0, 0.0])


def test_finishes_after_last_waypoint_and_stays_stopped():
    nav = Navigator(waypoints=[(1.0, 0.0)])
    for _ in range(3):
        cmd = nav.update([1.0, 0.0, 0.0], FREE_SCAN)
        assert cmd.finished and cmd.phase is NavPhase.DONE
        assert (cmd.v, cmd.w) == (0.0, 0.0)
    assert nav.target is None


def test_invalid_estimate_stops_the_robot():
    nav = Navigator(waypoints=[(3.0, 0.0)])
    cmd = nav.update([float("nan"), 0.0, 0.0], FREE_SCAN)
    assert (cmd.v, cmd.w) == (0.0, 0.0) and not cmd.finished


def test_avoid_turns_away_from_an_obstacle_ahead():
    obstacles = ((0.7, 0.1, 0.2),)         # slightly left of the straight path
    nav = Navigator(waypoints=[(3.0, 0.0)])
    cmd = nav.update([0.0, 0.0, 0.0], fake_scan((0.0, 0.0, 0.0), obstacles, BIG_ROOM))
    assert cmd.phase is NavPhase.AVOID
    assert cmd.w < 0                        # turns right, away from the obstacle


def test_avoid_does_not_drive_forward_when_surrounded():
    ring = make_lidar_scan(np.full(N_BEAMS, 0.4), ANGLES)
    nav = Navigator(waypoints=[(3.0, 0.0)])
    cmd = nav.update([0.0, 0.0, 0.0], ring)
    assert cmd.phase is NavPhase.AVOID
    assert cmd.v == 0.0


def test_returns_to_track_when_the_path_clears():
    nav = Navigator(waypoints=[(3.0, 0.0)])
    nav.update([0.0, 0.0, 0.0], fake_scan((0.0, 0.0, 0.0), ((0.7, 0.1, 0.2),), BIG_ROOM))
    assert nav.phase is NavPhase.AVOID
    for _ in range(30):
        cmd = nav.update([0.0, 0.0, 0.0], FREE_SCAN)
    assert cmd.phase is NavPhase.TRACK and cmd.v > 0


def test_reset_restores_initial_state():
    nav = Navigator(waypoints=[(1.0, 0.0)])
    nav.update([1.0, 0.0, 0.0], FREE_SCAN)
    assert nav.finished
    nav.reset()
    assert not nav.finished and nav.index == 0 and nav.phase is NavPhase.TRACK


# ------------------------------------------------------------------ closed loop
def test_closed_loop_reaches_all_waypoints_without_collision():
    dt = CONFIG.sim.control_dt
    hx, hy = CONFIG.env.room_half_size
    nav = Navigator()
    x = np.array(CONFIG.env.start_pose, dtype=float)

    min_obstacle_clearance = np.inf
    min_wall_clearance = np.inf
    finished = False
    for _ in range(int(CONFIG.sim.max_time / dt)):
        cmd = nav.update(x, fake_scan(x))
        if cmd.finished:
            finished = True
            break
        x = motion_model(x, cmd.v * dt, cmd.w * dt)
        for ox, oy, r in CONFIG.env.obstacles:
            min_obstacle_clearance = min(min_obstacle_clearance,
                                        math.hypot(x[0] - ox, x[1] - oy) - r - BODY)
        min_wall_clearance = min(min_wall_clearance, hx - abs(x[0]) - BODY, hy - abs(x[1]) - BODY)

    assert finished, "did not reach the last waypoint before the time limit"
    assert min_obstacle_clearance > 0.0
    assert min_wall_clearance > 0.0