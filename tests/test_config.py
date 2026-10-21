# Including libraries
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from config import CONFIG, ROOT_DIR


def _is_symmetric_positive_definite(m: np.ndarray) -> bool:
    return np.allclose(m, m.T) and np.all(np.linalg.eigvalsh(m) > 0)


def test_paths_are_inside_project():
    assert CONFIG.sim.model_path.is_relative_to(ROOT_DIR)


def test_control_period_is_multiple_of_physics_step():
    ratio = CONFIG.sim.control_dt / CONFIG.sim.dt
    assert ratio >= 1
    assert math.isclose(ratio, round(ratio))


def test_robot_parameters_are_positive():
    r = CONFIG.robot
    for value in (r.wheel_radius, r.wheel_base, r.body_radius,
                r.max_linear_vel, r.max_angular_vel):
        assert value > 0


def test_lidar_range_is_valid():
    assert 0 < CONFIG.lidar.range_min < CONFIG.lidar.range_max
    assert CONFIG.lidar.num_beams > 0


def test_ekf_matrices_shape_and_definiteness():
    assert CONFIG.ekf.P0().shape == (3, 3)
    assert CONFIG.ekf.Q().shape == (3, 3)
    assert CONFIG.R_camera().shape == (2, 2)
    assert _is_symmetric_positive_definite(CONFIG.ekf.P0())
    assert _is_symmetric_positive_definite(CONFIG.ekf.Q())
    assert _is_symmetric_positive_definite(CONFIG.R_camera())


def test_lqr_weights_are_valid():
    Q, R = CONFIG.controller.Q(), CONFIG.controller.R()
    assert Q.shape == (3, 3) and R.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(Q) >= 0)   # Q must be PSD
    assert np.all(np.linalg.eigvalsh(R) > 0)    # R must be PD


def test_start_pose_is_inside_room_and_free():
    x, y, _ = CONFIG.env.start_pose
    hx, hy = CONFIG.env.room_half_size
    margin = CONFIG.robot.body_radius
    assert abs(x) <= hx - margin and abs(y) <= hy - margin
    for ox, oy, orad in CONFIG.env.obstacles:
        assert math.hypot(x - ox, y - oy) > orad + margin


def test_waypoints_are_inside_room_and_clear_of_obstacles():
    hx, hy = CONFIG.env.room_half_size
    margin = CONFIG.robot.body_radius
    for x, y in CONFIG.env.waypoints:
        assert abs(x) <= hx - margin and abs(y) <= hy - margin
        for ox, oy, orad in CONFIG.env.obstacles:
            assert math.hypot(x - ox, y - oy) > orad + margin


def test_dock_is_inside_room():
    x, y, _ = CONFIG.env.dock_pose
    hx, hy = CONFIG.env.room_half_size
    assert abs(x) <= hx and abs(y) <= hy


def test_last_waypoint_is_in_front_of_dock():
    wx, wy = CONFIG.env.waypoints[-1]
    dx, dy, _ = CONFIG.env.dock_pose
    dist = math.hypot(dx - wx, dy - wy)
    assert dist > CONFIG.docking.target_distance


def test_config_is_immutable():
    with pytest.raises(FrozenInstanceError):
        CONFIG.sim.dt = 0.1


def test_aruco_dictionary_exists_in_opencv():
    cv2 = pytest.importorskip("cv2")
    assert hasattr(cv2.aruco, CONFIG.camera.aruco_dict)


# ------------------------------------------------- fixed checkpoint tags
WALL_INSET = 0.05   # [m] tag panels sit this far from the wall
TAGS = CONFIG.tags.fixed_tags


def _seg_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def test_no_tags_leaves_only_the_dock():
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, fixed_tags=()))
    table = cfg.tag_table()
    assert list(table) == [CONFIG.camera.marker_id]
    assert table[CONFIG.camera.marker_id] == pytest.approx(CONFIG.env.dock_pose)


def test_tag_ids_are_consecutive_and_unique():
    table = CONFIG.tag_table()
    mid = CONFIG.camera.marker_id
    assert len(table) == len(TAGS) + 1
    assert sorted(table) == list(range(mid, mid + len(TAGS) + 1))
    assert mid + len(TAGS) <= 49            # DICT_4X4_50 has ids 0..49


def test_tags_sit_on_the_walls_and_face_into_the_room():
    hx, hy = CONFIG.env.room_half_size
    for x, y, yaw in TAGS:
        assert abs(x) <= hx and abs(y) <= hy
        assert (math.isclose(abs(x), hx - WALL_INSET, abs_tol=1e-9)
                or math.isclose(abs(y), hy - WALL_INSET, abs_tol=1e-9))
        fx, fy = x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw)
        assert abs(fx) < hx and abs(fy) < hy
        assert math.hypot(fx, fy) < math.hypot(x, y)


def test_tags_keep_clear_of_the_dock():
    dx, dy, _ = CONFIG.env.dock_pose
    for x, y, _ in TAGS:
        assert math.hypot(x - dx, y - dy) >= 0.55


def test_default_tag_table_is_a_copy():
    table = CONFIG.tag_table()
    table.clear()
    assert len(CONFIG.tag_table()) == len(TAGS) + 1


def test_tag_standoff_points_are_clear_of_obstacles():
    s = CONFIG.tags.standoff
    for x, y, yaw in TAGS:
        px, py = x + s * math.cos(yaw), y + s * math.sin(yaw)
        for ox, oy, r in CONFIG.env.obstacles:
            assert math.hypot(px - ox, py - oy) > r + CONFIG.robot.body_radius


def test_obstacle_hides_the_dock_from_the_start_pose():
    sx, sy, _ = CONFIG.env.start_pose
    dx, dy, _ = CONFIG.env.dock_pose
    assert any(_seg_dist(ox, oy, sx, sy, dx, dy) < r for ox, oy, r in CONFIG.env.obstacles)


def test_obstacles_do_not_block_the_dock_front():
    dx, dy, dyaw = CONFIG.env.dock_pose
    front = (dx + CONFIG.tags.approach_dist * math.cos(dyaw),
             dy + CONFIG.tags.approach_dist * math.sin(dyaw))
    for ox, oy, r in CONFIG.env.obstacles:
        assert _seg_dist(ox, oy, dx, dy, *front) > r + CONFIG.robot.body_radius