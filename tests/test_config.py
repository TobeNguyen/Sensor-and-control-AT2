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

def test_waypoint_segments_are_clear_of_obstacles():
    pts = [np.array(CONFIG.env.start_pose[:2])] + [np.array(w) for w in CONFIG.env.waypoints]
    need = CONFIG.robot.body_radius
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        for ox, oy, r in CONFIG.env.obstacles:
            t = np.clip(np.dot(np.array([ox, oy]) - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            d = np.linalg.norm(a + t * ab - np.array([ox, oy]))
            assert d > r + need, f"segment {a}->{b} passes {d:.2f} m from obstacle ({ox}, {oy})"

# ------------------------------------------------- extra checkpoint tags
def _tag_config(n_tags):
    return replace(CONFIG, tags=replace(CONFIG.tags, n_tags=n_tags))


def test_zero_tags_leaves_only_the_dock():
    table = _tag_config(0).tag_table()
    assert list(table) == [CONFIG.camera.marker_id]
    assert table[CONFIG.camera.marker_id] == pytest.approx(CONFIG.env.dock_pose)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_tag_ids_are_consecutive_and_unique(n):
    table = _tag_config(n).tag_table()
    mid = CONFIG.camera.marker_id
    assert len(table) == n + 1
    assert sorted(table) == list(range(mid, mid + n + 1))


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_tags_sit_on_the_walls_and_face_into_the_room(n):
    cfg = _tag_config(n)
    hx, hy = cfg.env.room_half_size
    for tid, (x, y, yaw) in cfg.tag_table().items():
        if tid == cfg.camera.marker_id:
            continue
        assert abs(x) <= hx and abs(y) <= hy
        on_wall = (math.isclose(abs(x), hx - cfg.tags.inset, abs_tol=1e-9)
                or math.isclose(abs(y), hy - cfg.tags.inset, abs_tol=1e-9))
        assert on_wall
        # a point 0.5 m in front of the tag is inside the room and closer to the centre
        fx, fy = x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw)
        assert abs(fx) < hx and abs(fy) < hy
        assert math.hypot(fx, fy) < math.hypot(x, y)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_tags_do_not_overlap_the_dock(n):
    cfg = _tag_config(n)
    dx, dy, _ = cfg.env.dock_pose
    for tid, (x, y, _) in cfg.tag_table().items():
        if tid != cfg.camera.marker_id:
            assert math.hypot(x - dx, y - dy) > 0.3


def test_too_many_tags_raise_value_error():
    with pytest.raises(ValueError):
        _tag_config(50).tag_table()


def test_default_tag_table_is_a_copy():
    table = CONFIG.tag_table()
    table.clear()
    assert len(CONFIG.tag_table()) == CONFIG.tags.n_tags + 1


def test_tags_are_on_the_wall_opposite_to_the_dock():
    cfg = _tag_config(1)
    dx, dy, _ = cfg.env.dock_pose
    tx, ty, _ = cfg.tag_table()[CONFIG.camera.marker_id + 1]
    assert tx * dx < 0                       # opposite side of the room in x
    assert ty * dy < 0                       # staggered: other half in y
    assert tx == pytest.approx(-(dx - math.copysign(cfg.tags.inset, dx) * 0) , abs=0.2)


def test_first_tag_is_staggered_on_the_opposite_wall():
    cfg = _tag_config(1)
    dx, dy, _ = cfg.env.dock_pose
    tx, ty, _ = cfg.tag_table()[CONFIG.camera.marker_id + 1]
    assert tx * dx < 0 and ty * dy < 0


def test_four_tags_use_four_different_walls():
    cfg = _tag_config(4)
    yaws = {round(yaw, 3) for tid, (_, _, yaw) in cfg.tag_table().items()
            if tid != CONFIG.camera.marker_id}
    assert len(yaws) == 4


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_tags_keep_clear_of_the_dock(n):
    cfg = _tag_config(n)
    dx, dy, _ = cfg.env.dock_pose
    for tid, (x, y, _) in cfg.tag_table().items():
        if tid != cfg.camera.marker_id:
            assert math.hypot(x - dx, y - dy) >= 0.55