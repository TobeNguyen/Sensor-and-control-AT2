import math
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from config import CONFIG, RealismConfig
from simulation import (
    MESH_FILES, MESH_SUBDIR, MARKER_FILE, Simulation, build_scene_xml, generate_marker_image,
    marker_filename, meshes_available, point_segment_distance, random_obstacles,
    write_scene_files, wrap_angle,
)
IDEAL = replace(CONFIG, realism=RealismConfig(enabled=False))

@pytest.fixture
def sim(tmp_path):
    s = Simulation(IDEAL, scene_dir=tmp_path, seed=0)
    yield s
    s.close()


def run(sim, v, w, seconds):
    for _ in range(int(round(seconds / CONFIG.sim.control_dt))):
        sim.step(v, w)


def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi / 2), -math.pi / 2)
    assert math.isclose(wrap_angle(-math.pi / 2), -math.pi / 2)
    assert abs(wrap_angle(10.0)) <= math.pi


def test_scene_xml_contains_required_elements():
    xml = build_scene_xml(CONFIG)
    assert 'name="base"' in xml
    assert 'name="front_cam"' in xml
    assert 'name="dock_marker"' in xml
    for i in range(len(CONFIG.env.obstacles)):
        assert f'name="obstacle_{i}"' in xml


def test_marker_image_is_square_uint8_with_white_margin():
    img = generate_marker_image(CONFIG)
    assert img.dtype == np.uint8
    assert img.shape[0] == img.shape[1]
    assert img[0, 0] == 255


def test_reset_places_robot_at_start_pose(sim):
    np.testing.assert_allclose(sim.ground_truth(), CONFIG.env.start_pose, atol=1e-9)
    assert sim.time == 0.0
    np.testing.assert_array_equal(sim.read_encoders(), [0.0, 0.0])
    assert not sim.collided


def test_reset_with_custom_pose(sim):
    sim.reset((0.5, -1.0, 1.0))
    np.testing.assert_allclose(sim.ground_truth(), (0.5, -1.0, 1.0), atol=1e-9)


def test_drive_straight(sim):
    x0, y0, th0 = sim.ground_truth()
    run(sim, 0.2, 0.0, 1.0)
    x, y, th = sim.ground_truth()
    assert x - x0 == pytest.approx(0.2, abs=1e-3)
    assert y == pytest.approx(y0, abs=1e-3)
    assert th == pytest.approx(th0, abs=1e-3)
    assert sim.time == pytest.approx(1.0, abs=1e-6)


def test_rotate_in_place(sim):
    x0, y0, th0 = sim.ground_truth()
    run(sim, 0.0, 1.0, 1.0)
    x, y, th = sim.ground_truth()
    assert th - th0 == pytest.approx(1.0, abs=1e-2)
    assert (x, y) == pytest.approx((x0, y0), abs=1e-3)


def test_commands_are_clipped_to_limits(sim):
    x0 = sim.ground_truth()[0]
    run(sim, 10.0, 0.0, 1.0)
    travelled = sim.ground_truth()[0] - x0
    assert travelled == pytest.approx(CONFIG.robot.max_linear_vel, abs=1e-2)


def test_encoders_match_straight_motion(sim):
    run(sim, 0.2, 0.0, 1.0)
    expected = 0.2 / CONFIG.robot.wheel_radius
    left, right = sim.read_encoders()
    assert left == pytest.approx(expected, abs=0.05)
    assert right == pytest.approx(expected, abs=0.05)


def test_encoders_match_rotation(sim):
    run(sim, 0.0, 1.0, 1.0)
    expected = CONFIG.robot.wheel_base / CONFIG.robot.wheel_radius  # (right - left) for w=1, t=1
    left, right = sim.read_encoders()
    assert right - left == pytest.approx(expected, abs=0.05)


def test_lidar_shape_and_limits(sim):
    ranges = sim.read_lidar()
    assert ranges.shape == (CONFIG.lidar.num_beams,)
    assert np.all(ranges >= CONFIG.lidar.range_min)
    assert np.all(ranges <= CONFIG.lidar.range_max)


def test_lidar_measures_walls(sim):
    # Start pose is (-1.5, -1.5, 0): west wall and south wall are 0.5 m away.
    ranges = sim.read_lidar()
    n = CONFIG.lidar.num_beams
    assert ranges[n // 2] == pytest.approx(0.5, abs=0.05)       # angle pi  -> west
    assert ranges[3 * n // 4] == pytest.approx(0.5, abs=0.05)   # angle 3pi/2 -> south


def test_lidar_measures_obstacle(sim):
    ox, oy, radius = CONFIG.env.obstacles[0]
    sim.reset((ox, oy - 1.0, 0.0))
    ranges = sim.read_lidar()
    n = CONFIG.lidar.num_beams
    assert ranges[n // 4] == pytest.approx(1.0 - radius, abs=0.05)  # angle pi/2 -> +y


def test_lidar_is_reproducible_with_same_seed(tmp_path):
    a = Simulation(scene_dir=tmp_path / "a", seed=7)
    b = Simulation(scene_dir=tmp_path / "b", seed=7)
    np.testing.assert_array_equal(a.read_lidar(), b.read_lidar())
    a.close()
    b.close()


def test_collision_with_obstacle_is_detected(sim):
    ox, oy, _ = CONFIG.env.obstacles[0]
    sim.reset((ox, oy - 1.0, math.pi / 2))  # facing the obstacle
    assert not sim.collided
    hit = False
    for _ in range(int(6.0 / CONFIG.sim.control_dt)):
        sim.step(0.2, 0.0)
        if sim.collided:
            hit = True
            break
    assert hit


def test_time_exceeded_flag(sim):
    assert not sim.time_exceeded


def _render_or_skip(sim):
    try:
        return sim.read_camera()
    except Exception as exc:  # no OpenGL context available
        pytest.skip(f"Offscreen rendering not available: {exc}")


def test_camera_image_shape(sim):
    img = _render_or_skip(sim)
    assert img.shape == (CONFIG.camera.height, CONFIG.camera.width, 3)
    assert img.dtype == np.uint8


def test_camera_detects_aruco_marker(sim):
    cv2 = pytest.importorskip("cv2")
    dx, dy, _ = CONFIG.env.dock_pose
    sim.reset((dx - 0.7, dy, 0.0))  # 0.7 m in front of the marker, looking at it
    img = _render_or_skip(sim)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG.camera.aruco_dict))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert ids is not None and CONFIG.camera.marker_id in ids.flatten()

# ------------------------------------------------- TurtleBot3 visual meshes
def write_cube_stl(path, size_mm=100.0):
    """Tiny valid binary STL (a cube) used as a stand-in for the real meshes."""
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
            (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            coords = [float(x) * size_mm for idx in (a, b, c) for x in v[idx]]
            f.write(struct.pack("<12fH", 0.0, 0.0, 0.0, *coords, 0))


def write_fake_meshes(directory):
    mesh_dir = directory / MESH_SUBDIR
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for name in MESH_FILES.values():
        write_cube_stl(mesh_dir / name)


def test_no_meshes_means_plain_cylinder(tmp_path):
    with Simulation(scene_dir=tmp_path) as sim:
        assert not sim.uses_meshes
    assert 'type="mesh"' not in (tmp_path / CONFIG.sim.model_path.name).read_text()


def test_meshes_are_visual_only(tmp_path):
    import mujoco
    write_fake_meshes(tmp_path)
    assert meshes_available(tmp_path)
    with Simulation(scene_dir=tmp_path) as sim:
        assert sim.uses_meshes and sim.model.nmesh == len(MESH_FILES)
        mesh_geoms = np.flatnonzero(sim.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        assert len(mesh_geoms) == len(MESH_FILES)
        assert np.all(sim.model.geom_contype[mesh_geoms] == 0)
        assert np.all(sim.model.geom_conaffinity[mesh_geoms] == 0)


def test_physics_is_unchanged_with_meshes(tmp_path):
    write_fake_meshes(tmp_path)
    with Simulation(IDEAL, scene_dir=tmp_path, seed=0) as sim:
        x0 = sim.ground_truth()[0]
        run(sim, 0.2, 0.0, 1.0)
        assert sim.ground_truth()[0] - x0 == pytest.approx(0.2, abs=1e-3)
        sim.reset()
        ranges = sim.read_lidar()                 # the robot's own meshes must not be seen
        n = CONFIG.lidar.num_beams
        assert ranges[n // 2] == pytest.approx(0.5, abs=0.05)


def test_marker_still_detected_with_meshes(tmp_path):
    cv2 = pytest.importorskip("cv2")
    write_fake_meshes(tmp_path)
    with Simulation(scene_dir=tmp_path) as sim:
        dx, dy, _ = CONFIG.env.dock_pose
        sim.reset((dx - 0.7, dy, 0.0))
        try:
            img = sim.read_camera()
        except Exception as exc:
            pytest.skip(f"Offscreen rendering not available: {exc}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG.camera.aruco_dict))
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        _, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        assert ids is not None and CONFIG.camera.marker_id in ids.flatten()


def test_broken_meshes_fall_back_to_the_cylinder(tmp_path):
    mesh_dir = tmp_path / MESH_SUBDIR
    mesh_dir.mkdir()
    for name in MESH_FILES.values():
        (mesh_dir / name).write_bytes(b"not an stl file")
    with Simulation(scene_dir=tmp_path) as sim:
        assert not sim.uses_meshes
        assert sim.model.nmesh == 0

# ------------------------------------------------- Random obstacles
SEEDS = list(range(20))


def _layout(seed):
    return random_obstacles(CONFIG, np.random.default_rng(seed))


def test_default_keeps_fixed_obstacle_layout(sim):
    # randomize_obstacles=False must reproduce the layout from the config
    assert sim.obstacles == list(CONFIG.env.obstacles)


def test_random_layout_is_reproducible_with_same_seed():
    assert _layout(3) == _layout(3)


def test_random_layout_differs_between_seeds():
    layouts = {tuple(_layout(s)) for s in range(5)}
    assert len(layouts) > 1


@pytest.mark.parametrize("seed", SEEDS)
def test_random_layout_keeps_count_and_radii(seed):
    obs = _layout(seed)
    assert len(obs) == len(CONFIG.env.obstacles)
    assert [r for _, _, r in obs] == [r for _, _, r in CONFIG.env.obstacles]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_obstacles_stay_inside_room_with_robot_gap(seed):
    hx, hy = CONFIG.env.room_half_size
    robot_d = 2.0 * CONFIG.robot.body_radius
    for x, y, r in _layout(seed):
        assert abs(x) + r <= hx - robot_d + 1e-9
        assert abs(y) + r <= hy - robot_d + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_random_obstacles_keep_start_and_dock_free(seed):
    sx, sy, _ = CONFIG.env.start_pose
    dx, dy, _ = CONFIG.env.dock_pose
    robot_d = 2.0 * CONFIG.robot.body_radius
    for x, y, r in _layout(seed):
        assert math.hypot(x - sx, y - sy) >= r + robot_d
        assert math.hypot(x - dx, y - dy) >= r + robot_d


@pytest.mark.parametrize("seed", SEEDS)
def test_random_obstacles_leave_gap_between_each_other(seed):
    obs = _layout(seed)
    robot_d = 2.0 * CONFIG.robot.body_radius
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            (xi, yi, ri), (xj, yj, rj) = obs[i], obs[j]
            assert math.hypot(xi - xj, yi - yj) >= ri + rj + robot_d


def test_random_obstacles_raise_when_impossible():
    # Stub config with far too many obstacles for the room
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            room_half_size=(1.0, 1.0),
            start_pose=(0.0, 0.0, 0.0),
            dock_pose=(0.9, 0.0, 0.0),
            obstacles=[(0.0, 0.0, 0.2)] * 50,
        ),
        robot=SimpleNamespace(body_radius=CONFIG.robot.body_radius),
    )
    with pytest.raises(RuntimeError):
        random_obstacles(cfg, np.random.default_rng(0), max_tries=50)


def test_build_scene_xml_uses_custom_obstacles():
    custom = [(0.1, 0.2, 0.15), (-0.3, 0.4, 0.25)]
    xml = build_scene_xml(CONFIG, obstacles=custom)
    assert 'name="obstacle_0"' in xml and 'name="obstacle_1"' in xml
    assert 'name="obstacle_2"' not in xml
    assert 'pos="0.1 0.2 ' in xml and 'pos="-0.3 0.4 ' in xml


def test_randomized_simulation_matches_obstacle_list(tmp_path):
    import mujoco
    with Simulation(scene_dir=tmp_path, seed=5, randomize_obstacles=True) as sim:
        assert len(sim.obstacles) == len(CONFIG.env.obstacles)
        assert sim.obstacles != list(CONFIG.env.obstacles)
        for i, (ox, oy, r) in enumerate(sim.obstacles):
            gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
            assert gid >= 0
            np.testing.assert_allclose(sim.model.geom_pos[gid][:2], (ox, oy), atol=1e-9)
            assert sim.model.geom_size[gid][0] == pytest.approx(r)
        # the robot must not start inside an obstacle
        assert not sim.collided

@pytest.mark.parametrize("seed", SEEDS)
def test_random_obstacles_keep_the_waypoint_path_clear(seed):
    pts = ([tuple(CONFIG.env.start_pose[:2])]
        + [tuple(w[:2]) for w in CONFIG.env.waypoints]
        + [tuple(CONFIG.env.dock_pose[:2])])
    for x, y, r in _layout(seed):
        for a, b in zip(pts[:-1], pts[1:]):
            assert point_segment_distance(x, y, *a, *b) >= r + CONFIG.robot.body_radius

# ------------------------------------------------- extra checkpoint tags
DOCK_ID = CONFIG.camera.marker_id
EXTRA_TAG_IDS = sorted(i for i in CONFIG.tag_table() if i != DOCK_ID)


def test_scene_xml_contains_every_extra_tag():
    xml = build_scene_xml(CONFIG)
    assert 'name="dock_marker"' in xml
    for tid in EXTRA_TAG_IDS:
        assert f'name="tag_marker_{tid}"' in xml
        assert marker_filename(tid, CONFIG) in xml


def test_scene_xml_has_no_tags_when_n_tags_is_zero():
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, n_tags=0))
    xml = build_scene_xml(cfg)
    assert "tag_marker_" not in xml and 'name="dock_marker"' in xml


def test_write_scene_files_creates_every_marker_texture(tmp_path):
    write_scene_files(CONFIG, tmp_path)
    assert (tmp_path / MARKER_FILE).is_file()
    for tid in EXTRA_TAG_IDS:
        assert (tmp_path / marker_filename(tid, CONFIG)).is_file()


def test_marker_filename_keeps_the_dock_file():
    assert marker_filename(DOCK_ID, CONFIG) == MARKER_FILE
    assert marker_filename(DOCK_ID + 1, CONFIG) != MARKER_FILE


@pytest.mark.parametrize("tag_id", EXTRA_TAG_IDS)
def test_tag_panels_exist_in_the_model_at_the_configured_pose(sim, tag_id):
    import mujoco
    gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, f"tag_marker_{tag_id}")
    assert gid >= 0
    tx, ty, _ = CONFIG.tag_table()[tag_id]
    np.testing.assert_allclose(sim.model.geom_pos[gid][:2], (tx, ty), atol=1e-9)


@pytest.mark.parametrize("tag_id", EXTRA_TAG_IDS)
def test_camera_detects_each_extra_tag_from_its_standoff_point(sim, tag_id):
    cv2 = pytest.importorskip("cv2")
    tx, ty, tyaw = CONFIG.tag_table()[tag_id]
    s = CONFIG.tags.standoff
    sim.reset((tx + s * math.cos(tyaw), ty + s * math.sin(tyaw), wrap_angle(tyaw + math.pi)))
    img = _render_or_skip(sim)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG.camera.aruco_dict))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert ids is not None and tag_id in ids.flatten()


@pytest.mark.parametrize("seed", SEEDS)
def test_random_obstacles_keep_the_tag_standoff_points_free(seed):
    s = CONFIG.tags.standoff
    points = [(tx + s * math.cos(tyaw), ty + s * math.sin(tyaw))
            for tid, (tx, ty, tyaw) in CONFIG.tag_table().items() if tid != DOCK_ID]
    for x, y, r in _layout(seed):
        for px, py in points:
            assert math.hypot(x - px, y - py) >= r + CONFIG.robot.body_radius