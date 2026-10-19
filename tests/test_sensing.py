# Including libraries
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from dataclasses import replace
from config import CONFIG, RealismConfig
IDEAL = replace(CONFIG, realism=RealismConfig(enabled=False))
from robot_model import measurement_model
from sensing import (
    MAX_REPROJ_PX, MarkerMeasurement, Sensing, camera_forward_offset, camera_matrix,
    detect_marker, detect_markers, lidar_points, make_lidar_scan, sector_min_range,
)

N_BEAMS = CONFIG.lidar.num_beams
ANGLES = np.linspace(0.0, 2.0 * math.pi, N_BEAMS, endpoint=False)
BLACK_PX = 120   # side of the black marker in the synthetic image [px]


def free_ranges():
    return np.full(N_BEAMS, CONFIG.lidar.range_max)


def beam_index(angle):
    return int(round(angle / (2.0 * math.pi) * N_BEAMS)) % N_BEAMS


def synthetic_marker_image(shift_px=0, marker_id=None, black_px=BLACK_PX):
    """White image with the marker centred (plus a horizontal shift)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG.camera.aruco_dict))
    mid = CONFIG.camera.marker_id if marker_id is None else marker_id
    marker = cv2.aruco.generateImageMarker(dictionary, mid, black_px)
    pad = 20
    marker = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    h, w, size = CONFIG.camera.height, CONFIG.camera.width, marker.shape[0]
    img = np.full((h, w), 255, np.uint8)
    x0, y0 = w // 2 - size // 2 + shift_px, h // 2 - size // 2
    img[y0:y0 + size, x0:x0 + size] = marker
    return np.stack([img] * 3, axis=2)


class FakeSim:
    """Minimal stand-in for Simulation, so Sensing can be tested without MuJoCo."""

    def __init__(self):
        self.time = 0.0
        self.lidar_angles = ANGLES
        self.enc = np.zeros(2)
        self.image = np.full((CONFIG.camera.height, CONFIG.camera.width, 3), 255, np.uint8)
        self.camera_calls = 0

    def read_encoders(self):
        return self.enc.copy()

    def read_lidar(self):
        return free_ranges()

    def read_camera(self):
        self.camera_calls += 1
        return self.image


# --------------------------------------------------------------------- camera
def test_camera_matrix_values():
    K = camera_matrix()
    f = (CONFIG.camera.height / 2) / math.tan(math.radians(CONFIG.camera.fovy_deg) / 2)
    assert K[0, 0] == pytest.approx(f) and K[1, 1] == pytest.approx(f)
    assert K[0, 2] == pytest.approx(CONFIG.camera.width / 2)
    assert K[1, 2] == pytest.approx(CONFIG.camera.height / 2)


def test_detect_marker_returns_none_on_blank_image():
    blank = np.full((CONFIG.camera.height, CONFIG.camera.width, 3), 255, np.uint8)
    assert detect_marker(blank) is None


def test_detect_marker_ignores_wrong_id():
    assert detect_marker(synthetic_marker_image(marker_id=1)) is None


def test_detect_marker_range_and_bearing_centred():
    K = camera_matrix()
    m = detect_marker(synthetic_marker_image(), K, time=1.5)
    assert m is not None and m.time == 1.5 and m.marker_id == CONFIG.camera.marker_id
    z_cam = K[0, 0] * CONFIG.camera.marker_size / BLACK_PX
    assert m.range == pytest.approx(z_cam + camera_forward_offset(), abs=0.02)
    assert m.bearing == pytest.approx(0.0, abs=0.02)
    assert abs(m.view_angle) < 0.2
    np.testing.assert_allclose(m.z, [m.range, m.bearing])


def test_marker_on_the_left_of_image_has_positive_bearing():
    K = camera_matrix()
    m = detect_marker(synthetic_marker_image(shift_px=-80), K)
    z_cam = K[0, 0] * CONFIG.camera.marker_size / BLACK_PX
    x_cam = -80 * z_cam / K[0, 0]
    expected = math.atan2(-x_cam, z_cam + camera_forward_offset())
    assert m.bearing > 0
    assert m.bearing == pytest.approx(expected, abs=0.02)


def test_marker_on_the_right_of_image_has_negative_bearing():
    m = detect_marker(synthetic_marker_image(shift_px=80))
    assert m.bearing < 0

# ------------------------------------------------- noise model and quality gates
def test_marker_measurement_defaults():
    m = MarkerMeasurement(1.0, 0.0, 0.0, 0, 0.0)
    assert m.cov is None and math.isnan(m.reproj_error)


def test_detect_marker_returns_valid_covariance():
    m = detect_marker(synthetic_marker_image(), camera_matrix())
    assert m is not None and m.cov is not None and m.cov.shape == (2, 2)
    np.testing.assert_allclose(m.cov, m.cov.T)
    assert np.all(np.linalg.eigvalsh(m.cov) > 0)
    assert 0.0 <= m.reproj_error <= MAX_REPROJ_PX


def test_covariance_grows_with_distance():
    K = camera_matrix()
    near = detect_marker(synthetic_marker_image(black_px=120), K)
    far = detect_marker(synthetic_marker_image(black_px=60), K)
    assert near is not None and far is not None
    assert far.range > near.range
    assert far.cov[0, 0] > near.cov[0, 0]          # range variance increases with distance


def test_covariance_scales_with_sigma_px():
    K = camera_matrix()
    img = synthetic_marker_image()
    a = detect_marker(img, K, sigma_px=0.5)
    b = detect_marker(img, K, sigma_px=1.0)
    assert b.cov[1, 1] == pytest.approx(4.0 * a.cov[1, 1])   # variance ~ sigma^2


def test_reprojection_gate_rejects_marker():
    assert detect_marker(synthetic_marker_image(), camera_matrix(), max_reproj_px=-1.0) is None


def test_view_angle_gate_rejects_marker():
    assert detect_marker(synthetic_marker_image(), camera_matrix(), max_view_angle=-1.0) is None


# ---------------------------------------------------------------------- LiDAR
def test_make_lidar_scan_validity_mask():
    ranges = free_ranges()
    ranges[0] = 1.0                           # valid hit
    ranges[1] = CONFIG.lidar.range_min / 2    # too close
    ranges[2] = np.nan                        # corrupt
    scan = make_lidar_scan(ranges, ANGLES, time=2.0)
    assert scan.valid[0] and not scan.valid[1] and not scan.valid[2]
    assert not scan.valid[3]                  # no return (range_max)
    assert scan.time == 2.0 and scan.range_max == CONFIG.lidar.range_max


def test_make_lidar_scan_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        make_lidar_scan(free_ranges(), ANGLES[:-1])


def test_lidar_points_in_robot_frame():
    ranges = free_ranges()
    ranges[beam_index(math.pi / 2)] = 1.0     # obstacle on the left
    pts = lidar_points(make_lidar_scan(ranges, ANGLES))
    assert pts.shape == (1, 2)
    np.testing.assert_allclose(pts[0], [0.0, 1.0], atol=1e-9)


def test_sector_min_range_front_obstacle():
    ranges = free_ranges()
    ranges[0] = 0.5
    scan = make_lidar_scan(ranges, ANGLES)
    assert sector_min_range(scan, 0.0, 0.3) == pytest.approx(0.5)
    assert sector_min_range(scan, math.pi, 0.3) == CONFIG.lidar.range_max


def test_sector_min_range_wraps_around_zero():
    ranges = free_ranges()
    ranges[beam_index(math.radians(350))] = 0.8   # slightly to the right of front
    scan = make_lidar_scan(ranges, ANGLES)
    assert sector_min_range(scan, 0.0, math.radians(15)) == pytest.approx(0.8)


def test_sector_min_range_ignores_invalid_beams():
    ranges = free_ranges()
    ranges[0] = CONFIG.lidar.range_min / 2        # invalid
    scan = make_lidar_scan(ranges, ANGLES)
    assert sector_min_range(scan, 0.0, 0.3) == CONFIG.lidar.range_max


# ------------------------------------------------------------- Sensing (fake)
def test_sensing_odometry_is_incremental():
    sim = FakeSim()
    sensing = Sensing(sim, camera_period=None)
    sim.enc = np.array([3.0, 3.0])
    f1 = sensing.read()
    assert f1.odometry.ds == pytest.approx(CONFIG.robot.wheel_radius * 3.0)
    assert f1.odometry.dtheta == pytest.approx(0.0)
    f2 = sensing.read()                            # encoders unchanged
    assert f2.odometry.ds == pytest.approx(0.0)


def test_sensing_reset_resynchronises_odometry():
    sim = FakeSim()
    sensing = Sensing(sim, camera_period=None)
    sim.enc = np.array([5.0, 5.0])
    sensing.reset()
    assert sensing.read().odometry.ds == pytest.approx(0.0)


def test_sensing_camera_runs_at_its_own_rate():
    sim = FakeSim()
    sensing = Sensing(sim, camera_period=0.1)
    sensing.read()                                 # t = 0.00 -> camera frame
    sim.time = 0.02
    sensing.read()                                 # too early
    sim.time = 0.10
    sensing.read()                                 # due again
    assert sim.camera_calls == 2


def test_sensing_camera_can_be_disabled():
    sim = FakeSim()
    frame = Sensing(sim, camera_period=None).read()
    assert sim.camera_calls == 0 and frame.marker is None


def test_sensing_marker_is_none_when_not_visible():
    sim = FakeSim()
    assert Sensing(sim, camera_period=0.0).read().marker is None


def test_sensing_returns_marker_when_visible():
    sim = FakeSim()
    sim.image = synthetic_marker_image()
    frame = Sensing(sim, camera_period=0.0).read()
    assert frame.marker is not None and frame.marker.range > 0

def test_sensing_uses_the_calibration_module():
    import calibration
    import sensing
    assert sensing.camera_matrix is calibration.camera_matrix
    assert sensing.solve_marker_pose is calibration.solve_marker_pose

def test_sensing_exposes_new_calibration_functions():
    import calibration
    import sensing
    assert sensing.marker_measurement_covariance is calibration.marker_measurement_covariance
    assert sensing.MAX_REPROJ_PX == calibration.MAX_REPROJ_PX


def test_sensing_counts_camera_frames_and_detections():
    sim = FakeSim()
    sim.image = synthetic_marker_image()
    sensing = Sensing(sim, camera_period=0.0)
    sensing.read(); sensing.read()
    assert (sensing.n_camera, sensing.n_detect) == (2, 2)
    sim.image = np.full((CONFIG.camera.height, CONFIG.camera.width, 3), 255, np.uint8)
    sensing.read()
    assert (sensing.n_camera, sensing.n_detect) == (3, 2)
    sensing.reset()
    assert (sensing.n_camera, sensing.n_detect) == (0, 0)


def test_sensing_sigma_px_changes_covariance():
    sim = FakeSim()
    sim.image = synthetic_marker_image()
    lo = Sensing(sim, camera_period=0.0, marker_sigma_px=0.5).read().marker
    hi = Sensing(sim, camera_period=0.0, marker_sigma_px=2.0).read().marker
    assert hi.cov[1, 1] > lo.cov[1, 1]


def test_sensing_quality_gates_can_reject_everything():
    sim = FakeSim()
    sim.image = synthetic_marker_image()
    assert Sensing(sim, camera_period=0.0, max_view_angle=-1.0).read().marker is None
    assert Sensing(sim, camera_period=0.0, max_reproj_px=-1.0).read().marker is None


# ------------------------------------------------------ Sensing (real MuJoCo)
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


def test_real_odometry_matches_driven_distance(real_sim):
    sensing = Sensing(real_sim, camera_period=None)
    total_ds = total_dth = 0.0
    for _ in range(50):                            # 1 s at 50 Hz
        real_sim.step(0.2, 0.0)
        f = sensing.read()
        total_ds += f.odometry.ds
        total_dth += f.odometry.dtheta
    assert total_ds == pytest.approx(0.2, abs=0.01)
    assert total_dth == pytest.approx(0.0, abs=0.01)


def test_real_lidar_has_valid_hits(real_sim):
    f = Sensing(real_sim, camera_period=None).read()
    assert f.lidar.valid.any()


def test_real_marker_head_on(real_sim):
    dx, dy, _ = CONFIG.env.dock_pose
    real_sim.reset((dx - 0.7, dy, 0.0))
    m = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    assert m is not None
    assert m.range == pytest.approx(0.7, abs=0.05)
    assert m.bearing == pytest.approx(0.0, abs=0.05)
    assert abs(m.view_angle) < 0.2


def test_real_marker_matches_measurement_model(real_sim):
    dx, dy, _ = CONFIG.env.dock_pose
    pose = np.array([dx - 0.7, dy - 0.2, 0.0])
    real_sim.reset(tuple(pose))
    m = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    assert m is not None
    expected = measurement_model(pose, np.array([dx, dy]))
    assert m.range == pytest.approx(expected[0], abs=0.05)
    assert m.bearing == pytest.approx(expected[1], abs=0.05)


def test_real_marker_bearing_follows_heading(real_sim):
    dx, dy, _ = CONFIG.env.dock_pose
    real_sim.reset((dx - 0.7, dy, 0.2))            # robot turned left, marker on the right
    m = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    assert m is not None
    assert m.bearing == pytest.approx(-0.2, abs=0.05)


def test_real_marker_not_seen_when_facing_away(real_sim):
    dx, dy, _ = CONFIG.env.dock_pose
    real_sim.reset((dx - 0.7, dy, math.pi))
    assert read_or_skip(Sensing(real_sim, camera_period=0.0)).marker is None

def test_real_marker_has_covariance_and_small_reprojection_error(real_sim):
    dx, dy, _ = CONFIG.env.dock_pose
    real_sim.reset((dx - 0.7, dy, 0.0))
    m = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    assert m is not None and m.cov is not None
    assert m.reproj_error < MAX_REPROJ_PX


def test_real_marker_noise_grows_with_range(real_sim):
    dx, dy, _ = CONFIG.env.dock_pose
    real_sim.reset((dx - 0.5, dy, 0.0))
    near = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    real_sim.reset((dx - 1.2, dy, 0.0))
    far = read_or_skip(Sensing(real_sim, camera_period=0.0)).marker
    assert near is not None and far is not None
    assert far.cov[0, 0] > near.cov[0, 0]

# ------------------------------------------------- extra checkpoint tags
def synthetic_multi_marker_image(placements, black_px=BLACK_PX):
    """White image with several markers: placements = [(marker_id, shift_px), ...]."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG.camera.aruco_dict))
    h, w = CONFIG.camera.height, CONFIG.camera.width
    img = np.full((h, w), 255, np.uint8)
    for mid, shift_px in placements:
        marker = cv2.aruco.generateImageMarker(dictionary, mid, black_px)
        marker = cv2.copyMakeBorder(marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        size = marker.shape[0]
        x0, y0 = w // 2 - size // 2 + shift_px, h // 2 - size // 2
        img[y0:y0 + size, x0:x0 + size] = marker
    return np.stack([img] * 3, axis=2)


DOCK_ID = CONFIG.camera.marker_id
TAG_ID = DOCK_ID + 1


def test_detect_markers_returns_every_known_id():
    img = synthetic_multi_marker_image([(DOCK_ID, 160), (TAG_ID, -160)])
    found = detect_markers(img, camera_matrix())
    assert sorted(m.marker_id for m in found) == [DOCK_ID, TAG_ID]


def test_detect_markers_bearing_follows_the_image_side():
    img = synthetic_multi_marker_image([(DOCK_ID, 160), (TAG_ID, -160)])
    by_id = {m.marker_id: m for m in detect_markers(img, camera_matrix())}
    assert by_id[TAG_ID].bearing > 0       # left of the image
    assert by_id[DOCK_ID].bearing < 0      # right of the image


def test_detect_markers_ignores_ids_outside_the_tag_table():
    unknown = max(CONFIG.tag_table()) + 5
    img = synthetic_multi_marker_image([(unknown, 0)])
    assert detect_markers(img, camera_matrix()) == []


def test_detect_markers_is_empty_on_a_blank_image():
    blank = np.full((CONFIG.camera.height, CONFIG.camera.width, 3), 255, np.uint8)
    assert detect_markers(blank, camera_matrix()) == []


def test_detect_marker_still_returns_only_the_dock():
    img = synthetic_multi_marker_image([(DOCK_ID, 160), (TAG_ID, -160)])
    m = detect_marker(img, camera_matrix())
    assert m is not None and m.marker_id == DOCK_ID
    assert detect_marker(synthetic_multi_marker_image([(TAG_ID, 0)]), camera_matrix()) is None


def test_sensing_frame_separates_the_dock_from_the_tags():
    sim = FakeSim()
    sim.image = synthetic_multi_marker_image([(DOCK_ID, 160), (TAG_ID, -160)])
    sensing = Sensing(sim, camera_period=0.0)
    frame = sensing.read()
    assert frame.marker is not None and frame.marker.marker_id == DOCK_ID
    assert sorted(m.marker_id for m in frame.markers) == [DOCK_ID, TAG_ID]
    assert (sensing.n_detect, sensing.n_detect_any) == (1, 1)


def test_sensing_tag_only_frame_has_no_dock_marker():
    sim = FakeSim()
    sim.image = synthetic_multi_marker_image([(TAG_ID, 0)])
    sensing = Sensing(sim, camera_period=0.0)
    frame = sensing.read()
    assert frame.marker is None
    assert [m.marker_id for m in frame.markers] == [TAG_ID]
    assert (sensing.n_detect, sensing.n_detect_any) == (0, 1)
    sensing.reset()
    assert (sensing.n_detect, sensing.n_detect_any) == (0, 0)


def test_sensing_blank_frame_has_no_markers():
    frame = Sensing(FakeSim(), camera_period=0.0).read()
    assert frame.marker is None and frame.markers == ()


def test_real_extra_tag_is_seen_from_its_standoff_point(real_sim):
    from robot_model import wrap_angle
    for tid, (tx, ty, tyaw) in CONFIG.tag_table().items():
        if tid == DOCK_ID:
            continue
        s = CONFIG.tags.standoff
        real_sim.reset((tx + s * math.cos(tyaw), ty + s * math.sin(tyaw), wrap_angle(tyaw + math.pi)))
        frame = read_or_skip(Sensing(real_sim, camera_period=0.0))
        by_id = {m.marker_id: m for m in frame.markers}
        assert tid in by_id
        assert by_id[tid].range == pytest.approx(s, abs=0.1)
        assert by_id[tid].bearing == pytest.approx(0.0, abs=0.1)
        assert frame.marker is None          # the dock is not in view from here