import math
import numpy as np
import pytest
from spatialmath import SE3
import warnings, cv2
import calibration as cal
from config import CONFIG
from dataclasses import replace
OFF = cal.camera_forward_offset(CONFIG)


def marker_T(yaw, t):
    """Marker pose in the camera frame, normal pointing at the viewer."""
    R = (SE3.Ry(yaw) * SE3.Rx(math.pi)).R
    return SE3.Rt(R, np.array(t, dtype=float))


def test_camera_matrix_values():
    f = 240.0 / math.tan(math.radians(30.0))
    np.testing.assert_allclose(cal.camera_matrix(CONFIG), [[f, 0, 320], [0, f, 240], [0, 0, 1]])


def test_straight_ahead():
    np.testing.assert_allclose(cal.camera_to_robot([0, 0, 1.0], CONFIG), [1.0 + OFF, 0.0])


def test_image_right_is_robot_right():
    assert cal.camera_to_robot([0.3, 0, 1.0], CONFIG)[1] < 0


def test_head_on_marker():
    rng, bearing, view = cal.marker_pose_from_T(marker_T(0.0, (0, 0, 1.0)), CONFIG)
    assert rng == pytest.approx(1.0 + OFF)
    assert bearing == pytest.approx(0.0, abs=1e-9)
    assert view == pytest.approx(0.0, abs=1e-9)


def test_marker_on_the_left_has_positive_bearing():
    _, bearing, _ = cal.marker_pose_from_T(marker_T(0.0, (-0.2, 0, 1.0)), CONFIG)
    assert bearing > 0


def test_view_angle_sign_flips_with_yaw():
    v1 = cal.marker_pose_from_T(marker_T(0.3, (0, 0, 1.0)), CONFIG)[2]
    v2 = cal.marker_pose_from_T(marker_T(-0.3, (0, 0, 1.0)), CONFIG)[2]
    assert v1 == pytest.approx(-0.3, abs=1e-6)
    assert v1 == pytest.approx(-v2, abs=1e-6)

def calibrated_intr(rms=0.4, size=(640, 480), scale=1.0):
    K = cal.camera_matrix(CONFIG).copy()
    K[:2] *= scale
    return cal.CameraIntrinsics(K, np.zeros(5), size, cal.SOURCE_CALIBRATED, rms)


def test_nominal_calibration_warns():
    with pytest.warns(UserWarning):
        assert cal.get_calibration(CONFIG).intrinsics.source == cal.SOURCE_NOMINAL


def test_hardware_mode_refuses_nominal():
    with pytest.raises(ValueError):
        cal.get_calibration(CONFIG, require_calibrated=True)
    with pytest.raises(ValueError):
        cal.get_calibration(CONFIG, cal.nominal_intrinsics(CONFIG), require_calibrated=True)


def test_hardware_mode_accepts_good_and_rejects_bad_calibration():
    assert cal.get_calibration(CONFIG, calibrated_intr(0.4), require_calibrated=True).intrinsics.is_calibrated
    with pytest.raises(ValueError, match="RMS"):
        cal.get_calibration(CONFIG, calibrated_intr(2.0), require_calibrated=True)


def test_resolution_mismatch_is_rejected():
    with pytest.raises(ValueError, match="resolution"):
        cal.get_calibration(CONFIG, calibrated_intr(size=(320, 240), scale=0.5), require_calibrated=True)


def test_load_calibration_round_trip_and_nominal_file_rejected(tmp_path):
    path = cal.save_intrinsics(calibrated_intr(0.5), tmp_path / "intr.json")
    assert cal.load_calibration(path, CONFIG).intrinsics.reprojection_error == pytest.approx(0.5)
    nominal = cal.save_intrinsics(cal.nominal_intrinsics(CONFIG), tmp_path / "nominal.json")
    with pytest.raises(ValueError):
        cal.load_calibration(nominal, CONFIG)


def test_extrinsics_are_applied_and_consistent():
    ext = replace(CONFIG, camera=replace(CONFIG.camera, forward_offset=0.25, lateral_offset=0.03, yaw=0.1))
    np.testing.assert_allclose(cal.camera_to_robot([0, 0, 1.0], ext),
                               [math.cos(0.1) + 0.25, math.sin(0.1) + 0.03])
    q = np.array([1.3, -0.4])
    np.testing.assert_allclose(cal.camera_to_robot(cal.robot_to_camera(q, 0.0, ext), ext), q, atol=1e-9)
    assert cal.check_frame_conventions(ext) == []


def synthetic_views(n=8):
    K, obj = cal.camera_matrix(CONFIG), cal.chessboard_object_points((9, 6), 0.025)
    rng = np.random.default_rng(0)
    objs, imgs = [], []
    for _ in range(n):
        pts, _ = cv2.projectPoints(obj, rng.uniform(-0.5, 0.5, 3),
                                   np.array([-0.1, -0.06, rng.uniform(0.35, 0.6)]), K, np.zeros(5))
        objs.append(obj)
        imgs.append(pts.reshape(-1, 2))
    return objs, imgs


def test_intrinsics_are_recovered_from_synthetic_views():
    intr = cal.calibrate_intrinsics(*synthetic_views(), (640, 480))
    assert intr.is_calibrated and intr.reprojection_error < 0.5
    assert intr.K[0, 0] == pytest.approx(cal.camera_matrix(CONFIG)[0, 0], rel=0.05)


def test_too_few_views_are_rejected():
    with pytest.raises(ValueError):
        cal.calibrate_intrinsics(*synthetic_views(3), (640, 480))


def test_pnp_and_se3_paths_agree():
    T = marker_T(0.3, (0.1, 0.0, 1.0))
    a = cal.marker_pose_from_pnp(cv2.Rodrigues(T.R)[0], T.t, CONFIG)
    np.testing.assert_allclose(a, cal.marker_pose_from_T(T, CONFIG), atol=1e-9)


def test_solve_marker_pose_recovers_the_pose():
    T, K = marker_T(0.3, (0.1, 0.0, 1.0)), cal.camera_matrix(CONFIG)
    corners, _ = cv2.projectPoints(cal.marker_object_points(0.15), cv2.Rodrigues(T.R)[0],
                                   T.t.reshape(3, 1), K, np.zeros(5))
    best = cal.solve_marker_pose(corners.reshape(4, 2), K, np.zeros(5), 0.15)
    assert best is not None
    np.testing.assert_allclose(best[1].ravel(), T.t, atol=1e-3)
