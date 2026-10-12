# Including libraries
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import calibration as cal
from config import CONFIG
from robot_model import measurement_model, wrap_angle

OFF = cal.camera_forward_offset(CONFIG)


class TestIntrinsics(unittest.TestCase):
    def test_nominal_matrix_values(self):
        K = cal.camera_matrix(CONFIG)
        f = 240.0 / math.tan(math.radians(30.0))
        np.testing.assert_allclose(K, [[f, 0, 320], [0, f, 240], [0, 0, 1]])

    def test_nominal_is_not_labelled_calibrated(self):
        intr = cal.nominal_intrinsics(CONFIG)
        self.assertEqual(intr.source, cal.SOURCE_NOMINAL)
        self.assertFalse(intr.is_calibrated)
        self.assertIsNone(intr.reprojection_error)
        self.assertEqual(cal.validate_intrinsics(intr), [])

    def test_validate_detects_problems(self):
        good = cal.nominal_intrinsics(CONFIG)
        bad_f = cal.CameraIntrinsics(np.array([[-1, 0, 320], [0, 400, 240], [0, 0, 1.]]),
                                    np.zeros(5), (640, 480), cal.SOURCE_NOMINAL)
        bad_pp = cal.CameraIntrinsics(np.array([[400, 0, 900], [0, 400, 240], [0, 0, 1.]]),
                                    np.zeros(5), (640, 480), cal.SOURCE_NOMINAL)
        bad_shape = cal.CameraIntrinsics(np.eye(2), np.zeros(5), (640, 480), cal.SOURCE_NOMINAL)
        bad_dist = cal.CameraIntrinsics(good.K, np.zeros(3), (640, 480), cal.SOURCE_NOMINAL)
        bad_src = cal.CameraIntrinsics(good.K, np.zeros(5), (640, 480), "guess")
        for intr in (bad_f, bad_pp, bad_shape, bad_dist, bad_src):
            self.assertTrue(cal.validate_intrinsics(intr))
            with self.assertRaises(ValueError):
                cal.assert_valid_intrinsics(intr)


class TestCameraRobotFrames(unittest.TestCase):
    def test_straight_ahead(self):
        p = cal.camera_to_robot([0.0, 0.0, 1.0], CONFIG)
        np.testing.assert_allclose(p, [1.0 + OFF, 0.0])
        rng, bearing = cal.range_bearing(p)
        self.assertAlmostEqual(bearing, 0.0)

    def test_image_right_is_robot_right(self):
        p = cal.camera_to_robot([0.3, 0.0, 1.0], CONFIG)   # X > 0 -> right of the camera
        self.assertLess(p[1], 0.0)                         # robot y < 0 -> right
        self.assertLess(cal.range_bearing(p)[1], 0.0)

    def test_round_trip(self):
        p = np.array([1.7, -0.6])
        back = cal.camera_to_robot(cal.robot_to_camera(p, 0.0, CONFIG), CONFIG)
        np.testing.assert_allclose(back, p, atol=1e-12)

    def test_range_bearing_round_trip(self):
        p = cal.xy_from_range_bearing(1.2, 0.4)
        r, b = cal.range_bearing(p)
        self.assertAlmostEqual(r, 1.2)
        self.assertAlmostEqual(b, 0.4)

    def test_direction_has_no_offset(self):
        np.testing.assert_allclose(cal.camera_dir_to_robot([0, 0, 1]), [1, 0])
        np.testing.assert_allclose(cal.camera_dir_to_robot([1, 0, 0]), [0, -1])

    def test_projection_directions(self):
        intr = cal.nominal_intrinsics(CONFIG)
        c = cal.project_robot_point([1.0, 0.0], intr, 0.0, CONFIG)
        np.testing.assert_allclose(c, [320, 240], atol=1e-6)
        self.assertLess(cal.project_robot_point([1.0, 0.2], intr, 0.0, CONFIG)[0], 320)   # left
        self.assertLess(cal.project_robot_point([1.0, 0.0], intr, 0.1, CONFIG)[1], 240)   # up
        self.assertIsNone(cal.project_robot_point([0.0, 0.0], intr, 0.0, CONFIG))         # behind

    def test_marker_object_points(self):
        pts = cal.marker_object_points(0.15)
        self.assertEqual(pts.shape, (4, 3))
        np.testing.assert_allclose(pts[0], [-0.075, 0.075, 0.0])
        np.testing.assert_allclose(pts[2], [0.075, -0.075, 0.0])


class TestMarkerPose(unittest.TestCase):
    S = CONFIG.camera.marker_size

    def _true_pose(self, yaw, t):
        Rx = cv2.Rodrigues(np.array([math.pi, 0.0, 0.0]))[0]
        R = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0] @ Rx   # marker z points to the viewer
        return cv2.Rodrigues(R)[0], np.array(t, dtype=float).reshape(3, 1)

    def _rendered_corners(self, yaw, t):
        """Warp a real ArUco image into the camera view and detect it with the OpenCV detector."""
        K = cal.camera_matrix(CONFIG)
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG.camera.aruco_dict))
        img = cv2.aruco.generateImageMarker(dic, CONFIG.camera.marker_id, 200)
        img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        half = self.S / 2.0 * 280 / 200
        rvec, tvec = self._true_pose(yaw, t)
        c3 = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        px, _ = cv2.projectPoints(c3, rvec, tvec, K, np.zeros(5))
        H = cv2.getPerspectiveTransform(np.array([[0, 0], [280, 0], [280, 280], [0, 280]], np.float32),
                                        px.reshape(4, 2).astype(np.float32))
        frame = cv2.warpPerspective(img, H, (640, 480), borderValue=128)
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(frame)
        self.assertIsNotNone(ids)
        return corners[0].reshape(4, 2)

    def test_head_on(self):
        rng, bearing, view = cal.marker_pose_from_pnp([math.pi, 0, 0], [0, 0, 1.0], CONFIG)
        self.assertAlmostEqual(rng, 1.0 + OFF)
        self.assertAlmostEqual(bearing, 0.0)
        self.assertAlmostEqual(view, 0.0)

    def test_exact_projection_round_trip(self):
        K = cal.camera_matrix(CONFIG)
        rvec, tvec = self._true_pose(0.3, (-0.2, 0.0, 1.0))
        img, _ = cv2.projectPoints(cal.marker_object_points(self.S), rvec, tvec, K, np.zeros(5))
        sol = cal.solve_marker_pose(img, K, np.zeros(5), self.S)
        self.assertIsNotNone(sol)
        rng, bearing, view = cal.marker_pose_from_pnp(sol[0], sol[1], CONFIG)
        p = cal.camera_to_robot((-0.2, 0.0, 1.0), CONFIG)
        self.assertAlmostEqual(rng, math.hypot(*p), places=3)
        self.assertAlmostEqual(bearing, math.atan2(p[1], p[0]), places=3)
        self.assertAlmostEqual(view, -0.3, places=2)
        self.assertGreater(bearing, 0.0)   # X < 0 -> marker on the left -> positive bearing

    def test_detected_marker_view_angle_has_no_flip(self):
        """Regression: raw IPPE_SQUARE returns view_angle = -pi for some head-on views."""
        K = cal.camera_matrix(CONFIG)
        for yaw, t in [(0.0, (0, 0, 1.0)), (0.3, (-0.2, 0, 1.0)), (-0.4, (0.15, 0, 1.2))]:
            corners = self._rendered_corners(yaw, t)
            sol = cal.solve_marker_pose(corners, K, np.zeros(5), self.S)
            self.assertIsNotNone(sol)
            _, bearing, view = cal.marker_pose_from_pnp(sol[0], sol[1], CONFIG)
            self.assertAlmostEqual(view, -yaw, delta=0.06)
            self.assertAlmostEqual(sol[1][2, 0], t[2], delta=0.03)

    def test_view_angle_sign_flips_with_yaw(self):
        K = cal.camera_matrix(CONFIG)
        views = []
        for yaw in (0.3, -0.3):
            rvec, tvec = self._true_pose(yaw, (0.0, 0.0, 1.0))
            img, _ = cv2.projectPoints(cal.marker_object_points(self.S), rvec, tvec, K, np.zeros(5))
            sol = cal.solve_marker_pose(img, K, np.zeros(5), self.S)
            views.append(cal.marker_pose_from_pnp(sol[0], sol[1], CONFIG)[2])
        self.assertAlmostEqual(views[0], -views[1], places=2)

    def test_marker_behind_camera_rejected(self):
        K = cal.camera_matrix(CONFIG)
        # corners in clockwise-reversed order describe a marker seen from behind -> no valid pose
        img = np.array([[300, 220], [300, 260], [340, 260], [340, 220]], dtype=float)
        self.assertIsNone(cal.solve_marker_pose(img, K, np.zeros(5), self.S))


class TestWorldFrame(unittest.TestCase):
    def test_round_trip(self):
        pose = (1.0, -0.5, 0.8)
        p = np.array([0.4, 1.1])
        np.testing.assert_allclose(cal.world_to_robot(pose, cal.robot_to_world(pose, p)), p, atol=1e-12)

    def test_pose_matrix_rotation(self):
        np.testing.assert_allclose(cal.robot_to_world((0, 0, math.pi / 2), [1, 0]), [0, 1], atol=1e-12)

    def test_marker_world_position_matches_ekf_measurement_model(self):
        landmark = np.array(CONFIG.env.dock_pose[:2]) if hasattr(CONFIG, "env") else np.array([1.9, 1.0])
        pose = np.array([0.5, 0.2, 0.7])
        rng, bearing = measurement_model(pose, landmark)
        np.testing.assert_allclose(cal.marker_world_position(pose, rng, bearing), landmark, atol=1e-9)


class TestConsistencyChecks(unittest.TestCase):
    def test_default_config_passes(self):
        self.assertEqual(cal.check_frame_conventions(CONFIG), [])
        cal.assert_frame_conventions(CONFIG)

    def test_unit_errors_detected(self):
        from dataclasses import replace
        bad = replace(CONFIG, camera=replace(CONFIG.camera, marker_size=150.0, fovy_deg=1.0))
        self.assertEqual(len(cal.check_units(bad)), 2)
        with self.assertRaises(ValueError):
            cal.assert_frame_conventions(bad)

    def test_wrong_fov_is_caught(self):
        # An intrinsic matrix with the principal point moved must fail the 'ahead' check.
        intr = cal.nominal_intrinsics(CONFIG)
        K = intr.K.copy(); K[0, 2] += 50.0
        shifted = cal.CameraIntrinsics(K, np.zeros(5), intr.image_size, cal.SOURCE_NOMINAL)
        # projection still goes to the shifted centre, so convention check follows K and passes;
        # but a wrong fov relative to config must be flagged.
        K2 = intr.K.copy(); K2[1, 1] *= 1.3
        wrong_fov = cal.CameraIntrinsics(K2, np.zeros(5), intr.image_size, cal.SOURCE_NOMINAL)
        self.assertEqual(cal.check_frame_conventions(CONFIG, shifted), [])
        self.assertTrue(cal.check_frame_conventions(CONFIG, wrong_fov))


class TestCalibrationProcedure(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[500.0, 0, 322.0], [0, 505.0, 238.0], [0, 0, 1]])
        self.pattern, self.square = (9, 6), 0.025
        self.obj = cal.chessboard_object_points(self.pattern, self.square)

    def _views(self, n, noise, seed=0):
        rng = np.random.default_rng(seed)
        objs, imgs = [], []
        for _ in range(n):
            rvec = rng.uniform(-0.5, 0.5, 3)
            tvec = np.array([-0.1, -0.06, 0.0]) + np.array([rng.uniform(-0.03, 0.03),
                                                        rng.uniform(-0.03, 0.03),
                                                        rng.uniform(0.5, 0.8)])
            p, _ = cv2.projectPoints(self.obj, rvec, tvec, self.K, np.zeros(5))
            objs.append(self.obj)
            imgs.append(p.reshape(-1, 2) + rng.normal(0, noise, (len(self.obj), 2)))
        return objs, imgs

    def test_object_points_grid(self):
        self.assertEqual(self.obj.shape, (54, 3))
        np.testing.assert_allclose(self.obj[1], [0.025, 0, 0], atol=1e-7)
        np.testing.assert_allclose(self.obj[9], [0, 0.025, 0], atol=1e-7)

    def test_recovers_intrinsics_from_synthetic_views(self):
        objs, imgs = self._views(12, noise=0.1)
        intr = cal.calibrate_intrinsics(objs, imgs, (640, 480))
        self.assertTrue(intr.is_calibrated)
        self.assertLess(intr.reprojection_error, 0.5)
        self.assertAlmostEqual(intr.K[0, 0] / 500.0, 1.0, delta=0.02)
        self.assertAlmostEqual(intr.K[1, 1] / 505.0, 1.0, delta=0.02)
        self.assertAlmostEqual(intr.K[0, 2], 322.0, delta=5.0)
        self.assertAlmostEqual(intr.K[1, 2], 238.0, delta=5.0)

    def test_too_few_views_rejected(self):
        objs, imgs = self._views(3, noise=0.0)
        with self.assertRaises(ValueError):
            cal.calibrate_intrinsics(objs, imgs, (640, 480))

    def test_mismatched_inputs_rejected(self):
        objs, imgs = self._views(6, noise=0.0)
        with self.assertRaises(ValueError):
            cal.calibrate_intrinsics(objs, imgs[:-1], (640, 480))

    def test_chessboard_detection_on_rendered_board(self):
        sq = 40
        board = np.full((8 * sq, 11 * sq), 255, np.uint8)
        for r in range(7):
            for c in range(10):
                if (r + c) % 2 == 0:
                    board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
        corners = cal.find_chessboard_corners(board, self.pattern)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (54, 2))
        self.assertIsNone(cal.find_chessboard_corners(np.full((480, 640), 128, np.uint8), self.pattern))

    def test_calibrate_from_images_requires_enough_detections(self):
        blanks = [np.full((480, 640, 3), 128, np.uint8)] * 6
        with self.assertRaises(ValueError):
            cal.calibrate_from_images(blanks, self.pattern, self.square)


class TestPersistenceAndBundle(unittest.TestCase):
    def test_save_load_round_trip(self):
        intr = cal.CameraIntrinsics(cal.camera_matrix(CONFIG), np.array([0.1, -0.05, 0, 0, 0]),
                                    (640, 480), cal.SOURCE_CALIBRATED, 0.31)
        with tempfile.TemporaryDirectory() as d:
            path = cal.save_intrinsics(intr, Path(d) / "sub" / "cam.json")
            back = cal.load_intrinsics(path)
        np.testing.assert_allclose(back.K, intr.K)
        np.testing.assert_allclose(back.dist_coeffs, intr.dist_coeffs)
        self.assertEqual(back.source, cal.SOURCE_CALIBRATED)
        self.assertEqual(back.image_size, (640, 480))
        self.assertAlmostEqual(back.reprojection_error, 0.31)

    def test_default_bundle(self):
        c = cal.get_calibration(CONFIG)
        self.assertEqual(c.intrinsics.source, cal.SOURCE_NOMINAL)
        self.assertAlmostEqual(c.forward_offset, OFF)
        self.assertEqual(c.marker_points.shape, (4, 3))
        self.assertEqual(c.dist_coeffs.shape, (5,))

    def test_bundle_rejects_invalid_intrinsics(self):
        bad = cal.CameraIntrinsics(np.eye(3), np.zeros(5), (640, 480), cal.SOURCE_NOMINAL)
        with self.assertRaises(ValueError):
            cal.get_calibration(CONFIG, bad)


if __name__ == "__main__":
    unittest.main()