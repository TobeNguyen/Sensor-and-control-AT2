# Including libraries
import math
import unittest
from dataclasses import dataclass

import numpy as np

from config import CONFIG
from estimator import EKFEstimator, chi2_bounds, nees
from robot_model import (TagMap, default_landmark, landmark_table, lidar_jacobian, lidar_range_model,
                        measurement_jacobian, measurement_jacobian_pose, measurement_model,
                        measurement_model_pose, motion_model, observability_matrix,
                        view_angle_model, wrap_angle)

LM = default_landmark(CONFIG)
DOCK_YAW = CONFIG.env.dock_pose[2]


@dataclass
class Marker:                      # minimal stand-in for sensing.MarkerMeasurement
    range: float
    bearing: float
    marker_id: int = 0


def exact_marker(pose, rng=None) -> Marker:
    r, b = measurement_model(pose, LM)
    return Marker(r, b)


def is_psd(P, tol=1e-12) -> bool:
    return bool(np.all(np.linalg.eigvalsh(0.5 * (P + P.T)) >= -tol))


class TestInterface(unittest.TestCase):
    def test_reset_and_attributes(self):
        ekf = EKFEstimator(CONFIG)
        np.testing.assert_allclose(ekf.x, CONFIG.env.start_pose)
        np.testing.assert_allclose(ekf.P, CONFIG.ekf.P0())
        ekf.reset([1.0, 2.0, 4.0], np.eye(3) * 0.2)
        np.testing.assert_allclose(ekf.x, [1.0, 2.0, wrap_angle(4.0)])
        np.testing.assert_allclose(ekf.P, np.eye(3) * 0.2)
        self.assertEqual((ekf.n_updates, ekf.n_rejected), (0, 0))

    def test_reset_copies_inputs(self):
        x0, P0 = np.array([0.0, 0.0, 0.0]), np.eye(3)
        ekf = EKFEstimator(CONFIG)
        ekf.reset(x0, P0)
        ekf.predict(0.1, 0.0)
        np.testing.assert_allclose(x0, 0.0)
        np.testing.assert_allclose(P0, np.eye(3))


class TestPredict(unittest.TestCase):
    def test_mean_matches_motion_model(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset([0.0, 0.0, 0.3])
        expected = motion_model([0.0, 0.0, 0.3], 0.05, 0.02)
        ekf.predict(0.05, 0.02)
        np.testing.assert_allclose(ekf.x, expected)

    def test_covariance_grows_and_stays_valid(self):
        ekf = EKFEstimator(CONFIG)
        tr0 = np.trace(ekf.P)
        for _ in range(500):
            ekf.predict(0.003, 0.001)
            self.assertTrue(np.allclose(ekf.P, ekf.P.T))
            self.assertTrue(is_psd(ekf.P))
        self.assertGreater(np.trace(ekf.P), tr0)

    def test_noise_scale_and_q_extra_inflate_covariance(self):
        base = EKFEstimator(CONFIG)
        scaled = EKFEstimator(CONFIG, odom_cov_scale=100.0)
        extra = EKFEstimator(CONFIG, q_extra=np.eye(3) * 1e-4)
        for e in (base, scaled, extra):
            e.predict(0.003, 0.0)
        self.assertGreater(np.trace(scaled.P), np.trace(base.P))
        self.assertGreater(np.trace(extra.P), np.trace(base.P))

    def test_theta_wrapped(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset([0.0, 0.0, 3.1])
        ekf.predict(0.0, 0.2)
        self.assertLessEqual(abs(ekf.x[2]), math.pi)


class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([0.5, 0.2, 0.4])
        self.ekf = EKFEstimator(CONFIG)
        self.ekf.reset(self.truth + [0.10, -0.08, 0.06], np.diag([0.05, 0.05, 0.05]) ** 2 * 4)

    def test_update_pulls_estimate_towards_truth(self):
        err0 = np.linalg.norm(self.ekf.x[:2] - self.truth[:2])
        self.assertTrue(self.ekf.update(exact_marker(self.truth)))
        err1 = np.linalg.norm(self.ekf.x[:2] - self.truth[:2])
        self.assertLess(err1, err0)
        self.assertEqual(self.ekf.n_updates, 1)

    def test_repeated_updates_converge_in_range_bearing_directions(self):
        # One landmark observes range and bearing; the residual in those two outputs vanishes.
        for _ in range(30):
            self.ekf.update(exact_marker(self.truth))
        r_est, b_est = measurement_model(self.ekf.x, LM)
        r_true, b_true = measurement_model(self.truth, LM)
        self.assertAlmostEqual(r_est, r_true, places=3)
        self.assertAlmostEqual(wrap_angle(b_est - b_true), 0.0, places=3)

    def test_update_reduces_covariance(self):
        P_before = self.ekf.P.copy()
        self.ekf.update(exact_marker(self.truth))
        self.assertTrue(is_psd(P_before - self.ekf.P))     # P never increases
        self.assertLess(np.trace(self.ekf.P), np.trace(P_before))
        self.assertTrue(np.allclose(self.ekf.P, self.ekf.P.T))
        self.assertTrue(is_psd(self.ekf.P))

    def test_perfect_prediction_leaves_state_unchanged(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth)
        ekf.update(exact_marker(self.truth))
        np.testing.assert_allclose(ekf.x, self.truth, atol=1e-12)
        self.assertAlmostEqual(ekf.last_nis, 0.0)

    def test_none_marker_changes_nothing(self):
        x, P = self.ekf.x.copy(), self.ekf.P.copy()
        self.assertFalse(self.ekf.update(None))
        np.testing.assert_array_equal(self.ekf.x, x)
        np.testing.assert_array_equal(self.ekf.P, P)

    def test_invalid_measurements_ignored(self):
        x = self.ekf.x.copy()
        for bad in (Marker(float("nan"), 0.1), Marker(1.0, float("inf")),
                    Marker(-1.0, 0.1), Marker(0.0, 0.1), Marker(1.0, 0.1, marker_id=7)):
            self.assertFalse(self.ekf.update(bad))
        np.testing.assert_array_equal(self.ekf.x, x)
        self.assertEqual(self.ekf.n_updates, 0)

    def test_bearing_wrap_does_not_blow_up(self):
        # Robot facing away from the landmark: true bearing is near +-pi.
        truth = np.array([0.0, 0.0, wrap_angle(math.atan2(LM[1], LM[0]) + math.pi - 0.02)])
        ekf = EKFEstimator(CONFIG)
        ekf.reset(truth + [0.02, 0.0, 0.03])
        m = exact_marker(truth)
        self.assertGreater(abs(m.bearing), 3.0)
        ekf.update(m)
        err = ekf.x - truth
        err[2] = wrap_angle(err[2])
        self.assertLess(np.linalg.norm(err), 0.1)
        self.assertLessEqual(abs(ekf.x[2]), math.pi)

    def test_measurement_on_landmark_skipped(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset([LM[0], LM[1], 0.0])
        self.assertFalse(ekf.update(Marker(0.5, 0.0)))

    def test_gate_rejects_outlier_but_accepts_good_measurement(self):
        ekf = EKFEstimator(CONFIG, gate=9.21)
        ekf.reset(self.truth, np.diag([0.05, 0.05, 0.05]) ** 2)
        outlier = Marker(measurement_model(self.truth, LM)[0] + 2.0,
                        measurement_model(self.truth, LM)[1])
        self.assertFalse(ekf.update(outlier))
        self.assertEqual(ekf.n_rejected, 1)
        self.assertTrue(ekf.update(exact_marker(self.truth)))

    def test_ungated_filter_accepts_outlier(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth, np.diag([0.05, 0.05, 0.05]) ** 2)
        r, b = measurement_model(self.truth, LM)
        self.assertTrue(ekf.update(Marker(r + 2.0, b)))

    def test_dead_reckoning_mode_ignores_marker(self):
        ekf = EKFEstimator(CONFIG, use_marker=False)
        ekf.reset(self.truth + [0.1, 0.0, 0.0])
        self.assertFalse(ekf.update(exact_marker(self.truth)))

    def test_custom_landmark(self):
        lm = np.array([0.0, 1.0])
        ekf = EKFEstimator(CONFIG, landmark=lm)
        pose = np.array([0.0, 0.0, 0.0])
        r, b = measurement_model(pose, lm)
        ekf.reset(pose + [0.05, 0.0, 0.0])
        ekf.update(Marker(r, b))
        self.assertLess(abs(ekf.x[0] - pose[0]) + abs(ekf.x[1] - pose[1]), 0.05 + 1e-9)


class TestNees(unittest.TestCase):
    def test_known_value_and_wrap(self):
        P = np.diag([0.01, 0.04, 0.09])
        self.assertAlmostEqual(nees([0.1, 0.2, 0.3], [0, 0, 0], P), 3.0)
        self.assertAlmostEqual(nees([0, 0, math.pi - 0.01], [0, 0, -math.pi + 0.01], np.eye(3)),
                               0.02 ** 2, places=9)


class TestClosedLoopWithoutSimulator(unittest.TestCase):
    """EKF vs dead reckoning on a synthetic run with a systematic heading-odometry bias."""

    def run_filter(self, use_marker, seed=3, steps=1500):
        rng = np.random.default_rng(seed)
        truth = np.array(CONFIG.env.start_pose, dtype=float)
        ekf = EKFEstimator(CONFIG, use_marker=use_marker, q_extra=np.diag([1e-3, 1e-3, 2e-3]) ** 2)
        ekf.reset(truth)
        R = CONFIG.R_camera()
        errs, nees_vals = [], []
        for k in range(steps):
            ds, dth = 0.003, 0.0012 * math.sin(k / 150.0)
            truth = motion_model(truth, ds, dth)
            ekf.predict(ds, dth + 0.0006)                 # biased odometry -> drift
            if k % 5 == 0:                                # camera at 10 Hz
                r, b = measurement_model(truth, LM)
                ekf.update(Marker(r + rng.normal(0, math.sqrt(R[0, 0])),
                                b + rng.normal(0, math.sqrt(R[1, 1]))))
            errs.append(np.linalg.norm(ekf.x[:2] - truth[:2]))
            nees_vals.append(nees(ekf.x, truth, ekf.P))
        return np.array(errs), np.array(nees_vals), ekf

    def test_ekf_beats_dead_reckoning(self):
        dr, _, _ = self.run_filter(use_marker=False)
        ekf_err, _, ekf = self.run_filter(use_marker=True)
        self.assertGreater(dr[-1], 0.3)                  # dead reckoning really drifts
        self.assertLess(ekf_err[-300:].mean(), 0.5 * dr[-300:].mean())
        self.assertLess(ekf_err[-1], 0.15)
        self.assertGreater(ekf.n_updates, 250)
        self.assertTrue(is_psd(ekf.P))

    def test_uncertainty_grows_without_marker_and_shrinks_with_it(self):
        _, _, dr = self.run_filter(use_marker=False)
        _, _, ekf = self.run_filter(use_marker=True)
        self.assertGreater(np.trace(dr.P), np.trace(ekf.P))

    def test_marker_loss_then_recovery(self):
        ekf = EKFEstimator(CONFIG, q_extra=np.diag([1e-3, 1e-3, 2e-3]) ** 2)
        truth = np.array([0.0, 0.0, 0.0])
        ekf.reset(truth)
        for _ in range(100):                             # no marker: P grows
            truth = motion_model(truth, 0.003, 0.0)
            ekf.predict(0.003, 0.0)
        tr_lost = np.trace(ekf.P)
        ekf.update(exact_marker(truth))                  # marker reappears: P shrinks
        self.assertLess(np.trace(ekf.P), tr_lost)

@dataclass
class ViewMarker:                  # marker stand-in that also carries the view angle
    range: float
    bearing: float
    view_angle: float
    marker_id: int = 0


@dataclass
class Scan:                        # minimal stand-in for sensing.LidarScan
    ranges: np.ndarray
    angles: np.ndarray
    valid: np.ndarray


def exact_view_marker(pose) -> ViewMarker:
    r, b, v = measurement_model_pose(pose, LM, DOCK_YAW)
    return ViewMarker(r, b, v)


def exact_scan(pose) -> Scan:
    angles = np.linspace(0.0, 2.0 * math.pi, CONFIG.lidar.num_beams, endpoint=False)
    ranges = np.array([lidar_range_model(pose, a) for a in angles])
    return Scan(ranges, angles, ranges < CONFIG.lidar.range_max)


class TestViewMeasurement(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([0.5, 0.2, 0.4])

    def test_view_angle_zero_when_facing_the_marker_head_on(self):
        pose = np.array([1.0, 0.5, DOCK_YAW - math.pi])       # heading that faces the marker face
        self.assertAlmostEqual(view_angle_model(pose, DOCK_YAW), 0.0)

    def test_jacobian_matches_finite_differences(self):
        x = np.array([0.3, -0.2, 0.7])
        H = measurement_jacobian_pose(x, LM)
        H_num = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = 1e-6
            diff = measurement_model_pose(x + d, LM, DOCK_YAW) - measurement_model_pose(x - d, LM, DOCK_YAW)
            diff[1], diff[2] = wrap_angle(diff[1]), wrap_angle(diff[2])
            H_num[:, i] = diff / 2e-6
        np.testing.assert_allclose(H, H_num, atol=1e-5)

    def test_view_angle_makes_stationary_robot_fully_observable(self):
        F = np.eye(3)                                          # robot is not moving
        H2 = measurement_jacobian(self.truth, LM)
        H3 = measurement_jacobian_pose(self.truth, LM)
        self.assertEqual(np.linalg.matrix_rank(observability_matrix(F, H2)), 2)
        self.assertEqual(np.linalg.matrix_rank(observability_matrix(F, H3)), 3)

    def test_view_angle_corrects_heading_in_one_update(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth + [0.0, 0.0, 0.20], np.diag([0.05, 0.05, 0.2]) ** 2)
        ekf.update(exact_view_marker(self.truth))
        self.assertLess(abs(wrap_angle(ekf.x[2] - self.truth[2])), 0.05)
        self.assertEqual(ekf.nis_dof[-1], 3)

    def test_view_angle_gives_better_heading_than_range_bearing_only(self):
        # Large position uncertainty: range and bearing alone cannot separate position from heading.
        err = {}
        for use_view in (True, False):
            ekf = EKFEstimator(CONFIG, use_view=use_view)
            ekf.reset(self.truth + [0.15, -0.10, 0.20], np.diag([0.3, 0.3, 0.2]) ** 2)
            ekf.update(exact_view_marker(self.truth))
            err[use_view] = abs(wrap_angle(ekf.x[2] - self.truth[2]))
        self.assertLess(err[True], 0.5 * err[False])

    def test_perfect_prediction_gives_zero_nis(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth)
        self.assertTrue(ekf.update(exact_view_marker(self.truth)))
        self.assertAlmostEqual(ekf.last_nis, 0.0)
        np.testing.assert_allclose(ekf.x, self.truth, atol=1e-12)

    def test_non_finite_view_angle_falls_back_to_two_dof(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth + [0.05, 0.0, 0.0])
        r, b = measurement_model(self.truth, LM)
        self.assertTrue(ekf.update(ViewMarker(r, b, float("nan"))))
        self.assertEqual(ekf.nis_dof[-1], 2)

    def test_view_angle_wrap_near_pi(self):
        # Pose whose true view angle is close to +-pi: the residual must be wrapped, not ~2*pi.
        pose = np.array([0.0, 0.0, DOCK_YAW - math.pi + math.pi - 0.02])
        ekf = EKFEstimator(CONFIG)
        ekf.reset(pose + [0.0, 0.0, 0.04])
        m = exact_view_marker(pose)
        self.assertGreater(abs(m.view_angle), 3.0)
        ekf.update(m)
        self.assertLess(abs(ekf.last_innovation[2]), 0.1)

    def test_gate_scales_with_dof_and_rejects_bad_view(self):
        ekf = EKFEstimator(CONFIG, gate=9.21)
        self.assertAlmostEqual(ekf._gate_for(2), 9.21)
        self.assertGreater(ekf._gate_for(3), 9.21)
        ekf.reset(self.truth, np.diag([0.05, 0.05, 0.05]) ** 2)
        m = exact_view_marker(self.truth)
        self.assertFalse(ekf.update(ViewMarker(m.range, m.bearing, m.view_angle + 1.0)))
        self.assertEqual(ekf.n_rejected, 1)
        self.assertTrue(ekf.update(exact_view_marker(self.truth)))

    def test_nis_history_is_recorded_and_reset(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth + [0.05, -0.03, 0.04])
        for _ in range(3):
            ekf.update(exact_view_marker(self.truth))
        self.assertEqual(len(ekf.nis_history), 3)
        self.assertTrue(np.isfinite(ekf.mean_nis_per_dof()))
        ekf.reset(self.truth)
        self.assertEqual(ekf.nis_history, [])
        self.assertTrue(math.isnan(ekf.mean_nis_per_dof()))

class TestGateRecovery(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([0.5, 0.2, 0.4])

    def test_lockout_is_broken_after_consecutive_rejections(self):
        # Badly wrong and over-confident estimate: every measurement is gated out until recovery.
        ekf = EKFEstimator(CONFIG, gate=9.21, max_rejects=5)
        ekf.reset(self.truth + [0.4, -0.3, 0.3], np.diag([0.05, 0.05, 0.05]) ** 2)
        err0 = np.linalg.norm(ekf.x[:2] - self.truth[:2])
        results = [ekf.update(exact_view_marker(self.truth)) for _ in range(5)]
        self.assertEqual(results, [False, False, False, False, True])
        self.assertEqual(ekf.n_recovered, 1)
        self.assertLess(np.linalg.norm(ekf.x[:2] - self.truth[:2]), err0)
        self.assertTrue(is_psd(ekf.P))

    def test_an_accepted_measurement_resets_the_rejection_counter(self):
        ekf = EKFEstimator(CONFIG, gate=9.21, max_rejects=5)
        ekf.reset(self.truth, np.diag([0.05, 0.05, 0.05]) ** 2)
        m = exact_view_marker(self.truth)
        bad = ViewMarker(m.range + 2.0, m.bearing, m.view_angle)
        for _ in range(3):
            self.assertFalse(ekf.update(bad))
        self.assertTrue(ekf.update(m))                       # good measurement: counter back to 0
        for _ in range(3):
            self.assertFalse(ekf.update(bad))
        self.assertEqual(ekf.n_recovered, 0)

    def test_reset_clears_the_recovery_state(self):
        ekf = EKFEstimator(CONFIG, gate=9.21, max_rejects=2)
        ekf.reset(self.truth + [0.4, 0.4, 0.0], np.diag([0.05, 0.05, 0.05]) ** 2)
        for _ in range(2):
            ekf.update(exact_view_marker(self.truth))
        self.assertEqual(ekf.n_recovered, 1)
        ekf.reset(self.truth)
        self.assertEqual((ekf.n_recovered, ekf._reject_run), (0, 0))


class TestLidarMap(unittest.TestCase):
    def test_wall_ranges(self):
        hx = CONFIG.env.room_half_size[0]
        self.assertAlmostEqual(lidar_range_model([0.0, 0.0, 0.0], 0.0), hx)            # east wall
        self.assertAlmostEqual(lidar_range_model([0.5, 0.0, 0.0], math.pi), hx + 0.5)  # west wall
        self.assertAlmostEqual(lidar_range_model([0.0, 0.0, math.pi / 2], 0.0), hx)    # north wall

    def test_obstacle_range(self):
        # Obstacle 0 is at (0, -0.5) with radius 0.2: looking down from (0, 0.5) hits it at y = -0.3.
        self.assertAlmostEqual(lidar_range_model([0.0, 0.5, 0.0], -math.pi / 2), 0.8)

    def test_range_max_when_nothing_is_hit(self):
        cfg = CONFIG
        far_room = type(cfg)(env=type(cfg.env)(room_half_size=(10.0, 10.0), obstacles=()))
        self.assertAlmostEqual(lidar_range_model([0.0, 0.0, 0.0], 0.0, far_room), cfg.lidar.range_max)

    def test_jacobian_matches_analytic_wall_case(self):
        # Facing the east wall: r = (hx - x) / cos(theta + a)
        hx, th = CONFIG.env.room_half_size[0], 0.1
        H = lidar_jacobian([0.0, 0.0, th], 0.0)
        np.testing.assert_allclose(
            H, [-1.0 / math.cos(th), 0.0, hx * math.sin(th) / math.cos(th) ** 2], atol=1e-4)


class TestLidarUpdate(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([0.3, -0.2, 0.2])

    def test_lidar_update_pulls_estimate_towards_truth(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth + [0.08, -0.06, 0.05], np.diag([0.1, 0.1, 0.1]) ** 2)
        err0 = np.linalg.norm(ekf.x[:2] - self.truth[:2])
        scan = exact_scan(self.truth)
        for _ in range(5):
            ekf.update_lidar(scan)
        err1 = np.linalg.norm(ekf.x[:2] - self.truth[:2])
        self.assertLess(err1, 0.5 * err0)
        self.assertGreater(ekf.n_lidar_updates, 10)
        self.assertTrue(is_psd(ekf.P))

    def test_lidar_update_reduces_covariance(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth, np.diag([0.1, 0.1, 0.1]) ** 2)
        tr0 = np.trace(ekf.P)
        ekf.update_lidar(exact_scan(self.truth))
        self.assertLess(np.trace(ekf.P), tr0)

    def test_unmapped_object_is_gated_out(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth, np.diag([0.05, 0.05, 0.05]) ** 2)
        scan = exact_scan(self.truth)
        scan.ranges[::30] -= 0.5                            # something unexpected in front of every beam
        x = ekf.x.copy()
        self.assertEqual(ekf.update_lidar(scan), 0)
        np.testing.assert_allclose(ekf.x, x)
        self.assertGreater(ekf.n_lidar_rejected, 0)

    def test_disabled_or_missing_scan_changes_nothing(self):
        ekf = EKFEstimator(CONFIG, use_lidar=False)
        ekf.reset(self.truth + [0.05, 0.0, 0.0])
        x = ekf.x.copy()
        self.assertEqual(ekf.update_lidar(exact_scan(self.truth)), 0)
        np.testing.assert_array_equal(ekf.x, x)
        self.assertEqual(EKFEstimator(CONFIG).update_lidar(None), 0)


class TestConsistencyTools(unittest.TestCase):
    def test_chi2_bounds_contain_the_expected_value(self):
        for dof in (2, 3):
            lo, hi = chi2_bounds(dof, n=200)
            self.assertLess(lo, dof)
            self.assertGreater(hi, dof)

    def test_monte_carlo_nees_is_consistent(self):
        # Marker-only EKF with correct noise models: average NEES should be near 3.
        R = CONFIG.R_camera()
        nees_vals = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            truth = np.array(CONFIG.env.start_pose, dtype=float)
            ekf = EKFEstimator(CONFIG)
            ekf.reset(truth)
            for k in range(600):
                ds, dth = 0.003, 0.001 * math.sin(k / 100.0)
                truth = motion_model(truth, ds, dth)
                ekf.predict(ds, dth)
                if k % 5 == 0:
                    r, b, v = measurement_model_pose(truth, LM, DOCK_YAW)
                    ekf.update(ViewMarker(r + rng.normal(0, math.sqrt(R[0, 0])),
                                        b + rng.normal(0, math.sqrt(R[1, 1])),
                                        v + rng.normal(0, math.sqrt(ekf.R_view))))
            nees_vals.append(nees(ekf.x, truth, ekf.P))
        lo, hi = chi2_bounds(3, n=len(nees_vals), alpha=0.01)
        self.assertGreater(np.mean(nees_vals), lo)
        self.assertLess(np.mean(nees_vals), hi)

class TestTagMarkers(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([0.5, 1.0, 0.3])
        self.lm_tag, _ = landmark_table(CONFIG)[1]          # extra tag with id 1

    def tag_marker(self, pose):
        r, b = measurement_model(pose, self.lm_tag)
        return Marker(r, b, marker_id=1)

    def test_tag_update_uses_the_tag_landmark(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth + [0.10, -0.08, 0.06], np.diag([0.05, 0.05, 0.05]) ** 2 * 4)
        err0 = np.linalg.norm(ekf.x[:2] - self.truth[:2])
        for _ in range(30):
            self.assertTrue(ekf.update(self.tag_marker(self.truth)))
        # range and bearing to the TAG match the truth (the dock landmark would not give this)
        r_est, b_est = measurement_model(ekf.x, self.lm_tag)
        r_true, b_true = measurement_model(self.truth, self.lm_tag)
        self.assertAlmostEqual(r_est, r_true, places=3)
        self.assertAlmostEqual(wrap_angle(b_est - b_true), 0.0, places=3)
        self.assertLess(np.linalg.norm(ekf.x[:2] - self.truth[:2]), err0)

    def test_n_tag_updates_counts_only_extra_tags(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth)
        ekf.update(exact_marker(self.truth))                 # dock
        self.assertEqual((ekf.n_updates, ekf.n_tag_updates), (1, 0))
        ekf.update(self.tag_marker(self.truth))              # extra tag
        self.assertEqual((ekf.n_updates, ekf.n_tag_updates), (2, 1))
        ekf.reset(self.truth)
        self.assertEqual((ekf.n_updates, ekf.n_tag_updates), (0, 0))

    def test_unknown_tag_id_is_ignored(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth)
        x = ekf.x.copy()
        self.assertFalse(ekf.update(Marker(1.0, 0.1, marker_id=max(landmark_table(CONFIG)) + 1)))
        np.testing.assert_array_equal(ekf.x, x)

    def test_position_std_and_is_lost_follow_the_threshold(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth, np.diag([0.001, 0.001, 0.001]))
        self.assertAlmostEqual(ekf.position_std(), math.sqrt(0.002))
        self.assertFalse(ekf.is_lost())
        ekf.reset(self.truth, np.diag([0.1, 0.1, 0.01]))
        self.assertGreater(ekf.position_std(), CONFIG.tags.lost_std)
        self.assertTrue(ekf.is_lost())

    def test_inflate_makes_the_filter_lost_and_clears_the_reject_counter(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.truth, np.diag([0.001, 0.001, 0.001]))
        ekf._reject_run = 5
        ekf.inflate()
        self.assertTrue(ekf.is_lost())
        self.assertEqual(ekf._reject_run, 0)
        self.assertTrue(is_psd(ekf.P))

    def test_inflate_lets_a_tag_through_that_the_gate_rejected(self):
        ekf = EKFEstimator(CONFIG, gate=9.21)
        ekf.reset(self.truth + [0.3, -0.2, 0.1], np.diag([0.02, 0.02, 0.02]) ** 2)
        marker = self.tag_marker(self.truth)
        self.assertFalse(ekf.update(marker))                 # confident but wrong: gated out
        ekf.inflate()
        self.assertTrue(ekf.update(marker))                  # uncertain: accepted
        self.assertLess(np.linalg.norm(ekf.x[:2] - self.truth[:2]), 0.3)


class TestLearnedTagMap(unittest.TestCase):
    def setUp(self):
        self.pose = np.array(CONFIG.env.start_pose, dtype=float)
        self.dock_xy = np.array(CONFIG.env.dock_pose[:2], dtype=float)
        self.dock_id = int(CONFIG.camera.marker_id)

    def learned_map(self, shift=(0.0, 0.0)):
        """TagMap holding the dock, optionally shifted away from its true position."""
        tm = TagMap()
        r, b, v = measurement_model_pose(self.pose, self.dock_xy + np.asarray(shift), DOCK_YAW)
        tm.add(self.pose, ViewMarker(r, b, v, self.dock_id))
        return tm

    def test_empty_map_ignores_every_marker(self):
        ekf = EKFEstimator(CONFIG)
        ekf.reset(self.pose)
        ekf.set_tag_map(TagMap())
        self.assertEqual(ekf.landmarks, {})
        self.assertFalse(ekf.update(Marker(2.0, 0.1)))
        self.assertEqual(ekf.n_updates, 0)

    def test_learned_pose_replaces_the_config_pose(self):
        ekf = EKFEstimator(CONFIG)
        ekf.set_tag_map(self.learned_map())
        self.assertEqual(sorted(ekf.landmarks), [self.dock_id])
        np.testing.assert_allclose(ekf.landmarks[self.dock_id][0], self.dock_xy, atol=1e-9)
        np.testing.assert_allclose(ekf.landmark, self.dock_xy, atol=1e-9)

    def test_none_restores_the_poses_from_the_config(self):
        ekf = EKFEstimator(CONFIG)
        ekf.set_tag_map(TagMap())
        ekf.set_tag_map(None)
        self.assertEqual(sorted(ekf.landmarks), sorted(landmark_table(CONFIG)))

    def test_update_follows_the_learned_map_not_the_config(self):
        # Same real measurement: matches the config dock exactly, but not a wrongly learned dock.
        good, bad = EKFEstimator(CONFIG), EKFEstimator(CONFIG)
        for ekf in (good, bad):
            ekf.reset(self.pose, np.diag([0.05, 0.05, 0.05]) ** 2)
        good.set_tag_map(self.learned_map())
        bad.set_tag_map(self.learned_map(shift=(0.0, 0.3)))
        marker = exact_view_marker(self.pose)
        self.assertTrue(good.update(marker))
        self.assertTrue(bad.update(marker))
        self.assertAlmostEqual(good.last_nis, 0.0)
        self.assertGreater(bad.last_nis, 1.0)


if __name__ == "__main__":
    unittest.main()