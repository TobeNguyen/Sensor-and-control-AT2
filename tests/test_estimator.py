# Including libraries
import math
import unittest
from dataclasses import dataclass

import numpy as np

from config import CONFIG
from estimator import EKFEstimator, nees
from robot_model import default_landmark, measurement_model, motion_model, wrap_angle

LM = default_landmark(CONFIG)


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


if __name__ == "__main__":
    unittest.main()