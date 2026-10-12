# Including libraries
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config import CONFIG, Config
from robot_model import (default_landmark, innovation, jacobian_F, jacobian_G,
                        measurement_jacobian, measurement_model, motion_model, wrap_angle)

# Normalised Estimation Error Squared: e^T P^-1 e.
# For a consistent 3-state filter its average is about 3 (chi-square, 3 dof).
# Needs ground truth, so use it for EVALUATION only, never inside the controller.
def nees(x_est, x_true, P) -> float:
    e = np.asarray(x_est, dtype=float) - np.asarray(x_true, dtype=float)
    e[2] = wrap_angle(e[2])
    return float(e @ np.linalg.solve(np.asarray(P, dtype=float), e))

class EKFEstimator:
    def __init__(self, cfg: Config = CONFIG, landmark=None, q_extra: Optional[np.ndarray] = None,
                odom_cov_scale: float = 1.0, R: Optional[np.ndarray] = None,
                gate: Optional[float] = None, use_marker: bool = True):
        self.cfg = cfg
        self.landmark = default_landmark(cfg) if landmark is None else np.asarray(landmark, float)
        self.q_extra = np.zeros((3, 3)) if q_extra is None else np.asarray(q_extra, float)
        self.R = cfg.R_camera() if R is None else np.asarray(R, dtype=float)
        self.gate = gate
        self.use_marker = use_marker

        # Odometry noise: each wheel angle has std sigma_e. With ds = r (dl + dr) / 2 and
        # dtheta = r (dr - dl) / b, error propagation gives the diagonal covariance below.
        rob = cfg.robot
        s_ds = rob.wheel_radius * rob.encoder_noise_std / math.sqrt(2.0)
        s_dth = rob.wheel_radius * rob.encoder_noise_std * math.sqrt(2.0) / rob.wheel_base
        self._odom_cov = odom_cov_scale * np.diag([s_ds ** 2, s_dth ** 2])
        self.reset(cfg.env.start_pose)

    # Initialise belief N(x0, P0) and clear the diagnostics
    def reset(self, x0, P0: Optional[np.ndarray] = None) -> None:
        self.x = np.array(x0, dtype=float)
        self.x[2] = wrap_angle(self.x[2])
        self.P = self.cfg.ekf.P0() if P0 is None else np.array(P0, dtype=float)
        self.last_innovation: Optional[np.ndarray] = None
        self.last_nis: Optional[float] = None
        self.n_updates = 0
        self.n_rejected = 0

    # EKF prediction: propagate the mean through f and the covariance through F and G
    def predict(self, ds: float, dtheta: float) -> None:
        F = jacobian_F(self.x, ds, dtheta)
        G = jacobian_G(self.x, ds, dtheta)
        P = F @ self.P @ F.T + G @ self._odom_cov @ G.T + self.q_extra
        self.P = 0.5 * (P + P.T)
        self.x = motion_model(self.x, ds, dtheta)

    # A usable marker has the right id and finite, positive range
    def _is_valid(self, marker) -> bool:
        values = (marker.range, marker.bearing)
        return (all(math.isfinite(v) for v in values) and marker.range > 0.0
                and getattr(marker, "marker_id", self.cfg.camera.marker_id)
                == self.cfg.camera.marker_id)
    
    # EKF correction with a [range, bearing] marker measurement.
    # Returns True if the measurement was applied. With marker=None (lost marker) nothing
    # happens: the estimate keeps following odometry and P keeps growing.
    def update(self, marker) -> bool:
        if not self.use_marker or marker is None or not self._is_valid(marker):
            return False

        z = np.array([marker.range, marker.bearing], dtype=float)
        try:
            H = measurement_jacobian(self.x, self.landmark)
        except ValueError:                      # robot on top of the landmark: skip
            return False

        y = innovation(z, measurement_model(self.x, self.landmark))
        S = H @ self.P @ H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))
        self.last_innovation, self.last_nis = y, nis

        # Optional innovation gating: reject measurements that are statistically implausible
        if self.gate is not None and nis > self.gate:
            self.n_rejected += 1
            return False

        K = np.linalg.solve(S, H @ self.P).T     # K = P H^T S^-1 (P and S are symmetric)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        A = np.eye(3) - K @ H
        P = A @ self.P @ A.T + K @ self.R @ K.T  # Joseph form
        self.P = 0.5 * (P + P.T)
        self.n_updates += 1
        return True