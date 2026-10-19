# Including libraries
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import chi2

from config import CONFIG, Config
from robot_model import (default_landmark, innovation, jacobian_F, jacobian_G,
                        lidar_jacobian, lidar_range_model, measurement_jacobian,
                        measurement_jacobian_pose, measurement_model, measurement_model_pose,
                        motion_model, wrap_angle, landmark_table)

# Normalised Estimation Error Squared: e^T P^-1 e.
# For a consistent 3-state filter its average is about 3 (chi-square, 3 dof).
# Needs ground truth, so use it for EVALUATION only, never inside the controller.
def nees(x_est, x_true, P) -> float:
    e = np.asarray(x_est, dtype=float) - np.asarray(x_true, dtype=float)
    e[2] = wrap_angle(e[2])
    return float(e @ np.linalg.solve(np.asarray(P, dtype=float), e))

# Two-sided chi-square acceptance interval for the AVERAGE of n NEES (or NIS) samples with `dof`
# degrees of freedom. A consistent filter should have its average inside [lo, hi].
def chi2_bounds(dof: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    return (float(chi2.ppf(alpha / 2.0, n * dof) / n),
            float(chi2.ppf(1.0 - alpha / 2.0, n * dof) / n))

class EKFEstimator:
    def __init__(self, cfg: Config = CONFIG, landmark=None, q_extra: Optional[np.ndarray] = None,
            odom_cov_scale: float = 1.0, R: Optional[np.ndarray] = None,
            gate: Optional[float] = None, use_marker: bool = True,
            use_view: bool = True, R_view: Optional[float] = None,
            use_lidar: bool = True, lidar_noise_scale: float = 2.0,
            lidar_gate: float = 6.63, lidar_stride: int = 30, max_rejects: int = 10,
            use_marker_cov: bool = False):
        self.cfg = cfg
        # Start with NO known markers: poses are learned at run time (TagMap)
        self.landmarks = {}
        self.landmark, self.dock_yaw = None, None
        self.q_extra = np.zeros((3, 3)) if q_extra is None else np.asarray(q_extra, float)
        self.R = cfg.R_camera() if R is None else np.asarray(R, dtype=float)
        # Variance of the marker view angle (third camera measurement)
        view_std = getattr(cfg.camera, "view_noise_std", 0.03)
        self.R_view = view_std ** 2 if R_view is None else float(R_view)
        self.gate = gate                    # NIS threshold for a 2-dof measurement (None = off)
        self.max_rejects = max_rejects      # consecutive rejections before the lock-out recovery
        self.use_marker = use_marker
        self.use_view = use_view
        self.use_marker_cov = use_marker_cov
        # LiDAR update against the known room map (walls and obstacles)
        self.use_lidar = use_lidar
        self.lidar_noise_scale = lidar_noise_scale   # inflates sigma_lidar to cover map errors
        self.lidar_gate = lidar_gate                 # chi-square 99 %, 1 dof
        self.lidar_stride = lidar_stride             # use every n-th beam (360 / 30 = 12 beams)

        # Odometry noise: each wheel angle has std sigma_e. With ds = r (dl + dr) / 2 and
        # dtheta = r (dr - dl) / b, error propagation gives the diagonal covariance below.
        rob = cfg.robot
        s_ds = rob.wheel_radius * rob.encoder_noise_std / math.sqrt(2.0)
        s_dth = rob.wheel_radius * rob.encoder_noise_std * math.sqrt(2.0) / rob.wheel_base
        self._odom_cov = odom_cov_scale * np.diag([s_ds ** 2, s_dth ** 2])
        self.reset((0.0, 0.0, 0.0))   # origin = where the robot starts

    # Initialise belief N(x0, P0) and clear the diagnostics
    def reset(self, x0, P0: Optional[np.ndarray] = None) -> None:
        self.x = np.array(x0, dtype=float)
        self.x[2] = wrap_angle(self.x[2])
        self.P = self.cfg.ekf.P0() if P0 is None else np.array(P0, dtype=float)
        self.last_innovation: Optional[np.ndarray] = None
        self.last_nis: Optional[float] = None
        self.nis_history: list = []       # NIS of every accepted marker update
        self.nis_dof: list = []           # degrees of freedom of each NIS (2 or 3)
        self.n_updates = 0
        self.n_tag_updates = 0            # accepted updates from extra tags (not the dock)
        self.n_rejected = 0
        self.n_recovered = 0              # forced updates after a gate lock-out
        self._reject_run = 0              # consecutive gate rejections
        self.n_lidar_updates = 0
        self.n_lidar_rejected = 0

    # EKF prediction: propagate the mean through f and the covariance through F and G
    def predict(self, ds: float, dtheta: float) -> None:
        F = jacobian_F(self.x, ds, dtheta)
        G = jacobian_G(self.x, ds, dtheta)
        P = F @ self.P @ F.T + G @ self._odom_cov @ G.T + self.q_extra
        self.P = 0.5 * (P + P.T)
        self.x = motion_model(self.x, ds, dtheta)

    # A usable marker has a known id and finite, positive range
    def _is_valid(self, marker) -> bool:
        values = (marker.range, marker.bearing)
        return (all(math.isfinite(v) for v in values) and marker.range > 0.0
                and int(getattr(marker, "marker_id", self.cfg.camera.marker_id)) in self.landmarks)

    # Gate threshold for a measurement with `dof` degrees of freedom. The configured gate is
    # defined for 2 dof; for other sizes keep the same confidence level (same chi-square quantile).
    def _gate_for(self, dof: int) -> Optional[float]:
        if self.gate is None:
            return None
        if dof == 2:
            return self.gate
        return float(chi2.ppf(chi2.cdf(self.gate, 2), dof))

    # Mean NIS divided by its dof, averaged over all accepted marker updates (consistent: ~1)
    def mean_nis_per_dof(self) -> float:
        if not self.nis_history:
            return float("nan")
        return float(np.mean(np.array(self.nis_history) / np.array(self.nis_dof)))

    # EKF correction with a marker measurement.
    # Uses [range, bearing, view_angle] when the marker provides a finite view angle (3 dof,
    # full pose observable from one frame), otherwise [range, bearing] (2 dof).
    # Returns True if the measurement was applied. With marker=None (lost marker) nothing
    # happens: the estimate keeps following odometry and P keeps growing.
    def update(self, marker) -> bool:
        if not self.use_marker or marker is None or not self._is_valid(marker):
            return False
        marker_id = int(getattr(marker, "marker_id", self.cfg.camera.marker_id))
        landmark, tag_yaw = self.landmarks[marker_id]

        view = getattr(marker, "view_angle", None)
        use_view = self.use_view and view is not None and math.isfinite(view)

        R_cam = self.R
        cov = getattr(marker, "cov", None) if self.use_marker_cov else None
        if cov is not None:
            cov = np.asarray(cov, dtype=float)
            if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                R_cam = cov
        try:
            if use_view:
                z = np.array([marker.range, marker.bearing, view], dtype=float)
                H = measurement_jacobian_pose(self.x, landmark)
                z_pred = measurement_model_pose(self.x, landmark, tag_yaw)
                R = np.zeros((3, 3))
                R[:2, :2] = R_cam
                R[2, 2] = self.R_view
            else:
                z = np.array([marker.range, marker.bearing], dtype=float)
                H = measurement_jacobian(self.x, landmark)
                z_pred = measurement_model(self.x, landmark)
                R = self.R
        except ValueError:                      # robot on top of the landmark: skip
            return False

        y = innovation(z, z_pred)
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        self.last_innovation, self.last_nis = y, nis

        # Optional innovation gating: reject measurements that are statistically implausible
        gate = self._gate_for(len(z))
        if gate is not None and nis > gate:
            self.n_rejected += 1
            self._reject_run += 1
            if self._reject_run < self.max_rejects:
                return False
            # Gate lock-out recovery: the filter has rejected every recent measurement, so it is
            # probably the estimate that is wrong. Inflate P until this measurement is consistent
            # with the gate, then accept it.
            self.P = self.P * (nis / gate)
            S = H @ self.P @ H.T + R
            nis = float(y @ np.linalg.solve(S, y))
            self.n_recovered += 1
        self._reject_run = 0

        K = np.linalg.solve(S, H @ self.P).T     # K = P H^T S^-1 (P and S are symmetric)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        A = np.eye(3) - K @ H
        P = A @ self.P @ A.T + K @ R @ K.T       # Joseph form
        self.P = 0.5 * (P + P.T)
        self.nis_history.append(nis)
        self.n_updates += 1
        self.nis_dof.append(len(z))
        if marker_id != int(self.cfg.camera.marker_id):
            self.n_tag_updates += 1
        return True

    # EKF correction with a few LiDAR beams compared against the known room map.
    # Each beam is a scalar update (sequential), gated so that beams hitting something that is
    # not in the map (the dock panel, a wrongly associated edge) are ignored.
    # `scan` needs the attributes ranges, angles (robot frame) and valid, as sensing.LidarScan.
    # Returns the number of beams applied.
    def update_lidar(self, scan) -> int:
        if not self.use_lidar or scan is None:
            return 0
        lid = self.cfg.lidar
        r_var = (self.lidar_noise_scale * lid.noise_std) ** 2
        applied = 0
        for i in range(0, len(scan.ranges), self.lidar_stride):
            if not scan.valid[i]:
                continue
            angle, z = float(scan.angles[i]), float(scan.ranges[i])
            z_pred = lidar_range_model(self.x, angle, self.cfg)
            if z_pred >= lid.range_max:          # the map predicts no hit: nothing to compare
                continue
            H = lidar_jacobian(self.x, angle, self.cfg).reshape(1, 3)
            y = z - z_pred
            S = (H @ self.P @ H.T).item() + r_var
            if y * y / S > self.lidar_gate:
                self.n_lidar_rejected += 1
                continue
            K = (self.P @ H.T) / S               # 3 x 1
            self.x = self.x + (K * y).ravel()
            self.x[2] = wrap_angle(self.x[2])
            A = np.eye(3) - K @ H
            P = A @ self.P @ A.T + (K * r_var) @ K.T   # Joseph form, scalar measurement
            self.P = 0.5 * (P + P.T)
            self.n_lidar_updates += 1
            applied += 1
        return applied
    
    # Position uncertainty: sqrt(trace of the xy block of P) [m]
    def position_std(self) -> float:
        return float(math.sqrt(max(float(np.trace(self.P[:2, :2])), 0.0)))

    # True when the position uncertainty is above cfg.tags.lost_std
    def is_lost(self) -> bool:
        return self.position_std() > self.cfg.tags.lost_std

    # Declare the pose uncertain (e.g. after a failed docking): widen P so that the next
    # tag detections are accepted instead of being gated out as outliers.
    def inflate(self, pos_std: float = 0.5, yaw_std: float = 0.5) -> None:
        self.P = np.diag([pos_std ** 2, pos_std ** 2, yaw_std ** 2])
        self._reject_run = 0

    # Replace the known marker poses with a learned TagMap. None = no known markers (never the config poses).
    def set_tag_map(self, tag_map=None) -> None:
        self.landmarks = {} if tag_map is None else landmark_table(self.cfg, tag_map)
        dock = self.landmarks.get(int(self.cfg.camera.marker_id))
        self.landmark, self.dock_yaw = dock if dock is not None else (None, None)