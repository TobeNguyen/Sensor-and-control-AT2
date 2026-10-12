# Including libraries
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from config import CONFIG, Config
from calibration import (camera_matrix, camera_forward_offset,
                        solve_marker_pose, marker_pose_from_pnp)
from robot_model import odometry_increment
from simulation import Simulation

@dataclass(frozen=True, eq=False)
class LidarScan:
    ranges: np.ndarray    # [m], one per beam
    angles: np.ndarray    # [rad], robot frame, counter-clockwise from +x
    valid: np.ndarray     # bool, True where the beam hit something in range
    range_max: float
    time: float

@dataclass(frozen=True)
class MarkerMeasurement:
    range: float          # [m] distance from the robot centre to the marker centre
    bearing: float        # [rad] direction of the marker in the robot frame (left = +)
    view_angle: float     # [rad] 0 when the marker faces the robot head-on
    marker_id: int
    time: float

    @property
    def z(self) -> np.ndarray:
        return np.array([self.range, self.bearing])
    
@dataclass(frozen=True)
class OdometryIncrement:
    ds: float             # [m] distance travelled since the previous read
    dtheta: float         # [rad] heading change since the previous read
    time: float

@dataclass(frozen=True, eq=False)
class SensorFrame:
    time: float
    lidar: LidarScan
    odometry: OdometryIncrement
    marker: Optional[MarkerMeasurement]   # None = no fresh marker measurement

@lru_cache(maxsize=None)
def _detector(dict_name: str) -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# Detect the dock marker and return range, bearing and view angle in the robot frame.
# Camera parameters and frame conversions come from calibration.py (single source of truth).
def detect_marker(image_rgb: np.ndarray,
                K: Optional[np.ndarray] = None,
                dist_coeffs: Optional[np.ndarray] = None,
                cfg: Config = CONFIG,
                time: float = 0.0) -> Optional[MarkerMeasurement]:
    K = camera_matrix(cfg) if K is None else K
    dist_coeffs = np.zeros(5) if dist_coeffs is None else dist_coeffs

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector(cfg.camera.aruco_dict).detectMarkers(gray)
    if ids is None:
        return None

    candidates = [c for c, i in zip(corners, ids.flatten()) if i == cfg.camera.marker_id]
    if not candidates:
        return None
    # If the marker appears more than once, keep the largest detection.
    image_pts = max(candidates, key=lambda c: cv2.contourArea(c.reshape(4, 2))).reshape(4, 2)

    # Pose in the camera frame; rejects the flipped IPPE solution (marker facing away).
    sol = solve_marker_pose(image_pts, K, dist_coeffs, cfg.camera.marker_size)
    if sol is None:
        return None
    rvec, tvec, _ = sol

    # Convert to the robot frame (camera offset and axis conventions live in calibration.py).
    rng, bearing, view_angle = marker_pose_from_pnp(rvec, tvec, cfg)
    return MarkerMeasurement(range=rng,
                            bearing=bearing,
                            view_angle=view_angle,
                            marker_id=int(cfg.camera.marker_id),
                            time=float(time))

# Build a LidarScan and flag the beams that returned a usable hit
def make_lidar_scan(ranges, angles, cfg: Config = CONFIG, time: float = 0.0) -> LidarScan:
    ranges = np.asarray(ranges, dtype=float).copy()
    angles = np.asarray(angles, dtype=float).copy()
    if ranges.shape != angles.shape or ranges.ndim != 1:
        raise ValueError("ranges and angles must be 1-D arrays of the same length")
    lid = cfg.lidar
    valid = np.isfinite(ranges) & (ranges >= lid.range_min) & (ranges < lid.range_max)
    return LidarScan(ranges, angles, valid, float(lid.range_max), float(time))

# Valid hits as (N, 2) points in the robot frame
def lidar_points(scan: LidarScan) -> np.ndarray:
    r, a = scan.ranges[scan.valid], scan.angles[scan.valid]
    return np.column_stack([r * np.cos(a), r * np.sin(a)])

# Closest valid hit inside
def sector_min_range(scan: LidarScan, center: float = 0.0, half_width: float = 0.3) -> float:
    diff = np.abs((scan.angles - center + math.pi) % (2.0 * math.pi) - math.pi)
    mask = scan.valid & (diff <= half_width)
    return float(scan.ranges[mask].min()) if mask.any() else scan.range_max

class Sensing:
    def __init__(self, sim, cfg: Config = CONFIG,
                camera_matrix_: Optional[np.ndarray] = None,
                dist_coeffs: Optional[np.ndarray] = None,
                camera_period: Optional[float] = 0.1):
        self.sim = sim
        self.cfg = cfg
        self.K = camera_matrix(cfg) if camera_matrix_ is None else camera_matrix_
        self.dist_coeffs = np.zeros(5) if dist_coeffs is None else dist_coeffs
        self.camera_period = camera_period
        self.reset()

    def reset(self) -> None:
        self._prev_enc = np.asarray(self.sim.read_encoders(), dtype=float)
        self._last_cam_time = -math.inf

    def _camera_due(self, t: float) -> bool:
        if self.camera_period is None:
            return False
        return t - self._last_cam_time >= self.camera_period - 1e-9

    def read(self) -> SensorFrame:
        t = float(self.sim.time)

        enc = np.asarray(self.sim.read_encoders(), dtype=float)
        ds, dtheta = odometry_increment(self._prev_enc, enc, self.cfg)
        self._prev_enc = enc

        lidar = make_lidar_scan(self.sim.read_lidar(), self.sim.lidar_angles, self.cfg, t)

        marker = None
        if self._camera_due(t):
            self._last_cam_time = t
            marker = detect_marker(self.sim.read_camera(), self.K, self.dist_coeffs,
                                self.cfg, t)
        return SensorFrame(t, lidar, OdometryIncrement(ds, dtheta, t), marker)

if __name__ == "__main__":
    with Simulation() as sim:
        dx, dy, _ = CONFIG.env.dock_pose
        sim.reset((dx - 0.7, dy - 0.2, 0.0))
        frame = Sensing(sim, camera_period=0.0).read()
        print("LiDAR hits :", int(frame.lidar.valid.sum()), "/", len(frame.lidar.ranges))
        print("Front free :", round(sector_min_range(frame.lidar), 2), "m")
        print("Marker     :", frame.marker)
