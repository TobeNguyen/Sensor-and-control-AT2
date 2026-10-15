# Including libraries
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from config import CONFIG, Config
from calibration import (camera_matrix, camera_forward_offset,
                        solve_marker_pose, marker_pose_from_pnp,
                        marker_measurement_covariance, CameraIntrinsics,
                        SOURCE_NOMINAL, MAX_REPROJ_PX)
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
    cov: Optional[np.ndarray] = field(default=None, compare=False)   # 2x2 R for [range, bearing]
    reproj_error: float = field(default=float("nan"), compare=False) # RMS corner error [px]

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
    marker: Optional[MarkerMeasurement]   # dock marker only (None = not seen)
    markers: tuple = ()                   # every accepted marker (dock + extra tags) in this frame

@lru_cache(maxsize=None)
def _detector(dict_name: str) -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# Detect every known marker (dock + extra tags) and return one measurement per id.
def detect_markers(image_rgb: np.ndarray,
                K: Optional[np.ndarray] = None,
                dist_coeffs: Optional[np.ndarray] = None,
                cfg: Config = CONFIG,
                time: float = 0.0,
                sigma_px: float = 0.5,
                max_reproj_px: float = MAX_REPROJ_PX,
                max_view_angle: float = math.radians(70.0)) -> list[MarkerMeasurement]:
    K = camera_matrix(cfg) if K is None else K
    dist_coeffs = np.zeros(5) if dist_coeffs is None else dist_coeffs

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector(cfg.camera.aruco_dict).detectMarkers(gray)
    if ids is None:
        return []

    # Keep the largest detection of every known id.
    known = cfg.tag_table()
    best: dict = {}
    for c, i in zip(corners, ids.flatten()):
        i = int(i)
        if i not in known:
            continue
        pts = c.reshape(4, 2)
        if i not in best or cv2.contourArea(pts) > cv2.contourArea(best[i]):
            best[i] = pts

    intr = CameraIntrinsics(np.asarray(K, dtype=float), np.asarray(dist_coeffs, dtype=float),
                            (cfg.camera.width, cfg.camera.height), SOURCE_NOMINAL)
    out = []
    for marker_id, image_pts in sorted(best.items()):
        sol = solve_marker_pose(image_pts, K, dist_coeffs, cfg.camera.marker_size,
                                max_reproj_px=max_reproj_px)
        if sol is None:
            continue
        rvec, tvec, reproj = sol
        rng, bearing, view_angle = marker_pose_from_pnp(rvec, tvec, cfg)
        if abs(view_angle) > max_view_angle:      # grazing view: pose is unreliable
            continue
        cov = marker_measurement_covariance(rng, intr, cfg.camera.marker_size, sigma_px, cfg)
        out.append(MarkerMeasurement(range=rng, bearing=bearing, view_angle=view_angle,
                                    marker_id=marker_id, time=float(time),
                                    cov=cov, reproj_error=float(reproj)))
    return out

# Dock marker only (id = cfg.camera.marker_id). Same behaviour as before, kept for docking and tests.
def detect_marker(image_rgb: np.ndarray,
                K: Optional[np.ndarray] = None,
                dist_coeffs: Optional[np.ndarray] = None,
                cfg: Config = CONFIG,
                time: float = 0.0,
                sigma_px: float = 0.5,
                max_reproj_px: float = MAX_REPROJ_PX,
                max_view_angle: float = math.radians(70.0)) -> Optional[MarkerMeasurement]:
    for m in detect_markers(image_rgb, K, dist_coeffs, cfg, time, sigma_px,
                            max_reproj_px, max_view_angle):
        if m.marker_id == cfg.camera.marker_id:
            return m
    return None

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
                camera_period: Optional[float] = 0.1,
                marker_sigma_px: float = 0.5,
                max_reproj_px: float = MAX_REPROJ_PX,
                max_view_angle: float = math.radians(70.0)):
        self.sim = sim
        self.cfg = cfg
        self.K = camera_matrix(cfg) if camera_matrix_ is None else camera_matrix_
        self.dist_coeffs = np.zeros(5) if dist_coeffs is None else dist_coeffs
        self.camera_period = camera_period
        self.marker_sigma_px = marker_sigma_px
        self.max_reproj_px = max_reproj_px
        self.max_view_angle = max_view_angle
        self.reset()

    def reset(self) -> None:
        self._prev_enc = np.asarray(self.sim.read_encoders(), dtype=float)
        self._last_cam_time = -math.inf
        self.n_camera = 0      # camera frames processed
        self.n_detect = 0      # frames with an accepted DOCK measurement
        self.n_detect_any = 0  # frames with at least one accepted marker (dock or tag)

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

        marker, markers = None, ()
        if self._camera_due(t):
            self._last_cam_time = t
            found = detect_markers(self.sim.read_camera(), self.K, self.dist_coeffs,
                                self.cfg, t, sigma_px=self.marker_sigma_px,
                                max_reproj_px=self.max_reproj_px,
                                max_view_angle=self.max_view_angle)
            markers = tuple(found)
            marker = next((m for m in found if m.marker_id == self.cfg.camera.marker_id), None)
            self.n_camera += 1
            self.n_detect += int(marker is not None)
            self.n_detect_any += int(len(found) > 0)
        return SensorFrame(t, lidar, OdometryIncrement(ds, dtheta, t), marker, markers)

if __name__ == "__main__":
    with Simulation() as sim:
        dx, dy, _ = CONFIG.env.dock_pose
        sim.reset((dx - 0.7, dy - 0.2, 0.0))
        frame = Sensing(sim, camera_period=0.0).read()
        print("LiDAR hits :", int(frame.lidar.valid.sum()), "/", len(frame.lidar.ranges))
        print("Front free :", round(sector_min_range(frame.lidar), 2), "m")
        print("Marker     :", frame.marker)
