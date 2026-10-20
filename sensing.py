# Including libraries
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
from machinevisiontoolbox import Image

from config import CONFIG, Config
from calibration import camera_matrix, marker_pose_from_T
from robot_model import odometry_increment

@dataclass(frozen=True, eq=False)
class LidarScan:
    ranges: np.ndarray    # [m], one per beam
    angles: np.ndarray    # [rad], robot frame, counter-clockwise from +x
    valid: np.ndarray     # bool, True where the beam hit something in range
    range_max: float
    time: float

@dataclass(frozen=True)
class MarkerMeasurement:
    range: float          # [m] robot centre -> marker centre
    bearing: float        # [rad] direction of the marker in the robot frame (left = +)
    view_angle: float     # [rad] 0 when the marker faces the robot head-on
    marker_id: int
    time: float

    @property
    def z(self) -> np.ndarray:
        return np.array([self.range, self.bearing])

@dataclass(frozen=True)
class OdometryIncrement:
    ds: float
    dtheta: float
    time: float

@dataclass(frozen=True, eq=False)
class SensorFrame:
    time: float
    lidar: LidarScan
    odometry: OdometryIncrement
    marker: Optional[MarkerMeasurement]   # dock marker only (None = not seen)
    markers: tuple = ()                   # every accepted marker (dock + extra tags)
    image: Optional[np.ndarray] = None    # latest camera image the robot processed (RGB uint8)
    lidar_age: float = 0.0                # [s] age of the LiDAR data (0 in simulation)
    camera_age: float = 0.0               # [s] age of the camera data (0 in simulation)

# Detect every known marker with the Machine Vision Toolbox; one measurement per id.
def detect_markers(rgb, cfg: Config = CONFIG, time: float = 0.0,
                max_view_angle: float = math.radians(70.0), K=None) -> list[MarkerMeasurement]:
    first = int(cfg.camera.marker_id)
    known = range(first, first + int(cfg.tags.n_tags) + 1)
    name = cfg.camera.aruco_dict.replace("DICT_", "").lower()      # "4x4_50"
    out = []
    try:
        fiducials = Image(rgb, colororder="RGB").fiducial(
            name, K=camera_matrix(cfg) if K is None else K, side=cfg.camera.marker_size)
    except ValueError:                     # toolbox crashes when PnP returns NaN on a bad frame
        return []                          # treat it like a dropped frame
    out = []
    for f in fiducials:
        if int(f.id) not in known or f.pose is None or not np.all(np.isfinite(f.pose.A)):
            continue
        rng, brg, view = marker_pose_from_T(f.pose, cfg)
        if rng > 0.0 and abs(view) <= max_view_angle:              # skip grazing views
            out.append(MarkerMeasurement(rng, brg, view, int(f.id), float(time)))
    return out

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

# Closest valid hit inside a sector
def sector_min_range(scan: LidarScan, center: float = 0.0, half_width: float = 0.3) -> float:
    diff = np.abs((scan.angles - center + math.pi) % (2.0 * math.pi) - math.pi)
    mask = scan.valid & (diff <= half_width)
    return float(scan.ranges[mask].min()) if mask.any() else scan.range_max

class Sensing:
    """Reads a RobotEnvironment (MuJoCoEnvironment or HardwareEnvironment)."""

    def __init__(self, env, cfg: Config = CONFIG, camera_period: Optional[float] = 0.1,
                calibration=None):
        self.env, self.cfg, self.camera_period = env, cfg, camera_period
        self.calibration = calibration        # calibration.Calibration or None (= nominal MuJoCo)
        self.reset()

    def reset(self) -> None:
        self._prev_enc = np.asarray(self.env.read_encoders(), dtype=float)
        self._last_cam_time = -math.inf
        self._last_image = None
        self.n_camera = 0      # camera frames processed
        self.n_detect = 0      # frames with an accepted DOCK measurement
        self.n_detect_any = 0  # frames with at least one accepted marker

    def _camera_due(self, t: float) -> bool:
        return self.camera_period is not None and t - self._last_cam_time >= self.camera_period - 1e-9

    # Detect markers with the calibration (undistort first when it is a real calibration)
    def _detect(self, rgb: np.ndarray, t: float):
        cal = self.calibration
        if cal is None:
            return detect_markers(rgb, self.cfg, t)
        h, w = rgb.shape[:2]
        if (w, h) != tuple(int(v) for v in cal.intrinsics.image_size):
            raise ValueError(f"image {(w, h)} does not match the calibration resolution "
                            f"{tuple(cal.intrinsics.image_size)}")
        if cal.intrinsics.is_calibrated:
            rgb = cv2.undistort(rgb, cal.K, cal.dist_coeffs)
        return detect_markers(rgb, self.cfg, t, K=cal.K)

    def read(self) -> SensorFrame:
        env = self.env
        t = float(env.time)
        enc = np.asarray(env.read_encoders(), dtype=float)
        ds, dtheta = odometry_increment(self._prev_enc, enc, self.cfg)
        self._prev_enc = enc
        lidar = make_lidar_scan(env.read_lidar(), env.lidar_angles, self.cfg, t)

        marker, markers = None, ()
        if self._camera_due(t):
            self._last_cam_time = t
            rgb = env.read_camera()
            self._last_image = rgb
            found = self._detect(rgb, t)
            markers = tuple(found)
            marker = next((m for m in found if m.marker_id == self.cfg.camera.marker_id), None)
            self.n_camera += 1
            self.n_detect += int(marker is not None)
            self.n_detect_any += int(len(found) > 0)

        ages = env.sensor_ages() if hasattr(env, "sensor_ages") else (0.0, 0.0)
        return SensorFrame(t, lidar, OdometryIncrement(ds, dtheta, t), marker, markers,
                        self._last_image, float(ages[0]), float(ages[1]))

if __name__ == "__main__":
    from simulation import Simulation
    with Simulation() as sim:
        dx, dy, _ = CONFIG.env.dock_pose
        sim.reset((dx - 0.7, dy - 0.2, 0.0))
        frame = Sensing(sim, camera_period=0.0).read()
        print("LiDAR hits :", int(frame.lidar.valid.sum()), "/", len(frame.lidar.ranges))
        print("Front free :", round(sector_min_range(frame.lidar), 2), "m")
        print("Marker     :", frame.marker)
