# Including libraries
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from config import CONFIG, Config, RESULTS_DIR
from robot_model import wrap_angle

SOURCE_NOMINAL = "nominal_mujoco"   # derived from the known simulator model (NOT measured)
SOURCE_CALIBRATED = "calibrated"    # estimated from images of a calibration target
MIN_CALIBRATION_VIEWS = 5
VALID_DIST_LENGTHS = (4, 5, 8, 12, 14)
MAX_REPROJ_PX = 3.0                 # default reprojection-error gate for marker pose [px]
MAX_CALIB_RMS_PX = 1.0              # max accepted RMS reprojection error of a real calibration [px]

# Intrinsics
@dataclass(frozen=True, eq=False)
class CameraIntrinsics:
    K: np.ndarray                              # (3, 3) pinhole matrix [px]
    dist_coeffs: np.ndarray                    # OpenCV distortion vector
    image_size: tuple                          # (width, height) [px]
    source: str                                # SOURCE_NOMINAL | SOURCE_CALIBRATED
    reprojection_error: Optional[float] = None  # RMS [px]; None for nominal models

    @property
    def is_calibrated(self) -> bool:
        return self.source == SOURCE_CALIBRATED
    
# Pinhole matrix implied by the simulator's vertical field of view
def camera_matrix(cfg: Config = CONFIG) -> np.ndarray:
    cam = cfg.camera
    f = (cam.height / 2.0) / math.tan(math.radians(cam.fovy_deg) / 2.0)
    return np.array([[f, 0.0, cam.width / 2.0],
                    [0.0, f, cam.height / 2.0],
                    [0.0, 0.0, 1.0]])

# Camera position ahead of the robot centre [m]: measured value if configured, nominal otherwise
def camera_forward_offset(cfg: Config = CONFIG) -> float:
    measured = getattr(cfg.camera, "forward_offset", None)
    return cfg.robot.body_radius + 0.01 if measured is None else float(measured)

# Measured camera mounting in the robot frame: (forward [m], lateral-left [m], yaw [rad]).
# Values missing from the config fall back to the nominal simulator mounting (0 lateral, 0 yaw)
def camera_extrinsics(cfg: Config = CONFIG) -> tuple[float, float, float]:
    lateral = getattr(cfg.camera, "lateral_offset", None)
    yaw = getattr(cfg.camera, "yaw", None)
    return (camera_forward_offset(cfg),
            0.0 if lateral is None else float(lateral),
            0.0 if yaw is None else float(yaw))

# 2-D rotation matrix
def _rot2(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])

# Known (not measured) intrinsics of the simulated camera: ideal pinhole, no distortion
def nominal_intrinsics(cfg: Config = CONFIG) -> CameraIntrinsics:
    return CameraIntrinsics(camera_matrix(cfg), np.zeros(5),
                            (cfg.camera.width, cfg.camera.height), SOURCE_NOMINAL, None)

# Return a list of problems (empty list = intrinsics look sane)
def validate_intrinsics(intr: CameraIntrinsics) -> list[str]:
    problems = []
    K = np.asarray(intr.K, dtype=float)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return ["K must be a finite 3x3 matrix"]
    if not np.allclose(K[2], [0.0, 0.0, 1.0]):
        problems.append("last row of K must be [0, 0, 1]")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        problems.append("focal lengths must be positive")
    elif abs(K[0, 0] / K[1, 1] - 1.0) > 0.05:
        problems.append("fx and fy differ by more than 5 % (non-square pixels?)")
    if abs(K[0, 1]) > 1.0:
        problems.append("skew is unexpectedly large")
    w, h = intr.image_size
    if not (0.0 < K[0, 2] < w and 0.0 < K[1, 2] < h):
        problems.append("principal point lies outside the image")
    d = np.asarray(intr.dist_coeffs, dtype=float).ravel()
    if len(d) not in VALID_DIST_LENGTHS or not np.all(np.isfinite(d)):
        problems.append(f"distortion vector must be finite with length in {VALID_DIST_LENGTHS}")
    if intr.source not in (SOURCE_NOMINAL, SOURCE_CALIBRATED):
        problems.append(f"unknown source '{intr.source}'")
    return problems

def assert_valid_intrinsics(intr: CameraIntrinsics) -> None:
    problems = validate_intrinsics(intr)
    if problems:
        raise ValueError("Invalid intrinsics: " + "; ".join(problems))

# The runtime image resolution must match the resolution used for calibration
def check_resolution(intr: CameraIntrinsics, cfg: Config = CONFIG) -> list[str]:
    expected = (int(cfg.camera.width), int(cfg.camera.height))
    actual = tuple(int(v) for v in intr.image_size)
    if actual != expected:
        return [f"calibration resolution {actual} != runtime resolution {expected}"]
    return []
    
# Frame transforms: camera to robot
# Camera-frame point (X right, Y down, Z forward) -> robot ground-plane point (x, y)
def camera_to_robot(p_cam, cfg: Config = CONFIG) -> np.ndarray:
    X, _, Z = np.asarray(p_cam, dtype=float).reshape(3)
    fwd, lat, yaw = camera_extrinsics(cfg)
    return _rot2(yaw) @ np.array([Z, -X]) + np.array([fwd, lat])

# Robot ground-plane point -> camera frame. `height` is the point height above the camera axis [m]
def robot_to_camera(p_robot, height: float = 0.0, cfg: Config = CONFIG) -> np.ndarray:
    fwd, lat, yaw = camera_extrinsics(cfg)
    q = _rot2(-yaw) @ (np.asarray(p_robot, dtype=float).reshape(2) - np.array([fwd, lat]))
    return np.array([-q[1], -height, q[0]])

# Direction vector (no translation): camera frame -> robot ground plane
def camera_dir_to_robot(d_cam, cfg: Config = CONFIG) -> np.ndarray:
    d = np.asarray(d_cam, dtype=float).reshape(3)
    return _rot2(camera_extrinsics(cfg)[2]) @ np.array([d[2], -d[0]])

def range_bearing(p_robot) -> tuple[float, float]:
    x, y = np.asarray(p_robot, dtype=float).reshape(2)
    return math.hypot(x, y), math.atan2(y, x)

def xy_from_range_bearing(rng: float, bearing: float) -> np.ndarray:
    return np.array([rng * math.cos(bearing), rng * math.sin(bearing)])

# Square marker corners in the marker frame (same order as the ArUco detector output)
def marker_object_points(marker_size: float) -> np.ndarray:
    s = marker_size / 2.0
    return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])

# Marker pose (translation t, rotation R in the camera frame) -> (range, bearing, view_angle)
def _marker_pose(t, R, cfg: Config = CONFIG) -> tuple[float, float, float]:
    p_robot = camera_to_robot(t, cfg)
    rng, bearing = range_bearing(p_robot)
    n = camera_dir_to_robot(np.asarray(R, dtype=float)[:, 2], cfg)   # normal points towards the viewer
    view_angle = wrap_angle(math.atan2(n[1], n[0]) - math.pi)
    return rng, bearing, float(view_angle)

# Convert a solvePnP result (marker in camera frame) into (range, bearing, view_angle)
def marker_pose_from_pnp(rvec, tvec, cfg: Config = CONFIG) -> tuple[float, float, float]:
    R = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    return _marker_pose(np.asarray(tvec, dtype=float).reshape(3), R, cfg)

# Convert an SE3-like pose (attributes .t and .R, marker in the camera frame) into (range, bearing, view_angle)
def marker_pose_from_T(T, cfg: Config = CONFIG) -> tuple[float, float, float]:
    return _marker_pose(np.asarray(T.t, dtype=float).reshape(3), T.R, cfg)

# Marker pose in the camera frame from the 4 detected corners
def solve_marker_pose(image_pts, K: np.ndarray, dist_coeffs: np.ndarray,
                    marker_size: float, max_reproj_px: float = MAX_REPROJ_PX):
    obj = marker_object_points(marker_size)
    img = np.asarray(image_pts, dtype=np.float64).reshape(4, 2)
    dist = np.asarray(dist_coeffs, dtype=float)

    candidates = []
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates += list(zip(rvecs[:n], tvecs[:n]))
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        candidates.append((rv, tv))

    best = None
    for rv, tv in candidates:
        tv = np.asarray(tv, dtype=float).reshape(3)
        normal_z = cv2.Rodrigues(np.asarray(rv, dtype=float).reshape(3, 1))[0][2, 2]
        if tv[2] <= 0.0 or normal_z >= 0.0:
            continue

        # Reject candidates whose reprojection error exceeds the gate
        is_valid, err = validate_estimated_pose(img, rv, tv.reshape(3, 1), K, dist,
                                                marker_size, max_error_px=max_reproj_px)
        if not is_valid:
            continue
        if best is None or err < best[2]:
            best = (np.asarray(rv, dtype=float).reshape(3, 1), tv.reshape(3, 1), err)
    return best

# Validate the estimated pose by computing the reprojection error
def validate_estimated_pose(image_pts, rvec, tvec, K, dist_coeffs, marker_size,
                            max_error_px: float = MAX_REPROJ_PX) -> tuple[bool, float]:
    """Return (is_valid, RMS reprojection error in pixels)."""
    if image_pts is None or len(image_pts) != 4:
        return False, float('inf')

    obj_points = marker_object_points(marker_size)
    projected_pts, _ = cv2.projectPoints(obj_points, np.asarray(rvec, dtype=float).reshape(3, 1),
                                        np.asarray(tvec, dtype=float).reshape(3, 1), K, dist_coeffs)
    projected_pts = projected_pts.reshape(-1, 2)

    measured_pts = np.asarray(image_pts, dtype=np.float64).reshape(4, 2)
    errors = np.linalg.norm(measured_pts - projected_pts, axis=1)
    reproj_err = float(np.sqrt(np.mean(errors ** 2)))
    return reproj_err <= max_error_px, reproj_err

# Pixel (u, v) of a robot-frame point, or None if it is behind the camera
def project_robot_point(p_robot, intr: CameraIntrinsics, height: float = 0.0,
                        cfg: Config = CONFIG) -> Optional[np.ndarray]:
    pc = robot_to_camera(p_robot, height, cfg)
    if pc[2] <= 1e-9:
        return None
    zero = np.zeros(3)
    px, _ = cv2.projectPoints(pc.reshape(1, 1, 3), zero, zero, intr.K,
                            np.asarray(intr.dist_coeffs, dtype=float))
    return px.reshape(2)

# Frame transform: Robot to world (SE2)
def pose_matrix(pose) -> np.ndarray:
    x, y, th = np.asarray(pose, dtype=float).reshape(3)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

def robot_to_world(pose, p_robot) -> np.ndarray:
    p = np.append(np.asarray(p_robot, dtype=float).reshape(2), 1.0)
    return (pose_matrix(pose) @ p)[:2]

def world_to_robot(pose, p_world) -> np.ndarray:
    p = np.append(np.asarray(p_world, dtype=float).reshape(2), 1.0)
    return (np.linalg.inv(pose_matrix(pose)) @ p)[:2]

# World position of a marker seen at (range, bearing) from a robot at `pose`
def marker_world_position(pose, rng: float, bearing: float) -> np.ndarray:
    return robot_to_world(pose, xy_from_range_bearing(rng, bearing))

# Measurement noise
# Covariance R of the marker measurement [range, bearing] for the EKF.
# First-order model: corner noise sigma_px -> depth error ~ Z^2 * sigma_px / (f * marker_size),
# bearing error ~ sigma_px / (2 f) (centroid of 4 corners). Verify/tune with Monte-Carlo trials.
def marker_measurement_covariance(rng: float, intr: CameraIntrinsics, marker_size: float,
                                sigma_px: float = 0.5, cfg: Config = CONFIG) -> np.ndarray:
    f = float(intr.K[0, 0])
    z = max(rng - camera_forward_offset(cfg), 1e-3)
    sigma_r = z * z * sigma_px / (f * marker_size)
    sigma_b = sigma_px / (2.0 * f)
    return np.diag([sigma_r ** 2, sigma_b ** 2])

# LiDAR scan (ranges, angles in the robot frame) -> valid points (N, 2) in the robot frame.
# Defaults follow the TurtleBot3 LDS-01 (0.12 - 3.5 m). `offset` = LiDAR position in the robot frame [m].
def lidar_to_robot(ranges, angles, r_min: float = 0.12, r_max: float = 3.5, offset=(0.0, 0.0)):
    r = np.asarray(ranges, dtype=float).ravel()
    a = np.asarray(angles, dtype=float).ravel()
    ok = np.isfinite(r) & (r >= r_min) & (r <= r_max)
    pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1) + np.asarray(offset, dtype=float)
    return pts, ok

# Consistency checks
def check_units(cfg: Config = CONFIG) -> list[str]:
    problems = []
    if not 0.01 <= cfg.camera.marker_size <= 1.0:
        problems.append("marker_size outside 0.01-1.0 m (millimetres instead of metres?)")
    if not 10.0 <= cfg.camera.fovy_deg <= 170.0:
        problems.append("fovy_deg outside 10-170 deg (radians instead of degrees?)")
    fwd, lat, yaw = camera_extrinsics(cfg)
    if not 0.0 < fwd < 0.5:
        problems.append("camera forward offset outside 0-0.5 m")
    if abs(lat) > 0.5:
        problems.append("camera lateral offset outside +-0.5 m")
    if abs(yaw) > math.pi / 2.0:
        problems.append("camera yaw outside +-90 deg (degrees instead of radians?)")
    return problems

# Return a list of violated conventions (empty list = all consistent)
def check_frame_conventions(cfg: Config = CONFIG, intr: Optional[CameraIntrinsics] = None,
                            tol: float = 1e-6) -> list[str]:
    intr = nominal_intrinsics(cfg) if intr is None else intr
    problems = check_units(cfg)
    cx, cy = intr.K[0, 2], intr.K[1, 2]
    fwd, lat, yaw = camera_extrinsics(cfg)
    origin = np.array([fwd, lat])
    axis = np.array([math.cos(yaw), math.sin(yaw)])       # camera optical axis on the ground plane
    left_dir = np.array([-axis[1], axis[0]])

    ahead = project_robot_point(origin + 1.0 * axis, intr, 0.0, cfg)
    if ahead is None or abs(ahead[0] - cx) > 1e-3 or abs(ahead[1] - cy) > 1e-3:
        problems.append("a point straight ahead of the camera must project to the principal point")
    left = project_robot_point(origin + 1.0 * axis + 0.2 * left_dir, intr, 0.0, cfg)
    if left is None or left[0] >= cx:
        problems.append("a point on the robot's left (y>0) must appear left of the image centre")
    above = project_robot_point(origin + 1.0 * axis, intr, 0.1, cfg)
    if above is None or above[1] >= cy:
        problems.append("a point above the camera axis must appear above the image centre")

    p = np.array([1.3, -0.4])
    if np.linalg.norm(camera_to_robot(robot_to_camera(p, 0.0, cfg), cfg) - p) > tol:
        problems.append("robot->camera->robot round trip is not the identity")

    # Head-on marker: OpenCV pose with the marker y axis flipped (rotation pi about X).
    # Only meaningful when the camera is mounted on the robot axis and looks straight ahead.
    if abs(lat) < tol and abs(yaw) < tol:
        rng, bearing, view = marker_pose_from_pnp(np.array([math.pi, 0.0, 0.0]),
                                                np.array([0.0, 0.0, 1.0]), cfg)
        if abs(bearing) > tol or abs(view) > tol or abs(rng - (1.0 + fwd)) > tol:
            problems.append("head-on marker must give bearing 0, view angle 0 and range Z + offset")

    half_v = math.atan(cy / intr.K[1, 1])
    if abs(half_v - math.radians(cfg.camera.fovy_deg) / 2.0) > 1e-6 and not intr.is_calibrated:
        problems.append("nominal K does not reproduce the configured vertical field of view")
    return problems

def assert_frame_conventions(cfg: Config = CONFIG, intr: Optional[CameraIntrinsics] = None) -> None:
    problems = check_frame_conventions(cfg, intr)
    if problems:
        raise ValueError("Frame convention check failed: " + "; ".join(problems))
    
# Basic chessboard intrinsic calibration
# 3-D corner grid of a chessboard with `pattern_size` = (inner cols, inner rows)
def chessboard_object_points(pattern_size: tuple, square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return grid

# Sub-pixel chessboard corners (N, 2), or None if the board is not found
def find_chessboard_corners(image: np.ndarray, pattern_size: tuple) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)

# Estimate K and distortion from several views of a planar target.
# Default flags keep tangential distortion and k3 at zero, which is stable with few views.
def calibrate_intrinsics(object_points: Sequence[np.ndarray], image_points: Sequence[np.ndarray],
                        image_size: tuple,
                        flags: int = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
                        ) -> CameraIntrinsics:
    if len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must have the same number of views")
    if len(object_points) < MIN_CALIBRATION_VIEWS:
        raise ValueError(f"need at least {MIN_CALIBRATION_VIEWS} views, got {len(object_points)}")
    obj = [np.asarray(o, dtype=np.float32).reshape(-1, 3) for o in object_points]
    img = [np.asarray(i, dtype=np.float32).reshape(-1, 2) for i in image_points]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None, None, flags=flags)
    intr = CameraIntrinsics(K, dist.ravel(), tuple(image_size), SOURCE_CALIBRATED, float(rms))
    assert_valid_intrinsics(intr)
    return intr

# Convenience wrapper: detect the chessboard in every image, then calibrate
def calibrate_from_images(images: Sequence[np.ndarray], pattern_size: tuple,
                        square_size: float, drop_worst: int = 8,
                        flags: int = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3) -> CameraIntrinsics:
    sizes = {img.shape[:2] for img in images}
    if len(sizes) != 1:
        raise ValueError(f"all calibration images must have the same resolution, got {sizes}")
    template = chessboard_object_points(pattern_size, square_size)
    obj, img = [], []
    for image in images:
        corners = find_chessboard_corners(image, pattern_size)
        if corners is not None:
            obj.append(template)
            img.append(corners)
    if len(obj) < MIN_CALIBRATION_VIEWS:
        raise ValueError(f"chessboard found in only {len(obj)} of {len(images)} images "
                        f"(need {MIN_CALIBRATION_VIEWS})")
    h, w = images[0].shape[:2]
    intr = calibrate_intrinsics(obj, img, (w, h), flags)
    for _ in range(drop_worst):
        if len(obj) <= MIN_CALIBRATION_VIEWS + 2 or (intr.reprojection_error or 0.0) <= MAX_CALIB_RMS_PX:
            break
        errs = []
        for o, i in zip(obj, img):
            ok, rv, tv = cv2.solvePnP(o, i, intr.K, intr.dist_coeffs)
            proj, _ = cv2.projectPoints(o, rv, tv, intr.K, intr.dist_coeffs)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i) ** 2, axis=1)))))
        worst = int(np.argmax(errs))
        print(f"[calib] dropping view {worst} (RMS {errs[worst]:.2f} px)", flush=True)
        del obj[worst], img[worst]
        intr = calibrate_intrinsics(obj, img, (w, h), flags)
    return intr

# Persistence
def save_intrinsics(intr: CameraIntrinsics, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "K": np.asarray(intr.K).tolist(),
        "dist_coeffs": np.asarray(intr.dist_coeffs).ravel().tolist(),
        "image_size": list(intr.image_size),
        "source": intr.source,
        "reprojection_error": intr.reprojection_error,
    }, indent=2))
    return path

def load_intrinsics(path) -> CameraIntrinsics:
    d = json.loads(Path(path).read_text())
    intr = CameraIntrinsics(np.array(d["K"], dtype=float), np.array(d["dist_coeffs"], dtype=float),
                            tuple(d["image_size"]), d["source"], d.get("reprojection_error"))
    assert_valid_intrinsics(intr)
    return intr

@dataclass(frozen=True, eq=False)
class Calibration:
    intrinsics: CameraIntrinsics
    forward_offset: float
    marker_size: float
    lateral_offset: float = 0.0     # camera offset to the robot's left [m]
    yaw: float = 0.0                # camera yaw relative to the robot heading [rad]

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.K

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.intrinsics.dist_coeffs, dtype=float)

    @property
    def marker_points(self) -> np.ndarray:
        return marker_object_points(self.marker_size)

# Nominal MuJoCo intrinsics unless a calibrated set is supplied.
# With require_calibrated=True (hardware runs) a real, saved calibration is mandatory.
def get_calibration(cfg: Config = CONFIG, intrinsics: Optional[CameraIntrinsics] = None,
                    sample_image: Optional[np.ndarray] = None,
                    require_calibrated: bool = False) -> Calibration:
    """
    Provide a Calibration object with the camera intrinsics, camera extrinsics, and marker size.
    If no intrinsics are provided, the nominal MuJoCo intrinsics are used (a warning is issued).
    If require_calibrated is True, nominal intrinsics are rejected and the reprojection error is gated.
    If a sample image is provided, the effectiveness of lens distortion removal is validated and reported.
    """
    if intrinsics is None:
        if require_calibrated:
            raise ValueError("A calibrated intrinsics file is required; refusing to use nominal_mujoco")
        warnings.warn("Using nominal MuJoCo intrinsics (NOT measured); do not use on hardware")
    intr = nominal_intrinsics(cfg) if intrinsics is None else intrinsics
    assert_valid_intrinsics(intr)

    problems = check_resolution(intr, cfg)
    if require_calibrated:
        if not intr.is_calibrated:
            problems.append("intrinsics are not from a real calibration")
        elif intr.reprojection_error is None or intr.reprojection_error > MAX_CALIB_RMS_PX:
            problems.append(f"calibration RMS error {intr.reprojection_error} px exceeds {MAX_CALIB_RMS_PX} px")
    if problems:
        raise ValueError("Calibration rejected: " + "; ".join(problems))

    assert_frame_conventions(cfg, intr)
    fwd, lat, yaw = camera_extrinsics(cfg)
    calib_obj = Calibration(intr, fwd, cfg.camera.marker_size, lat, yaw)
    
    # Validate the effectiveness of lens distortion removal if a sample image is provided
    if sample_image is not None and intr.is_calibrated:
        val_res = validate_calibration_image_effect(sample_image, calib_obj.K, calib_obj.dist_coeffs)
        if val_res["status"] == "success":
            print(f"[Calibration Check] Undistort check passed. Mean pixel shift: {val_res['mean_pixel_difference']:.3f} px")
            
    return calib_obj

# Hardware entry point: load the saved calibration file and require it to be real
def load_calibration(path, cfg: Config = CONFIG, sample_image: Optional[np.ndarray] = None) -> Calibration:
    return get_calibration(cfg, load_intrinsics(path), sample_image, require_calibrated=True)

# Validate the effectiveness of lens distortion removal by comparing the raw and undistorted images.
def validate_calibration_image_effect(raw_image: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> dict:
    if raw_image is None or raw_image.size == 0:
        return {"status": "error", "message": "Raw image is empty"}

    undistorted_image = cv2.undistort(raw_image, K, dist_coeffs)

    if raw_image.ndim == 3:
        raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        und_gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)
    else:
        raw_gray, und_gray = raw_image, undistorted_image

    mean_diff = float(np.mean(cv2.absdiff(raw_gray, und_gray)))

    return {
        "status": "success",
        "mean_pixel_difference": mean_diff,
        "undistorted_image": undistorted_image
    }

# Calibrate the SIMULATED camera like the real one: chessboard views -> corners -> OpenCV calibration.
# Accepted only with a small reprojection error. `world_cfg` = the world the images come from.
def calibrate_camera_in_sim(world_cfg: Config, believed_cfg: Config = CONFIG, n_views: int = 20,
                            pattern: tuple = (9, 6), square: float = 0.04, seed: int = 0) -> Calibration:
    from simulation import render_chessboard_views     # local import: simulation does not import calibration
    images = render_chessboard_views(world_cfg, n_views, pattern, square, seed)
    sim_flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                 | cv2.CALIB_FIX_K3)                       # simulated lens has no distortion
    intr = calibrate_from_images(images, pattern, square, flags=sim_flags)
    if intr.reprojection_error is None or intr.reprojection_error > MAX_CALIB_RMS_PX:
        raise ValueError(f"camera calibration rejected: RMS {intr.reprojection_error} px "
                        f"> {MAX_CALIB_RMS_PX} px")
    return get_calibration(believed_cfg, intr)

# Render chessboard views of the hidden world, save a preview grid, then calibrate the camera
def chessboard_demo(cfg: Config, seed: int = 0, show: bool = True) -> Calibration:
    from config import RESULTS_DIR
    from simulation import Simulation, render_chessboard_views   # local import: simulation does not import calibration
    with Simulation(cfg, seed=seed) as s:
        world = s.world_cfg                      # same hidden world as the main run
    images = render_chessboard_views(world, 20, (9, 6), 0.04, seed)
    out = RESULTS_DIR / "chessboard"
    out.mkdir(parents=True, exist_ok=True)

    if show:
        try:
            tiles = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (320, 240)) for im in images[:6]]
            grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
            cv2.imwrite(str(out / "views.png"), grid)
            cv2.imshow("Chessboard views (calibration)", grid)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        except cv2.error:
            pass                                 # headless: the image is still saved

    cal = calibrate_camera_in_sim(world, cfg, seed=seed)
    save_intrinsics(cal.intrinsics, RESULTS_DIR / "camera_intrinsics_sim.json")
    print(f"[Calibration] RMS {cal.intrinsics.reprojection_error:.3f} px | fx {cal.K[0, 0]:.1f}")
    return cal

# ---------------------------------------------------------------- camera -> robot extrinsics
# Chessboard at MEASURED poses in front of the robot (Zhang intrinsics + PnP).
# Each placement: board centre + normal in the camera frame (PnP) vs in the robot frame (tape / jig).
SIM_EXTRINSIC_BOARD = (1.5, -0.3, math.pi)   # sim only: board centre (x, y) and normal yaw in the world


def board_in_camera(rgb, K, dist, pattern=(9, 6), square=0.025):
    """Board centre as a camera planar point [Z, -X] and the heading of its normal, or None."""
    corners = find_chessboard_corners(rgb, pattern)
    if corners is None:
        return None
    obj = chessboard_object_points(pattern, square)
    ok, rv, tv = cv2.solvePnP(obj, corners.astype(np.float64), K, dist)
    if not ok:
        return None
    R = cv2.Rodrigues(rv)[0]
    centre = R @ obj.mean(axis=0) + tv.ravel()           # (X right, Y down, Z forward)
    n = R[:, 2]
    if n @ centre > 0.0:                                  # normal must point back at the camera
        n = -n
    return np.array([centre[2], -centre[0]]), math.atan2(-n[0], n[2])


def solve_extrinsics(samples):
    """samples: (q_cam, a_cam, p_robot, a_robot) per placement -> (fwd, lat, yaw, report)."""
    if len(samples) < 3:
        raise ValueError(f"only {len(samples)} usable board placements (need >= 3, better >= 5)")
    yaws = np.array([wrap_angle(ar - ac) for _, ac, _, ar in samples])
    yaw = math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean())
    R = _rot2(yaw)
    t = np.array([p - R @ q for q, _, p, _ in samples])
    fwd, lat = t.mean(axis=0)
    err = np.linalg.norm(t - t.mean(axis=0), axis=1)
    report = {"n": len(samples),
              "pos_rms_mm": float(1000.0 * np.sqrt(np.mean(err ** 2))),
              "yaw_std_deg": float(np.degrees(np.std([wrap_angle(y - yaw) for y in yaws])))}
    return float(fwd), float(lat), float(yaw), report


def apply_extrinsics(cfg: Config, fwd: float, lat: float, yaw: float) -> Config:
    """Robot BELIEF uses the calibrated camera mounting (the world is unchanged)."""
    return replace(cfg, camera=replace(cfg.camera, forward_offset=fwd, lateral_offset=lat, yaw=yaw))


def calibrate_extrinsics_in_sim(cfg: Config = CONFIG, seed=None, calibration=None):
    """Same procedure in MuJoCo: the robot is placed at known poses in front of a board (the 'jig')."""
    import tempfile
    from simulation import Simulation, CALIB_BOARD_PATTERN, CHESSBOARD_SQUARE   # local: no import cycle
    K = camera_matrix(cfg) if calibration is None else calibration.K
    dist = np.zeros(5) if calibration is None else calibration.dist_coeffs
    bx, by, byaw = SIM_EXTRINSIC_BOARD
    samples = []
    with tempfile.TemporaryDirectory() as tmp, Simulation(cfg, seed=seed, scene_dir=Path(tmp),
                                                          calib_board=SIM_EXTRINSIC_BOARD) as sim:
        for d in (0.8, 1.0, 1.2):
            for l in (-0.1, 0.0, 0.1):
                for th in (-0.15, 0.0, 0.15):
                    pose = (bx - d, by + l, th)
                    sim.reset(pose)
                    img = next((im for im in (sim.read_camera() for _ in range(5)) if im.any()), None)
                    obs = None if img is None else board_in_camera(img, K, dist, CALIB_BOARD_PATTERN,
                                                                   CHESSBOARD_SQUARE)
                    if obs is None:
                        continue
                    c, s = math.cos(th), math.sin(th)
                    dx, dy = bx - pose[0], by - pose[1]
                    samples.append((obs[0], obs[1], np.array([c * dx + s * dy, -s * dx + c * dy]),
                                    wrap_angle(byaw - th)))
        hidden = sim.cam_offset                                    # EVALUATION ONLY
    fwd, lat, yaw, rep = solve_extrinsics(samples)
    rep["hidden_forward"] = hidden
    return fwd, lat, yaw, rep


def extrinsics_from_csv(csv_path, intr: CameraIntrinsics, pattern=(9, 6), square=0.025):
    """Hardware: CSV with columns file,x,y,normal_deg (board centre and normal in the ROBOT frame)."""
    import csv
    folder, samples = Path(csv_path).parent, []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            img = cv2.cvtColor(cv2.imread(str(folder / row["file"])), cv2.COLOR_BGR2RGB)
            obs = board_in_camera(img, intr.K, np.asarray(intr.dist_coeffs, float), pattern, square)
            if obs is None:
                print("board not found in", row["file"])
                continue
            samples.append((obs[0], obs[1], np.array([float(row["x"]), float(row["y"])]),
                            math.radians(float(row["normal_deg"]))))
    return solve_extrinsics(samples)


def save_extrinsics(fwd: float, lat: float, yaw: float, rep: dict, path=None) -> Path:
    path = Path(path) if path is not None else RESULTS_DIR / "camera_extrinsics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"forward_offset": fwd, "lateral_offset": lat, "yaw": yaw, **rep},
                               indent=2, default=float))
    return path

def load_extrinsics(path) -> tuple[float, float, float]:
    d = json.loads(Path(path).read_text())
    return float(d["forward_offset"]), float(d["lateral_offset"]), float(d["yaw"])


def calibrate_hardware(cfg: Config = CONFIG,
                    images_glob: str = "calib_images/*.jpg",
                    extr_csv: str = "extr_images/poses.csv",
                    pattern: tuple = (9, 6), square: float = 0.025,
                    force: bool = False):
    """Intrinsics + extrinsics for the real camera. Reuses saved files unless force=True.
    Returns (Calibration, cfg_with_extrinsics)."""
    import glob
    intr_path = RESULTS_DIR / "camera_intrinsics.json"
    extr_path = RESULTS_DIR / "camera_extrinsics.json"

    if force or not (intr_path.exists() and extr_path.exists()):
        paths = sorted(glob.glob(images_glob))
        images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                for im in (cv2.imread(p) for p in paths) if im is not None]
        if not images:
            raise FileNotFoundError(f"No calibration images found for '{images_glob}'")
        if not Path(extr_csv).exists():
            raise FileNotFoundError(f"Missing extrinsic pose file: {extr_csv}")

        intr = calibrate_from_images(images, pattern, square)
        save_intrinsics(intr, intr_path)
        print(f"[hw-calib] intrinsics: RMS {intr.reprojection_error:.3f} px | fx {intr.K[0, 0]:.1f}")
        if intr.reprojection_error > MAX_CALIB_RMS_PX:
            raise ValueError(f"RMS {intr.reprojection_error:.3f} px > {MAX_CALIB_RMS_PX} px: "
                            "remove blurry images and retake")

        fwd, lat, yaw, rep = extrinsics_from_csv(extr_csv, intr, pattern, square)
        save_extrinsics(fwd, lat, yaw, rep, extr_path)
        print(f"[hw-calib] extrinsics: fwd {fwd:.4f} m | lat {lat:+.4f} m | yaw {math.degrees(yaw):+.2f} deg "
            f"({rep['n']} placements, RMS {rep['pos_rms_mm']:.1f} mm)")
    else:
        print("[hw-calib] using saved calibration files")

    intr = load_intrinsics(intr_path)
    fwd, lat, yaw = load_extrinsics(extr_path)
    cfg_hw = apply_extrinsics(cfg, fwd, lat, yaw)
    cal = get_calibration(cfg_hw, intr, require_calibrated=True)
    return cal, cfg_hw

# ---------------------------------------------------------------- capture images from the real camera
def capture_calib_images(get_frame, out_dir="calib_images", pattern=(9, 6),
                         n_target=20, min_shift_px=40.0, timeout=600.0):
    """Intrinsics: live view, auto-save an image whenever the chessboard appears in a view
    different enough from the last saved one. get_frame() -> RGB image from the robot camera."""
    import time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("calib_*.jpg"):
        old.unlink()                                  # avoid mixing old and new images
    print(f"Hold the chessboard tilted/rotated, near/far, and move it into the frame corners. "
          f"Need {n_target} images. Press q in the window to abort.")
    last, saved, t0 = None, 0, time.time()
    while saved < n_target:
        if time.time() - t0 > timeout:
            cv2.destroyAllWindows()
            raise TimeoutError(f"Timed out: only {saved}/{n_target} images saved")
        img = get_frame()
        corners = find_chessboard_corners(img, pattern)

        view = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if corners is not None:
            cv2.drawChessboardCorners(view, tuple(pattern), corners.reshape(-1, 1, 2).astype(np.float32), True)
        cv2.putText(view, f"saved {saved}/{n_target} | {'BOARD OK' if corners is not None else 'no board'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if corners is not None else (0, 0, 255), 2)
        cv2.imshow("calib capture", view)

        if corners is not None and (last is None or np.abs(corners - last).mean() > min_shift_px):
            cv2.imwrite(str(out / f"calib_{saved:02d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            last, saved = corners, saved + 1
            print(f"  saved {saved}/{n_target}", flush=True)
        if cv2.waitKey(300) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("capture aborted")
    cv2.destroyAllWindows()


def capture_extrinsic_images(get_frame, out_dir="extr_images", pattern=(9, 6)):
    """Extrinsics: for each placement, type 'x y normal_deg' (robot frame);
    the code grabs a frame (live view, retries 5 s) and writes poses.csv."""
    import time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Place the chessboard FIRST (whole board visible, 0.7-1.2 m away), then type its centre "
          "(x forward, y left, metres) and normal_deg.")
    print("A board facing the robot head-on has normal_deg = 180.")
    while True:
        s = input(f"Placement {len(rows) + 1}: enter 'x y normal_deg' (empty = done): ").strip()
        if not s:
            break
        try:
            x, y, n = (float(v) for v in s.split())
        except ValueError:
            print("  need exactly 3 numbers separated by spaces")
            continue
        img, corners, t0 = None, None, time.time()
        while time.time() - t0 < 5.0:
            img = get_frame()
            corners = find_chessboard_corners(img, pattern)
            view = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if corners is not None:
                cv2.drawChessboardCorners(view, tuple(pattern),
                                          corners.reshape(-1, 1, 2).astype(np.float32), True)
            cv2.imshow("extrinsic capture", view)
            cv2.waitKey(200)
            if corners is not None:
                break
        if corners is None:
            print("  chessboard not found: make sure the WHOLE board is in the image, then re-enter")
            continue
        name = f"extr_{len(rows):02d}.jpg"
        cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append(f"{name},{x},{y},{n}")
        print(f"  saved {name}")
    cv2.destroyAllWindows()
    if len(rows) < 3:
        raise ValueError(f"only {len(rows)} placements (need >= 3, better >= 5)")
    (out / "poses.csv").write_text("file,x,y,normal_deg\n" + "\n".join(rows) + "\n")

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Camera calibration tools")
    parser.add_argument("--calibrate", metavar="IMAGE_GLOB", default=None,
                        help='chessboard images of the REAL camera, e.g. "calib_images/*.jpg"')
    parser.add_argument("--pattern", type=int, nargs=2, default=(9, 6),
                        help="inner corners of the chessboard (cols rows)")
    parser.add_argument("--square", type=float, default=0.025, help="chessboard square size [m]")
    parser.add_argument("--out", default="results/camera_intrinsics.json")
    parser.add_argument("--check", metavar="INTRINSICS_JSON", default=None,
                        help="load a saved calibration and verify it is usable on hardware")
    parser.add_argument("--sim", action="store_true",
                        help="calibrate the simulated camera with rendered chessboard views")
    parser.add_argument("--seed", type=int, default=None, help="simulation seed (with --sim)")
    parser.add_argument("--extrinsic-sim", action="store_true",
                        help="camera -> robot extrinsics in MuJoCo (chessboard at known poses)")
    parser.add_argument("--extrinsic-poses", metavar="CSV", default=None,
                        help="hardware: CSV file,x,y,normal_deg of the board in the ROBOT frame")
    parser.add_argument("--intrinsics", default=str(RESULTS_DIR / "camera_intrinsics.json"),
                        help="intrinsics JSON used with --extrinsic-poses")
    parser.add_argument("--images", default="calib_images/*.jpg",
                        help="chessboard images for the intrinsic calibration (default mode)")
    parser.add_argument("--extr-csv", default="extr_images/poses.csv",
                        help="CSV file,x,y,normal_deg for the extrinsic calibration (default mode)")
    parser.add_argument("--capture", action="store_true",
                        help="capture chessboard images from the real robot camera, then calibrate")
    parser.add_argument("--ip", default=None, help="robot IP (used with --capture)")
    parser.add_argument("--extr-only", action="store_true",
                        help="with --capture: skip intrinsic images, only capture the extrinsic step")
    args = parser.parse_args()
    if args.extrinsic_sim or args.extrinsic_poses:
        if args.extrinsic_sim:
            from simulation import Simulation
            with Simulation(CONFIG, seed=args.seed) as s:           # same hidden world as the main run
                world = s.world_cfg
            cal = calibrate_camera_in_sim(world, CONFIG, seed=args.seed or 0)     # Zhang (intrinsic) first
            print(f"intrinsic: RMS {cal.intrinsics.reprojection_error:.3f} px | fx {cal.K[0, 0]:.1f}")
            fwd, lat, yaw, rep = calibrate_extrinsics_in_sim(CONFIG, args.seed, calibration=cal)
        else:
            fwd, lat, yaw, rep = extrinsics_from_csv(args.extrinsic_poses, load_intrinsics(args.intrinsics),
                                                    tuple(args.pattern), args.square)
        print(f"forward {fwd:.4f} m | lateral {lat:+.4f} m | yaw {math.degrees(yaw):+.2f} deg")
        print(f"{rep['n']} placements | position RMS {rep['pos_rms_mm']:.1f} mm | yaw std {rep['yaw_std_deg']:.2f} deg")
        if rep.get("hidden_forward") is not None:
            print(f"hidden true forward offset: {rep['hidden_forward']:.4f} m (evaluation only)")
        print("saved:", save_extrinsics(fwd, lat, yaw, rep))
    elif args.calibrate:
        paths = sorted(glob.glob(args.calibrate))
        images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
        intr = calibrate_from_images(images, tuple(args.pattern), args.square)
        save_intrinsics(intr, args.out)
        print(f"Saved {args.out} | RMS reprojection error: {intr.reprojection_error:.3f} px")
    elif args.check:
        cal = load_calibration(args.check)
        print("Hardware calibration OK | RMS:", round(cal.intrinsics.reprojection_error, 3), "px")
        print("Extrinsics (fwd, lat, yaw):", cal.forward_offset, cal.lateral_offset, cal.yaw)
    elif args.sim:
        from simulation import Simulation
        with Simulation(CONFIG, seed=args.seed) as s:        # hidden world: used for COMPARISON only
            world = s.world_cfg
        cal = calibrate_camera_in_sim(world, CONFIG, seed=args.seed or 0)
        f_nom = float(nominal_intrinsics(CONFIG).K[0, 0])
        f_true = (world.camera.height / 2.0) / math.tan(math.radians(world.camera.fovy_deg) / 2.0)
        print(f"RMS reprojection error : {cal.intrinsics.reprojection_error:.3f} px")
        print(f"focal length  nominal  : {f_nom:.1f} px")
        print(f"focal length  calibrated: {cal.K[0, 0]:.1f} px")
        print(f"focal length  hidden true: {f_true:.1f} px (evaluation only)")
        print("distortion k1, k2      :", np.round(cal.dist_coeffs[:2], 4), "(simulated lens has none)")
        print("saved:", save_intrinsics(cal.intrinsics, "results/camera_intrinsics_sim.json"))
    else:
        if args.capture:
            from hardware import ROBOT_IP, RosbridgeEnvironment, hardware_config
            from rosbridge_api import RosbridgeRobot
            with RosbridgeRobot(args.ip or ROBOT_IP) as robot:
                env = RosbridgeEnvironment(robot, CONFIG)
                try:
                    if not args.extr_only:
                        capture_calib_images(env.read_camera, Path(args.images).parent,
                                            tuple(args.pattern))
                        input("Intrinsic images done. Set up the board for the extrinsic step, press Enter... ")
                    capture_extrinsic_images(env.read_camera, Path(args.extr_csv).parent,
                                            tuple(args.pattern))
                finally:
                    env.emergency_stop()
        # Default mode: calibrate EVERYTHING from the saved images
        paths = sorted(glob.glob(args.images))
        images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                for im in (cv2.imread(p) for p in paths) if im is not None]
        if not images:
            raise SystemExit(f"No images found for '{args.images}'")
        pattern = tuple(args.pattern)

        # 1) intrinsics
        intr = calibrate_from_images(images, pattern, args.square)
        save_intrinsics(intr, args.out)
        print(f"[1/2] Intrinsics saved: {args.out} | RMS {intr.reprojection_error:.3f} px | fx {intr.K[0, 0]:.1f}")
        if intr.reprojection_error > MAX_CALIB_RMS_PX:
            print(f"  !! RMS > {MAX_CALIB_RMS_PX} px: hardware.py will reject this file. Remove blurry images and rerun.")

        # 2) extrinsics (needs the measured board poses in the CSV)
        fwd, lat, yaw, rep = extrinsics_from_csv(args.extr_csv, intr, pattern, args.square)
        print(f"[2/2] Extrinsics: forward {fwd:.4f} m | lateral {lat:+.4f} m | yaw {math.degrees(yaw):+.2f} deg")
        print(f"      {rep['n']} placements | position RMS {rep['pos_rms_mm']:.1f} mm | yaw std {rep['yaw_std_deg']:.2f} deg")
        print("      saved:", save_extrinsics(fwd, lat, yaw, rep))