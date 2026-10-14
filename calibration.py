# Including libraries
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from config import CONFIG, Config
from robot_model import wrap_angle

SOURCE_NOMINAL = "nominal_mujoco"   # derived from the known simulator model (NOT measured)
SOURCE_CALIBRATED = "calibrated"    # estimated from images of a calibration target
MIN_CALIBRATION_VIEWS = 5
VALID_DIST_LENGTHS = (4, 5, 8, 12, 14)

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

# Camera position ahead of the robot centre [m]
def camera_forward_offset(cfg: Config = CONFIG) -> float:
    return cfg.robot.body_radius + 0.01

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
    
# Frame transforms: camera to robot
# Camera-frame point (X right, Y down, Z forward) -> robot ground-plane point (x, y)
def camera_to_robot(p_cam, cfg: Config = CONFIG) -> np.ndarray:
    X, _, Z = np.asarray(p_cam, dtype=float).reshape(3)
    return np.array([Z + camera_forward_offset(cfg), -X])

# Robot ground-plane point -> camera frame. `height` is the point height above the camera axis [m]
def robot_to_camera(p_robot, height: float = 0.0, cfg: Config = CONFIG) -> np.ndarray:
    x_r, y_r = np.asarray(p_robot, dtype=float).reshape(2)
    return np.array([-y_r, -height, x_r - camera_forward_offset(cfg)])

# Direction vector (no translation): camera frame -> robot ground plane
def camera_dir_to_robot(d_cam) -> np.ndarray:
    d = np.asarray(d_cam, dtype=float).reshape(3)
    return np.array([d[2], -d[0]])

def range_bearing(p_robot) -> tuple[float, float]:
    x, y = np.asarray(p_robot, dtype=float).reshape(2)
    return math.hypot(x, y), math.atan2(y, x)

def xy_from_range_bearing(rng: float, bearing: float) -> np.ndarray:
    return np.array([rng * math.cos(bearing), rng * math.sin(bearing)])

# Square marker corners in the marker frame (same order as the ArUco detector output)
def marker_object_points(marker_size: float) -> np.ndarray:
    s = marker_size / 2.0
    return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])

# Convert a solvePnP result (marker in camera frame) into (range, bearing, view_angle)
def marker_pose_from_pnp(rvec, tvec, cfg: Config = CONFIG) -> tuple[float, float, float]:
    p_robot = camera_to_robot(tvec, cfg)
    rng, bearing = range_bearing(p_robot)
    normal_cam = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0][:, 2]
    n = camera_dir_to_robot(normal_cam)       # normal points towards the viewer
    view_angle = wrap_angle(math.atan2(n[1], n[0]) - math.pi)
    return rng, bearing, float(view_angle)

# Marker pose in the camera frame from the 4 detected corners
def solve_marker_pose(image_pts, K: np.ndarray, dist_coeffs: np.ndarray,
                    marker_size: float):
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

        # Validate the estimated pose by computing the reprojection error
        is_valid, reproj_err = validate_estimated_pose(img, rv, tv, K, dist, marker_size, max_error_px=1.5)
        
        if not is_valid:
            continue  # Skip this candidate if the reprojection error exceeds the threshold

        proj, _ = cv2.projectPoints(obj, rv, tv.reshape(3, 1), K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, axis=1))))
        if best is None or err < best[2]:
            best = (np.asarray(rv, dtype=float).reshape(3, 1), tv.reshape(3, 1), err)
    return best

# Validate the estimated pose by computing the reprojection error
def validate_estimated_pose(image_pts, rvec, tvec, K, dist_coeffs, marker_size, max_error_px: float = 1.5) -> tuple[bool, float]:
    """
    Return (True/False, RMS reprojection error in pixels).
    Change the 'max_error_px' threshold to control the tolerance for accepting the estimated pose.
    """
    if image_pts is None or len(image_pts) != 4:
        return False, float('inf')
    
    # Compute the 3D corner points of the ArUco marker in the marker coordinate system
    s = marker_size / 2.0
    obj_points = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float32)
    
    # Reproject the 3D points to 2D image points using the estimated pose (rvec, tvec)
    projected_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    projected_pts = projected_pts.reshape(-1, 2)

    # Compare the projected points with the actual measured points in the image
    measured_pts = np.asarray(image_pts, dtype=np.float32).reshape(4, 2)
    errors = np.linalg.norm(measured_pts - projected_pts, axis=1)
    reproj_err = float(np.sqrt(np.mean(errors ** 2)))

    # Return True/False, reprojection error in pixels
    # Return True if the reprojection error is within the acceptable threshold, otherwise False
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

# Consistency checks
def check_units(cfg: Config = CONFIG) -> list[str]:
    problems = []
    if not 0.01 <= cfg.camera.marker_size <= 1.0:
        problems.append("marker_size outside 0.01-1.0 m (millimetres instead of metres?)")
    if not 10.0 <= cfg.camera.fovy_deg <= 170.0:
        problems.append("fovy_deg outside 10-170 deg (radians instead of degrees?)")
    if not 0.0 < camera_forward_offset(cfg) < 0.5:
        problems.append("camera forward offset outside 0-0.5 m")
    return problems

# Return a list of violated conventions (empty list = all consistent)
def check_frame_conventions(cfg: Config = CONFIG, intr: Optional[CameraIntrinsics] = None,
                            tol: float = 1e-6) -> list[str]:
    intr = nominal_intrinsics(cfg) if intr is None else intr
    problems = check_units(cfg)
    cx, cy = intr.K[0, 2], intr.K[1, 2]

    ahead = project_robot_point([1.0, 0.0], intr, 0.0, cfg)
    if ahead is None or abs(ahead[0] - cx) > 1e-3 or abs(ahead[1] - cy) > 1e-3:
        problems.append("a point straight ahead must project to the principal point")
    left = project_robot_point([1.0, 0.2], intr, 0.0, cfg)
    if left is None or left[0] >= cx:
        problems.append("a point on the robot's left (y>0) must appear left of the image centre")
    above = project_robot_point([1.0, 0.0], intr, 0.1, cfg)
    if above is None or above[1] >= cy:
        problems.append("a point above the camera axis must appear above the image centre")

    p = np.array([1.3, -0.4])
    if np.linalg.norm(camera_to_robot(robot_to_camera(p, 0.0, cfg), cfg) - p) > tol:
        problems.append("robot->camera->robot round trip is not the identity")

    # Head-on marker: OpenCV pose with the marker y axis flipped (rotation pi about X)
    rng, bearing, view = marker_pose_from_pnp(np.array([math.pi, 0.0, 0.0]),
                                            np.array([0.0, 0.0, 1.0]), cfg)
    if abs(bearing) > tol or abs(view) > tol or abs(rng - (1.0 + camera_forward_offset(cfg))) > tol:
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
                        square_size: float) -> CameraIntrinsics:
    obj_template = chessboard_object_points(pattern_size, square_size)
    obj, img = [], []
    for image in images:
        corners = find_chessboard_corners(image, pattern_size)
        if corners is not None:
            obj.append(obj_template)
            img.append(corners)
    if len(obj) < MIN_CALIBRATION_VIEWS:
        raise ValueError(f"chessboard found in only {len(obj)} of {len(images)} images "
                        f"(need {MIN_CALIBRATION_VIEWS})")
    h, w = images[0].shape[:2]
    return calibrate_intrinsics(obj, img, (w, h))

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

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.K

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.intrinsics.dist_coeffs, dtype=float)

    @property
    def marker_points(self) -> np.ndarray:
        return marker_object_points(self.marker_size)

# Nominal MuJoCo intrinsics unless a calibrated set is supplied
def get_calibration(cfg: Config = CONFIG, intrinsics: Optional[CameraIntrinsics] = None, sample_image: Optional[np.ndarray] = None) -> Calibration:
    """
    Provide a Calibration object with the camera intrinsics, forward offset, and marker size. 
    If no intrinsics are provided, the nominal MuJoCo intrinsics are used. 
    If a sample image is provided, the effectiveness of lens distortion removal is validated and reported.
    """
    intr = nominal_intrinsics(cfg) if intrinsics is None else intrinsics
    assert_valid_intrinsics(intr)
    assert_frame_conventions(cfg, intr)
    calib_obj = Calibration(intr, camera_forward_offset(cfg), cfg.camera.marker_size)
    
    # Validate the effectiveness of lens distortion removal if a sample image is provided
    if sample_image is not None:
        val_res = validate_calibration_image_effect(sample_image, calib_obj.K, calib_obj.dist_coeffs)
        if val_res["status"] == "success":
            print(f"[Calibration Check] Undistort check passed. Mean pixel shift: {val_res['mean_pixel_difference']:.3f} px")
            
    return calib_obj

# Validate the effectiveness of lens distortion removal by comparing the raw and undistorted images.
def validate_calibration_image_effect(raw_image: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> dict:
    if raw_image is None or raw_image.size == 0:
        return {"status": "error", "message": "Raw image is empty"}
    
    # Use the calibration parameters to undistort the raw image
    undistorted_image = cv2.undistort(raw_image, K, dist_coeffs)

    # Calculate the mean pixel difference between the raw and undistorted images as a quality metric
    diff = cv2.absdiff(cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY), 
                       cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY))
    mean_diff = float(np.mean(diff))
    
    return {
        "status": "success",
        "mean_pixel_difference": mean_diff,
        "undistorted_image": undistorted_image
    }

if __name__ == "__main__":
    cal = get_calibration()
    print("Source        :", cal.intrinsics.source)
    print("K             :\n", np.round(cal.K, 2))
    print("Forward offset:", round(cal.forward_offset, 3), "m")
    print("Convention check:", check_frame_conventions() or "OK")