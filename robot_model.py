# Including libraries
from __future__ import annotations
from typing import Optional
import math

import numpy as np

from config import CONFIG, Config

N_STATE = 3

# Wrap an angle to [-pi, pi]
def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

# World position (x, y) of the ArUco marker
def default_landmark(cfg: Config = CONFIG) -> np.ndarray:
    return np.array(cfg.env.dock_pose[:2], dtype=float)

# Tag poses learned during SURVEY: {id: (position (2,), yaw)} in the world frame of the start pose.
class TagMap:
    def __init__(self, min_obs: int = 1):
        self.min_obs = int(min_obs)
        self._obs: dict = {}                     # id -> list of (x, y, yaw, weight)

    # Add one observation made from a known robot pose (the EKF estimate).
    # Observations farther than max_range are ignored: their error grows with range^2.
    def add(self, robot_pose, marker, max_range: Optional[float] = None) -> None:
        if max_range is not None and marker.range > max_range:
            return
        px, py, th = (float(v) for v in robot_pose)
        a = th + marker.bearing
        self._obs.setdefault(int(marker.marker_id), []).append((
            px + marker.range * math.cos(a),
            py + marker.range * math.sin(a),
            wrap_angle(th + marker.view_angle + math.pi),       # inverse of view_angle_model
            1.0 / max(marker.range, 0.1) ** 2))                 # close observations weigh more

    def count(self, marker_id: int) -> int:
        return len(self._obs.get(int(marker_id), []))

    # Weighted mean pose of every tag with at least min_obs observations (yaw averaged on the circle)
    def table(self) -> dict:
        out = {}
        for i, obs in self._obs.items():
            if len(obs) < self.min_obs:
                continue
            o = np.array(obs, dtype=float)
            w = o[:, 3] / o[:, 3].sum()
            yaw = math.atan2(float((w * np.sin(o[:, 2])).sum()), float((w * np.cos(o[:, 2])).sum()))
            out[int(i)] = ((o[:, :2] * w[:, None]).sum(axis=0), yaw)
        return out

# {marker_id: (landmark position (2,), marker yaw)}.
# tag_map = learned TagMap; None = the poses written in the config (used by old code and tests).
def landmark_table(cfg: Config = CONFIG, tag_map=None) -> dict:
    if tag_map is not None:
        return {int(i): (np.array(p, dtype=float), float(yaw)) for i, (p, yaw) in tag_map.table().items()}
    return {int(i): (np.array(p[:2], dtype=float), float(p[2])) for i, p in cfg.tag_table().items()}

# Convert two cumulative encoder readings
def odometry_increment(enc_prev, enc_now, cfg: Config = CONFIG) -> tuple[float, float]:
    dl, dr = np.asarray(enc_now, dtype=float) - np.asarray(enc_prev, dtype=float)
    r, b = cfg.robot.wheel_radius, cfg.robot.wheel_base
    ds = r * (dl + dr) / 2.0
    dtheta = r * (dr - dl) / b
    return float(ds), float(dtheta)

# Discrete pose update
def motion_model(x, ds: float, dtheta: float) -> np.ndarray:
    px, py, th = x
    mid = th + dtheta / 2.0
    return np.array([px + ds * math.cos(mid),
                     py + ds * math.sin(mid),
                    wrap_angle(th + dtheta)])

# Covariance prediction in the EKF
def jacobian_F(x, ds: float, dtheta: float) -> np.ndarray:
    mid = x[2] + dtheta / 2.0
    return np.array([[1.0, 0.0, -ds * math.sin(mid)],
                     [0.0, 1.0, ds * math.cos(mid)],
                    [0.0, 0.0, 1.0]])

# Mapping odometry noise into state noise
def jacobian_G(x, ds: float, dtheta: float) -> np.ndarray:
    mid = x[2] + dtheta / 2.0
    return np.array([[math.cos(mid), -0.5 * ds * math.sin(mid)],
                     [math.sin(mid), 0.5 * ds * math.cos(mid)],
                    [0.0, 1.0]])

# Expected camera measurement of the landmark from pose x
def measurement_model(x, landmark) -> np.ndarray:
    dx, dy = landmark[0] - x[0], landmark[1] - x[1]
    return np.array([math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - x[2])])

def measurement_jacobian(x, landmark) -> np.ndarray:
    dx, dy = landmark[0] - x[0], landmark[1] - x[1]
    q = dx * dx + dy * dy
    if q < 1e-12:
        raise ValueError("Robot is on top of the landmark: Jacobian undefined.")
    r = math.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                    [dy / q, -dx / q, -1.0]])

# Measurement residual with the bearing (and the view angle, if present) wrapped to (-pi, pi)
def innovation(z, z_pred) -> np.ndarray:
    nu = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
    nu[1] = wrap_angle(nu[1])
    if nu.shape[0] > 2:
        nu[2] = wrap_angle(nu[2])
    return nu

# Expected view angle of the dock marker: 0 when the robot looks straight at the marker face.
# The marker normal points along dock_yaw, so the robot heading that faces it is dock_yaw - pi.
def view_angle_model(x, dock_yaw: float) -> float:
    return wrap_angle(dock_yaw - math.pi - x[2])

# Expected [range, bearing, view_angle] measurement of the dock marker from pose x
def measurement_model_pose(x, landmark, dock_yaw: float) -> np.ndarray:
    r, b = measurement_model(x, landmark)
    return np.array([r, b, view_angle_model(x, dock_yaw)])

# 3x3 Jacobian of [range, bearing, view_angle]; the view angle depends on theta only
def measurement_jacobian_pose(x, landmark) -> np.ndarray:
    return np.vstack([measurement_jacobian(x, landmark), [0.0, 0.0, -1.0]])

# Expected range of one LiDAR beam in the known room map (4 walls + circular obstacles).
# beam_angle is in the robot frame; range_max is returned when nothing is hit.
def lidar_range_model(x, beam_angle: float, cfg: Config = CONFIG) -> float:
    px, py, th = x
    a = th + beam_angle
    c, s = math.cos(a), math.sin(a)
    hx, hy = cfg.env.room_half_size
    best = cfg.lidar.range_max
    # Inner wall faces at x = +-hx and y = +-hy
    if c > 1e-9:
        best = min(best, (hx - px) / c)
    elif c < -1e-9:
        best = min(best, (-hx - px) / c)
    if s > 1e-9:
        best = min(best, (hy - py) / s)
    elif s < -1e-9:
        best = min(best, (-hy - py) / s)
    # Ray-circle intersection for every obstacle (nearest entry point)
    for ox, oy, r in cfg.env.obstacles:
        qx, qy = px - ox, py - oy
        b = qx * c + qy * s
        disc = b * b - (qx * qx + qy * qy - r * r)
        if disc >= 0.0:
            t = -b - math.sqrt(disc)
            if 0.0 < t < best:
                best = t
    return float(best)

# Gradient of the expected beam range w.r.t. [x, y, theta] (central finite differences)
def lidar_jacobian(x, beam_angle: float, cfg: Config = CONFIG, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    H = np.zeros(3)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        H[i] = (lidar_range_model(x + d, beam_angle, cfg)
                - lidar_range_model(x - d, beam_angle, cfg)) / (2.0 * eps)
    return H

# Tracking error expressed in the robot frame
def pose_error(x, x_ref) -> np.ndarray:
    dx, dy = x_ref[0] - x[0], x_ref[1] - x[1]
    c, s = math.cos(x[2]), math.sin(x[2])
    return np.array([c * dx + s * dy,
                     -s * dx + c * dy,
                    wrap_angle(x_ref[2] - x[2])])

# Discrete linear error model
def tracking_error_model(v_ref: float, w_ref: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([[0.0, w_ref, 0.0],
                [-w_ref, 0.0, v_ref],
                [0.0, 0.0, 0.0]])
    B = np.array([[-1.0, 0.0],
                [0.0, 0.0],
                [0.0, -1.0]])
    return np.eye(N_STATE) + A * dt, B * dt

def controllability_matrix(A, B) -> np.ndarray:
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)

def observability_matrix(F, H) -> np.ndarray:
    n = F.shape[0]
    blocks = [H]
    for _ in range(n - 1):
        blocks.append(blocks[-1] @ F)
    return np.vstack(blocks)

def is_controllable(A, B) -> bool:
    return np.linalg.matrix_rank(controllability_matrix(A, B)) == A.shape[0]

def is_observable(F, H) -> bool:
    return np.linalg.matrix_rank(observability_matrix(F, H)) == F.shape[0]

# Ranks of the controllability and observability matrices at an operating point
def analyse_model(x, v: float, w: float, landmark=None, cfg: Config = CONFIG) -> dict:
    dt = cfg.sim.control_dt
    landmark = default_landmark(cfg) if landmark is None else landmark
    Ad, Bd = tracking_error_model(v, w, dt)
    F = jacobian_F(x, v * dt, w * dt)
    H = measurement_jacobian(x, landmark)
    H3 = measurement_jacobian_pose(x, landmark)      # range + bearing + marker view angle
    return {
        "controllability_rank": int(np.linalg.matrix_rank(controllability_matrix(Ad, Bd))),
        "observability_rank": int(np.linalg.matrix_rank(observability_matrix(F, H))),
        "observability_rank_with_view": int(np.linalg.matrix_rank(observability_matrix(F, H3))),
        "n_states": N_STATE,
    }

if __name__ == "__main__":
    print("Moving  :", analyse_model(np.array([0.0, 0.0, 0.3]), 0.2, 0.0))
    print("Stopped :", analyse_model(np.array([0.0, 0.0, 0.3]), 0.0, 0.0))
