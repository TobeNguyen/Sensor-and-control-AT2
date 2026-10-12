# Including libraries
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are

from config import CONFIG, Config
from robot_model import is_controllable, pose_error, tracking_error_model, motion_model

DEFAULT_CRUISE_SPEED = 0.15   # [m/s] nominal speed, also the LQR design point
MIN_TRACK_SPEED = 0.03        # [m/s] never creep slower than this while tracking
SLOW_RADIUS = 0.40            # [m] start slowing down this far before the segment end
STOP_MARGIN = 0.05            # [m] stop translating this far outside the body radius
SLOW_BAND = 0.30              # [m] slow-down band added to lidar.safe_distance

# Reference pose and feed-forward velocities
@dataclass(frozen=True)
class Reference:
    x: float
    y: float
    theta: float
    v: float = 0.0
    w: float = 0.0

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
# Discrete-time LQR gain K
def lqr_gain(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

# Saturate the velocity command to the robot limits
def limit_command(v: float, w: float, cfg: Config = CONFIG,
                allow_reverse: bool = False) -> tuple[float, float]:
    rob = cfg.robot
    v_min = -rob.max_linear_vel if allow_reverse else 0.0
    return (float(np.clip(v, v_min, rob.max_linear_vel)),
            float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel)))

# Scale forward speed fown to zero as the front obstale gets close
def safe_command(v: float, w: float, front_range: float,
                cfg: Config = CONFIG) -> tuple[float, float]:
    stop = cfg.robot.body_radius + STOP_MARGIN
    slow = cfg.lidar.safe_distance + SLOW_BAND
    if v > 0.0:
        v *= float(np.clip((front_range - stop) / (slow - stop), 0.0, 1.0))
    return v, w

# Reference point on the line for path tracking
def segment_reference(x_est, p_start, p_end,
                    v_cruise: float = DEFAULT_CRUISE_SPEED,
                    slow_radius: float = SLOW_RADIUS,
                    v_min: float = MIN_TRACK_SPEED) -> Reference:
    p0, p1 = np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float)
    d = p1 - p0
    length = float(np.hypot(*d))
    if length < 1e-9:
        return Reference(p1[0], p1[1], float(x_est[2]), 0.0, 0.0)

    u = d / length
    s = float(np.clip(np.dot(np.asarray(x_est[:2], dtype=float) - p0, u), 0.0, length))
    point = p0 + s * u
    remaining = length - s
    v_ref = float(np.clip(v_cruise * remaining / slow_radius, v_min, v_cruise))
    return Reference(point[0], point[1], math.atan2(u[1], u[0]), v_ref, 0.0)

class LQRTracker:
    def __init__(self, cfg: Config = CONFIG,
                v_design: float = DEFAULT_CRUISE_SPEED, w_design: float = 0.0):
        self.cfg = cfg
        self.Ad, self.Bd = tracking_error_model(v_design, w_design, cfg.sim.control_dt)
        if not is_controllable(self.Ad, self.Bd):
            raise ValueError("Error model is not controllable at the design point "
                            "(v_design and w_design cannot both be zero).")
        self.Q = cfg.controller.Q()
        self.R = cfg.controller.R()
        self.K = lqr_gain(self.Ad, self.Bd, self.Q, self.R)

    def closed_loop_poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.Ad - self.Bd @ self.K)
    
    # Velocity command from the estimated pose and a reference
    def compute(self, x_est, ref: Reference,
                front_range: Optional[float] = None) -> tuple[float, float]:
        x_est = np.asarray(x_est, dtype=float)
        ref_vals = np.array([ref.x, ref.y, ref.theta, ref.v, ref.w], dtype=float)
        if x_est.shape != (3,) or not np.all(np.isfinite(x_est)) \
                or not np.all(np.isfinite(ref_vals)):
            return 0.0, 0.0

        du = -self.K @ pose_error(x_est, ref.pose)
        v, w = limit_command(ref.v + du[0], ref.w + du[1], self.cfg)
        if front_range is not None:
            v, w = safe_command(v, w, front_range, self.cfg)
        return v, w
    
if __name__ == "__main__":
    tracker = LQRTracker()
    print("Gain K:\n", np.round(tracker.K, 3))
    print("Closed-loop pole magnitudes:", np.round(np.abs(tracker.closed_loop_poles()), 3))

    dt = CONFIG.sim.control_dt
    x, p0, p1 = np.array([0.0, 0.2, 0.0]), (0.0, 0.0), (2.0, 0.0)
    for _ in range(3000):
        v, w = tracker.compute(x, segment_reference(x, p0, p1))
        x = motion_model(x, v * dt, w * dt)
        if np.hypot(*(np.array(p1) - x[:2])) < 0.05:
            break
    print("Final pose:", np.round(x, 3))