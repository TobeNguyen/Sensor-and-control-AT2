# Including libraries
from __future__ import annotations

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

# Measurement residual with the bearing wrapped to (-pi, pi)
def innovation(z, z_pred) -> np.ndarray:
    nu = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
    nu[1] = wrap_angle(nu[1])
    return nu

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
    return {
        "controllability_rank": int(np.linalg.matrix_rank(controllability_matrix(Ad, Bd))),
        "observability_rank": int(np.linalg.matrix_rank(observability_matrix(F, H))),
        "n_states": N_STATE,
    }

if __name__ == "__main__":
    print("Moving  :", analyse_model(np.array([0.0, 0.0, 0.3]), 0.2, 0.0))
    print("Stopped :", analyse_model(np.array([0.0, 0.0, 0.3]), 0.0, 0.0))
