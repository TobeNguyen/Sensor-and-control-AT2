# Last line of defence between the controllers and the actuators
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import CONFIG, Config
from sensing import LidarScan, sector_min_range


@dataclass(frozen=True)
class SafetyConfig:
    max_lidar_age: float = 0.5        # [s] older LiDAR data -> stop
    estop_lidar_age: float = 2.0      # [s] older than this -> latched E-STOP
    min_valid_beams: int = 30         # fewer valid beams -> LiDAR treated as invalid
    forward_half_angle: float = math.pi / 3
    translate_margin: float = 0.05    # [m] beyond body_radius when moving
    rotate_margin: float = 0.01       # [m] beyond body_radius when rotating (full ring is checked)
    sigma_slow: float = 0.30          # [m] position std: start slowing down
    sigma_stop: float = 1.00          # [m] position std: no translation
    max_lin_acc: float = 0.5          # [m/s^2] (speeding up only, braking is never delayed)
    max_ang_acc: float = 3.0          # [rad/s^2]


class SafetyGuard:
    def __init__(self, cfg: Config = CONFIG, safety: Optional[SafetyConfig] = None):
        self.cfg = cfg
        self.sc = SafetyConfig() if safety is None else safety
        self.reset()

    def reset(self) -> None:
        """Clear a latched fault (operator action) and the rate-limiter state."""
        self.fault = ""
        self._v = 0.0
        self._w = 0.0

    def _halt(self, reason: str) -> tuple[float, float, str]:
        self._v = self._w = 0.0
        return 0.0, 0.0, reason

    @staticmethod
    def _ramp(cur: float, target: float, step: float) -> float:
        if abs(target) < abs(cur) and target * cur >= 0.0:
            return target                           # braking is never delayed
        return cur + float(np.clip(target - cur, -step, step))

    # Returns (v, w, reason). reason == "" means the command passed unchanged (apart from limits).
    def filter(self, v: float, w: float, scan: LidarScan, dt: float,
               lidar_age: float = 0.0, sigma_xy: float = 0.0) -> tuple[float, float, str]:
        sc, rob = self.sc, self.cfg.robot
        if self.fault:
            return 0.0, 0.0, self.fault
        if not (math.isfinite(v) and math.isfinite(w)):
            return self._halt("invalid command")
        if lidar_age > sc.estop_lidar_age:
            self.fault = "E-STOP: LiDAR data lost"
            return self._halt(self.fault)
        if lidar_age > sc.max_lidar_age or int(scan.valid.sum()) < sc.min_valid_beams:
            return self._halt("LiDAR data stale or invalid")

        reasons = []
        v = float(np.clip(v, -rob.max_linear_vel, rob.max_linear_vel))
        w = float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel))

        # Position uncertainty too high: slow down, then stop translating (rotation stays allowed)
        if abs(v) > 0.0 and sigma_xy > sc.sigma_slow:
            scale = (sc.sigma_stop - sigma_xy) / (sc.sigma_stop - sc.sigma_slow)
            v *= float(np.clip(scale, 0.0, 1.0))
            reasons.append("position uncertainty high")

        # Translation: check the sector in the direction of motion (forward or reverse)
        if abs(v) > 1e-9:
            centre = 0.0 if v > 0.0 else math.pi
            if sector_min_range(scan, centre, sc.forward_half_angle) < rob.body_radius + sc.translate_margin:
                v = 0.0
                reasons.append("translation blocked")

        # Rotation: the footprint sweeps a full circle, so check the whole ring
        if abs(w) > 1e-9 and sector_min_range(scan, 0.0, math.pi) < rob.body_radius + sc.rotate_margin:
            w = 0.0
            reasons.append("rotation blocked")

        v = self._ramp(self._v, v, sc.max_lin_acc * dt)
        w = self._ramp(self._w, w, sc.max_ang_acc * dt)
        self._v, self._w = v, w
        return v, w, "; ".join(reasons)