# Odometry calibration in simulation: drive a known distance, then spin a known angle.
# The "tape measure" is the simulator's true pose, used ONLY here (like a tape / mocap on a real robot).
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import CONFIG, RESULTS_DIR, Config
from robot_model import wrap_angle
from simulation import Simulation

OUT_FILE = RESULTS_DIR / "odometry_calibration.json"


def _run(sim: Simulation, v: float, w: float, steps: int, settle: int = 40):
    """Apply (v, w) for `steps`, then stop and let the robot settle. Returns (dphi, distance, turn)."""
    enc0 = sim.read_encoders()
    prev = sim.ground_truth()
    dist = turn = 0.0
    for k in range(steps + settle):
        sim.step(v, w) if k < steps else sim.step(0.0, 0.0)
        cur = sim.ground_truth()
        dist += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        turn += wrap_angle(cur[2] - prev[2])          # accumulate small wrapped steps (no 2*pi jump)
        prev = cur
    return sim.read_encoders() - enc0, dist, turn


def calibrate_odometry(cfg: Config = CONFIG, seed: int | None = None,
                       straight: float = 1.0, turns: float = 2.0) -> tuple[float, float]:
    """Return (wheel_radius, wheel_base) estimated from a straight run and an in-place spin."""
    dt = cfg.sim.control_dt
    with Simulation(cfg, seed=seed) as sim:
        # 1) straight line: radius = distance / wheel rotation
        sim.reset((-1.5, -1.5, 0.0))
        v = 0.15
        dphi, dist, _ = _run(sim, v, 0.0, int(round(straight / v / dt)))
        r_est = dist / float(np.mean(dphi))

        # 2) spin in place: base = r * (right - left wheel rotation) / body rotation
        sim.reset((0.0, 1.5, 0.0))
        w = 0.5
        dphi, _, turn = _run(sim, 0.0, w, int(round(2.0 * math.pi * turns / w / dt)))
        b_est = r_est * float(dphi[1] - dphi[0]) / turn
    return float(r_est), float(b_est)


def apply_odometry(cfg: Config, wheel_radius: float, wheel_base: float) -> Config:
    """Config copy where the robot's BELIEF uses the calibrated values (the world is unchanged)."""
    return replace(cfg, robot=replace(cfg.robot, wheel_radius=wheel_radius, wheel_base=wheel_base))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Odometry calibration (simulation)")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    r, b = calibrate_odometry(CONFIG, seed=a.seed)
    print(f"config   : wheel_radius {CONFIG.robot.wheel_radius:.4f} m | wheel_base {CONFIG.robot.wheel_base:.4f} m")
    print(f"estimated: wheel_radius {r:.4f} m | wheel_base {b:.4f} m")
    with Simulation(CONFIG, seed=a.seed) as s:      # EVALUATION ONLY: hidden true values
        print(f"hidden   : wheel_radius {s.r_true:.4f} m | wheel_base {s.b_true:.4f} m (before slip)")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"wheel_radius": r, "wheel_base": b, "seed": a.seed}, indent=2))
    print("saved:", OUT_FILE)