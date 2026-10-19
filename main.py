# Including libraries
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from calibration import get_calibration, load_intrinsics, marker_measurement_covariance
from config import CONFIG, RESULTS_DIR, Config, RealismConfig
from controller import LQRTracker
from docking import DockingController
from estimator import EKFEstimator, chi2_bounds, nees
from navigation import Navigator
from recovery import RecoveryPhase, TagRecovery
from robot_model import TagMap, analyse_model, wrap_angle
from sensing import Sensing, sector_min_range
from simulation import Simulation
from visualization import RouteVisualizer

MAX_RECOVERIES = 3   # maximum number of tag recoveries per run (avoid endless retry loops)
EXPLORE_SCAN_ANGLE = math.pi   # EXPLORE: rotation at every stop [rad] (use math.pi for a faster 180 deg scan)
EXPLORE_STEP = 1.0                   # EXPLORE: distance driven between two scans [m]

class Phase(Enum):
    SURVEY = "survey"
    EXPLORE = "explore"
    NAVIGATE = "navigate"
    RECOVER = "recover"
    DOCK = "dock"
    DONE = "done"
    FAILED = "failed"

# Docking quality of a pose relative to the dock
def dock_errors(pose, cfg: Config = CONFIG) -> dict:
    dx, dy, yaw = cfg.env.dock_pose
    x, y, theta = pose
    return {
        "dock_range": math.hypot(dx - x, dy - y),
        "dock_lateral": abs(math.cos(yaw) * (y - dy) - math.sin(yaw) * (x - dx)),
        "dock_heading": abs(wrap_angle(theta - (yaw - math.pi))),
    }

# Pose / tag poses from the estimator frame (origin = robot start) to the world frame.
# EVALUATION and DISPLAY only: the controller never sees the world frame.
def est_to_world(pose_local, start) -> np.ndarray:
    sx, sy, sth = (float(v) for v in start)
    x, y, th = (float(v) for v in pose_local)
    c, s = math.cos(sth), math.sin(sth)
    return np.array([sx + c * x - s * y, sy + s * x + c * y, wrap_angle(th + sth)])

def table_to_world(table: dict, start) -> dict:
    out = {}
    for i, (p, yaw) in table.items():
        w = est_to_world((p[0], p[1], yaw), start)
        out[int(i)] = (w[:2], float(w[2]))
    return out

@dataclass
class TrialLog:
    t: list = field(default_factory=list)
    truth: list = field(default_factory=list)   # ground truth: EVALUATION ONLY
    est: list = field(default_factory=list)
    sigma: list = field(default_factory=list)   # sqrt(diag(P))
    phase: list = field(default_factory=list)
    v: list = field(default_factory=list)
    w: list = field(default_factory=list)
    nees: list = field(default_factory=list)         # evaluation only (uses ground truth)
    lidar_valid: list = field(default_factory=list)  # valid LiDAR beams per step
    updates: list = field(default_factory=list)      # cumulative EKF marker updates
    tag_updates: list = field(default_factory=list)  # cumulative EKF updates from extra tags
    mode: list = field(default_factory=list)         # TRACK / AVOID / DOCK-... / RECOVER-... / DONE

    def record(self, t, truth, est, P, phase, v, w,
            lidar_valid: int = 0, updates: int = 0, mode: str = "",
            tag_updates: int = 0) -> None:
        self.t.append(float(t))
        self.truth.append(np.array(truth, dtype=float))
        self.est.append(np.array(est, dtype=float))
        self.sigma.append(np.sqrt(np.diag(P)))
        self.phase.append(phase)
        self.v.append(float(v))
        self.w.append(float(w))
        try:
            self.nees.append(nees(est, truth, P))
        except np.linalg.LinAlgError:
            self.nees.append(float("nan"))
        self.lidar_valid.append(int(lidar_valid))
        self.updates.append(int(updates))
        self.tag_updates.append(int(tag_updates))
        self.mode.append(str(mode))

    def arrays(self) -> dict:
        return {
            "t": np.array(self.t),
            "truth": np.array(self.truth).reshape(-1, 3),
            "est": np.array(self.est).reshape(-1, 3),
            "sigma": np.array(self.sigma).reshape(-1, 3),
            "v": np.array(self.v),
            "w": np.array(self.w),
            "nees": np.array(self.nees),
            "lidar_valid": np.array(self.lidar_valid),
            "updates": np.array(self.updates),
            "tag_updates": np.array(self.tag_updates),
            "mode": np.array(self.mode, dtype=str),
        }

# ------------------------------------------------- live table + reports
# Column titles and widths of the real-time monitor# Column titles and widths of the real-time monitor
_COLS = (("Time (s)", 8), ("Pose est. x,y,th(deg)", 23), ("sigma xy", 9),
        ("LiDAR valid", 12), ("upd (tag)", 10), ("Tags seen", 10),
        ("v, w", 12), ("Mode", 14))

def _border(left: str, mid: str, right: str) -> str:
    return left + mid.join("─" * (w + 2) for _, w in _COLS) + right

def _line(cells) -> str:
    return "│ " + " │ ".join(str(c).ljust(w) for c, (_, w) in zip(cells, _COLS)) + " │"

def table_header() -> str:
    return "\n".join([
        "TRACK A - REAL-TIME SIMULATION MONITOR",
        _border("┌", "┬", "┐"),
        _line([title for title, _ in _COLS]),
        _border("├", "┼", "┤"),
    ])

def table_footer() -> str:
    return _border("└", "┴", "┘")

# One table row: the pose shown is the ESTIMATED pose (never ground truth)
def table_row(t, pose, lidar_valid, lidar_total, updates, v, w, mode,
            sigma_xy: float = float("nan"), tag_updates: int = 0, tags_seen=()) -> str:
    seen = ",".join(str(i) for i in tags_seen) if len(tags_seen) else "-"
    return _line([
        f"{t:.1f}",
        f"{pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.1f}°",
        f"{sigma_xy:.3f}",
        f"{lidar_valid}/{lidar_total}",
        f"{updates} ({tag_updates})",
        seen,
        f"{v:.2f},{w:.2f}",
        mode,
    ])

# Mode label shown in the table (navigation, recovery, docking or final state)
def mode_label(phase: Phase, ncmd=None, dcmd=None, rphase=None) -> str:
    if phase is Phase.NAVIGATE and ncmd is not None:
        return ncmd.phase.value.upper()
    if phase is Phase.DOCK and dcmd is not None:
        return "DOCK-" + dcmd.phase.value.upper()
    if phase is Phase.RECOVER and rphase is not None:
        return "RECOVER-" + rphase.value.upper()
    return phase.value.upper()

# New navigator after a recovery. Lost while navigating: continue with the remaining waypoints.
# Docking failed: drive to the approach point in front of the dock (not the wall-side last waypoint).
def resume_navigator(nav: Navigator, cfg: Config, after_dock: bool, approach=None) -> Navigator:
    if after_dock and approach is not None:
        remaining = [approach]
    elif after_dock:
        remaining = [nav.waypoints[-1]]
    else:
        remaining = nav.waypoints[nav.index:]
    if not remaining:
        remaining = [nav.waypoints[-1]]
    return Navigator(cfg, tracker=nav.tracker, waypoints=remaining)

# Point in front of the dock (learned pose only)
def dock_approach_point(cfg: Config, lm, yaw) -> np.ndarray:
    return np.array([lm[0] + cfg.tags.approach_dist * math.cos(yaw),
                    lm[1] + cfg.tags.approach_dist * math.sin(yaw)])


# Rotate a little, stand still, observe, repeat: the camera image is sharp and not delayed while still
class StopGoScan:
    def __init__(self, omega: float = 0.8, step: float = 0.5, hold: float = 0.7, settle: float = 0.3):
        self.omega, self.step, self.hold, self.settle = omega, step, hold, settle
        self.reset()

    def reset(self) -> None:
        self.spun = 0.0          # total rotation since reset [rad]
        self.step_spun = 0.0     # rotation since the last stop [rad]
        self.hold_t0 = None      # time when the robot stopped
        self.started = False     # the first stop happens right after reset (look before turning)

    # Returns (w, observe). observe is True only when the robot has been still long enough
    def update(self, t: float, dtheta: float):
        self.spun += abs(dtheta)
        if not self.started:
            self.started = True
            self.hold_t0 = t
        if self.hold_t0 is not None:
            if t - self.hold_t0 >= self.hold:
                self.hold_t0, self.step_spun = None, 0.0
                return self.omega, False
            return 0.0, t - self.hold_t0 >= self.settle
        self.step_spun += abs(dtheta)
        if self.step_spun >= self.step:
            self.hold_t0 = t
            return 0.0, False
        return self.omega, False

# Direction (robot frame) with the most free space, with a small preference for straight ahead
def freest_direction(lidar, half_width: float = 0.4) -> float:
    best, best_score = 0.0, -1e9
    for a in np.linspace(-math.pi, math.pi, 36, endpoint=False):
        score = sector_min_range(lidar, float(a), half_width) - 0.1 * abs(float(a))
        if score > best_score:
            best, best_score = float(a), score
    return best


# Map seen by the EKF: the dock stays hidden until it has been re-learned from a close view
class MapView:
    def __init__(self, tag_map, dock_id: int):
        self.tag_map, self.dock_id, self.show_dock = tag_map, int(dock_id), False

    def table(self) -> dict:
        t = self.tag_map.table()
        if not self.show_dock:
            t.pop(self.dock_id, None)
        return t

# Static theory block printed once before the run: quantities that do not change in time
def theory_block(cfg: Config = CONFIG, estimator=None) -> str:
    x0 = np.array(cfg.env.start_pose, dtype=float)
    moving = analyse_model(x0, 0.15, 0.0, cfg=cfg)
    stopped = analyse_model(x0, 0.0, 0.0, cfg=cfg)
    tracker = LQRTracker(cfg)
    poles = np.abs(tracker.closed_loop_poles())
    cal = get_calibration(cfg)
    n = moving["n_states"]
    ekf_cfg = cfg.ekf

    # Extra checkpoint tags (everything in the tag table except the dock)
    dock_id = int(cfg.camera.marker_id)
    tags = {int(i): p for i, p in cfg.tag_table().items() if int(i) != dock_id}
    tag_lines = []
    for tid, (tx, ty, tyaw) in sorted(tags.items()):
        sx = tx + cfg.tags.standoff * math.cos(tyaw)
        sy = ty + cfg.tags.standoff * math.sin(tyaw)
        tag_lines.append(f"  tag id {tid}: x = {tx:+.2f}, y = {ty:+.2f}, yaw = {math.degrees(tyaw):+.0f} deg"
                        f" | stand-off ({sx:+.2f}, {sy:+.2f})")

    return "\n".join([
        "=== A. MODEL (robot_model.py) ===",
        f"Differential drive, state [x y theta], input [v w], control dt = {cfg.sim.control_dt} s",
        f"Controllability rank (moving | stopped): {moving['controllability_rank']}/{n} | "
        f"{stopped['controllability_rank']}/{n}",
        f"Observability rank   (moving | stopped): {moving['observability_rank']}/{n} | "
        f"{stopped['observability_rank']}/{n}",
        "=== B. SENSING + CALIBRATION (sensing.py, calibration.py) ===",
        f"Camera intrinsics: {cal.intrinsics.source} | focal {cal.K[0, 0]:.1f} px | "
        f"frame conventions OK | camera offset {cal.forward_offset:.2f} m",
        f"LiDAR: {cfg.lidar.num_beams} beams, {cfg.lidar.range_min}-{cfg.lidar.range_max} m, "
        f"sigma {cfg.lidar.noise_std} m | camera period 0.1 s (Sensing default)",
                "Marker noise model (sigma_px = 0.5): " + ", ".join(
        f"{d} m -> ({math.sqrt(marker_measurement_covariance(d, cal.intrinsics, cfg.camera.marker_size)[0, 0]) * 1000:.1f} mm, "
        f"{math.sqrt(marker_measurement_covariance(d, cal.intrinsics, cfg.camera.marker_size)[1, 1]):.4f} rad)"
        for d in (0.5, 1.0, 2.0)),
        "=== C. ESTIMATION (estimator.py) ===",
        f"{type(estimator).__name__ if estimator is not None else 'EKF'} | "
        f"R std (range, bearing) = ({cfg.camera.range_noise_std} m, {cfg.camera.bearing_noise_std} rad) | "
        f"P0 std = {ekf_cfg.init_pose_std} | gate = {getattr(estimator, 'gate', 'n/a')} | "
        f"marker R: {'per-measurement' if getattr(estimator, 'use_marker_cov', False) else 'fixed (config)'}",
        "=== D. CONTROL (controller.py, LQR) ===",
        f"Q = diag{cfg.controller.q_diag}, R = diag{cfg.controller.r_diag}",
        f"K = {np.round(tracker.K, 3).tolist()}",
        f"Closed-loop |poles| max = {poles.max():.3f} "
        f"({'stable' if poles.max() < 1.0 else 'UNSTABLE'})",
        "=== E. INTEGRATION (navigation.py, docking.py, main.py) ===",
        f"{len(cfg.env.waypoints)} waypoints, {len(cfg.env.obstacles)} obstacles, "
        f"dock at {tuple(round(v, 2) for v in cfg.env.dock_pose)}",
        "Controller input: ESTIMATED pose only (ground truth is logged for evaluation)",
        "=== F. RECOVERY (recovery.py, tags in config.py) ===",
        f"{len(tags)} extra tags (ids {', '.join(str(i) for i in sorted(tags)) or '-'}) | "
        f"lost when sqrt(trace(Pxy)) > {cfg.tags.lost_std} m | stand-off {cfg.tags.standoff} m | "
        f"spin {cfg.tags.spin_omega} rad/s | max {MAX_RECOVERIES} recoveries",
        *tag_lines,
        "",
    ])


# Obstacles that were really in the scene during the run (falls back to the config layout)
def result_obstacles(result: RunResult, cfg: Config = CONFIG) -> list:
    return list(result.obstacles) if result.obstacles is not None else list(cfg.env.obstacles)

# Print the obstacle layout used in a run (to verify that it is random)
def obstacles_text(obstacles) -> str:
    lines = [f"Obstacles ({len(obstacles)}):"]
    for i, (ox, oy, r) in enumerate(obstacles):
        lines.append(f"  #{i}: x = {ox:+.3f} m, y = {oy:+.3f} m, r = {r:.2f} m")
    return "\n".join(lines)

# Quantitative metrics of one run (ground truth is used here for EVALUATION only)
def compute_report(result: RunResult, cfg: Config = CONFIG) -> dict:
    a = result.log.arrays()
    if len(a["t"]) == 0:
        return {}
    err = a["est"] - a["truth"]
    err[:, 2] = (err[:, 2] + math.pi) % (2.0 * math.pi) - math.pi
    modes = a["mode"]
    nav_steps = np.isin(modes, ["TRACK", "AVOID"])
    recover_steps = np.char.startswith(modes, "RECOVER")
    rob = cfg.robot
    hx, hy = cfg.env.room_half_size
    truth = a["truth"]
    clearance = min([float(np.min(np.hypot(truth[:, 0] - ox, truth[:, 1] - oy)) - r - rob.body_radius)
                    for ox, oy, r in result_obstacles(result, cfg)] or [float("inf")])
    wall = float(np.min(np.minimum(hx - np.abs(truth[:, 0]), hy - np.abs(truth[:, 1])))
                - rob.body_radius)
    saturated = (a["v"] >= 0.99 * rob.max_linear_vel) | (np.abs(a["w"]) >= 0.99 * rob.max_angular_vel)
    finite = np.isfinite(a["nees"])
    finite_nees = a["nees"][finite]
    max_nees, max_nees_time = float("nan"), float("nan")
    if len(finite_nees):
        i_max = int(np.argmax(finite_nees))
        max_nees, max_nees_time = float(finite_nees[i_max]), float(a["t"][finite][i_max])
    return {
        "position_rmse": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
        "heading_rmse": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "mean_nees": float(finite_nees.mean()) if len(finite_nees) else float("nan"),
        "max_nees": max_nees,
        "max_nees_time": max_nees_time,
        "final_sigma": a["sigma"][-1],
        "marker_updates": int(a["updates"][-1]),
        "mean_lidar_valid": float(a["lidar_valid"].mean()),
        "mean_v": float(a["v"].mean()), "max_v": float(a["v"].max()),
        "mean_abs_w": float(np.abs(a["w"]).mean()), "max_abs_w": float(np.abs(a["w"]).max()),
        "saturated_fraction": float(saturated.mean()),
        "avoid_fraction": float(np.sum(modes == "AVOID") / max(1, nav_steps.sum())),
        "recover_time": float(recover_steps.sum() * cfg.sim.control_dt),
        "min_obstacle_clearance": clearance,
        "min_wall_clearance": wall,
    }

# Final report grouped by the marking criteria (printed once, after the run)
def print_report(result: RunResult, cfg: Config = CONFIG, dr_rmse: Optional[float] = None) -> None:
    r = compute_report(result, cfg)
    if not r:
        print("No data logged.")
        return
    m = result.metrics(cfg)
    dr = "not run (use --compare)" if dr_rmse is None else f"{dr_rmse:.3f} m"
    d = result.diag
    print("=== FINAL REPORT ===")
    n_cam, n_det = result.camera_stats
    sensing_line = (f"B. Sensing    : mean valid LiDAR beams {r['mean_lidar_valid']:.0f}/{cfg.lidar.num_beams} | "
        f"dock marker accepted {n_det}/{n_cam} camera frames ({100 * n_det / max(1, n_cam):.0f} %)")
    if d and "n_detect_any" in d:
        sensing_line += (f" | any marker (dock or tag) {d['n_detect_any']}/{n_cam} "
                        f"({100 * d['n_detect_any'] / max(1, n_cam):.0f} %)")
    print(sensing_line)
    est_line = f"C. Estimation : marker updates {r['marker_updates']}"
    if d and "n_tag_updates" in d:
        est_line += f" (dock {r['marker_updates'] - d['n_tag_updates']}, tags {d['n_tag_updates']})"
    print(est_line + f" | position RMSE {r['position_rmse']:.3f} m | heading RMSE {r['heading_rmse']:.3f} rad")
    print(f"                mean NEES {r['mean_nees']:.2f} (consistent filter: ~3) | "
        f"max NEES {r['max_nees']:.1f} at t = {r['max_nees_time']:.1f} s | "
        f"final sigma {np.round(r['final_sigma'], 4).tolist()}")
    print(f"                dead-reckoning baseline RMSE: {dr}")
    if d:
        print(f"                gate rejected {d['n_rejected']} (forced recoveries {d['n_recovered']}) | "
            f"mean NIS/dof {d['nis_per_dof']:.2f} (consistent: ~1) | "
            f"LiDAR updates {d['n_lidar_updates']} (rejected {d['n_lidar_rejected']})")
    print(f"D. Control    : mean/max v {r['mean_v']:.2f}/{r['max_v']:.2f} m/s | "
        f"mean/max |w| {r['mean_abs_w']:.2f}/{r['max_abs_w']:.2f} rad/s | "
          f"saturated {100 * r['saturated_fraction']:.1f} % of steps")
    print(f"E. Integration: AVOID {100 * r['avoid_fraction']:.1f} % of navigation | "
        f"min obstacle clearance {r['min_obstacle_clearance']:.2f} m | "
        f"min wall clearance {r['min_wall_clearance']:.2f} m | collision {result.collided}")
    print(f"                docking range {m['dock_range']:.3f} m, lateral {m['dock_lateral']:.3f} m, "
        f"heading {m['dock_heading']:.3f} rad")
    n_tags = len(cfg.tag_table()) - 1
    print(f"F. Recovery   : {n_tags} extra tags | recoveries started {d.get('n_tag_recoveries', 0)}/{MAX_RECOVERIES} | "
        f"tag updates {d.get('n_tag_updates', 0)} | time in recovery {r['recover_time']:.1f} s")
    errs = d.get("tag_map_errors")
    if errs:
        print("                learned tag map error [m]: "
            + ", ".join(f"id {i}: {e:.3f}" for i, e in sorted(errs.items())))
    for event in result.events:
        print(f"                {event}")

    # Automatic sanity warnings (loose single-sample chi-square band: runs are correlated in time)
    warnings = []
    lo, hi = chi2_bounds(3, 1, alpha=0.01)
    if math.isfinite(r["mean_nees"]) and not lo <= r["mean_nees"] <= hi:
        warnings.append(f"mean NEES {r['mean_nees']:.1f} outside [{lo:.1f}, {hi:.1f}]: filter over-confident "
                        f"or biased (worst at t = {r['max_nees_time']:.1f} s, NEES {r['max_nees']:.1f})")
    if d.get("n_recovered", 0) > 0:
        warnings.append(f"EKF gate lock-out recovery forced {d['n_recovered']} times: measurements were "
                        f"rejected repeatedly, the estimate or the map may be wrong")
    if d and math.isfinite(d.get("nis_per_dof", float("nan"))) and d["nis_per_dof"] > 3.0:
        warnings.append(f"mean NIS/dof {d['nis_per_dof']:.2f} > 3: measurement noise is underestimated")
    if result.collided:
        warnings.append("collision detected during the run")
    if warnings:
        print("WARNINGS      :")
        for w_ in warnings:
            print(f"                - {w_}")

    print(f"Result        : {'SUCCESS' if result.success else 'FAILURE'} ({result.reason}) "
        f"in {result.time:.1f} s | ground truth used in control: NO (evaluation only)")

@dataclass
class RunResult:
    success: bool
    reason: str                 # docked | docking failed | collision | timeout | viewer closed
    time: float                 # simulated seconds
    nav_time: Optional[float]   # time when navigation handed over to docking
    collided: bool
    phase: Phase
    estimate: np.ndarray        # final estimated pose
    truth: np.ndarray           # final true pose (EVALUATION ONLY)
    log: TrialLog
    camera_stats: tuple = (0, 0)                # (camera frames processed, accepted detections)
    diag: dict = field(default_factory=dict)    # EKF counters + NIS (evaluation/diagnostics)
    obstacles: Optional[list] = None            # obstacle layout used in the run (None = config layout)
    events: list = field(default_factory=list)  # timeline of LOST / RELOCALIZED / docking events
    world_cfg: Optional[Config] = None          # the REAL world (dock moved, ...): EVALUATION ONLY

    def metrics(self, cfg: Config = CONFIG) -> dict:
        err = self.estimate - self.truth
        out = {"position_error": math.hypot(err[0], err[1]),
            "heading_error": abs(wrap_angle(err[2]))}
        out.update(dock_errors(self.truth, cfg))
        return out

    def summary(self, cfg: Config = CONFIG) -> str:
        m = self.metrics(cfg)
        nav = "n/a" if self.nav_time is None else f"{self.nav_time:.1f} s"
        return "\n".join([
            f"Result            : {'SUCCESS' if self.success else 'FAILURE'} ({self.reason})",
            f"Total time        : {self.time:.1f} s (navigation finished at {nav})",
            f"Collision         : {self.collided}",
            f"Estimate error    : {m['position_error']:.3f} m, {m['heading_error']:.3f} rad",
            f"Dock range        : {m['dock_range']:.3f} m",
            f"Dock lateral error: {m['dock_lateral']:.3f} m",
            f"Dock heading error: {m['dock_heading']:.3f} rad",
        ])

@contextmanager
def _viewer(sim: Simulation, enabled: bool):
    if not enabled:
        yield None
        return
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as handle:
        handle.cam.lookat[:] = [0.0, 0.0, 0.0]
        handle.cam.distance = 6.5
        handle.cam.azimuth = 90.0
        handle.cam.elevation = -75.0
        yield handle

def run_pipeline(cfg: Config = CONFIG, seed: Optional[int] = None, estimator=None,
                scene_dir: Optional[Path] = None, viewer: bool = False,
                stop_on_collision: bool = True, verbose: bool = False,
                dock_use_estimate: bool = False, marker_sigma_px: float = 0.5,
                use_marker_cov: bool = False,
                visualizer: Optional[RouteVisualizer] = None,
                randomize_obstacles: bool = False,
                use_recovery: bool = True,
                use_survey: Optional[bool] = None,
                use_lidar_map: bool = False) -> RunResult:
    log = TrialLog()
    phase, reason = Phase.NAVIGATE, "timeout"
    nav_time: Optional[float] = None
    collided_any = False
    cam_stats = (0, 0)
    n_detect_any = 0
    viz_outputs = None
    n_tag_recoveries = 0                 # how many times the tag recovery was started
    recovery_after_dock = False          # True when the recovery was started by a failed docking
    events: list = []                    # timeline strings, also printed live when verbose
    last_target: Optional[int] = None    # tag the recovery is currently driving to
    last_tags: tuple = ()                # ids seen in the latest processed camera frame
    last_n_cam = -1
    tag_map: Optional[TagMap] = None     # tag poses learned during SURVEY
    dock_id = int(cfg.camera.marker_id)
    dock_lm = dock_yaw = None            # dock pose from the learned map (None = config pose)
    survey_t0: Optional[float] = None
    explore_t0: Optional[float] = None   # start time of the EXPLORE phase
    spun = 0.0                           # rotation accumulated during SURVEY [rad]
    scan = StopGoScan()         # SURVEY / EXPLORE: rotate a little, stop, observe
    ex_state = "PICK"                    # EXPLORE: PICK -> TURN -> MOVE -> SCAN -> PICK ...
    ex_target = 0.0                      # EXPLORE: heading to turn to [rad]
    ex_moved = 0.0                       # EXPLORE: distance driven in the current MOVE [m]
    dock_near = TagMap(min_obs=3)        # dock observations made from close range only
    dock_refined = False                 # True once the dock pose was re-learned from a close view
    aim_mode = False                     # NAVIGATE driven by "look - turn - move" instead of a fixed waypoint
    ap_state = "LOOK"                    # LOOK -> TURN -> MOVE -> LOOK ... (SEARCH when the dock is not seen)
    ap_t0 = None                         # time when the current LOOK started
    ap_obs = []                          # (range, bearing) of the dock seen while standing still
    ap_target = 0.0                      # heading to turn to [rad]
    ap_goal = 0.0                        # distance to drive in the current MOVE [m]
    ap_moved = 0.0                       # distance driven so far in the current MOVE [m]
    ap_idle = 0.0
    obstacles = list(cfg.env.obstacles)
    world_cfg = cfg
    start = np.array(cfg.env.start_pose, dtype=float)
    dt = cfg.sim.control_dt
    report_every = max(1, int(round(1.0 / dt)))      # one table row per simulated second

    # Record an event and print it right away (verbose runs only)
    def note(t: float, text: str) -> None:
        events.append(f"[t={t:.1f}s] {text}")
        if verbose:
            print(f"  >> [t={t:5.1f}s] {text}", flush=True)

    with Simulation(cfg, seed=seed, scene_dir=scene_dir,
                    randomize_obstacles=randomize_obstacles) as sim:
        # Effective config: the obstacles that are really in the scene (random or fixed)
        obstacles = list(sim.obstacles)
        world_cfg = sim.world_cfg
        run_cfg = replace(cfg, env=replace(cfg.env, obstacles=tuple(obstacles)))
        nav, dock = Navigator(run_cfg), DockingController(run_cfg)
        recovery = TagRecovery(run_cfg, tracker=nav.tracker)

        # The EKF LiDAR map must contain the obstacles that are really in the scene
        if estimator is None:
            ekf_env = replace(run_cfg.env, obstacles=()) if cfg.realism.enabled else run_cfg.env
            estimator = EKFEstimator(replace(run_cfg, env=ekf_env), gate=9.21,
                                    use_marker_cov=use_marker_cov, use_lidar=use_lidar_map)
        if verbose:
            print(obstacles_text(obstacles))
            print(theory_block(run_cfg, estimator))
            print(table_header())

        sim.reset(tuple(start))              # `start` places the robot in the hidden world (simulation + evaluation only)
        intr_file = RESULTS_DIR / "camera_intrinsics.json"
        cal = get_calibration(cfg, intrinsics=load_intrinsics(intr_file) if intr_file.exists() else None)
        sensing = Sensing(sim, cfg, camera_matrix_=cal.K, dist_coeffs=cal.dist_coeffs,
                        marker_sigma_px=marker_sigma_px)
        estimator.reset((0.0, 0.0, 0.0))     # the robot's own frame: origin = start pose
        # Always learn the markers at run time: no tag / dock poses from the config
        tag_map = TagMap(min_obs=run_cfg.tags.survey_min_obs)
        map_view = MapView(tag_map, dock_id)
        estimator.set_tag_map(map_view)       # empty map: markers are ignored until learned
        phase = Phase.SURVEY
        if visualizer is not None:
            visualizer.start(sim)

        with _viewer(sim, viewer) as view:
            step = 0
            last_rot_t = -1.0
            while not sim.time_exceeded:
                t0 = time.perf_counter()
                if view is not None and not view.is_running():
                    reason = "viewer closed"
                    break

                # 1) sensing and state estimation (no ground truth)
                frame = sensing.read()
                estimator.predict(frame.odometry.ds, frame.odometry.dtheta)
                if abs(frame.odometry.dtheta) / dt > 0.05:
                    last_rot_t = frame.time
                if frame.time - last_rot_t >= 0.3:
                    for marker in frame.markers:     # dock + extra tags, matched by id in the EKF
                        estimator.update(marker)
                estimator.update_lidar(frame.lidar)
                x_est = np.array(estimator.x, dtype=float)
                x_eval = est_to_world(x_est, start)      # world frame, for logging / display only
                sigma_xy = float(math.sqrt(max(float(np.trace(estimator.P[:2, :2])), 0.0)))
                if sensing.n_camera != last_n_cam:   # a new camera frame was processed
                    last_n_cam = sensing.n_camera
                    last_tags = tuple(sorted(m.marker_id for m in frame.markers))

                # 2) decide the velocity command from the ESTIMATED pose only
                v = w = 0.0
                ncmd = dcmd = None

                # 2-pre) SURVEY: spin at the known start pose and learn the tag poses (no ground truth).
                # dock_first: scan a full turn (or until the dock is seen) and use the learned dock as the goal.
                if phase is Phase.SURVEY:
                    if survey_t0 is None:
                        survey_t0 = frame.time
                        note(frame.time, "SURVEY: scanning 180 deg at the start pose to learn the tag map")
                    first = run_cfg.tags.dock_first
                    w_scan, observe = scan.update(frame.time, frame.odometry.dtheta)
                    if observe:                      # learn only while standing still
                        for marker in frame.markers:
                            is_dock = int(marker.marker_id) == dock_id
                            limit = run_cfg.tags.dock_max_range if (first and is_dock) else run_cfg.tags.survey_max_range
                            tag_map.add(x_est, marker, max_range=limit)
                    dock_seen = first and tag_map.count(dock_id) >= run_cfg.tags.survey_min_obs
                    angle_limit = run_cfg.tags.survey_angle          # 180 deg maximum
                    if dock_seen or scan.spun >= angle_limit or frame.time - survey_t0 > run_cfg.tags.survey_max_time:
                        learned = tag_map.table()
                        estimator.set_tag_map(map_view)
                        recovery = TagRecovery(run_cfg, tracker=nav.tracker, tag_map=tag_map)
                        scan.reset()
                        if dock_id in learned:
                            dock_lm, dock_yaw = learned[dock_id]
                        elif not first:
                            note(frame.time, "SURVEY: dock too far to learn from the start pose "
                                            "-> navigate and scan for it on the way")
                        if visualizer is not None:
                            visualizer.set_tag_map(table_to_world(learned, start))
                        if first and dock_id not in learned:
                            note(frame.time, "SURVEY: dock not seen -> EXPLORE (move, stop, scan)")
                            phase, explore_t0, ex_state = Phase.EXPLORE, frame.time, "PICK"
                        elif not learned:
                            note(frame.time, "SURVEY FAILED: no tag could be learned from the start pose")
                            phase, reason = Phase.FAILED, "survey failed"
                        else:
                            note(frame.time, f"SURVEY done: learned ids {sorted(learned)}")
                            if first:      # goal comes from the scan, not from the config waypoints
                                aim_mode, ap_state, ap_t0 = True, "LOOK", None
                            phase = Phase.NAVIGATE
                    else:
                        v, w = 0.0, w_scan

                # 2-pre-a) EXPLORE (dock_first only): drive straight, turn left near obstacles/walls,
                # until the dock has been seen enough times to learn its pose
                if phase is Phase.EXPLORE:
                    observe = False
                    if ex_state == "SCAN":                   # rotate-stop-rotate, observe while still
                        w, observe = scan.update(frame.time, frame.odometry.dtheta)
                        if scan.spun >= EXPLORE_SCAN_ANGLE:
                            ex_state = "PICK"
                    elif ex_state == "PICK":                 # head for the most open direction (LiDAR only)
                        ex_target = wrap_angle(x_est[2] + freest_direction(frame.lidar))
                        ex_state = "TURN"
                    elif ex_state == "TURN":
                        err = wrap_angle(ex_target - x_est[2])
                        if abs(err) < 0.1:
                            ex_state, ex_moved = "MOVE", 0.0
                        else:
                            w = float(np.clip(2.0 * err, -0.6, 0.6))
                    elif ex_state == "MOVE":                 # drive ~1 m, then stop and scan again
                        ex_moved += frame.odometry.ds
                        if ex_moved >= EXPLORE_STEP or sector_min_range(frame.lidar, 0.0, 0.5) < 0.6:
                            ex_state = "SCAN"
                            scan.reset()
                        else:
                            v = 0.2
                    if observe:
                        for marker in frame.markers:
                            is_dock = int(marker.marker_id) == dock_id
                            limit = run_cfg.tags.dock_max_range if is_dock else run_cfg.tags.survey_max_range
                            tag_map.add(x_est, marker, max_range=limit)
                    if tag_map.count(dock_id) >= run_cfg.tags.survey_min_obs:
                        learned = tag_map.table()
                        estimator.set_tag_map(map_view)
                        recovery = TagRecovery(run_cfg, tracker=nav.tracker, tag_map=tag_map)
                        dock_lm, dock_yaw = learned[dock_id]
                        scan.reset()
                        aim_mode, ap_state, ap_t0 = True, "LOOK", None
                        if visualizer is not None:
                            visualizer.set_tag_map(table_to_world(learned, start))
                        note(frame.time, f"EXPLORE done: dock learned, ids {sorted(learned)}")
                        phase = Phase.NAVIGATE
                    elif frame.time - explore_t0 > 60.0:
                        note(frame.time, "EXPLORE FAILED: dock not seen within 60 s")
                        phase, reason = Phase.FAILED, "explore failed"

                # 2-pre-c) re-learn the dock from CLOSE observations (accurate), then let the EKF use it
                if phase in (Phase.NAVIGATE, Phase.DOCK) and frame.marker is not None:
                    m_near = frame.marker
                    if m_near.range < 1.5 and abs(m_near.view_angle) < 0.6:
                        dock_near.add(x_est, m_near, max_range=1.5)
                        if dock_near.count(dock_id) >= 3:
                            tag_map._obs[dock_id] = list(dock_near._obs[dock_id])
                            dock_lm, dock_yaw = dock_near.table()[dock_id]
                            map_view.show_dock = True
                            estimator.set_tag_map(map_view)
                            dock.set_landmark(dock_lm, dock_yaw)
                            if not dock_refined:
                                dock_refined = True
                                note(frame.time, "dock re-learned from a close view")

                # 2-pre-b) learn the tags / dock that were too far during SURVEY.
                # Only close observations while the filter is confident.
                if tag_map is not None and phase in (Phase.NAVIGATE, Phase.DOCK) and not estimator.is_lost():
                    grew = False
                    for marker in frame.markers:
                        mid = int(marker.marker_id)
                        if tag_map.count(mid) < run_cfg.tags.survey_min_obs:
                            tag_map.add(x_est, marker, max_range=run_cfg.tags.survey_max_range)
                            grew = grew or tag_map.count(mid) >= run_cfg.tags.survey_min_obs
                    if grew:
                        learned = tag_map.table()
                        estimator.set_tag_map(map_view)
                        recovery = TagRecovery(run_cfg, tracker=nav.tracker, tag_map=tag_map)
                        if dock_id in learned and dock_lm is None:
                            dock_lm, dock_yaw = learned[dock_id]
                            dock.set_landmark(dock_lm, dock_yaw)
                        if visualizer is not None:
                            visualizer.set_tag_map(table_to_world(learned, start))
                        note(frame.time, f"map updated while navigating: learned ids {sorted(learned)}")

                # 2a) position uncertainty too large while navigating: start the tag recovery
                if (use_recovery and phase is Phase.NAVIGATE and not aim_mode and recovery.available
                        and n_tag_recoveries < MAX_RECOVERIES and estimator.is_lost()):
                    recovery.start()
                    n_tag_recoveries += 1
                    recovery_after_dock = False
                    last_target = None
                    phase = Phase.RECOVER
                    note(frame.time, f"LOST (sigma {sigma_xy:.3f} m > {cfg.tags.lost_std} m) "
                                    f"-> tag recovery #{n_tag_recoveries} started (spinning to look for a tag)")

                # 2b) recovery: spin / drive to the tags until the EKF is localised again
                if phase is Phase.RECOVER:
                    v, w = recovery.update(x_est, frame.lidar, not estimator.is_lost(),
                                        frame.odometry.dtheta)
                    if recovery.phase is RecoveryPhase.GOTO and recovery.target_id != last_target:
                        last_target = recovery.target_id
                        sx, sy = recovery.standoff_point(last_target)
                        note(frame.time, f"recovery: no tag seen, driving to tag {last_target} "
                                        f"stand-off ({sx:+.2f}, {sy:+.2f})")
                    if recovery.phase is RecoveryPhase.MAP_GOTO and last_target != -1:
                        last_target = -1
                        ax_, ay_ = recovery.approach_point()
                        note(frame.time, f"recovery: no tag found -> following the learned map to the "
                                        f"dock approach point ({ax_:+.2f}, {ay_:+.2f})")
                    if recovery.phase is RecoveryPhase.DONE:
                        if recovery.arrived_by_map:
                            note(frame.time, "map-follow finished (not relocalized) -> docking")
                            nav_time = frame.time if nav_time is None else nav_time
                            phase = Phase.DOCK
                            dock.begin(x_est, dock_lm, dock_yaw)
                        else:
                            note(frame.time, f"RELOCALIZED (sigma {sigma_xy:.3f} m, tag updates "
                                            f"{int(getattr(estimator, 'n_tag_updates', 0))}) -> resume navigation")
                            approach = None if dock_lm is None else dock_approach_point(run_cfg, dock_lm, dock_yaw)
                            nav = resume_navigator(nav, run_cfg, recovery_after_dock, approach)
                            phase = Phase.NAVIGATE
                        v = w = 0.0
                    elif recovery.phase is RecoveryPhase.FAILED:
                        note(frame.time, "recovery FAILED: nothing left to try")
                        phase, reason = Phase.FAILED, "recovery failed"
                        v = w = 0.0

                # 2c-pre) AIM: look at the dock while standing still, turn towards it, drive a short step, repeat
                if phase is Phase.NAVIGATE and aim_mode:
                    if ap_state == "LOOK":
                        if ap_t0 is None:
                            ap_t0, ap_obs = frame.time, []
                        if frame.time - ap_t0 >= 0.4 and frame.marker is not None:
                            ap_obs.append((frame.marker.range, frame.marker.bearing))
                        if frame.time - ap_t0 >= 1.2:
                            ap_t0 = None
                            if len(ap_obs) >= 2:
                                rng_d = float(np.median([o[0] for o in ap_obs]))
                                brg_d = float(np.median([o[1] for o in ap_obs]))
                                a_dock = x_est[2] + brg_d
                                gx = x_est[0] + rng_d * math.cos(a_dock) + run_cfg.tags.approach_dist * math.cos(dock_yaw)
                                gy = x_est[1] + rng_d * math.sin(a_dock) + run_cfg.tags.approach_dist * math.sin(dock_yaw)
                                gd = math.hypot(gx - x_est[0], gy - x_est[1])
                                if gd < 0.25:
                                    aim_mode = False
                                    phase, nav_time = Phase.DOCK, frame.time
                                    dock.begin(x_est, dock_lm, dock_yaw)
                                    note(frame.time, f"AIM done: in front of the dock ({rng_d:.2f} m) -> docking")
                                else:
                                    ap_target = wrap_angle(math.atan2(gy - x_est[1], gx - x_est[0]))
                                    ap_goal = max(0.3, min(1.0, gd))
                                    ap_state = "TURN"
                                    note(frame.time, f"AIM: dock {rng_d:.2f} m -> approach point {gd:.2f} m away")
                            else:
                                ap_state = "SEARCH"
                                scan.reset()
                                note(frame.time, "AIM: dock not seen -> scanning")
                    elif ap_state == "SEARCH":               # rotate-stop-rotate until the dock shows up
                        w, observe = scan.update(frame.time, frame.odometry.dtheta)
                        if observe and frame.marker is not None:
                            ap_state = "LOOK"
                        elif scan.spun >= 2.0 * math.pi:     # nothing in a full turn: move to another spot
                            ap_target = wrap_angle(x_est[2] + freest_direction(frame.lidar))
                            ap_goal, ap_state = 0.8, "TURN"
                    elif ap_state == "TURN":
                        err = wrap_angle(ap_target - x_est[2])
                        if abs(err) < 0.08:
                            ap_state, ap_moved = "MOVE", 0.0
                        else:
                            w = float(np.clip(2.0 * err, -0.6, 0.6))
                    elif ap_state == "MOVE":
                        ap_moved += frame.odometry.ds
                        if sector_min_range(frame.lidar, 0.0, 0.5) < 0.6:     # blocked: sidestep to the open side
                            ap_target = wrap_angle(x_est[2] + freest_direction(frame.lidar))
                            ap_goal, ap_state = 0.8, "TURN"
                        elif ap_moved >= ap_goal:
                            ap_state = "LOOK"
                        else:
                            v = 0.2
                    if ap_state in ("TURN", "MOVE") and abs(v) < 1e-6 and abs(w) < 1e-6:
                        ap_idle += dt
                        if ap_idle > 1.0:
                            left = sector_min_range(frame.lidar, math.pi / 2, 0.5)
                            right = sector_min_range(frame.lidar, -math.pi / 2, 0.5)
                            side = math.pi / 2 if left >= right else -math.pi / 2
                            ap_target = wrap_angle(x_est[2] + side)
                            ap_goal, ap_state, ap_idle = 0.5, "TURN", 0.0
                            note(frame.time, "AIM: stuck -> forced 90 deg turn")
                    else:
                        ap_idle = 0.0
                    v, w = (v, w) if phase is Phase.NAVIGATE else (0.0, 0.0)

                # 2c) navigation
                if phase is Phase.NAVIGATE and not aim_mode:
                    ncmd = nav.update(x_est, frame.lidar)

                # 2d) docking (a failed docking widens P and starts the tag recovery)
                if phase is Phase.DOCK:
                    dcmd = dock.update(frame.marker, frame.odometry,
                                    x_est if (dock_use_estimate and dock_lm is not None) else None)
                    if dcmd.finished:
                        if dcmd.docked:
                            phase, reason = Phase.DONE, "docked"
                            note(frame.time, "DOCKED")
                        elif use_recovery and recovery.available and n_tag_recoveries < MAX_RECOVERIES:
                            estimator.inflate()              # accept the next tag detections
                            recovery.start()
                            n_tag_recoveries += 1
                            recovery_after_dock = True
                            last_target = None
                            phase = Phase.RECOVER
                            note(frame.time, f"docking failed -> P inflated -> tag recovery "
                                            f"#{n_tag_recoveries} started")
                        else:
                            phase, reason = Phase.FAILED, "docking failed"
                            note(frame.time, "docking failed (no recovery left)")
                    else:
                        v, w = dcmd.v, dcmd.w

                # 3) logging and evaluation (the ONLY place ground truth is read)
                hit = sim.collided
                collided_any = collided_any or hit
                mode = mode_label(phase, ncmd, dcmd,
                                recovery.phase if phase is Phase.RECOVER else None)
                if aim_mode and phase is Phase.NAVIGATE:
                    mode = "AIM-" + ap_state
                n_valid = int(frame.lidar.valid.sum())
                n_updates = int(getattr(estimator, "n_updates", 0))
                n_tag_upd = int(getattr(estimator, "n_tag_updates", 0))
                log.record(frame.time, sim.ground_truth(), x_eval, estimator.P,
                        phase.value, v, w, n_valid, n_updates, mode, n_tag_upd)
                if visualizer is not None:
                    visualizer.record(frame.time, x_eval, frame.lidar, mode, frame.marker, sim,
                                    markers=frame.markers)
                    if phase is Phase.DONE:
                        visualizer.capture_target(sim, label="Docked at charging station")
                if verbose and step % report_every == 0:
                    print(table_row(frame.time, x_est, n_valid, frame.lidar.valid.size,
                                    n_updates, v, w, mode, sigma_xy, n_tag_upd, last_tags), flush=True)

                if phase in (Phase.DONE, Phase.FAILED):
                    break
                if hit and stop_on_collision:
                    phase, reason = Phase.FAILED, "collision"
                    note(frame.time, "COLLISION")
                    break

                # 4) act
                if view is None:
                    sim.step(v, w)
                else:
                    with view.lock():
                        sim.step(v, w)
                    view.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))   # real-time pacing
                step += 1

        if visualizer is not None:
            viz_outputs = visualizer.finish(sim, success=phase is Phase.DONE,
                                            reason=reason, sim_time=sim.time)
        cam_stats = (sensing.n_camera, sensing.n_detect)
        n_detect_any = sensing.n_detect_any
        truth = sim.ground_truth()           # final pose, evaluation only
        sim_time = sim.time

    if verbose:
        print(table_footer())
        if viz_outputs:
            print("Route visualization saved to:", viz_outputs["visualization"])

    diag = {k: int(getattr(estimator, k, 0)) for k in
            ("n_updates", "n_rejected", "n_recovered", "n_lidar_updates", "n_lidar_rejected",
            "n_tag_updates")}
    diag["n_tag_recoveries"] = n_tag_recoveries
    diag["n_detect_any"] = int(n_detect_any)
    diag["nis_per_dof"] = (estimator.mean_nis_per_dof()
                        if hasattr(estimator, "mean_nis_per_dof") else float("nan"))
    if tag_map is not None:                  # EVALUATION ONLY: compare the learned map with the config poses
        truth_tags = world_cfg.tag_table()
        diag["tag_map_errors"] = {i: float(np.hypot(*(p - np.array(truth_tags[i][:2], dtype=float))))
                                    for i, (p, _) in table_to_world(tag_map.table(), start).items() if i in truth_tags}

    return RunResult(phase is Phase.DONE, reason, sim_time, nav_time, collided_any, phase,
                    est_to_world(estimator.x, start), truth, log,
                    camera_stats=cam_stats, diag=diag, obstacles=obstacles, events=events,
                    world_cfg=world_cfg)

# Save trajectory and estimation error with its uncertainty
def plot_run(result: RunResult, cfg: Config = CONFIG, path: Optional[Path] = None) -> Path:
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    path = RESULTS_DIR / "run.png" if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = result.log.arrays()

    fig = Figure(figsize=(12, 5.5))
    ax, ax2 = fig.subplots(1, 2)

    hx, hy = cfg.env.room_half_size
    ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, edgecolor="gray"))
    for ox, oy, r in result_obstacles(result, cfg):
        ax.add_patch(Circle((ox, oy), r, color="tab:red", alpha=0.4))
    wp = np.array(cfg.env.waypoints)
    ax.plot(wp[:, 0], wp[:, 1], "k^", label="waypoints")
    ax.plot(cfg.env.dock_pose[0], cfg.env.dock_pose[1], "gs", markersize=10, label="dock marker")
    ax.plot(a["truth"][:, 0], a["truth"][:, 1], label="true pose (evaluation)")
    ax.plot(a["est"][:, 0], a["est"][:, 1], "--", label="estimate")
    ax.set_aspect("equal")
    ax.set_xlim(-hx - 0.2, hx + 0.2)
    ax.set_ylim(-hy - 0.2, hy + 0.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory: {result.reason}")
    ax.legend(loc="upper left", fontsize=8)

    err = np.hypot(*(a["est"][:, :2] - a["truth"][:, :2]).T)
    band = 2.0 * np.hypot(a["sigma"][:, 0], a["sigma"][:, 1])
    ax2.plot(a["t"], err, label="position error")
    ax2.plot(a["t"], band, "--", label="2 sigma (sqrt(sx^2 + sy^2))")
    if result.nav_time is not None:
        ax2.axvline(result.nav_time, color="gray", linestyle=":", label="docking starts")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("[m]")
    ax2.set_title("Estimation error and uncertainty")
    ax2.legend(fontsize=8)

    fig.savefig(path, dpi=150)
    return path

# On macOS the live viewer needs mjpython; re-run this script under it
def _ensure_mjpython() -> None:
    if sys.platform != "darwin" or os.environ.get("_MJ_RELAUNCHED"):
        return
    mjpython = Path(sys.executable).parent / "mjpython"
    if not mjpython.exists():
        return
    os.environ["_MJ_RELAUNCHED"] = "1"
    # Launch via python so a broken shebang (spaces in the path) is bypassed
    os.execv(sys.executable, [sys.executable, str(mjpython), *sys.argv])

# Repeated trials with different noise seeds: success rate and mean +- std of the key metrics
def run_trials(n: int, seed0: int = 0, cfg: Config = CONFIG, use_marker_cov: bool = False,
            marker_sigma_px: float = 0.5, scene_dir: Optional[Path] = None,
            randomize_obstacles: bool = False, use_lidar_map: bool = False) -> int:
    rows = []
    print(f"marker R: {'per-measurement' if use_marker_cov else 'fixed'} | sigma_px {marker_sigma_px}")
    print(f"{'seed':>5} {'result':>7} {'reason':>15} {'time':>6} {'pos RMSE':>9} "
        f"{'NEES':>6} {'NIS/dof':>8} {'det %':>6} {'rej':>4} {'rec':>4} {'dock lat':>9}")
    for i in range(n):
        res = run_pipeline(cfg, seed=seed0 + i, scene_dir=scene_dir,
                        marker_sigma_px=marker_sigma_px, use_marker_cov=use_marker_cov,
                        randomize_obstacles=randomize_obstacles, use_lidar_map=use_lidar_map)
        rep = compute_report(res, cfg)
        n_cam, n_det = res.camera_stats
        row = {"ok": res.success, "reason": res.reason, "time": res.time,
            "rmse": rep.get("position_rmse", float("nan")),
            "nees": rep.get("mean_nees", float("nan")),
            "nis": res.diag.get("nis_per_dof", float("nan")),
               "det": 100.0 * n_det / max(1, n_cam),
            "rej": res.diag.get("n_rejected", 0),
            "rec": res.diag.get("n_tag_recoveries", 0),
            "lat": res.metrics(res.world_cfg or cfg)["dock_lateral"]}
        rows.append(row)
        print(f"{seed0 + i:>5} {'OK' if res.success else 'FAIL':>7} {res.reason:>15} {row['time']:>6.1f} "
            f"{row['rmse']:>9.3f} {row['nees']:>6.2f} {row['nis']:>8.2f} {row['det']:>6.0f} "
            f"{row['rej']:>4} {row['rec']:>4} {row['lat']:>9.3f}", flush=True)

    ok = np.array([r["ok"] for r in rows])
    print(f"\nSuccess rate : {int(ok.sum())}/{n} ({100 * ok.mean():.0f} %)")
    for label, key in (("Position RMSE [m]", "rmse"), ("Mean NEES (~3)", "nees"),
                    ("Mean NIS/dof (~1)", "nis"), ("Marker accepted [%]", "det"),
                    ("Gate rejections", "rej"), ("Tag recoveries", "rec"),
                    ("Dock lateral [m]", "lat")):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"{label:<20}: {np.nanmean(v):.3f} +- {np.nanstd(v):.3f}")
    failures = {}
    for r in rows:
        if not r["ok"]:
            failures[r["reason"]] = failures.get(r["reason"], 0) + 1
    if failures:
        print("Failure causes:", failures)
    return 0 if ok.all() else 1

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track A: navigation + visual docking in MuJoCo")
    parser.add_argument("--seed", type=int, default=None, help="noise seed (default: config)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="run headless without the live MuJoCo window")
    parser.add_argument("--plot", action="store_true", help="save results/run.png")
    parser.add_argument("--compare", action="store_true",
                        help="also run a dead-reckoning baseline (no marker updates) for comparison")
    parser.add_argument("--dock-raw", action="store_true",
                        help="dock on the raw marker measurement instead of the EKF estimate")
    parser.add_argument("--trials", type=int, default=0,
                        help="run N headless trials (seeds seed..seed+N-1) and print statistics")
    parser.add_argument("--marker-cov", action="store_true",
                        help="EKF uses the per-measurement marker covariance from sensing.py")
    parser.add_argument("--sigma-px", type=float, default=0.5,
                        help="assumed ArUco corner noise [px] used to build the marker covariance")
    parser.add_argument("--no-viz", action="store_true",
                        help="do not save the route picture / snapshots")
    parser.add_argument("--no-live-camera", action="store_true",
                        help="do not open the live camera window")
    parser.add_argument("--fixed-obstacles", action="store_true",
                        help="use the fixed obstacle layout from config instead of a random one")
    parser.add_argument("--no-survey", action="store_true",
                        help="skip the start-pose survey and use the tag poses written in the config")
    parser.add_argument("--ideal", action="store_true",
                        help="ideal simulation (no hidden errors, delays or sensor degradation)")
    parser.add_argument("--lidar-map", action="store_true",
                        help="EKF uses LiDAR against the fixed room walls (needs the world frame; off by default)")
    return parser

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = replace(CONFIG, realism=RealismConfig(enabled=False)) if args.ideal else CONFIG
    randomize = not args.fixed_obstacles

    # A random obstacle layout needs a random seed; it is printed so the run can be reproduced
    seed = args.seed
    if randomize and seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    if randomize:
        print(f"Random obstacles | seed {seed} (reproduce with --seed {seed})")

    if args.trials > 0:
        return run_trials(args.trials, seed or 0, cfg,
                        use_marker_cov=args.marker_cov, marker_sigma_px=args.sigma_px,
                        randomize_obstacles=randomize, use_lidar_map=args.lidar_map)
    use_viewer = not args.no_viewer
    if use_viewer:
        _ensure_mjpython()
    viz = None if args.no_viz else RouteVisualizer(
        cfg, live_view=use_viewer and not args.no_live_camera)
    result = run_pipeline(cfg, seed=seed, viewer=use_viewer, verbose=True,
                        dock_use_estimate=False,
                        marker_sigma_px=args.sigma_px, use_marker_cov=args.marker_cov,
                        visualizer=viz, randomize_obstacles=randomize,
                        use_lidar_map=args.lidar_map)
    ecfg = result.world_cfg or cfg          # the real world, for evaluation only

    dr_rmse = None
    if args.compare:
        # Same seed -> same obstacle layout as the main run
        baseline = run_pipeline(cfg, seed=seed,
                                estimator=EKFEstimator(cfg, use_marker=False, use_lidar=False),
                                randomize_obstacles=randomize)
        dr_rmse = compute_report(baseline, ecfg).get("position_rmse")

    print()
    print(result.summary(ecfg))
    print()
    print_report(result, ecfg, dr_rmse)
    if args.plot:
        print("Plot saved to:", plot_run(result, ecfg))
    return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())