# Including libraries
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from calibration import get_calibration
from config import CONFIG, RESULTS_DIR, Config
from controller import LQRTracker
from docking import DockingController
from estimator import EKFEstimator, nees
from navigation import Navigator
from robot_model import analyse_model, wrap_angle
from sensing import Sensing
from simulation import Simulation

class Phase(Enum):
    NAVIGATE = "navigate"
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
    mode: list = field(default_factory=list)         # TRACK / AVOID / DOCK-... / DONE

    def record(self, t, truth, est, P, phase, v, w,
            lidar_valid: int = 0, updates: int = 0, mode: str = "") -> None:
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
            "mode": np.array(self.mode, dtype=str),
        }

# ------------------------------------------------- live table + reports
# Column titles and widths of the real-time monitor
_COLS = (("Time (s)", 8), ("Pose est. x,y,th(deg)", 23), ("LiDAR valid", 12),
        ("EKF upd", 8), ("v, w", 12), ("Mode", 14))

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
def table_row(t, pose, lidar_valid, lidar_total, updates, v, w, mode) -> str:
    return _line([
        f"{t:.1f}",
        f"{pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.1f}°",
        f"{lidar_valid}/{lidar_total}",
        f"{updates}",
        f"{v:.2f},{w:.2f}",
        mode,
    ])

# Mode label shown in the table (navigation, docking or final state)
def mode_label(phase: Phase, ncmd=None, dcmd=None) -> str:
    if phase is Phase.NAVIGATE and ncmd is not None:
        return ncmd.phase.value.upper()
    if phase is Phase.DOCK and dcmd is not None:
        return "DOCK-" + dcmd.phase.value.upper()
    return phase.value.upper()

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
        "=== C. ESTIMATION (estimator.py) ===",
        f"{type(estimator).__name__ if estimator is not None else 'EKF'} | "
        f"R std (range, bearing) = ({cfg.camera.range_noise_std} m, {cfg.camera.bearing_noise_std} rad) | "
        f"P0 std = {ekf_cfg.init_pose_std} | gate = {getattr(estimator, 'gate', 'n/a')}",
        "=== D. CONTROL (controller.py, LQR) ===",
        f"Q = diag{cfg.controller.q_diag}, R = diag{cfg.controller.r_diag}",
        f"K = {np.round(tracker.K, 3).tolist()}",
        f"Closed-loop |poles| max = {poles.max():.3f} "
        f"({'stable' if poles.max() < 1.0 else 'UNSTABLE'})",
        "=== E. INTEGRATION (navigation.py, docking.py, main.py) ===",
        f"{len(cfg.env.waypoints)} waypoints, {len(cfg.env.obstacles)} obstacles, "
        f"dock at {tuple(round(v, 2) for v in cfg.env.dock_pose)}",
        "Controller input: ESTIMATED pose only (ground truth is logged for evaluation)",
        "",
    ])

# Quantitative metrics of one run (ground truth is used here for EVALUATION only)
def compute_report(result: RunResult, cfg: Config = CONFIG) -> dict:
    a = result.log.arrays()
    if len(a["t"]) == 0:
        return {}
    err = a["est"] - a["truth"]
    err[:, 2] = (err[:, 2] + math.pi) % (2.0 * math.pi) - math.pi
    modes = a["mode"]
    nav_steps = np.isin(modes, ["TRACK", "AVOID"])
    rob = cfg.robot
    hx, hy = cfg.env.room_half_size
    truth = a["truth"]
    clearance = min([float(np.min(np.hypot(truth[:, 0] - ox, truth[:, 1] - oy)) - r - rob.body_radius)
                    for ox, oy, r in cfg.env.obstacles] or [float("inf")])
    wall = float(np.min(np.minimum(hx - np.abs(truth[:, 0]), hy - np.abs(truth[:, 1])))
                - rob.body_radius)
    saturated = (a["v"] >= 0.99 * rob.max_linear_vel) | (np.abs(a["w"]) >= 0.99 * rob.max_angular_vel)
    finite_nees = a["nees"][np.isfinite(a["nees"])]
    return {
        "position_rmse": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
        "heading_rmse": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "mean_nees": float(finite_nees.mean()) if len(finite_nees) else float("nan"),
        "final_sigma": a["sigma"][-1],
        "marker_updates": int(a["updates"][-1]),
        "mean_lidar_valid": float(a["lidar_valid"].mean()),
        "mean_v": float(a["v"].mean()), "max_v": float(a["v"].max()),
        "mean_abs_w": float(np.abs(a["w"]).mean()), "max_abs_w": float(np.abs(a["w"]).max()),
        "saturated_fraction": float(saturated.mean()),
        "avoid_fraction": float(np.sum(modes == "AVOID") / max(1, nav_steps.sum())),
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
    print("=== FINAL REPORT ===")
    print(f"B. Sensing    : mean valid LiDAR beams {r['mean_lidar_valid']:.0f}/{cfg.lidar.num_beams}")
    print(f"C. Estimation : marker updates {r['marker_updates']} | "
        f"position RMSE {r['position_rmse']:.3f} m | heading RMSE {r['heading_rmse']:.3f} rad")
    print(f"                mean NEES {r['mean_nees']:.2f} (consistent filter: ~3) | "
        f"final sigma {np.round(r['final_sigma'], 4).tolist()}")
    print(f"                dead-reckoning baseline RMSE: {dr}")
    print(f"D. Control    : mean/max v {r['mean_v']:.2f}/{r['max_v']:.2f} m/s | "
        f"mean/max |w| {r['mean_abs_w']:.2f}/{r['max_abs_w']:.2f} rad/s | "
          f"saturated {100 * r['saturated_fraction']:.1f} % of steps")
    print(f"E. Integration: AVOID {100 * r['avoid_fraction']:.1f} % of navigation | "
        f"min obstacle clearance {r['min_obstacle_clearance']:.2f} m | "
        f"min wall clearance {r['min_wall_clearance']:.2f} m | collision {result.collided}")
    print(f"                docking range {m['dock_range']:.3f} m, lateral {m['dock_lateral']:.3f} m, "
        f"heading {m['dock_heading']:.3f} rad")
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
                stop_on_collision: bool = True, verbose: bool = False) -> RunResult:
    estimator = EKFEstimator(cfg) if estimator is None else estimator
    nav, dock, log = Navigator(cfg), DockingController(cfg), TrialLog()
    phase, reason = Phase.NAVIGATE, "timeout"
    nav_time: Optional[float] = None
    collided_any = False
    start = np.array(cfg.env.start_pose, dtype=float)
    dt = cfg.sim.control_dt
    report_every = max(1, int(round(1.0 / dt)))      # one table row per simulated second

    if verbose:
        print(theory_block(cfg, estimator))
        print(table_header())

    with Simulation(cfg, seed=seed, scene_dir=scene_dir) as sim:
        sim.reset(tuple(start))
        sensing = Sensing(sim, cfg)
        estimator.reset(start)               # known start pose (a prior, not ground truth)

        with _viewer(sim, viewer) as view:
            step = 0
            while not sim.time_exceeded:
                t0 = time.perf_counter()
                if view is not None and not view.is_running():
                    reason = "viewer closed"
                    break

                # 1) sensing and state estimation (no ground truth)
                frame = sensing.read()
                estimator.predict(frame.odometry.ds, frame.odometry.dtheta)
                estimator.update(frame.marker)
                x_est = np.array(estimator.x, dtype=float)

                # 2) decide the velocity command from the ESTIMATED pose only
                v = w = 0.0
                ncmd = dcmd = None
                if phase is Phase.NAVIGATE:
                    ncmd = nav.update(x_est, frame.lidar)
                    if ncmd.finished:
                        phase, nav_time = Phase.DOCK, frame.time
                        dock.begin(x_est)
                    else:
                        v, w = ncmd.v, ncmd.w
                if phase is Phase.DOCK:
                    dcmd = dock.update(frame.marker, frame.odometry)
                    if dcmd.finished:
                        phase = Phase.DONE if dcmd.docked else Phase.FAILED
                        reason = "docked" if dcmd.docked else "docking failed"
                    else:
                        v, w = dcmd.v, dcmd.w

                # 3) logging and evaluation (the ONLY place ground truth is read)
                hit = sim.collided
                collided_any = collided_any or hit
                mode = mode_label(phase, ncmd, dcmd)
                n_valid = int(frame.lidar.valid.sum())
                n_updates = int(getattr(estimator, "n_updates", 0))
                log.record(frame.time, sim.ground_truth(), x_est, estimator.P,
                        phase.value, v, w, n_valid, n_updates, mode)
                if verbose and step % report_every == 0:
                    print(table_row(frame.time, x_est, n_valid, frame.lidar.valid.size,
                                    n_updates, v, w, mode), flush=True)

                if phase in (Phase.DONE, Phase.FAILED):
                    break
                if hit and stop_on_collision:
                    phase, reason = Phase.FAILED, "collision"
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

        truth = sim.ground_truth()           # final pose, evaluation only
        sim_time = sim.time

    if verbose:
        print(table_footer())

    return RunResult(phase is Phase.DONE, reason, sim_time, nav_time, collided_any, phase,
                    np.array(estimator.x, dtype=float), truth, log)

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
    for ox, oy, r in cfg.env.obstacles:
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track A: navigation + visual docking in MuJoCo")
    parser.add_argument("--seed", type=int, default=None, help="noise seed (default: config)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="run headless without the live MuJoCo window")
    parser.add_argument("--plot", action="store_true", help="save results/run.png")
    parser.add_argument("--compare", action="store_true",
                        help="also run a dead-reckoning baseline (no marker updates) for comparison")
    return parser

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    use_viewer = not args.no_viewer
    if use_viewer:
        _ensure_mjpython()
    result = run_pipeline(CONFIG, seed=args.seed, viewer=use_viewer, verbose=True)

    dr_rmse = None
    if args.compare:
        baseline = run_pipeline(CONFIG, seed=args.seed,
                                estimator=EKFEstimator(CONFIG, use_marker=False))
        dr_rmse = compute_report(baseline, CONFIG).get("position_rmse")

    print()
    print(result.summary(CONFIG))
    print()
    print_report(result, CONFIG, dr_rmse)
    if args.plot:
        print("Plot saved to:", plot_run(result, CONFIG))
    return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())