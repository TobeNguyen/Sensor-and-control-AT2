"""
TurtleBot3 route visualization (passive logger, never sends control commands).

Records:
  - one camera image at the starting point
  - one camera image of the target (ArUco marker / "painting") once it is close and
    centred, or a clearly labelled final-view image if it was never confirmed
  - the EKF-estimated trajectory (NOT MuJoCo ground truth)
  - decimated LiDAR hits projected into the world frame with the EKF pose
  - the tag poses learned during the start-pose SURVEY (estimated, not ground truth)
  - a live camera window while the simulation runs (display only, nothing stored)

It never records video and never keeps a history of camera frames, so memory
use stays small: two images + a capped list of path/LiDAR points.
It does NOT perform SLAM.

Typical use in main.py:
    viz = RouteVisualizer()
    sim.reset(start); viz.start(sim)
    loop:
        ... sensing, EKF update ...
        viz.record(frame.time, ekf.x, frame.lidar, mode, frame.marker, sim,
                   markers=frame.markers)
    viz.set_tag_map(learned)    # once, after the SURVEY
    viz.finish(sim, success=ok, reason=why, sim_time=sim.time)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
import multiprocessing as mp
import queue
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from config import CONFIG, RESULTS_DIR, Config
from estimator import chi2_bounds, nees
from robot_model import Phase, RunResult, wrap_angle

MAX_RECOVERIES = 1   # maximum number of tag recoveries per run (avoid endless retry loops)


def as_rgb(image: Any) -> np.ndarray:
    """Validate a camera image (RGB uint8 HxWx3) and return a copy."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Camera must return an HxWx3 RGB image")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb.copy()


def _camera_window_worker(q, window: str) -> None:
    import cv2
    while True:
        try:
            frame = q.get(timeout=0.05)
        except queue.Empty:
            cv2.waitKey(1)
            continue
        if frame is None:
            break
        cv2.imshow(window, frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

class LiveCameraView:
    """Live camera window in a separate process (works with mjpython on macOS)."""

    def __init__(self, window: str = "TurtleBot3 camera", period: float = 0.10,
                scale: float = 0.6, enabled: bool = True):
        self.window = window
        self.period = float(period)
        self.scale = float(scale)
        self.enabled = enabled
        self._last_t = -math.inf
        self._proc = None
        self._q = None

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue(maxsize=2)
        self._proc = ctx.Process(target=_camera_window_worker,
                                args=(self._q, self.window), daemon=True)
        self._proc.start()

    # NEW: `markers` = every marker accepted in the latest frame (dock + extra tags)
    def update(self, t: float, image: Any, mode: str = "", marker: Any = None,
            markers: Any = ()) -> None:
        if not self.enabled or t - self._last_t < self.period - 1e-9:
            return
        self._last_t = t
        if self._proc is None:
            self._start()
        bgr = cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2BGR)
        if self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale)
        # NEW: show the ids of every visible marker, not only the dock
        ids = ",".join(str(m.marker_id) for m in markers) if markers else "-"
        if marker is not None:
            text = (f"{mode} | dock r={marker.range:.2f} m  b={math.degrees(marker.bearing):+.0f} deg"
                    f" | ids {ids}")
            color = (0, 220, 0)
        elif markers:
            text = f"{mode} | tag ids {ids} (dock not seen)"
            color = (0, 200, 255)
        else:
            text = f"{mode} | no marker"
            color = (0, 0, 255)
        cv2.putText(bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        try:
            self._q.put_nowait(bgr)
        except queue.Full:
            pass
        
    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None


class RouteVisualizer:
    """Passive logger: does not issue any robot control commands."""

    def __init__(self,
                 cfg: Config = CONFIG,
                 output_dir: Path | str | None = None,
                 path_period: float = 0.20,
                 lidar_period: float = 0.50,
                 beam_stride: int = 6,
                 max_lidar_points: int = 25_000,
                 target_capture_range: float = 0.80,
                 target_max_bearing: float = 0.25,
                 live_view: bool = True,
                 live_period: float = 0.10):
        if path_period <= 0 or lidar_period <= 0:
            raise ValueError("Sampling periods must be positive.")
        if beam_stride < 1 or max_lidar_points < 1:
            raise ValueError("beam_stride and max_lidar_points must be positive.")
        self.cfg = cfg
        self.output_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR / "visualization"
        self.path_period = float(path_period)
        self.lidar_period = float(lidar_period)
        self.beam_stride = int(beam_stride)
        self.max_lidar_points = int(max_lidar_points)
        self.target_capture_range = float(target_capture_range)
        self.target_max_bearing = float(target_max_bearing)
        self.live = LiveCameraView(period=live_period, enabled=live_view)
        self.reset()

    def reset(self) -> None:
        self._routes: list[tuple] = []          # (t, x, y, heading_deg, mode, n_lidar, marker_seen)
        self._lidar_batches: list[np.ndarray] = []
        self._point_count = 0
        self._last_path_t = -math.inf
        self._last_lidar_t = -math.inf
        self._start_image: np.ndarray | None = None
        self._target_image: np.ndarray | None = None
        self._target_confirmed = False
        self._target_caption = ""
        self._learned_tags: dict = {}           # NEW: {id: (x, y)} learned during SURVEY (estimated)
        self.outputs: dict[str, Path] = {}
        self._last_image: np.ndarray | None = None   # latest image the robot processed

    # ------------------------------------------------------------------ capture
    def set_world(self, world_cfg: Config, obstacles) -> None:
        """Real world (moved dock, real obstacles). EVALUATION drawing only, never used for control."""
        self.eval_cfg = world_cfg
        self.eval_obstacles = list(obstacles)

    def start(self, sim: Any = None) -> None:
        """Call right after sim.reset(). The start image is taken from the first processed camera frame."""
        self.reset()

    def capture_target(self, sim: Any = None, *, label: str = "Destination view") -> None:
        """Store the target image (once), from the latest processed camera frame."""
        if self._last_image is None:
            return
        self._target_image = self._last_image.copy()
        self._target_confirmed = True
        self._target_caption = label

    # NEW
    def set_tag_map(self, table: dict) -> None:
        """Tag poses learned by the SURVEY: {id: (position (2,), yaw)} (estimated, not ground truth)."""
        self._learned_tags = {int(i): (float(p[0]), float(p[1])) for i, (p, _) in table.items()}

    def record(self, t: float, pose_est: Any, lidar: Any,
               mode: str, marker: Any = None, sim: Any = None, markers: Any = (),
               frame: Any = None) -> None:
        """Call once per control step, after the EKF update.

        pose_est is [x, y, theta] from the EKF, NOT ground truth. Images come from `frame.image`,
        so the visualizer never touches the simulator (no extra random draws, no side effects).
        """
        t = float(t)
        pose = np.asarray(pose_est, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return
        x, y, theta = map(float, pose)

        # Estimated path (decimated in time)
        if t - self._last_path_t >= self.path_period - 1e-9:
            self._routes.append((t, x, y, math.degrees(theta), str(mode),
                                 int(np.count_nonzero(lidar.valid)), int(marker is not None)))
            self._last_path_t = t

        # LiDAR hits in the world frame (decimated in time and in beams, capped in total)
        if t - self._last_lidar_t >= self.lidar_period - 1e-9:
            self._last_lidar_t = t
            r = np.asarray(lidar.ranges, dtype=float)
            angles = np.asarray(lidar.angles, dtype=float)
            mask = np.asarray(lidar.valid, dtype=bool)
            if r.shape != angles.shape or r.shape != mask.shape:
                raise ValueError("LiDAR ranges, angles, valid must have the same shape.")
            indices = np.flatnonzero(mask)[::self.beam_stride]
            remaining = self.max_lidar_points - self._point_count
            if remaining > 0 and len(indices):
                indices = indices[:remaining]
                phi = angles[indices] + theta
                points = np.column_stack((x + r[indices] * np.cos(phi),
                                          y + r[indices] * np.sin(phi)))
                self._lidar_batches.append(points)
                self._point_count += len(points)

        image = None if frame is None else getattr(frame, "image", None)
        if image is None:
            return
        self._last_image = image
        has_content = bool(np.any(image))                        # a dropped frame is all zeros
        if self._start_image is None and has_content:
            self._start_image = np.array(image, copy=True)

        # Target snapshot: exactly once, when the marker is near and roughly centred
        if (has_content and not self._target_confirmed and marker is not None
                and math.isfinite(float(marker.range)) and math.isfinite(float(marker.bearing))
                and float(marker.range) <= self.target_capture_range
                and abs(float(marker.bearing)) <= self.target_max_bearing):
            self.capture_target(label="ArUco marker / destination")

        # Live camera window (display only)
        self.live.update(t, image, mode, marker, markers)

    # ------------------------------------------------------------------- output
    @staticmethod
    def _save_rgb(path: Path, rgb: np.ndarray) -> None:
        ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError(f"Could not write camera image: {path}")

    def finish(self, sim: Any | None = None, *, success: bool = False, reason: str = "",
               sim_time: float | None = None, extra: dict | None = None) -> dict[str, Path]:
        """Write JPEG snapshots, trajectory CSV, summary PNG and JSON. No video."""
        self.live.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.outputs = {}

        if self._start_image is not None:
            p = self.output_dir / "start_camera.jpg"
            self._save_rgb(p, self._start_image)
            self.outputs["start_camera"] = p

        if self._target_image is None and self._last_image is not None:
            # Do not pretend a camera view proves the target was located.
            self._target_image = self._last_image.copy()
            self._target_caption = "Final view (target NOT confirmed)"

        if self._target_image is not None:
            name = "target_camera.jpg" if self._target_confirmed else "final_view.jpg"
            p = self.output_dir / name
            self._save_rgb(p, self._target_image)
            self.outputs["target_camera" if self._target_confirmed else "final_view"] = p

        csv_path = self.output_dir / "estimated_path.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "x_est_m", "y_est_m", "heading_deg",
                             "mode", "valid_lidar_beams", "marker_detected"])
            writer.writerows(self._routes)
        self.outputs["path_csv"] = csv_path

        map_path = self.output_dir / "route_and_snapshots.png"
        self._make_figure(map_path, success, reason)
        self.outputs["visualization"] = map_path

        metadata = {
            "success": bool(success),
            "reason": str(reason),
            "sim_time_s": sim_time,
            "target_confirmed": self._target_confirmed,
            "camera_target_label": self._target_caption,
            "estimated_path_samples": len(self._routes),
            "lidar_map_points": self._point_count,
            "learned_tag_ids": sorted(self._learned_tags),
            "method": "EKF trajectory + LiDAR projected with estimated pose; not SLAM",
            "ground_truth_used_for_control": False,
            "ground_truth_drawn_for_evaluation": True,   # true dock / obstacles are drawn as reference
            "video_recorded": False,
        }
        if extra:
            metadata.update(extra)
        metadata_path = self.output_dir / "summary.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        self.outputs["summary"] = metadata_path
        return self.outputs.copy()

    def _make_figure(self, path: Path, success: bool, reason: str) -> None:
        figure = Figure(figsize=(13, 7), facecolor="white", constrained_layout=True)
        layout = figure.add_gridspec(2, 2, width_ratios=(1.5, 1), height_ratios=(1, 1))
        ax_map = figure.add_subplot(layout[:, 0])
        ax_start = figure.add_subplot(layout[0, 1])
        ax_goal = figure.add_subplot(layout[1, 1])

        if self._lidar_batches:
            pts = np.concatenate(self._lidar_batches, axis=0)
            ax_map.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.20,
                           c="#60798c", rasterized=True, label="LiDAR returns")

        if self._routes:
            poses = np.asarray([p[1:3] for p in self._routes], dtype=float)
            ax_map.plot(poses[:, 0], poses[:, 1], color="#1674b8", lw=2.3,
                        label="Estimated robot path")
            ax_map.scatter(*poses[0], c="#2d9960", s=95, marker="o", zorder=5, label="Start")
            ax_map.scatter(*poses[-1], c="#f2a035", s=95, marker="X", zorder=5,
                           label="Last position")
            for j in np.linspace(0, len(self._routes) - 1, min(10, len(self._routes)), dtype=int):
                angle = math.radians(self._routes[j][3])
                px, py = poses[j]
                ax_map.arrow(px, py, 0.13 * math.cos(angle), 0.13 * math.sin(angle),
                             head_width=0.045, head_length=0.06, color="#1674b8",
                             alpha=0.65, length_includes_head=True, zorder=3)

        # Real world (evaluation only): moved dock, real obstacles, extra tags
        hx, hy = self.cfg.env.room_half_size
        ax_map.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, edgecolor="#777777",
                                   lw=1.1, linestyle=":", label="Configured room bounds"))
        world = self.eval_cfg
        for ox, oy, r in self.eval_obstacles:
            ax_map.add_patch(Circle((ox, oy), r, color="tab:red", alpha=0.35))
        dock_x, dock_y, _ = world.env.dock_pose
        ax_map.scatter([dock_x], [dock_y], marker="*", s=200, c="#ba559b",
                       label="ArUco target (true, evaluation)", zorder=6)

        dock_id = int(world.camera.marker_id)
        tags_cfg = {i: p for i, p in world.tag_table().items() if i != dock_id}
        if tags_cfg:
            ax_map.scatter([p[0] for p in tags_cfg.values()], [p[1] for p in tags_cfg.values()],
                           s=45, marker="s", facecolors="none", edgecolors="#ba559b",
                           label="Tags (true, evaluation)")
        if self._learned_tags:
            ax_map.scatter([p[0] for p in self._learned_tags.values()],
                           [p[1] for p in self._learned_tags.values()],
                           s=30, marker="x", c="#d04040", zorder=6, label="Learned tags (SURVEY)")
            for i, (px, py) in self._learned_tags.items():
                ax_map.annotate(str(i), (px, py), textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax_map.set_xlim(-hx - 0.35, hx + 0.35)
        ax_map.set_ylim(-hy - 0.35, hy + 0.35)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("World x [m]")
        ax_map.set_ylabel("World y [m]")
        ax_map.grid(alpha=0.15)
        ax_map.set_title("LiDAR observations and EKF-estimated trajectory")
        ax_map.legend(loc="lower right", fontsize=7)

        for ax, img, caption in (
            (ax_start, self._start_image, "Camera: starting view"),
            (ax_goal, self._target_image, self._target_caption or "Camera: destination view"),
        ):
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image not available", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(caption, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        result = "SUCCESS" if success else "NOT CONFIRMED"
        figure.suptitle(f"TurtleBot3 visual route record  |  {result}  |  {reason}",
                        fontsize=14, weight="bold")
        figure.savefig(path, dpi=140)
        figure.clear()

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
# Column titles and widths of the real-time monitor
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

# Save trajectory and estimation error with its uncertainty
def plot_run(result: RunResult, cfg: Config = CONFIG, path: Optional[Path] = None) -> Path:
    path = RESULTS_DIR / "run.png" if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = result.log.arrays()

    fig = Figure(figsize=(12, 5.5))
    ax, ax2 = fig.subplots(1, 2)

    hx, hy = cfg.env.room_half_size
    ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, edgecolor="gray"))
    for ox, oy, r in result_obstacles(result, cfg):
        ax.add_patch(Circle((ox, oy), r, color="tab:red", alpha=0.4))
    ax.plot(cfg.env.dock_pose[0], cfg.env.dock_pose[1], "gs", markersize=10, label="Tags (true, evaluation)")
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