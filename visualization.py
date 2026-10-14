"""
TurtleBot3 route visualization (passive logger, never sends control commands).

Records:
  - one camera image at the starting point
  - one camera image of the target (ArUco marker / "painting") once it is close and
    centred, or a clearly labelled final-view image if it was never confirmed
  - the EKF-estimated trajectory (NOT MuJoCo ground truth)
  - decimated LiDAR hits projected into the world frame with the EKF pose
  - a live camera window while the simulation runs (display only, nothing stored)

It never records video and never keeps a history of camera frames, so memory
use stays small: two images + a capped list of path/LiDAR points.
It does NOT perform SLAM.

Typical use in main.py:
    viz = RouteVisualizer()
    sim.reset(start); viz.start(sim)
    loop:
        ... sensing, EKF update ...
        viz.record(frame.time, ekf.x, frame.lidar, mode, frame.marker, sim)
    viz.finish(sim, success=ok, reason=why, sim_time=sim.time)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any
import multiprocessing as mp
import queue
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from config import CONFIG, RESULTS_DIR, Config


def get_rgb(sim: Any) -> np.ndarray:
    """Simulation.read_camera() -> RGB uint8 HxWx3 copy."""
    rgb = np.asarray(sim.read_camera())
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

    def update(self, t: float, sim: Any, mode: str = "", marker: Any = None) -> None:
        if not self.enabled or t - self._last_t < self.period - 1e-9:
            return
        self._last_t = t
        if self._proc is None:
            self._start()
        bgr = cv2.cvtColor(get_rgb(sim), cv2.COLOR_RGB2BGR)
        if self.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale)
        if marker is not None:
            text = f"{mode} | marker r={marker.range:.2f} m  b={math.degrees(marker.bearing):+.0f} deg"
            color = (0, 220, 0)
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
        self.outputs: dict[str, Path] = {}

    # ------------------------------------------------------------------ capture
    def start(self, sim: Any) -> None:
        """Call right after sim.reset(): stores the starting-point image."""
        self.reset()
        self._start_image = get_rgb(sim)

    def capture_target(self, sim: Any, *, label: str = "Destination view") -> None:
        """Store the target image (once). Also callable from main.py when DOCKED."""
        self._target_image = get_rgb(sim)
        self._target_confirmed = True
        self._target_caption = label

    def record(self, t: float, pose_est: Any, lidar: Any,
               mode: str, marker: Any = None, sim: Any = None) -> None:
        """Call once per control step, after the EKF update.

        pose_est is [x, y, theta] from the EKF, NOT ground truth.
        lidar is a sensing.LidarScan; marker is a MarkerMeasurement or None.
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

        # Target snapshot: exactly once, when the marker is near and roughly centred
        if (sim is not None and not self._target_confirmed and marker is not None
                and math.isfinite(float(marker.range)) and math.isfinite(float(marker.bearing))
                and float(marker.range) <= self.target_capture_range
                and abs(float(marker.bearing)) <= self.target_max_bearing):
            self.capture_target(sim, label="ArUco marker / destination")

        # Live camera window (display only)
        if sim is not None:
            self.live.update(t, sim, mode, marker)

    # ------------------------------------------------------------------- output
    @staticmethod
    def _save_rgb(path: Path, rgb: np.ndarray) -> None:
        ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError(f"Could not write camera image: {path}")

    def finish(self, sim: Any | None = None, *, success: bool = False, reason: str = "",
               sim_time: float | None = None) -> dict[str, Path]:
        """Write JPEG snapshots, trajectory CSV, summary PNG and JSON. No video."""
        self.live.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.outputs = {}

        if self._start_image is not None:
            p = self.output_dir / "start_camera.jpg"
            self._save_rgb(p, self._start_image)
            self.outputs["start_camera"] = p

        if self._target_image is None and sim is not None:
            # Do not pretend a camera view proves the target was located.
            self._target_image = get_rgb(sim)
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
            "method": "EKF trajectory + LiDAR projected with estimated pose; not SLAM",
            "ground_truth_used": False,
            "video_recorded": False,
        }
        metadata_path = self.output_dir / "summary.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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

        # Configured scene objects are context, not map-derived observations.
        hx, hy = self.cfg.env.room_half_size
        ax_map.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, edgecolor="#777777",
                                   lw=1.1, linestyle=":", label="Configured room bounds"))
        ax_map.scatter([w[0] for w in self.cfg.env.waypoints],
                       [w[1] for w in self.cfg.env.waypoints],
                       s=27, marker="^", c="#858585", label="Configured waypoints")
        dock_x, dock_y, _ = self.cfg.env.dock_pose
        ax_map.scatter([dock_x], [dock_y], marker="*", s=200, c="#ba559b",
                       label="Configured ArUco target", zorder=6)
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