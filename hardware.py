# Including libraries
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np
import cv2

from config import CONFIG, RESULTS_DIR, Config
from environment import HardwareEnvironment

# Robot settings
ROBOT_IP = "192.168.1.207"                       # TurtleBot ID: 192.168.1.2XX
# ===== Real-robot configuration (config.py stays untouched, it is for the simulation) =====
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480   # must equal the size of the images the robot sends
MARKER_SIZE = 0.15                       # [m] width of the BLACK part of the printed marker
N_EXTRA_TAGS = 3                         # extra tags in the real arena (0 = only the dock marker)
CONTROL_DT = 0.05                        # [s] 20 Hz: realistic with WiFi and a 5 Hz LiDAR
SPEED_SCALE = 0.5                                # fraction of max speed (keep 0.5 for the first runs)
CAMERA_DELAY = 0.2                               # [s] camera latency compensated by the EKF
ODOM_SCALE = 4.0                                 # x odometry covariance (dead reckoning is noisy)
SCAN_METHOD = None          # set the exact method name of RosbridgeRobot here if the guesses below fail
SCAN_NAMES = ("query_laser_scan", "query_scan", "query_laser", "query_lidar",
            "get_laser_scan", "get_scan", "laser_scan", "scan")
MAX_BEAM_GAP = math.radians(3.0)     # a target beam farther than this from any real beam = no hit
CONTROL_DT = 0.05      # 20 Hz: realistic for WiFi + 5 Hz LiDAR (the sim uses 0.02)
ODOM_SCALE = 1.5       # real wheel encoders: much less noise than dead reckoning

def hardware_config(speed_scale: float = 1.0) -> Config:
    """CONFIG with the real-robot values. Only this function decides what differs from the simulation."""
    c = CONFIG
    extra = tuple(c.tags.fixed_tags[i] if i < len(c.tags.fixed_tags) else (0.0, 0.0, 0.0)
                for i in range(N_EXTRA_TAGS))                 # only the COUNT matters for control
    return replace(
        c,
        camera=replace(c.camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, marker_size=MARKER_SIZE),
        robot=replace(c.robot, max_linear_vel=c.robot.max_linear_vel * speed_scale,
                      max_angular_vel=c.robot.max_angular_vel * speed_scale),
        sim=replace(c.sim, control_dt=CONTROL_DT),
        tags=replace(c.tags, fixed_tags=extra),
        env=replace(c.env, obstacles=()),                       # the real arena is unknown to the robot
        realism=replace(c.realism, enabled=False),              # simulated imperfections do not apply
    )


def with_fov(cfg: Config, calibration) -> Config:
    """Vertical field of view from the calibrated focal length (instead of the simulator's 60 deg)."""
    fy = float(calibration.K[1, 1])
    return replace(cfg, camera=replace(cfg.camera, fovy_deg=math.degrees(2.0 * math.atan(cfg.camera.height / 2.0 / fy))))

# Helper
# Call a query method
def _query(fn, timeout: float):
    for kw in (dict(timeout=timeout, wait_for_new=True), dict(timeout=timeout), {}):
        try:
            return fn(**kw)
        except TypeError:
            continue
    return None

def _find_scan_method(robot):
    names = (SCAN_METHOD,) if SCAN_METHOD else SCAN_NAMES
    for n in names:
        fn = getattr(robot, n, None)
        if callable(fn):
            return fn
    public = [n for n in dir(robot) if not n.startswith("_") and callable(getattr(robot, n, None))]
    raise AttributeError("No laser-scan method found on RosbridgeRobot. Public methods: "
                        f"{public}. Put the right name in SCAN_METHOD at the top of hardware.py.")

# Accept (angles, ranges), dict or object with angles/ranges
def _parse_scan(scan):
    if scan is None:
        return None
    if isinstance(scan, dict):
        a, r = scan.get("angles", scan.get("angle")), scan.get("ranges", scan.get("range"))
    elif isinstance(scan, (tuple, list)) and len(scan) >= 2:
        a, r = scan[0], scan[1]
    else:
        a = getattr(scan, "angles", getattr(scan, "angle", None))
        r = getattr(scan, "ranges", getattr(scan, "range", None))
    if a is None or r is None:
        raise ValueError(f"cannot read angles / ranges from a scan of type {type(scan)}")
    a, r = np.asarray(a, dtype=float).ravel(), np.asarray(r, dtype=float).ravel()
    if a.size != r.size:
        raise ValueError("scan angles and ranges have different lengths")
    if a.size > 1 and np.any(np.diff(a) < -1e-9) and not np.any(np.diff(r) < -1e-9):
        a, r = r, a                       # arrays came in (ranges, angles) order
    return a, r

# Nearest-beam resampling onto the beam angles the pipeline expects
def resample_scan(a, r, target, range_min: float, range_max: float) -> np.ndarray:
    r = np.where(np.isfinite(r) & (r >= range_min) & (r < range_max), r, range_max)
    a = np.mod(a, 2.0 * math.pi)
    diff = np.abs((np.asarray(target)[:, None] - a[None, :] + math.pi) % (2.0 * math.pi) - math.pi)
    out = r[diff.argmin(axis=1)]
    out[diff.min(axis=1) > MAX_BEAM_GAP] = range_max
    return out

# ------------------------------------------------------------ environment on the real robot
class RosbridgeEnvironment(HardwareEnvironment):
    def __init__(self, robot, cfg: Config = CONFIG, lidar_timeout: float = 0.0,
                camera_timeout: float = 0.1):
        super().__init__(cfg)
        self.robot = robot
        self.lidar_timeout, self.camera_timeout = float(lidar_timeout), float(camera_timeout)
        self._scan_fn = _find_scan_method(robot)
        self._enc = np.zeros(2)                       # dead-reckoning fallback
        self._enc_last = np.zeros(2)                  # last real wheel angles
        self._cmd = (0.0, 0.0)
        self._t_int = time.monotonic()
        self._ranges = np.full(cfg.lidar.num_beams, cfg.lidar.range_max)
        self._image = np.zeros((cfg.camera.height, cfg.camera.width, 3), np.uint8)
        self._joint_ok = self._wait_joint_states(2.0)
        print("Odometry:", "wheel encoders from /joint_states" if self._joint_ok
            else "NO /joint_states -> dead reckoning from commands (less accurate)")

    @staticmethod
    def _wheel_angles(js):
        try:
            names = [str(n).lower() for n in js["name"]]
            pos = js["position"]
            return np.array([pos[next(i for i, n in enumerate(names) if "left" in n)],
                            pos[next(i for i, n in enumerate(names) if "right" in n)]], dtype=float)
        except (KeyError, TypeError, StopIteration, IndexError):
            return None

    def _wait_joint_states(self, timeout: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            a = self._wheel_angles(self.robot.query_joint_states() or {})
            if a is not None:
                self._enc_last = a
                return True
            time.sleep(0.05)
        return False

    def _driver_encoders(self) -> np.ndarray:
        if self._joint_ok:
            a = self._wheel_angles(self.robot.query_joint_states() or {})
            if a is not None:
                self._enc_last = a
            return self._enc_last.copy()
        self._integrate()
        return self._enc.copy()

    def _driver_lidar(self):
        # timeout 0: never block the control loop (the LiDAR is ~5 Hz); None = no new scan yet
        parsed = _parse_scan(_query(self._scan_fn, self.lidar_timeout))
        if parsed is None:
            return self._ranges, self._lidar_stamp
        lid = self.cfg.lidar
        self._ranges = resample_scan(*parsed, self._angles, lid.range_min, lid.range_max)
        return self._ranges, time.monotonic()

    def _driver_camera(self):
        bgr = _query(self.robot.query_image, self.camera_timeout)
        if bgr is None:
            return self._image, self._camera_stamp
        cam = self.cfg.camera
        if bgr.shape[0] != cam.height or bgr.shape[1] != cam.width:
            raise ValueError(f"camera image is {bgr.shape[1]}x{bgr.shape[0]} but config / calibration "
                            f"expect {cam.width}x{cam.height}: fix cfg.camera or recalibrate")
        self._image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._image, time.monotonic()

    # encoders are integrated from the commanded wheel speeds
    def _integrate(self) -> None:
        now = time.monotonic()
        dt, self._t_int = now - self._t_int, now
        v, w = self._cmd
        r, b = self.cfg.robot.wheel_radius, self.cfg.robot.wheel_base
        self._enc += np.array([(v - w * b / 2.0) / r, (v + w * b / 2.0) / r]) * dt

    def step(self, v: float, w: float) -> None:
        rob = self.cfg.robot
        v = float(np.clip(v, -rob.max_linear_vel, rob.max_linear_vel))
        w = float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel))
        self._integrate()                             # close the previous period with the previous command
        self.robot.send_command(v, w)
        self._cmd = (v, w)
        remaining = self.cfg.sim.control_dt - (time.monotonic() - self._last_step)
        if remaining > 0.0:
            time.sleep(remaining)
        self._last_step = time.monotonic()

    def emergency_stop(self) -> None:
        self._integrate()
        self._cmd = (0.0, 0.0)
        for _ in range(3):                            # repeat: one lost WiFi packet must not keep the robot moving
            self.robot.send_command(0.0, 0.0)

    def close(self) -> None:
        try:
            self.emergency_stop()
        except Exception:
            pass

# ------------------------------------------------------------ stand-in for Simulation (used by main.run_pipeline)
class HardwareSim:
    def __init__(self, env: RosbridgeEnvironment, cfg: Config):
        self.env, self.cfg, self.world_cfg = env, cfg, cfg
        self.obstacles: list = []
        self.model = self.data = None
        self.collided = False                         # no contact sensor: LiDAR + SafetyGuard protect the robot

    @property
    def time(self) -> float:
        return self.env.time

    def reset(self, pose=None) -> None:
        pass

    def ground_truth(self) -> np.ndarray:
        return np.full(3, np.nan)                     # NEES / truth columns become NaN: measure the dock by hand

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

# ------------------------------------------------------------ pre-flight check (robot does not move)
def preflight(env: RosbridgeEnvironment, cfg: Config) -> bool:
    from sensing import detect_markers, make_lidar_scan
    ok = True
    ranges = None
    for _ in range(3):
        ranges = env.read_lidar()
        time.sleep(0.1)
    scan = make_lidar_scan(ranges, env.lidar_angles, cfg)
    n_valid = int(scan.valid.sum())
    print(f"LiDAR : {n_valid}/{len(ranges)} valid beams, min range {ranges.min():.2f} m, "
        f"age {env.sensor_ages()[0]:.2f} s")
    if n_valid < 20:
        print("  !! very few valid beams: check the LiDAR topic / angle convention"); ok = False
    img = env.read_camera()
    print(f"Camera: {img.shape[1]}x{img.shape[0]}, mean brightness {img.mean():.0f}, age {env.sensor_ages()[1]:.2f} s")
    if not np.any(img):
        print("  !! camera image is empty"); ok = False
    found = detect_markers(img, cfg, env.time)
    print("Markers in view:", [(m.marker_id, round(m.range, 2), round(math.degrees(m.bearing), 1)) for m in found] or "none")
    env.emergency_stop()
    print("Stop command sent.")
    return ok

# ------------------------------------------------------------ full run
def run_hardware(args) -> int:
    import main
    from estimator import EKFEstimator
    from rosbridge_api import RosbridgeRobot
    from visualization import RouteVisualizer, print_report

    s = float(args.speed_scale)
    cfg = hardware_config(s)
    from calibration import calibrate_hardware
    try:
        calibration, cfg = calibrate_hardware(cfg, force=args.recalibrate, square=args.square)
    except (FileNotFoundError, ValueError) as e:
        print(f"Calibration failed: {e}")
        return 2
    cfg = with_fov(cfg, calibration)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / "hardware" / stamp
    out.mkdir(parents=True, exist_ok=True)

    with RosbridgeRobot(args.ip) as robot:
        env = RosbridgeEnvironment(robot, cfg)
        try:
            if not preflight(env, cfg) and not args.force:
                print("Pre-flight failed. Fix the problems above or use --force.")
                return 2
            input("Pre-flight OK. Clear the area, keep a hand on the robot. Press Enter to start (Ctrl+C = stop)... ")

            ekf_env = replace(cfg.env, obstacles=())
            ekf = EKFEstimator(replace(cfg, env=ekf_env), gate=9.21, odom_cov_scale=args.odom_scale,
                            camera_delay=args.camera_delay, use_lidar=False,
                               R=np.diag([0.15 ** 2, 0.05 ** 2]), R_view=0.15 ** 2)
            viz = None if args.no_viz else RouteVisualizer(cfg, output_dir=out, live_view=not args.no_live_camera)

            orig = (main.Simulation, main.MuJoCoEnvironment)
            main.Simulation = lambda c=None, seed=None, scene_dir=None: HardwareSim(env, cfg)
            main.MuJoCoEnvironment = lambda sim: sim.env
            try:
                result = main.run_pipeline(cfg, seed=None, estimator=ekf, viewer=False, verbose=True,
                                        dock_use_estimate=True, use_recovery=True,
                                        calib_odometry=False, calib_camera=False,
                                        calibration=calibration, camera_delay=args.camera_delay,
                                        visualizer=viz)
            finally:
                main.Simulation, main.MuJoCoEnvironment = orig
        except KeyboardInterrupt:
            print("\nInterrupted by user: stopping the robot.")
            return 1
        finally:
            env.emergency_stop()

    print()
    print_report(result, cfg)
    manual = {}
    if not args.no_prompt:
        print("\nMeasure the final dock position with a ruler (the robot has no ground truth).")
        for key, text in (("dock_range_m", "distance robot -> dock marker [m]"),
                        ("dock_lateral_m", "sideways offset from the marker axis [m]"),
                        ("dock_heading_deg", "heading error [deg]")):
            try:
                manual[key] = float(input(f"  {text}: "))
            except ValueError:
                manual[key] = None
    a = result.log.arrays()
    np.savez(out / "log.npz", t=a["t"], est=a["est"], sigma=a["sigma"], v=a["v"], w=a["w"], mode=a["mode"])
    (out / "summary.json").write_text(json.dumps({
        "success": result.success, "reason": result.reason, "time_s": result.time,
        "camera_stats": result.camera_stats, "events": result.events, "manual_dock_measurement": manual,
        "settings": {"speed_scale": s, "odom_scale": args.odom_scale, "camera_delay": args.camera_delay,
                    "calibration": "calibrate_hardware (results/camera_intrinsics.json + camera_extrinsics.json)", "odometry": "dead reckoning from commands (no encoders)"},
    }, indent=2, default=str), encoding="utf-8")
    print("Saved to", out)
    return 0 if result.success else 1

def main_cli() -> int:
    import os
    p = argparse.ArgumentParser(description="Run Track A on a real TurtleBot3 (rosbridge)")
    p.add_argument("ip", nargs="?", default=ROBOT_IP, help="robot IP (default: ROBOT_IP above)")
    p.add_argument("--check", action="store_true", help="read-only test only, the robot does not move")
    p.add_argument("--speed-scale", type=float, default=SPEED_SCALE)
    p.add_argument("--recalibrate", action="store_true", help="force recalibration from images")
    p.add_argument("--square", type=float, default=0.025, help="chessboard square size [m]")
    p.add_argument("--camera-delay", type=float, default=CAMERA_DELAY)
    p.add_argument("--odom-scale", type=float, default=ODOM_SCALE)
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--no-live-camera", action="store_true")
    p.add_argument("--no-prompt", action="store_true", help="skip the manual dock measurement questions")
    p.add_argument("--force", action="store_true", help="run even if the pre-flight check fails")
    args = p.parse_args()

    if not args.check:                            # plain `python hardware.py` = full run
        return run_hardware(args)
    from calibration import calibrate_hardware
    from rosbridge_api import RosbridgeRobot
    cfg0 = hardware_config()
    cfg_hw = cfg0
    try:
        cal, cfg_hw = calibrate_hardware(cfg0, force=args.recalibrate, square=args.square)
        cfg_hw = with_fov(cfg_hw, cal)
        print(f"Calibration OK | RMS {cal.intrinsics.reprojection_error:.3f} px | "
            f"extrinsics fwd {cal.forward_offset:.3f} m, lat {cal.lateral_offset:+.3f} m, "
            f"yaw {math.degrees(cal.yaw):+.2f} deg")
    except (FileNotFoundError, ValueError) as e:
        print(f"Calibration not ready ({e}) -> sensor check only, marker ranges are approximate")

    with RosbridgeRobot(args.ip) as robot:
        env = RosbridgeEnvironment(robot, cfg_hw)
        try:
            ok = preflight(env, cfg_hw)
        finally:
            env.emergency_stop()

    if ok and input("Calibration + pre-flight OK. Run hardware now? [y/N] ").strip().lower() == "y":
        return run_hardware(args)
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main_cli())