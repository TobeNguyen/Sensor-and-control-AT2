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
from recovery import Explorer, StopGoScan, MapView
from sensing import Sensing, sector_min_range
from robot_model import wrap_angle
import docking
from controller import limit_command
# Robot settings
ROBOT_IP = "192.168.1.213"                       # TurtleBot ID: 192.168.1.2XX
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
SIMPLE_SAFETY = True
IGNORE_NEAR = 0.30
MAX_ANGULAR_HW = 0.3
SURVEY_ANGLE_HW = 2.0 * math.pi
HW_SCAN_STEP = math.radians(30.0)
DOCK_LOCK_TIME = 1.5     # [s] stand still this long after seeing the dock
HW_LEARN_SIGMA = 0.15    # [m] real odometry: 0.08 is too strict
TAG_LOCK_TIME = 3.0      # [s] max stop time per tag
TAG_LOST_TIME = 1.0      # [s] tag not seen this long while stopped -> give up on it
CENTER_TOL = 0.12        # [rad] dock counts as centred below this bearing
MAX_TURN = 0.25          # [rad/s]
MIN_TURN = 0.08          # [rad/s] smaller commands may not move the robot
LEARN_RANGE = 2.0        # [m] drive closer if the dock is farther than this
APPROACH_V = 0.10        # [m/s]
DOCK_MEMORY = 10.0       # [s] remember the dock direction this long (odometry-propagated)
MAX_DOCK_LOCK = 6.0      # [s] SURVEY: longest stop (turn + learn) for one sighting
HW_K_BEARING = 1.2      # [1/s] centre the marker (sim: 5.5)
HW_K_AXIS = 0.6         # [1/s] gentle pull onto the marker axis (sim: 4.0)
HW_VIEW_CLIP = 0.2      # [rad] limit of the axis term
HW_EDGE = 0.15          # [rad] marker near the image edge: only turn TOWARDS it
HW_BETA_MAX = 0.2       # [rad] no forward motion if the marker is this far off-centre
HW_DOCK_TURN = 0.15     # [rad/s] max turn while docking (sim: 0.8)
HW_DOCK_SPEED = 0.10    # [m/s]
ODOM_FILE = RESULTS_DIR / "odometry_hw.json"
DOCK_TARGET_HW = 0.85    # [m] robot centre -> marker at the end of docking (sim: 0.55)
DOCK_TOL_HW = 0.08       # [m] (sim: 0.05)
APPROACH_DIST_HW = 1.4   # [m] AIM approach point in front of the dock (sim: 1.0)
def _turn(b: float) -> float:
    return math.copysign(float(np.clip(abs(b), MIN_TURN, MAX_TURN)), b)


def hw_docking_command(rng, bearing, view, cfg):
    d = cfg.docking
    if rng <= d.target_distance + d.distance_tolerance:          # arrived: only centre
        return limit_command(0.0, float(np.clip(0.8 * bearing, -HW_DOCK_TURN, HW_DOCK_TURN)), cfg)
    speed = float(np.clip(docking.K_RANGE * (rng - d.target_distance),
                        docking.DOCK_MIN_SPEED, HW_DOCK_SPEED))
    speed *= float(np.clip(1.0 - abs(bearing) / HW_BETA_MAX, 0.0, 1.0))
    if abs(bearing) > HW_EDGE:                                   # keep the marker in the narrow FOV
        w = HW_K_BEARING * bearing
    else:
        w = HW_K_BEARING * bearing - HW_K_AXIS * float(np.clip(view, -HW_VIEW_CLIP, HW_VIEW_CLIP))
    return limit_command(speed, float(np.clip(w, -HW_DOCK_TURN, HW_DOCK_TURN)), cfg)


DOCK_PATCH = dict(docking_command=hw_docking_command,
                MEASUREMENT_TIMEOUT=1.0,     # tolerate short WiFi frame drops before SEARCH
                SEARCH_OMEGA=0.2)            # search slowly (sim: 0.7)

class HwSensing(Sensing):
    dock_t, dock_b, dock_r = -99.0, 0.0, 0.0     # last dock sighting (bearing / range propagated)

    def read(self):
        f = super().read()
        C = HwSensing
        if f.marker is not None:
            C.dock_t, C.dock_b, C.dock_r = f.time, f.marker.bearing, f.marker.range
        else:                                    # between camera frames: follow the robot's motion
            C.dock_b = wrap_angle(C.dock_b - f.odometry.dtheta)
            C.dock_r -= f.odometry.ds * math.cos(C.dock_b)
        return f


class HwStopGoScan(StopGoScan):
    def __init__(self, *a, **k):
        k["step"] = HW_SCAN_STEP
        super().__init__(*a, **k)
        self.lock_t0 = None

    def update(self, t, dtheta):
        locked = t - HwSensing.dock_t < DOCK_LOCK_TIME
        if not locked:
            self.lock_t0 = None
        elif self.lock_t0 is None:
            self.lock_t0 = t
        if locked and t - self.lock_t0 < MAX_DOCK_LOCK:
            self.spun += abs(dtheta)
            self.hold_t0, self.step_spun = None, 0.0
            b = HwSensing.dock_b
            if abs(b) > CENTER_TOL:
                return _turn(b), False           # turn BACK towards the dock
            return 0.0, abs(dtheta) < 1e-3       # centred: stand still and observe
        return super().update(t, dtheta)


class HwExplorer(Explorer):
    ekf = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lock_id = None
        self.lock_t0 = self.lock_seen = 0.0
        self.skipped = set()

    def update(self, frame, x_est, sigma_xy):
        n_min = self.run_cfg.tags.survey_min_obs
        still = abs(frame.odometry.dtheta) < 1e-3 and abs(frame.odometry.ds) < 1e-3

        # A) dock seen recently and not learned yet: face it -> get closer -> stand still and learn
        if (self.tag_map.count(self.dock_id) < n_min
                and frame.time - HwSensing.dock_t < DOCK_MEMORY):
            b, r = HwSensing.dock_b, HwSensing.dock_r
            if abs(b) > CENTER_TOL:
                return 0.0, _turn(b)
            if r > LEARN_RANGE and sector_min_range(frame.lidar, 0.0, 0.4) > 0.6:
                return APPROACH_V, 1.0 * b       # drive towards it, keep it centred
            if still and sigma_xy < HW_LEARN_SIGMA and frame.marker is not None:
                self.tag_map.add(x_est, frame.marker, max_range=self.run_cfg.tags.dock_max_range)
            return 0.0, 0.0

        # B) other tags: stop and learn them (dock handled above, never skipped)
        if self.lock_id is None:
            new = [m for m in frame.markers
                   if int(m.marker_id) != self.dock_id
                   and self.tag_map.count(m.marker_id) < n_min and m.marker_id not in self.skipped]
            if new:
                self.lock_id = int(new[0].marker_id)
                self.lock_t0 = self.lock_seen = frame.time
        if self.lock_id is None:
            return super().update(frame, x_est, sigma_xy)
        if any(int(m.marker_id) == self.lock_id for m in frame.markers):
            self.lock_seen = frame.time
        learned = self.tag_map.count(self.lock_id) >= n_min
        if learned or frame.time - self.lock_t0 > TAG_LOCK_TIME or frame.time - self.lock_seen > TAG_LOST_TIME:
            if learned and self.ekf is not None:
                self.ekf.set_tag_map(MapView(self.tag_map, self.dock_id))
            elif not learned:
                self.skipped.add(self.lock_id)
            self.lock_id, self.t0 = None, frame.time
            return 0.0, 0.0
        if still and sigma_xy < HW_LEARN_SIGMA:
            tags = self.run_cfg.tags
            for m in frame.markers:
                self.tag_map.add(x_est, m, max_range=min(tags.survey_max_range, 2.0))
        return 0.0, 0.0

def load_odometry_hw():
    """(wheel_radius, wheel_base) measured on the real robot, or None."""
    try:
        d = json.loads(ODOM_FILE.read_text())
        return float(d["wheel_radius"]), float(d["wheel_base"])
    except (OSError, ValueError, KeyError):
        return None
    
def hardware_config(speed_scale: float = 1.0) -> Config:
    c = CONFIG
    extra = tuple(c.tags.fixed_tags[i] if i < len(c.tags.fixed_tags) else (0.0, 0.0, 0.0)
                for i in range(N_EXTRA_TAGS))
    odo = load_odometry_hw()
    r, b = odo if odo is not None else (c.robot.wheel_radius, c.robot.wheel_base)
    return replace(
        c,
        camera=replace(c.camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, marker_size=MARKER_SIZE),
        robot=replace(c.robot, wheel_radius=r, wheel_base=b,
                      max_linear_vel=c.robot.max_linear_vel * speed_scale,
                      max_angular_vel=min(c.robot.max_angular_vel * speed_scale, MAX_ANGULAR_HW)),
        sim=replace(c.sim, control_dt=CONTROL_DT),
        docking=replace(c.docking, target_distance=DOCK_TARGET_HW,
                        distance_tolerance=DOCK_TOL_HW),
        tags=replace(c.tags, fixed_tags=extra, survey_angle=SURVEY_ANGLE_HW,
                    approach_dist=APPROACH_DIST_HW),
        env=replace(c.env, obstacles=()),
        realism=replace(c.realism, enabled=False),
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
    def __init__(self, robot, cfg: Config = CONFIG, lidar_timeout: float = 0.15,
                camera_timeout: float = 0.3):
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
        parsed = _parse_scan(_query(self._scan_fn, self.lidar_timeout))
        lid = self.cfg.lidar
        if parsed is not None:
            r = resample_scan(*parsed, self._angles, lid.range_min, lid.range_max)
            if SIMPLE_SAFETY:
                r[r < IGNORE_NEAR] = lid.range_max
            self._ranges = r
            return self._ranges, time.monotonic()
        stamp = time.monotonic() if SIMPLE_SAFETY else self._lidar_stamp
        return self._ranges, stamp

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
    i = int(np.argmin(ranges))
    print(f"  nearest beam: {math.degrees(env.lidar_angles[i]):.0f} deg, {ranges[i]:.2f} m")
    if n_valid < 20:
        print("  !! very few valid beams: check the LiDAR topic / angle convention"); ok = False
    img = env.read_camera()
    t0 = time.time()
    while not np.any(img) and time.time() - t0 < 5.0:
        time.sleep(0.2)
        img = env.read_camera()
    print(f"Camera: {img.shape[1]}x{img.shape[0]}, mean brightness {img.mean():.0f}, age {env.sensor_ages()[1]:.2f} s")
    if not np.any(img):
        print("  !! camera image is empty"); ok = False
    found = detect_markers(img, cfg, env.time)
    print("Markers in view:", [(m.marker_id, round(m.range, 2), round(math.degrees(m.bearing), 1)) for m in found] or "none")
    env.emergency_stop()
    print("Stop command sent.")
    return ok

def _drive(env, v: float, w: float, duration: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        env.step(v, w)
    for _ in range(int(1.0 / env.cfg.sim.control_dt)):     # 1 s settle
        env.step(0.0, 0.0)


def calibrate_odometry_hw(env: RosbridgeEnvironment, cfg: Config):
    if not env._joint_ok:
        print("!! no /joint_states: odometry calibration needs the real wheel encoders")
        return None
    r0, b0 = cfg.robot.wheel_radius, cfg.robot.wheel_base

    # 1) straight line -> wheel radius
    input("\n[1/2] STRAIGHT: clear ~1.3 m in front, mark the wheel axle on the floor. "
          "Enter = drive ~1 m... ")
    e0 = env.read_encoders(); _drive(env, 0.10, 0.0, 10.0); e1 = env.read_encoders()
    dphi = e1 - e0
    print(f"  wheel rotation L {dphi[0]:.3f} rad, R {dphi[1]:.3f} rad | "
          f"odometry says {r0 * dphi.mean():.3f} m (r = {r0:.4f})")
    if dphi.mean() <= 0:
        print("!! wheel angles decreased while driving forward: check /joint_states sign"); return None
    d_true = float(input("  measured distance [m]: "))
    r = d_true / float(dphi.mean())

    # 2) spin in place -> wheel base
    input("\n[2/2] SPIN: mark the robot heading with tape. Enter = spin ~2 turns left... ")
    e0 = env.read_encoders(); _drive(env, 0.0, 0.3, 4.0 * math.pi / 0.3); e1 = env.read_encoders()
    dphi = e1 - e0
    print(f"  odometry says {math.degrees(r * (dphi[1] - dphi[0]) / b0):.0f} deg (b = {b0:.4f})")
    extra = input("  2 full turns (720 deg) + how many EXTRA deg? (+ more, - less, empty = 0): ").strip()
    theta = math.radians(720.0 + (float(extra) if extra else 0.0))
    b = r * float(dphi[1] - dphi[0]) / theta

    print(f"\n  wheel_radius {r0:.4f} -> {r:.4f} m | wheel_base {b0:.4f} -> {b:.4f} m")
    if not (0.025 < r < 0.045 and 0.12 < b < 0.32):
        print("!! values look wrong (TB3: r ~0.033, Burger b ~0.160, Waffle b ~0.287) -> NOT saved")
        return None
    ODOM_FILE.parent.mkdir(parents=True, exist_ok=True)
    ODOM_FILE.write_text(json.dumps({"wheel_radius": r, "wheel_base": b,
                                     "straight_m": d_true, "spin_deg": math.degrees(theta)}, indent=2))
    print("  saved:", ODOM_FILE, "(used automatically by hardware_config)")
    return r, b

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

            orig = (main.Simulation, main.MuJoCoEnvironment, main.StopGoScan, main.Explorer, main.Sensing)
            main.Simulation = lambda c=None, seed=None, scene_dir=None: HardwareSim(env, cfg)
            main.MuJoCoEnvironment = lambda sim: sim.env
            main.StopGoScan, main.Explorer, main.Sensing = HwStopGoScan, HwExplorer, HwSensing
            HwExplorer.ekf = ekf
            HwSensing.dock_t = -99.0
            dock_orig = {k: getattr(docking, k) for k in DOCK_PATCH}
            for k, v in DOCK_PATCH.items():
                setattr(docking, k, v)
            try:
                result = main.run_pipeline(cfg, seed=None, estimator=ekf, viewer=False, verbose=True,
                                        dock_use_estimate=True, use_recovery=True,
                                        calib_odometry=False, calib_camera=False,
                                        calibration=calibration, camera_delay=args.camera_delay,
                                        visualizer=viz)
            finally:
                (main.Simulation, main.MuJoCoEnvironment, main.StopGoScan, main.Explorer, main.Sensing) = orig
                for k, v in dock_orig.items():
                    setattr(docking, k, v)
            env.emergency_stop()
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
    p.add_argument("--calib-odom", action="store_true",
            help="with --check: drive 1 m + spin 2 turns to calibrate the wheel odometry (robot MOVES)")
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
            odo = load_odometry_hw()
            print("Odometry params:", f"r {cfg_hw.robot.wheel_radius:.4f} m, b {cfg_hw.robot.wheel_base:.4f} m",
                "(calibrated)" if odo else "(nominal config, not calibrated)")
            if args.calib_odom and ok:
                calibrate_odometry_hw(env, cfg_hw)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user: stopping the robot.")
            return 1
        except Exception as e:
            print(f"\nERROR: {e!r} -> stopping the robot.")
            return 1
        finally:
            for _ in range(10):
                try:
                    env.robot.send_command(0.0, 0.0)
                except Exception:
                    pass
                time.sleep(0.05)
            time.sleep(0.3)

    if ok:
        print(f"Check OK. Now run:  python hardware.py {args.ip}")
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main_cli())