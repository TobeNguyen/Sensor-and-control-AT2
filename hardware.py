# hardware.py - BACKUP runner for the real TurtleBot3 (rosbridge), Track A.
# Stand-alone: it does NOT use main.run_pipeline and does NOT touch real_robot.py or the simulation files.
# Same idea as main.py, simplified:  SURVEY -> (EXPLORE) -> AIM -> DOCK -> DOCKED
#   - pose = wheel odometry (no EKF, no tag recovery: simpler and enough for one dock)
#   - the dock pose is stored in the odometry frame and refreshed at every sighting
#   - every decision is taken while STANDING STILL (camera delay + narrow 28 deg FOV)
#
#   python hardware.py 192.168.1.213 --check               sensors only, robot does not move
#   python hardware.py 192.168.1.213 --check --calib-odom  + odometry calibration (robot MOVES)
#   python hardware.py 192.168.1.213                       full run
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2

from config import CONFIG, RESULTS_DIR, Config
from environment import HardwareEnvironment
from sensing import Sensing, sector_min_range
from robot_model import wrap_angle, motion_model

# ============================================================ robot / sensors
ROBOT_IP = "192.168.1.213"
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
MARKER_SIZE = 0.10           # [m] black part of the printed marker
CONTROL_DT = 0.05            # [s] 20 Hz
CAMERA_PERIOD = 0.1          # [s] camera processed at 10 Hz
CAMERA_DELAY = 0.2           # [s] image age: the pose of that moment is used for the marker
SPEED_SCALE = 0.5
MAX_ANGULAR_HW = 0.3         # [rad/s]
SCAN_METHOD = None
SCAN_NAMES = ("query_laser_scan", "query_scan", "query_laser", "query_lidar",
              "get_laser_scan", "get_scan", "laser_scan", "scan")
MAX_BEAM_GAP = math.radians(3.0)
IGNORE_NEAR = 0.30           # [m] LiDAR returns closer than this = robot parts
ODOM_FILE = RESULTS_DIR / "odometry_hw.json"

# ============================================================ mission
RUN_TIMEOUT = 300.0          # [s]
SCAN_STEP = math.radians(30.0)   # SURVEY: turn 30 deg, stop, look (12 stops = 360 deg)
SCAN_W = 0.3                 # [rad/s]
SCAN_HOLD = 1.2              # [s] standing still at every stop
STILL_TIME = 0.3             # [s] robot must be still this long before a marker counts
OBS_MIN = 3                  # detections needed for a decision (median)
LOOK_TIME = 2.5              # [s] max time to collect them
CENTER_TOL = 0.12            # [rad]
MIN_TURN = 0.08              # [rad/s] smaller commands do not move the robot
MAX_TURN = 0.3               # [rad/s]
EXPLORE_STEP = 1.0           # [m] drive this far between two surveys
MAX_EXPLORES = 6
AIM_DIST = 1.3               # [m] pre-dock point on the marker axis
LAT_TOL = 0.15               # [m] lateral offset accepted to start docking
DOCK_START_MAX = 2.0         # [m] start docking only closer than this
MAX_AIM_TRIES = 3
DRIVE_V = 0.12               # [m/s]
REVERSE_V = 0.08             # [m/s]

# ============================================================ docking
DOCK_TARGET = 0.85           # [m] robot centre -> marker at the end (marker must still fit the image)
DOCK_TOL = 0.08              # [m]
DOCK_BEARING_TOL = 0.12      # [rad]
DOCK_AXIS_TOL = 0.25         # [rad] |bearing - view|
DOCK_CONFIRM = 3             # consecutive real detections inside the tolerances
DOCK_SPEED = 0.10            # [m/s]
DOCK_MIN_SPEED = 0.03
K_BEARING = 1.2
K_AXIS = 0.6
BETA_MAX = 0.2               # [rad] no forward motion if the marker is this far off-centre
DOCK_TURN = 0.15             # [rad/s]
DOCK_LOST = 2.0              # [s] marker not seen this long -> back to AIM
ALIGN_TIME = 4.0             # [s] arrived but not aligned this long -> back to AIM
DOCK_TIMEOUT = 45.0          # [s]

# ============================================================ safety
FRONT_STOP = 0.40            # [m] LiDAR, +-25 deg
REAR_STOP = 0.35             # [m] LiDAR, +-35 deg behind
LIDAR_MAX_AGE = 1.0          # [s] older LiDAR -> stop


# ============================================================ config
def load_odometry_hw():
    try:
        d = json.loads(ODOM_FILE.read_text())
        return float(d["wheel_radius"]), float(d["wheel_base"])
    except (OSError, ValueError, KeyError):
        return None


def hardware_config(speed_scale: float = 1.0) -> Config:
    c = CONFIG
    odo = load_odometry_hw()
    r, b = odo if odo is not None else (c.robot.wheel_radius, c.robot.wheel_base)
    return replace(
        c,
        camera=replace(c.camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, marker_size=MARKER_SIZE),
        robot=replace(c.robot, wheel_radius=r, wheel_base=b,
                      max_linear_vel=c.robot.max_linear_vel * speed_scale,
                      max_angular_vel=min(c.robot.max_angular_vel * speed_scale, MAX_ANGULAR_HW)),
        sim=replace(c.sim, control_dt=CONTROL_DT),
        env=replace(c.env, obstacles=()),
        realism=replace(c.realism, enabled=False),
    )


def with_fov(cfg: Config, calibration) -> Config:
    fy = float(calibration.K[1, 1])
    return replace(cfg, camera=replace(cfg.camera, fovy_deg=math.degrees(2.0 * math.atan(cfg.camera.height / 2.0 / fy))))


# ============================================================ small helpers
def turn_cmd(err: float) -> float:
    """Turn rate towards a heading error, never below MIN_TURN (dead band of the motors)."""
    return math.copysign(float(np.clip(1.5 * abs(err), MIN_TURN, MAX_TURN)), err)


def lateral(r, b, v) -> float:
    return r * math.sin(wrap_angle(b - v))          # >0: marker axis on the robot's left


class DockPose:
    """Dock in the odometry frame: marker centre (x, y) and yaw_in = heading that faces it head-on."""

    def __init__(self, x, y, yaw_in):
        self.x, self.y, self.yaw_in = float(x), float(y), float(yaw_in)

    @staticmethod
    def from_marker(pose, r, b, v):
        x, y, th = pose
        return DockPose(x + r * math.cos(th + b), y + r * math.sin(th + b), th + v)

    @staticmethod
    def median(obs):
        return DockPose(float(np.median([o.x for o in obs])), float(np.median([o.y for o in obs])),
                        math.atan2(np.mean([math.sin(o.yaw_in) for o in obs]),
                                   np.mean([math.cos(o.yaw_in) for o in obs])))

    def relative(self, pose):
        """(range, bearing, view) as the camera would measure them from `pose`."""
        x, y, th = pose
        r = math.hypot(self.x - x, self.y - y)
        b = wrap_angle(math.atan2(self.y - y, self.x - x) - th)
        return r, b, wrap_angle(self.yaw_in - th)

    def axis_point(self, d):
        return self.x - d * math.cos(self.yaw_in), self.y - d * math.sin(self.yaw_in)


class GoTo:
    """Turn -> drive (forwards, or backwards if the goal is close behind) -> face a point. Odometry only."""

    def __init__(self, goal, face=None, allow_reverse=True):
        self.goal, self.face, self.allow_reverse = goal, face, allow_reverse
        self.state, self.rev, self.stall = "TURN1", False, 0.0

    def step(self, pose, ds, dt):
        """(v, w, done)."""
        x, y, th = pose
        gx, gy = self.goal
        dist = math.hypot(gx - x, gy - y)
        head = math.atan2(gy - y, gx - x)
        if self.state == "TURN1":
            if dist < 0.08:
                self.state = "TURN2"
            else:
                self.rev = self.allow_reverse and dist < 0.8 and math.cos(head - th) < -0.2
                err = wrap_angle(head + (math.pi if self.rev else 0.0) - th)
                if abs(err) > 0.08:
                    return 0.0, turn_cmd(err), False
                self.state = "DRIVE"
        if self.state == "DRIVE":
            if dist > 0.08:
                err = wrap_angle(head + (math.pi if self.rev else 0.0) - th)
                v = min(REVERSE_V if self.rev else DRIVE_V, 0.5 * dist + 0.03) * max(0.0, math.cos(err))
                self.stall = self.stall + dt if abs(ds) < 1e-4 else 0.0
                if self.stall < 3.0:                      # blocked by the safety stop: give up on this goal
                    return (-v if self.rev else v), float(np.clip(1.5 * err, -MAX_TURN, MAX_TURN)), False
            self.state = "TURN2"
        if self.face is None:
            return 0.0, 0.0, True
        err = wrap_angle(math.atan2(self.face[1] - y, self.face[0] - x) - th)
        if abs(err) > 0.06:
            return 0.0, turn_cmd(err), False
        return 0.0, 0.0, True


class StopGoScan:
    """Turn SCAN_STEP, stand still SCAN_HOLD, repeat until `total` is covered."""

    def __init__(self, total=2.0 * math.pi):
        self.total, self.spun, self.turned = total, 0.0, 0.0
        self.phase, self.t0, self.done = "HOLD", None, False

    def update(self, t, dtheta):
        if self.phase == "HOLD":
            if self.t0 is None:
                self.t0 = t
            if t - self.t0 >= SCAN_HOLD:
                if self.spun >= self.total - 1e-3:
                    self.done = True
                    return 0.0
                self.phase, self.turned = "TURN", 0.0
            return 0.0
        self.turned += abs(dtheta)
        self.spun += abs(dtheta)
        if self.turned >= SCAN_STEP:
            self.phase, self.t0 = "HOLD", None
            return 0.0
        return SCAN_W


def freest_heading(scan) -> float:
    """Relative heading with the most free space (front preferred)."""
    best, best_cost = 0.0, -1e9
    for a in np.radians(np.arange(-180, 180, 15)):
        free = sector_min_range(scan, float(a), math.radians(15))
        cost = min(free, 2.5) - 0.3 * abs(a) / math.pi
        if cost > best_cost:
            best, best_cost = float(a), cost
    return best


def dock_command(r, b, v):
    """Gentle visual servoing (narrow camera): keep the marker in view, small pull onto the axis."""
    if r <= DOCK_TARGET + DOCK_TOL:
        w = turn_cmd(b) if abs(b) > DOCK_BEARING_TOL else 0.0
        return 0.0, float(np.clip(w, -DOCK_TURN, DOCK_TURN))
    speed = float(np.clip(1.0 * (r - DOCK_TARGET), DOCK_MIN_SPEED, DOCK_SPEED))
    speed *= float(np.clip(1.0 - abs(b) / BETA_MAX, 0.0, 1.0))
    w = K_BEARING * b - (K_AXIS * float(np.clip(v, -0.2, 0.2)) if abs(b) < 0.15 else 0.0)
    return speed, float(np.clip(w, -DOCK_TURN, DOCK_TURN))


def is_docked(r, b, v) -> bool:
    return (r <= DOCK_TARGET + DOCK_TOL and abs(b) <= DOCK_BEARING_TOL
            and abs(wrap_angle(b - v)) <= DOCK_AXIS_TOL)


# ============================================================ mission (state machine)
class Mission:
    def __init__(self):
        self.pose = np.zeros(3)                  # odometry frame, start = (0, 0, 0)
        self.hist = []                           # (t, pose) for the camera delay
        self.state, self.t_state = "SURVEY", 0.0
        self.events: list[str] = []
        self.dock: DockPose | None = None        # median estimate
        self.last: DockPose | None = None        # latest sighting
        self.last_t = -99.0
        self.obs: list = []                      # still observations of the current LOOK
        self.still_t0 = 0.0
        self.scan = StopGoScan()
        self.goto: GoTo | None = None
        self.after_lock = "AIM"
        self.n_explore = self.n_aim = 0
        self.ok_count = 0
        self.arrived = 0.0
        self.done, self.success, self.reason = False, False, "timeout"

    # ---------------------------------------------------------- bookkeeping
    def note(self, t, text):
        s = f"[t={t:6.1f}s] {text}"
        self.events.append(s)
        print("  >> " + s, flush=True)

    def go(self, t, state, text=None):
        self.state, self.t_state = state, t
        if text:
            self.note(t, text)

    def pose_at(self, t):
        for tt, p in reversed(self.hist):
            if tt <= t:
                return p
        return self.hist[0][1] if self.hist else self.pose

    def sense(self, f):
        self.pose = motion_model(self.pose, f.odometry.ds, f.odometry.dtheta)
        self.hist.append((f.time, self.pose.copy()))
        self.hist = [h for h in self.hist if f.time - h[0] < 2.0]
        moving = abs(f.odometry.ds) > 1e-3 or abs(f.odometry.dtheta) > 1e-3
        if moving:
            self.still_t0 = f.time
        m = f.marker
        if m is not None:
            p = self.pose_at(f.time - CAMERA_DELAY)
            self.last = DockPose.from_marker(p, m.range, m.bearing, m.view_angle)
            self.last_t = f.time
            if f.time - self.still_t0 >= STILL_TIME:
                self.obs.append((f.time, self.last, m))

    def seen_recently(self, t, age=1.0):
        return t - self.last_t < age

    def fresh_obs(self, t):
        self.obs = [o for o in self.obs if t - o[0] < LOOK_TIME]
        return self.obs

    # ---------------------------------------------------------- one control step
    def step(self, f, dt):
        t = f.time
        self.sense(f)
        S = self.state

        if S in ("SURVEY", "SEARCH"):
            if self.seen_recently(t):
                self.after_lock = "AIM"
                self.obs = []
                return self.go(t, "LOCK", f"{S}: dock seen -> stop and look") or (0.0, 0.0)
            w = self.scan.update(t, f.odometry.dtheta)
            if self.scan.done:
                if self.n_explore >= MAX_EXPLORES:
                    self.done, self.reason = True, "dock not found"
                    self.note(t, "FAILED: dock not found")
                    return 0.0, 0.0
                self.n_explore += 1
                a = freest_heading(f.lidar)
                L = float(np.clip(sector_min_range(f.lidar, a, 0.3) - 0.7, 0.3, EXPLORE_STEP))
                th = self.pose[2] + a
                self.goto = GoTo((self.pose[0] + L * math.cos(th), self.pose[1] + L * math.sin(th)),
                                allow_reverse=False)
                return self.go(t, "EXPLORE", f"{S}: no dock in 360 deg -> EXPLORE #{self.n_explore} "
                                            f"({math.degrees(a):+.0f} deg, {L:.2f} m)") or (0.0, 0.0)
            return 0.0, w

        if S == "EXPLORE":
            if self.seen_recently(t):
                self.obs = []
                return self.go(t, "LOCK", "EXPLORE: dock seen -> stop and look") or (0.0, 0.0)
            v, w, fin = self.goto.step(self.pose, f.odometry.ds, dt)
            if fin:
                self.scan = StopGoScan()
                return self.go(t, "SURVEY", "EXPLORE step done -> SURVEY 360 deg") or (0.0, 0.0)
            return v, w

        if S == "LOCK":                                      # centre the dock, stand still, collect OBS_MIN
            if not self.seen_recently(t, 3.0):
                self.scan = StopGoScan()
                return self.go(t, "SEARCH", "LOCK: dock lost -> SEARCH 360 deg") or (0.0, 0.0)
            _, b, _ = self.last.relative(self.pose)
            if abs(b) > CENTER_TOL:
                self.obs = []
                return 0.0, turn_cmd(b)
            obs = self.fresh_obs(t)
            if len(obs) >= OBS_MIN:
                self.dock = DockPose.median([o[1] for o in obs])
                r, b, v = self.dock.relative(self.pose)
                self.n_aim = 0
                self.obs = []
                return self.go(t, "AIM-LOOK", f"dock learned: {r:.2f} m, {math.degrees(b):+.0f} deg, "
                                              f"{lateral(r, b, v):+.2f} m off the axis -> AIM") or (0.0, 0.0)
            return 0.0, 0.0

        if S == "AIM-GO":
            v, w, fin = self.goto.step(self.pose, f.odometry.ds, dt)
            if fin:
                self.obs = []
                return self.go(t, "AIM-LOOK") or (0.0, 0.0)
            return v, w

        if S == "AIM-LOOK":                                  # stand still, look, decide
            if self.seen_recently(t, 1.0):
                _, b, _ = self.last.relative(self.pose)
                if abs(b) > CENTER_TOL:
                    self.obs = []
                    return 0.0, turn_cmd(b)
            obs = self.fresh_obs(t)
            if len(obs) >= OBS_MIN:
                self.dock = DockPose.median([o[1] for o in obs])
                rs = np.median([o[2].range for o in obs])
                bs = np.median([o[2].bearing for o in obs])
                vs = np.median([o[2].view_angle for o in obs])
                lat = lateral(rs, bs, vs)
                self.obs = []
                if (abs(lat) <= LAT_TOL and rs <= DOCK_START_MAX) or \
                        (self.n_aim >= MAX_AIM_TRIES and rs <= DOCK_START_MAX + 0.2):
                    self.ok_count, self.arrived = 0, 0.0
                    return self.go(t, "DOCK", f"AIM: dock {rs:.2f} m, {lat:+.2f} m off the axis -> DOCK") or (0.0, 0.0)
                self.n_aim += 1
                ax, ay = self.dock.axis_point(AIM_DIST)
                self.goto = GoTo((ax, ay), face=(self.dock.x, self.dock.y))
                d = math.hypot(ax - self.pose[0], ay - self.pose[1])
                return self.go(t, "AIM-GO", f"AIM #{self.n_aim}: dock {rs:.2f} m, {lat:+.2f} m off the axis "
                                            f"-> axis point {AIM_DIST:.1f} m in front of it ({d:.2f} m away)") or (0.0, 0.0)
            if t - self.t_state > LOOK_TIME + 0.5 and not self.seen_recently(t, 1.0):
                if self.dock is not None and self.n_aim < MAX_AIM_TRIES + 2:   # face the stored dock first
                    _, b, _ = self.dock.relative(self.pose)
                    if abs(b) > CENTER_TOL:
                        return 0.0, turn_cmd(b)
                    if t - self.t_state < LOOK_TIME + 3.0:
                        return 0.0, 0.0
                self.scan = StopGoScan()
                return self.go(t, "SEARCH", "AIM: dock not in view -> SEARCH 360 deg") or (0.0, 0.0)
            return 0.0, 0.0

        if S == "DOCK":
            m = f.marker
            if m is not None:
                self.ok_count = self.ok_count + 1 if is_docked(m.range, m.bearing, m.view_angle) else 0
                if self.ok_count >= DOCK_CONFIRM:
                    self.done, self.success, self.reason = True, True, "docked"
                    self.note(t, f"DOCKED: {m.range:.2f} m, bearing {math.degrees(m.bearing):+.1f} deg, "
                                f"axis {math.degrees(wrap_angle(m.bearing - m.view_angle)):+.1f} deg")
                    return 0.0, 0.0
            if not self.seen_recently(t, DOCK_LOST):
                self.obs = []
                return self.go(t, "AIM-LOOK", "DOCK: marker lost -> AIM") or (0.0, 0.0)
            if t - self.t_state > DOCK_TIMEOUT:
                self.obs = []
                return self.go(t, "AIM-LOOK", "DOCK: too long -> AIM") or (0.0, 0.0)
            r, b, v = self.last.relative(self.pose)
            if r <= DOCK_TARGET + DOCK_TOL:
                self.arrived += dt
                if self.arrived > ALIGN_TIME:
                    self.obs = []
                    return self.go(t, "AIM-LOOK", f"DOCK: arrived but {lateral(r, b, v):+.2f} m off the axis "
                                                f"-> AIM (back to the axis point)") or (0.0, 0.0)
            else:
                self.arrived = 0.0
            if r < DOCK_TARGET - 0.15:
                return -0.05, 0.0                             # too close: back off slowly
            return dock_command(r, b, v)

        return 0.0, 0.0


def safety(v, w, scan, lidar_age):
    if lidar_age > LIDAR_MAX_AGE:
        return 0.0, 0.0, "LiDAR too old"
    if v > 0.0 and sector_min_range(scan, 0.0, math.radians(25)) < FRONT_STOP:
        return 0.0, w, "front stop"
    if v < 0.0 and sector_min_range(scan, math.pi, math.radians(35)) < REAR_STOP:
        return 0.0, w, "rear stop"
    return v, w, ""


# ============================================================ rosbridge helpers
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
    raise AttributeError(f"No laser-scan method found on RosbridgeRobot. Public methods: {public}. "
                        "Put the right name in SCAN_METHOD at the top of hardware.py.")


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
        a, r = r, a
    return a, r


def resample_scan(a, r, target, range_min: float, range_max: float) -> np.ndarray:
    r = np.where(np.isfinite(r) & (r >= range_min) & (r < range_max), r, range_max)
    a = np.mod(a, 2.0 * math.pi)
    diff = np.abs((np.asarray(target)[:, None] - a[None, :] + math.pi) % (2.0 * math.pi) - math.pi)
    out = r[diff.argmin(axis=1)]
    out[diff.min(axis=1) > MAX_BEAM_GAP] = range_max
    return out


class RosbridgeEnvironment(HardwareEnvironment):
    def __init__(self, robot, cfg: Config = CONFIG, lidar_timeout: float = 0.15, camera_timeout: float = 0.3):
        super().__init__(cfg)
        self.robot = robot
        self.lidar_timeout, self.camera_timeout = float(lidar_timeout), float(camera_timeout)
        self._scan_fn = _find_scan_method(robot)
        self._enc = np.zeros(2)
        self._enc_last = np.zeros(2)
        self._cmd = (0.0, 0.0)
        self._t_int = time.monotonic()
        self._ranges = np.full(cfg.lidar.num_beams, cfg.lidar.range_max)
        self._image = np.zeros((cfg.camera.height, cfg.camera.width, 3), np.uint8)
        self.n_scans = 0                               # real scans received (0 = no /scan)
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
            r[r < IGNORE_NEAR] = lid.range_max
            self._ranges = r
            self.n_scans += 1
            return self._ranges, time.monotonic()
        return self._ranges, self._lidar_stamp           # no new scan: the age grows -> safety stop

    def _driver_camera(self):
        bgr = _query(self.robot.query_image, self.camera_timeout)
        if bgr is None:
            return self._image, self._camera_stamp
        cam = self.cfg.camera
        if bgr.shape[0] != cam.height or bgr.shape[1] != cam.width:
            raise ValueError(f"camera image is {bgr.shape[1]}x{bgr.shape[0]} but config / calibration "
                            f"expect {cam.width}x{cam.height}")
        self._image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._image, time.monotonic()

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
        self._integrate()
        self.robot.send_command(v, w)
        self._cmd = (v, w)
        remaining = self.cfg.sim.control_dt - (time.monotonic() - self._last_step)
        if remaining > 0.0:
            time.sleep(remaining)
        self._last_step = time.monotonic()

    def emergency_stop(self) -> None:
        self._integrate()
        self._cmd = (0.0, 0.0)
        for _ in range(10):                      # ~0.5 s of zeros: one lost WiFi packet cannot keep it moving
            try:
                self.robot.send_command(0.0, 0.0)
            except Exception:
                pass
            time.sleep(0.05)


# ============================================================ pre-flight check (robot does not move)
def preflight(env: RosbridgeEnvironment, cfg: Config) -> bool:
    from sensing import detect_markers, make_lidar_scan
    ok = True
    ranges = None
    for _ in range(5):
        ranges = env.read_lidar()
        time.sleep(0.1)
    scan = make_lidar_scan(ranges, env.lidar_angles, cfg)
    n_valid = int(scan.valid.sum())
    print(f"LiDAR : {n_valid}/{len(ranges)} valid beams, min range {ranges.min():.2f} m, "
        f"scans received {env.n_scans}")
    if env.n_scans == 0:
        print("  !! NO LiDAR scan received: is the LiDAR spinning? restart bringup / rosbridge on the robot")
        ok = False
    elif n_valid < 20:
        print("  !! very few valid beams: check the LiDAR / angle convention"); ok = False
    else:
        i = int(np.argmin(ranges))
        print(f"  nearest beam: {math.degrees(env.lidar_angles[i]):.0f} deg, {ranges[i]:.2f} m")
    img = env.read_camera()
    t0 = time.time()
    while not np.any(img) and time.time() - t0 < 5.0:
        time.sleep(0.2)
        img = env.read_camera()
    print(f"Camera: {img.shape[1]}x{img.shape[0]}, mean brightness {img.mean():.0f}")
    if not np.any(img):
        print("  !! camera image is empty"); ok = False
    found = detect_markers(img, cfg, env.time)
    print("Markers in view:", [(m.marker_id, round(m.range, 2), round(math.degrees(m.bearing), 1))
                            for m in found] or "none")
    env.emergency_stop()
    print("Stop command sent.")
    return ok


# ============================================================ odometry calibration (robot MOVES)
def _drive(env, v: float, w: float, duration: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        env.step(v, w)
    for _ in range(int(1.0 / env.cfg.sim.control_dt)):
        env.step(0.0, 0.0)


def _ask_float(text, default=None):
    while True:
        s = input(text).strip()
        if not s and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print("  please type a number" + (" (or just Enter)" if default is not None else ""))


def calibrate_odometry_hw(env: RosbridgeEnvironment, cfg: Config):
    if not env._joint_ok:
        print("!! no /joint_states: odometry calibration needs the real wheel encoders")
        return None
    r0, b0 = cfg.robot.wheel_radius, cfg.robot.wheel_base
    input("\n[1/2] STRAIGHT: clear ~1.3 m in front, mark the wheel axle on the floor. Enter = drive ~1 m... ")
    e0 = env.read_encoders(); _drive(env, 0.10, 0.0, 10.0); e1 = env.read_encoders()
    dphi = e1 - e0
    print(f"  wheel rotation L {dphi[0]:.3f} rad, R {dphi[1]:.3f} rad | odometry says {r0 * dphi.mean():.3f} m")
    if dphi.mean() <= 0:
        print("!! wheel angles decreased while driving forward: check /joint_states sign"); return None
    d_true = _ask_float("  measured distance [m]: ")
    r = d_true / float(dphi.mean())
    input("\n[2/2] SPIN: mark the robot heading with tape. Enter = spin ~2 turns left... ")
    e0 = env.read_encoders(); _drive(env, 0.0, 0.3, 4.0 * math.pi / 0.3); e1 = env.read_encoders()
    dphi = e1 - e0
    print(f"  odometry says {math.degrees(r * (dphi[1] - dphi[0]) / b0):.0f} deg (b = {b0:.4f})")
    extra = _ask_float("  2 full turns (720 deg) + how many EXTRA deg? (+ more, - less, just Enter = 0): ", 0.0)
    theta = math.radians(720.0 + extra)
    b = r * float(dphi[1] - dphi[0]) / theta
    print(f"\n  wheel_radius {r0:.4f} -> {r:.4f} m | wheel_base {b0:.4f} -> {b:.4f} m")
    if not (0.025 < r < 0.045 and 0.12 < b < 0.32):
        print("!! values look wrong (TB3: r ~0.033, Burger b ~0.160, Waffle b ~0.287) -> NOT saved")
        return None
    ODOM_FILE.parent.mkdir(parents=True, exist_ok=True)
    ODOM_FILE.write_text(json.dumps({"wheel_radius": r, "wheel_base": b,
                                    "straight_m": d_true, "spin_deg": math.degrees(theta)}, indent=2))
    print("  saved:", ODOM_FILE)
    return r, b


# ============================================================ full run
def save_plot(path_xy, mission, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    p = np.asarray(path_xy)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.5, label="robot (odometry)")
    ax.plot(p[0, 0], p[0, 1], "go", label="start")
    ax.plot(p[-1, 0], p[-1, 1], "ro", label="end")
    if mission.dock is not None:
        d = mission.dock
        ax.plot(d.x, d.y, "ks", ms=10, label="dock (estimated)")
        ax.plot(*zip((d.x, d.y), d.axis_point(AIM_DIST)), "k--", lw=1)
    ax.set_aspect("equal"); ax.grid(True); ax.legend(); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"hardware.py: {mission.reason}")
    fig.savefig(out / "route.png", dpi=120)
    plt.close(fig)


def run_hardware(args) -> int:
    from rosbridge_api import RosbridgeRobot
    from calibration import calibrate_hardware

    cfg = hardware_config(float(args.speed_scale))
    try:
        calibration, cfg = calibrate_hardware(cfg, force=args.recalibrate, square=args.square)
    except (FileNotFoundError, ValueError) as e:
        print(f"Calibration failed: {e}")
        return 2
    cfg = with_fov(cfg, calibration)
    odo = load_odometry_hw()
    print(f"Odometry params: r {cfg.robot.wheel_radius:.4f} m, b {cfg.robot.wheel_base:.4f} m",
          "(calibrated)" if odo else "(nominal: run --check --calib-odom)")
    out = RESULTS_DIR / "hardware_backup" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    mission = Mission()
    path, rows = [], []
    with RosbridgeRobot(args.ip) as robot:
        env = RosbridgeEnvironment(robot, cfg)
        try:
            if not preflight(env, cfg) and not args.force:
                print("Pre-flight failed. Fix the problems above (or --force at your own risk).")
                return 2
            input("Pre-flight OK. Clear the area, keep a hand on the robot. Enter = start (Ctrl+C = stop)... ")
            sensing = Sensing(env, cfg, camera_period=CAMERA_PERIOD, calibration=calibration)
            dt = cfg.sim.control_dt
            t0 = env.time
            last_print, last_guard = -99.0, ""
            print(f"{'t[s]':>6} {'x':>6} {'y':>6} {'th':>6} {'state':<9} {'v':>5} {'w':>6} dock")
            while True:
                f = sensing.read()
                v, w = mission.step(f, dt)
                v, w, guard = safety(v, w, f.lidar, f.lidar_age)
                if guard != last_guard:
                    last_guard = guard
                    if guard:
                        mission.note(f.time, f"SAFETY: {guard}")
                t = f.time - t0
                x = mission.pose
                path.append((x[0], x[1]))
                rows.append((t, x[0], x[1], x[2], mission.state, v, w,
                            -1.0 if f.marker is None else f.marker.range))
                if t - last_print >= 1.0:
                    last_print = t
                    dk = (f"{f.marker.range:.2f} m {math.degrees(f.marker.bearing):+.0f} deg"
                        if f.marker is not None else "-")
                    print(f"{t:6.1f} {x[0]:6.2f} {x[1]:6.2f} {math.degrees(x[2]):6.0f} "
                        f"{mission.state:<9} {v:5.2f} {w:6.2f} {dk}", flush=True)
                if mission.done:
                    break
                if t > RUN_TIMEOUT:
                    mission.reason = "timeout"
                    mission.note(f.time, "TIMEOUT")
                    break
                env.step(v, w)
        except KeyboardInterrupt:
            print("\nInterrupted by user: stopping the robot.")
            mission.reason = "interrupted"
        finally:
            env.emergency_stop()

    print("\n=== RESULT ===")
    print(f"{'SUCCESS' if mission.success else 'FAILURE'} ({mission.reason}) after "
        f"{rows[-1][0] if rows else 0:.1f} s, camera frames {sensing.n_camera if rows else 0}, "
        f"dock detections {sensing.n_detect if rows else 0}")
    for e in mission.events:
        print("  " + e)
    manual = {}
    if not args.no_prompt and rows:
        print("\nMeasure the final dock position with a ruler.")
        for key, text in (("dock_range_m", "distance robot centre -> dock marker [m]"),
                        ("dock_lateral_m", "sideways offset from the marker axis [m]"),
                        ("dock_heading_deg", "heading error [deg]")):
            try:
                manual[key] = float(input(f"  {text}: "))
            except ValueError:
                manual[key] = None
    if rows:
        np.savetxt(out / "log.csv", np.array([(r[0], r[1], r[2], r[3], r[5], r[6], r[7]) for r in rows]),
                delimiter=",", header="t,x,y,theta,v,w,dock_range", comments="")
        (out / "states.txt").write_text("\n".join(f"{r[0]:.2f} {r[4]}" for r in rows))
        save_plot(path, mission, out)
    (out / "summary.json").write_text(json.dumps({
        "success": mission.success, "reason": mission.reason, "events": mission.events,
        "manual_dock_measurement": manual,
        "settings": {"dock_target_m": DOCK_TARGET, "aim_dist_m": AIM_DIST, "lat_tol_m": LAT_TOL,
                    "axis_tol_rad": DOCK_AXIS_TOL, "speed_scale": args.speed_scale,
                    "odometry": f"r {cfg.robot.wheel_radius:.4f} b {cfg.robot.wheel_base:.4f}"
                                + (" (calibrated)" if odo else " (nominal)")},
    }, indent=2), encoding="utf-8")
    print("Saved to", out)
    return 0 if mission.success else 1


def main_cli() -> int:
    p = argparse.ArgumentParser(description="BACKUP: simplified Track A on the real TurtleBot3")
    p.add_argument("ip", nargs="?", default=ROBOT_IP)
    p.add_argument("--check", action="store_true", help="sensors only, the robot does not move")
    p.add_argument("--calib-odom", action="store_true", help="with --check: odometry calibration (robot MOVES)")
    p.add_argument("--speed-scale", type=float, default=SPEED_SCALE)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--square", type=float, default=0.025)
    p.add_argument("--no-prompt", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not args.check:
        return run_hardware(args)

    from calibration import calibrate_hardware
    from rosbridge_api import RosbridgeRobot
    cfg = hardware_config()
    try:
        cal, cfg = calibrate_hardware(cfg, force=args.recalibrate, square=args.square)
        cfg = with_fov(cfg, cal)
        print(f"Calibration OK | RMS {cal.intrinsics.reprojection_error:.3f} px")
    except (FileNotFoundError, ValueError) as e:
        print(f"Calibration not ready ({e}) -> marker ranges are approximate")
    ok = False
    with RosbridgeRobot(args.ip) as robot:
        env = RosbridgeEnvironment(robot, cfg)
        try:
            ok = preflight(env, cfg)
            odo = load_odometry_hw()
            print(f"Odometry params: r {cfg.robot.wheel_radius:.4f} m, b {cfg.robot.wheel_base:.4f} m",
                "(calibrated)" if odo else "(nominal)")
            if args.calib_odom and ok:
                calibrate_odometry_hw(env, cfg)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
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
    if ok:
        print(f"Check OK. Now run:  python {Path(__file__).name} {args.ip}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main_cli())