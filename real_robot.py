# Real TurtleBot3 (rosbridge) for Track A.
# config.py / main.py / docking.py / recovery.py stay untouched (they are the simulation).
# Everything that differs on the real robot is defined HERE and patched in only while run_hardware runs.
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2

from config import CONFIG, RESULTS_DIR, Config
from environment import HardwareEnvironment
from recovery import Explorer, StopGoScan, MapView
from sensing import Sensing, sector_min_range, lidar_points
from robot_model import wrap_angle, motion_model
from safety import SafetyGuard
from controller import limit_command
from navigation import AimController
import docking

# ============================================================ robot / sensors
ROBOT_IP = "192.168.1.213"               # TurtleBot ID: 192.168.1.2XX
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480   # must equal the size of the images the robot sends
MARKER_SIZE = 0.15                       # [m] width of the BLACK part of the printed marker
N_EXTRA_TAGS = 3                         # extra tags in the real arena (0 = only the dock marker)
CONTROL_DT = 0.05                        # [s] 20 Hz: realistic with WiFi and a 5 Hz LiDAR
SPEED_SCALE = 0.5                        # fraction of max speed (keep 0.5 for the first runs)
CAMERA_DELAY = 0.2                       # [s] camera latency compensated by the EKF
ODOM_SCALE = 1.5                         # x odometry covariance (real wheel encoders)
MAX_ANGULAR_HW = 0.3                     # [rad/s]
SCAN_METHOD = None                       # exact RosbridgeRobot method name if the guesses below fail
SCAN_NAMES = ("query_laser_scan", "query_scan", "query_laser", "query_lidar",
              "get_laser_scan", "get_scan", "laser_scan", "scan")
MAX_BEAM_GAP = math.radians(3.0)         # a target beam farther than this from any real beam = no hit
SIMPLE_SAFETY = True
IGNORE_NEAR = 0.30                       # [m] LiDAR returns closer than this are ignored (robot parts)
ODOM_FILE = RESULTS_DIR / "odometry_hw.json"

# ============================================================ SURVEY / EXPLORE
SURVEY_ANGLE_HW = 2.0 * math.pi          # full turn at the start pose
HW_SCAN_STEP = math.radians(30.0)        # 12 stops per turn (camera FOV ~28 deg)
DOCK_LOCK_TIME = 1.5     # [s] dock seen this recently -> the scan stops / turns back
MAX_DOCK_LOCK = 6.0      # [s] longest stop (turn + learn) for one sighting during a scan
HW_LEARN_SIGMA = 0.15    # [m] learn markers only below this position std (sim: 0.08)
TAG_LOCK_TIME = 3.0      # [s] max stop time per extra tag
TAG_LOST_TIME = 1.0      # [s] tag not seen this long while stopped -> give up on it
CENTER_TOL = 0.12        # [rad] dock counts as centred below this bearing
MAX_TURN = 0.25          # [rad/s]
MIN_TURN = 0.08          # [rad/s] smaller commands may not move the robot
LEARN_RANGE = 2.0        # [m] drive closer if the dock is farther than this
APPROACH_V = 0.10        # [m/s]
DOCK_MEMORY = 10.0       # [s] remember the dock direction this long (odometry-propagated)

# ============================================================ docking
DOCK_TARGET_HW = 0.85    # [m] robot centre -> marker at the end (sim 0.55: marker would not fit the image)
DOCK_TOL_HW = 0.08       # [m]
DOCK_BEARING_TOL_HW = 0.12   # [rad]
APPROACH_DIST_HW = 1.4   # [m] AIM approach point in front of the dock (sim: 1.0)
HW_K_BEARING = 1.2       # [1/s] centre the marker (sim: 5.5)
HW_K_AXIS = 0.6          # [1/s] gentle pull onto the marker axis (sim: 4.0)
HW_VIEW_CLIP = 0.2       # [rad] limit of the axis term
HW_EDGE = 0.15           # [rad] marker near the image edge: only turn TOWARDS it
HW_BETA_MAX = 0.2        # [rad] no forward motion if the marker is this far off-centre
HW_DOCK_TURN = 0.15      # [rad/s] max turn while docking (sim: 0.8)
HW_DOCK_SPEED = 0.10     # [m/s]
USE_PREDOCK_MANEUVER = True
HW_LAT_TOL = 0.15        # [m] larger lateral offset -> go to the pre-dock point first
HW_PREDOCK = 1.2         # [m] pre-dock point on the marker axis
HW_MANEUVER_V = 0.12     # [m/s]
HW_MANEUVER_V_REV = 0.08 # [m/s] reversing (pre-dock point behind the robot)
HW_MANEUVER_W = 0.3      # [rad/s]
HW_MAX_MANEUVERS = 2     # per docking attempt / per AIM
HW_MANEUVER_TIMEOUT = 20.0
HW_OBS_WINDOW = 1.0      # [s] lateral offset = median of the dock detections in this window
HW_OBS_MIN = 3           # ... needs at least this many detections

# ============================================================ AIM (close to the dock)
HW_AIM_CLOSE = 1.8       # [m] dock seen closer than this: AIM stops, looks and decides itself
HW_AIM_MEMORY = 1.0      # [s] dock counts as "in view" this long after the last detection

# ============================================================ LiDAR in docking
USE_LIDAR_VIEW = True    # marker plane angle from a LiDAR line fit (camera view angle is ~5 deg noisy)
LV_RADIUS = 0.25         # [m] LiDAR points this close to the marker centre are fitted
LV_MIN_POINTS = 6
LV_MAX_RMS = 0.02        # [m] residual of the line fit
LV_MIN_SPAN = 0.15       # [m] fitted segment must be at least this long
LV_MAX_DIFF = 0.35       # [rad] LiDAR and camera disagree more -> keep the camera value
HW_FRONT_STOP = 0.45     # [m] LiDAR brake in front (must be > IGNORE_NEAR)
HW_REAR_STOP = 0.40      # [m] LiDAR brake when reversing


# ============================================================ config
def load_odometry_hw():
    """(wheel_radius, wheel_base) measured on the real robot, or None."""
    try:
        d = json.loads(ODOM_FILE.read_text())
        return float(d["wheel_radius"]), float(d["wheel_base"])
    except (OSError, ValueError, KeyError):
        return None


def hardware_config(speed_scale: float = 1.0) -> Config:
    """CONFIG with the real-robot values. Only this function decides what differs from the simulation."""
    c = CONFIG
    extra = tuple(c.tags.fixed_tags[i] if i < len(c.tags.fixed_tags) else (0.0, 0.0, 0.0)
                  for i in range(N_EXTRA_TAGS))                   # only the COUNT matters for control
    odo = load_odometry_hw()
    r, b = odo if odo is not None else (c.robot.wheel_radius, c.robot.wheel_base)
    return replace(
        c,
        camera=replace(c.camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, marker_size=MARKER_SIZE),
        robot=replace(c.robot, wheel_radius=r, wheel_base=b,
                      max_linear_vel=c.robot.max_linear_vel * speed_scale,
                      max_angular_vel=min(c.robot.max_angular_vel * speed_scale, MAX_ANGULAR_HW)),
        sim=replace(c.sim, control_dt=CONTROL_DT),
        docking=replace(c.docking, target_distance=DOCK_TARGET_HW, distance_tolerance=DOCK_TOL_HW,
                        bearing_tolerance=DOCK_BEARING_TOL_HW),
        tags=replace(c.tags, fixed_tags=extra, survey_angle=SURVEY_ANGLE_HW,
                     approach_dist=APPROACH_DIST_HW),
        env=replace(c.env, obstacles=()),                         # the real arena is unknown to the robot
        realism=replace(c.realism, enabled=False),                # simulated imperfections do not apply
    )


def with_fov(cfg: Config, calibration) -> Config:
    """Vertical field of view from the calibrated focal length (instead of the simulator's 60 deg)."""
    fy = float(calibration.K[1, 1])
    return replace(cfg, camera=replace(cfg.camera, fovy_deg=math.degrees(2.0 * math.atan(cfg.camera.height / 2.0 / fy))))


def _turn(b: float) -> float:
    return math.copysign(float(np.clip(abs(b), MIN_TURN, MAX_TURN)), b)


# ============================================================ sensing (+ LiDAR marker plane)
def lidar_view_angle(m, scan) -> float | None:
    """View angle of the marker from a line fitted to the LiDAR points around it (None = no good fit)."""
    pts = lidar_points(scan)
    if len(pts) < LV_MIN_POINTS:
        return None
    c = np.array([m.range * math.cos(m.bearing), m.range * math.sin(m.bearing)])
    near = pts[np.linalg.norm(pts - c, axis=1) < LV_RADIUS]
    if len(near) < LV_MIN_POINTS:
        return None
    mu = near.mean(axis=0)
    _, s, vt = np.linalg.svd(near - mu, full_matrices=False)
    d = vt[0]
    proj = (near - mu) @ d
    if s[1] / math.sqrt(len(near)) > LV_MAX_RMS or np.ptp(proj) < LV_MIN_SPAN:
        return None
    n = np.array([-d[1], d[0]])
    if n @ mu > 0.0:                                  # normal must point back at the robot
        n = -n
    view = float(wrap_angle(math.atan2(n[1], n[0]) - math.pi))   # same convention as calibration._marker_pose
    if abs(wrap_angle(view - m.view_angle)) > LV_MAX_DIFF:
        return None
    return view


class HwSensing(Sensing):
    dock_t, dock_b, dock_r = -99.0, 0.0, 0.0     # last dock sighting (bearing / range propagated)
    n_lidar_view = 0                              # markers whose view angle came from the LiDAR

    def read(self):
        f = super().read()
        if USE_LIDAR_VIEW and f.markers:
            new = []
            for m in f.markers:
                v = lidar_view_angle(m, f.lidar)
                if v is not None:
                    HwSensing.n_lidar_view += 1
                    m = replace(m, view_angle=v)
                new.append(m)
            dock = next((m for m in new if m.marker_id == self.cfg.camera.marker_id), None)
            f = replace(f, markers=tuple(new), marker=dock)
        C = HwSensing
        if f.marker is not None:
            C.dock_t, C.dock_b, C.dock_r = f.time, f.marker.bearing, f.marker.range
        else:                                    # between camera frames: follow the robot's motion
            C.dock_b = wrap_angle(C.dock_b - f.odometry.dtheta)
            C.dock_r -= f.odometry.ds * math.cos(C.dock_b)
        return f


# ============================================================ SURVEY / EXPLORE
class HwStopGoScan(StopGoScan):
    def __init__(self, *a, **k):
        k["step"] = HW_SCAN_STEP                 # main.py passes step=0.8: override it
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
    ekf = None           # set in run_hardware: learned tags go to the EKF right away

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
                return APPROACH_V, 1.0 * b
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


# ============================================================ docking
def hw_docking_command(rng, bearing, view, cfg):
    """Gentle visual servoing for the narrow real camera (keeps the marker in view)."""
    d = cfg.docking
    if rng <= d.target_distance + d.distance_tolerance:          # arrived: only centre
        return limit_command(0.0, float(np.clip(0.8 * bearing, -HW_DOCK_TURN, HW_DOCK_TURN)), cfg)
    speed = float(np.clip(docking.K_RANGE * (rng - d.target_distance),
                          docking.DOCK_MIN_SPEED, HW_DOCK_SPEED))
    speed *= float(np.clip(1.0 - abs(bearing) / HW_BETA_MAX, 0.0, 1.0))
    if abs(bearing) > HW_EDGE:                                   # marker near the image edge
        w = HW_K_BEARING * bearing
    else:
        w = HW_K_BEARING * bearing - HW_K_AXIS * float(np.clip(view, -HW_VIEW_CLIP, HW_VIEW_CLIP))
    return limit_command(speed, float(np.clip(w, -HW_DOCK_TURN, HW_DOCK_TURN)), cfg)


DOCK_PATCH = dict(docking_command=hw_docking_command,
                  MEASUREMENT_TIMEOUT=1.0,     # tolerate short WiFi frame drops before SEARCH
                  SEARCH_OMEGA=0.2,            # search slowly (sim: 0.7)
                  AXIS_TOLERANCE=0.25,         # ~14 deg (camera view angle noise ~5 deg)
                  RETREAT_DELAY=1e9)           # no back-off loop


def _median_obs(obs, t):
    """Median (range, bearing, view) of the dock detections of the last HW_OBS_WINDOW s (None = too few)."""
    obs[:] = [o for o in obs if t - o[0] <= HW_OBS_WINDOW]
    if len(obs) < HW_OBS_MIN:
        return None
    return tuple(float(np.median([o[k] for o in obs])) for k in (1, 2, 3))


def _lateral(r, b, v) -> float:
    return r * math.sin(wrap_angle(b - v))                # >0: marker axis is on the robot's left


class AxisManeuver:
    """Drive (forwards OR backwards) to the point HW_PREDOCK in front of the marker, then face it.
    Uses only odometry after the start, so a wrong map does not matter."""

    def __init__(self, r, b, v, t0):
        self.mk = (r * math.cos(b), r * math.sin(b))      # marker in the robot frame at the start
        self.goal = (self.mk[0] - HW_PREDOCK * math.cos(v), self.mk[1] - HW_PREDOCK * math.sin(v))
        self.pose = np.zeros(3)
        self.state, self.t0, self.rev = "TURN1", float(t0), False

    @staticmethod
    def _w(err):
        return math.copysign(float(np.clip(1.5 * abs(err), MIN_TURN, HW_MANEUVER_W)), err)

    def step(self, odo):
        """(v, w) while running, None when finished."""
        self.pose = motion_model(self.pose, odo.ds, odo.dtheta)
        x, y, th = self.pose
        if float(odo.time) - self.t0 > HW_MANEUVER_TIMEOUT:
            return None
        gx, gy = self.goal
        dist = math.hypot(gx - x, gy - y)
        if self.state == "TURN1":
            if dist < 0.06:
                self.state = "TURN2"
            else:
                head = math.atan2(gy - y, gx - x)
                self.rev = math.cos(head - th) < -0.2         # goal behind: reverse instead of a U-turn
                err = wrap_angle(head + (math.pi if self.rev else 0.0) - th)
                if abs(err) > 0.06:
                    return 0.0, self._w(err)
                self.state = "DRIVE"
        if self.state == "DRIVE":
            if dist > 0.06:
                head = math.atan2(gy - y, gx - x)
                err = wrap_angle(head + (math.pi if self.rev else 0.0) - th)
                vmax = HW_MANEUVER_V_REV if self.rev else HW_MANEUVER_V
                v = min(vmax, 0.5 * dist + 0.03) * max(0.0, math.cos(err))
                return (-v if self.rev else v), float(np.clip(1.5 * err, -HW_MANEUVER_W, HW_MANEUVER_W))
            self.state = "TURN2"
        mx, my = self.mk                                      # TURN2: face the marker again
        err = wrap_angle(math.atan2(my - y, mx - x) - th)
        if abs(err) > 0.05:
            return 0.0, self._w(err)
        return None


class HwDockingController(docking.DockingController):
    """Docking on the marker. Off the axis (median of several detections), at ANY range:
    go to the pre-dock point on the axis first (forwards or reversing), then normal docking."""

    def reset(self):
        super().reset()
        self._mv = None
        self._n_mv = 0
        self._obs = []

    def update(self, marker, odometry, x_est=None):
        t = float(odometry.time)
        if self._mv is not None:
            cmd = self._mv.step(odometry)
            if cmd is not None:
                self._last_t, self._unseen_time, self._search_time = t, 0.0, 0.0
                self._meas = None                              # old measurement is stale after the maneuver
                self.phase = docking.DockPhase.APPROACH
                return self._command(*cmd)
            self._mv, self._obs = None, []
        if docking._is_valid(marker):
            self._obs.append((t, marker.range, marker.bearing, marker.view_angle))
        med = _median_obs(self._obs, t)
        if (USE_PREDOCK_MANEUVER and med is not None and not self.finished
                and self._n_mv < HW_MAX_MANEUVERS and abs(_lateral(*med)) > HW_LAT_TOL):
            self._mv = AxisManeuver(*med, t)
            self._n_mv += 1
            self._obs = []
            print(f"  >> [t={t:5.1f}s] DOCK: {_lateral(*med):+.2f} m off the axis at {med[0]:.2f} m "
                  f"-> pre-dock point on the axis (maneuver {self._n_mv})", flush=True)
            return self._command(0.0, 0.0)
        return super().update(marker, odometry, x_est)


class HwAimController(AimController):
    """AIM + close-range rule: dock in view closer than HW_AIM_CLOSE -> stop, look, then
    on the axis -> DOCK right away (no 0.9 m lower limit), off the axis -> AxisManeuver, then look again."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mv, self.n_mv, self.close_obs = None, 0, []

    def restart_look(self):                    # called by main.py when docking gives up
        super().restart_look()
        self.mv, self.n_mv, self.close_obs = None, 0, []

    def _look_again(self):
        self.state, self.t0, self.goal_pt, self.goal_id = "LOOK", None, None, None
        self.close_obs = []

    def update(self, frame, x_est, nav, scan, dock_lm, dock_yaw, tag_map, dock_id, dt, note):
        t = frame.time
        if self.mv is not None:
            cmd = self.mv.step(frame.odometry)
            if cmd is not None:
                self.state = "AXIS"
                return cmd[0], cmd[1], nav, None
            self.mv = None
            self._look_again()
            note(t, "AIM: pre-dock point reached -> looking at the dock again")
            return 0.0, 0.0, nav, None

        S = HwSensing
        if not (t - S.dock_t < HW_AIM_MEMORY and S.dock_r < HW_AIM_CLOSE):
            if self.state in ("CLOSE", "AXIS"):          # dock lost again: back to the normal AIM
                self._look_again()
            return super().update(frame, x_est, nav, scan, dock_lm, dock_yaw, tag_map, dock_id, dt, note)

        # dock close and in view: stop, centre it, collect a few detections while standing still
        self.state = "CLOSE"
        if abs(S.dock_b) > CENTER_TOL:
            self.close_obs = []
            return 0.0, _turn(S.dock_b), nav, None
        still = abs(frame.odometry.dtheta) < 1e-3 and abs(frame.odometry.ds) < 1e-3
        if still and frame.marker is not None:
            m = frame.marker
            self.close_obs.append((t, m.range, m.bearing, m.view_angle))
        med = _median_obs(self.close_obs, t)
        if med is None:
            return 0.0, 0.0, nav, None
        lat = _lateral(*med)
        if abs(lat) <= HW_LAT_TOL or self.n_mv >= HW_MAX_MANEUVERS:
            note(t, f"AIM: dock {med[0]:.2f} m, {lat:+.2f} m off the axis -> docking")
            self._look_again()
            return 0.0, 0.0, nav, "DOCK"
        self.mv = AxisManeuver(*med, t)
        self.n_mv += 1
        self.close_obs = []
        note(t, f"AIM: dock {med[0]:.2f} m but {lat:+.2f} m off the axis -> "
                f"driving to the pre-dock point {HW_PREDOCK:.1f} m in front of it")
        return 0.0, 0.0, nav, None


# ============================================================ safety
class HwSafetyGuard(SafetyGuard):
    """SafetyGuard + LiDAR brakes that work with IGNORE_NEAR (the default 0.27 m check never fires)."""

    def filter(self, v, w, scan, dt, lidar_age=0.0, sigma_xy=0.0):
        v, w, reason = super().filter(v, w, scan, dt, lidar_age, sigma_xy)
        if v > 0.0 and sector_min_range(scan, 0.0, math.radians(25)) < HW_FRONT_STOP:
            v, self._v = 0.0, 0.0
            reason = (reason + "; " if reason else "") + "LiDAR front stop"
        if v < 0.0 and sector_min_range(scan, math.pi, math.radians(35)) < HW_REAR_STOP:
            v, self._v = 0.0, 0.0
            reason = (reason + "; " if reason else "") + "LiDAR rear stop"
        return v, w, reason


@contextmanager
def patched(module, values: dict):
    """Temporarily replace attributes of a module (restored even after an error / Ctrl+C)."""
    old = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


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
    raise AttributeError("No laser-scan method found on RosbridgeRobot. Public methods: "
                         f"{public}. Put the right name in SCAN_METHOD at the top of hardware.py.")


def _parse_scan(scan):
    """Accept (angles, ranges), a dict or an object with angles / ranges."""
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


def resample_scan(a, r, target, range_min: float, range_max: float) -> np.ndarray:
    """Nearest-beam resampling onto the beam angles the pipeline expects."""
    r = np.where(np.isfinite(r) & (r >= range_min) & (r < range_max), r, range_max)
    a = np.mod(a, 2.0 * math.pi)
    diff = np.abs((np.asarray(target)[:, None] - a[None, :] + math.pi) % (2.0 * math.pi) - math.pi)
    out = r[diff.argmin(axis=1)]
    out[diff.min(axis=1) > MAX_BEAM_GAP] = range_max
    return out


# ============================================================ environment on the real robot
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

    def _integrate(self) -> None:
        """Encoders integrated from the commanded wheel speeds (fallback without /joint_states)."""
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
        for _ in range(3):                            # one lost WiFi packet must not keep the robot moving
            self.robot.send_command(0.0, 0.0)

    def close(self) -> None:
        try:
            self.emergency_stop()
        except Exception:
            pass


class HardwareSim:
    """Stand-in for Simulation inside main.run_pipeline (no ground truth on the real robot)."""

    def __init__(self, env: RosbridgeEnvironment, cfg: Config):
        self.env, self.cfg, self.world_cfg = env, cfg, cfg
        self.obstacles: list = []
        self.model = self.data = None
        self.collided = False

    @property
    def time(self) -> float:
        return self.env.time

    def reset(self, pose=None) -> None:
        pass

    def ground_truth(self) -> np.ndarray:
        return np.full(3, np.nan)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


# ============================================================ pre-flight check (robot does not move)
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


# ============================================================ odometry calibration (robot MOVES)
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

    input("\n[2/2] SPIN: mark the robot heading with tape. Enter = spin ~2 turns left... ")
    e0 = env.read_encoders(); _drive(env, 0.0, 0.3, 4.0 * math.pi / 0.3); e1 = env.read_encoders()
    dphi = e1 - e0
    print(f"  odometry says {math.degrees(r * (dphi[1] - dphi[0]) / b0):.0f} deg (b = {b0:.4f})")
    extra = input("  2 full turns (720 deg) + how many EXTRA deg? (+ more, - less, just Enter = 0): ").strip()
    try:
        extra_deg = float(extra) if extra else 0.0
    except ValueError:
        print(f"  '{extra}' is not a number -> using 0")
        extra_deg = 0.0
    theta = math.radians(720.0 + extra_deg)
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


# ============================================================ full run
def run_hardware(args) -> int:
    import main
    from estimator import EKFEstimator
    from rosbridge_api import RosbridgeRobot
    from visualization import RouteVisualizer, print_report
    from calibration import calibrate_hardware

    s = float(args.speed_scale)
    cfg = hardware_config(s)
    try:
        calibration, cfg = calibrate_hardware(cfg, force=args.recalibrate, square=args.square)
    except (FileNotFoundError, ValueError) as e:
        print(f"Calibration failed: {e}")
        return 2
    cfg = with_fov(cfg, calibration)
    odo = load_odometry_hw()
    print(f"Odometry params: r {cfg.robot.wheel_radius:.4f} m, b {cfg.robot.wheel_base:.4f} m",
          "(calibrated)" if odo else "(nominal config, not calibrated: run --check --calib-odom)")

    out = RESULTS_DIR / "hardware" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    with RosbridgeRobot(args.ip) as robot:
        env = RosbridgeEnvironment(robot, cfg)
        try:
            if not preflight(env, cfg) and not args.force:
                print("Pre-flight failed. Fix the problems above or use --force.")
                return 2
            input("Pre-flight OK. Clear the area, keep a hand on the robot. Press Enter to start (Ctrl+C = stop)... ")

            ekf = EKFEstimator(replace(cfg, env=replace(cfg.env, obstacles=())), gate=9.21,
                               odom_cov_scale=args.odom_scale, camera_delay=args.camera_delay,
                               use_lidar=False, R=np.diag([0.15 ** 2, 0.05 ** 2]), R_view=0.15 ** 2)
            viz = None if args.no_viz else RouteVisualizer(cfg, output_dir=out, live_view=not args.no_live_camera)

            HwExplorer.ekf = ekf
            HwSensing.dock_t, HwSensing.n_lidar_view = -99.0, 0
            main_patch = dict(
                Simulation=lambda c=None, seed=None, scene_dir=None: HardwareSim(env, cfg),
                MuJoCoEnvironment=lambda sim: sim.env,
                StopGoScan=HwStopGoScan, Explorer=HwExplorer, Sensing=HwSensing,
                SafetyGuard=HwSafetyGuard, DockingController=HwDockingController,
                AimController=HwAimController)
            with patched(main, main_patch), patched(docking, DOCK_PATCH):
                result = main.run_pipeline(cfg, seed=None, estimator=ekf, viewer=False, verbose=True,
                                           dock_use_estimate=False,      # dock on the marker, not the map
                                           use_recovery=True, calib_odometry=False, calib_camera=False,
                                           calibration=calibration, camera_delay=args.camera_delay,
                                           visualizer=viz)
            env.emergency_stop()
        except KeyboardInterrupt:
            print("\nInterrupted by user: stopping the robot.")
            return 1
        finally:
            env.emergency_stop()
        joint_ok = env._joint_ok

    print()
    print_report(result, cfg)
    print(f"LiDAR-refined marker view angles: {HwSensing.n_lidar_view}")
    manual = {}
    if not args.no_prompt:
        print("\nMeasure the final dock position with a ruler (the robot has no ground truth).")
        for key, text in (("dock_range_m", "distance robot centre -> dock marker [m]"),
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
        "settings": {
            "speed_scale": s, "odom_scale": args.odom_scale, "camera_delay": args.camera_delay,
            "calibration": "calibrate_hardware (results/camera_intrinsics.json + camera_extrinsics.json)",
            "odometry": ("wheel encoders (/joint_states)" if joint_ok else "dead reckoning from commands")
                        + f", r {cfg.robot.wheel_radius:.4f} m, b {cfg.robot.wheel_base:.4f} m"
                        + (" (calibrated)" if odo else " (nominal)"),
            "dock_target_m": DOCK_TARGET_HW, "axis_tolerance_rad": DOCK_PATCH["AXIS_TOLERANCE"],
            "lidar_view_refinements": HwSensing.n_lidar_view},
    }, indent=2, default=str), encoding="utf-8")
    print("Saved to", out)
    return 0 if result.success else 1


def main_cli() -> int:
    p = argparse.ArgumentParser(description="Run Track A on a real TurtleBot3 (rosbridge)")
    p.add_argument("ip", nargs="?", default=ROBOT_IP, help="robot IP (default: ROBOT_IP above)")
    p.add_argument("--check", action="store_true", help="read-only test only, the robot does not move")
    p.add_argument("--calib-odom", action="store_true",
                   help="with --check: drive 1 m + spin 2 turns to calibrate the wheel odometry (robot MOVES)")
    p.add_argument("--speed-scale", type=float, default=SPEED_SCALE)
    p.add_argument("--recalibrate", action="store_true", help="force camera recalibration from images")
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
    cfg_hw = hardware_config()
    try:
        cal, cfg_hw = calibrate_hardware(cfg_hw, force=args.recalibrate, square=args.square)
        cfg_hw = with_fov(cfg_hw, cal)
        print(f"Calibration OK | RMS {cal.intrinsics.reprojection_error:.3f} px | "
              f"extrinsics fwd {cal.forward_offset:.3f} m, lat {cal.lateral_offset:+.3f} m, "
              f"yaw {math.degrees(cal.yaw):+.2f} deg")
    except (FileNotFoundError, ValueError) as e:
        print(f"Calibration not ready ({e}) -> sensor check only, marker ranges are approximate")

    ok = False
    with RosbridgeRobot(args.ip) as robot:
        env = RosbridgeEnvironment(robot, cfg_hw)
        try:
            ok = preflight(env, cfg_hw)
            odo = load_odometry_hw()
            print(f"Odometry params: r {cfg_hw.robot.wheel_radius:.4f} m, b {cfg_hw.robot.wheel_base:.4f} m",
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
        print(f"Check OK. Now run:  python {Path(__file__).name} {args.ip}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main_cli())
