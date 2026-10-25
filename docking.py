# Including libraries
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from config import CONFIG, Config
from calibration import camera_forward_offset
from controller import limit_command
from robot_model import TagMap, measurement_model, view_angle_model, wrap_angle
from sensing import MarkerMeasurement, OdometryIncrement, Sensing

DOCK_MAX_SPEED = 0.12        # [m/s]
DOCK_MIN_SPEED = 0.03        # [m/s] never creep slower than this while approaching
K_RANGE = 1.0                # [1/s] speed per metre of range error
K_BEARING = 1.5              # [1/s] centre the marker in the image
K_AXIS = 4.0                 # [1/s] converge onto the marker axis
K_CENTER = 2.0               # [1/s] in-place centring during ALIGN
BETA_MAX = 0.6               # [rad] forward speed is zero when |bearing| >= this
AXIS_TOLERANCE = 0.15        # [rad] max |delta| to count as docked
SEARCH_OMEGA = 0.7           # [rad/s] rotation speed while searching
MEASUREMENT_TIMEOUT = 0.5    # [s] how long a propagated measurement stays usable
ALIGN_TIMEOUT = 20.0         # [s] give up if close but never aligned
RETREAT_DELAY = 2.0          # [s] arrived but off the marker axis this long -> back off and re-approach
RETREAT_EXTRA = 0.40         # [m] back off until this much beyond the arrival range (short: no LiDAR behind the robot)
RETREAT_SPEED = -0.08        # [m/s] reverse speed while backing off
MAX_DOCK_TIME = 60.0         # [s] total time limit for docking
MARKER_FACE_OFFSET = 0.005   # [m] half thickness of the marker panel (see simulation.py)
VISIBILITY_MARGIN = 1.2      # marker must fit in the image with this safety factor
AXIS_SLOW = 0.6              # [rad] forward speed is reduced as |axis| grows towards this
BLIND_TIMEOUT = 8.0          # [s] estimate-driven docking: longest time without a real detection
DOCK_CONFIRM_FRAMES = 2      # DOCKED needs this many consecutive REAL detections inside the tolerances
REAL_HOLD_TIME = 1.0         # [s] a real measurement (propagated with odometry) beats the map estimate this long
BACKOFF_SPEED = -0.05        # [m/s] too close to see the marker: reverse slowly instead of spinning
DOCK_MAX_TURN = 0.8          # [rad/s] max turning rate while docking (avoids overshoot with the camera delay)

class DockPhase(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    ALIGN = "align"
    DOCKED = "docked"
    FAILED = "failed"

@dataclass(frozen=True)
class DockCommand:
    v: float
    w: float
    phase: DockPhase
    marker_range: float      # [m] (nan if no marker)
    marker_bearing: float    # [rad] (nan if no marker)
    axis_angle: float        # [rad] delta = bearing - view angle (nan if no marker)
    finished: bool           # True when DOCKED or FAILED

    @property
    def docked(self) -> bool:
        return self.phase is DockPhase.DOCKED
    
@dataclass
class _Measurement:
    range: float
    bearing: float
    view_angle: float
    time: float              # time of the last REAL camera measurement

# Smallest robot-to-marker range
def min_visible_range(cfg: Config = CONFIG, margin: float = VISIBILITY_MARGIN) -> float:
    half_fov = math.radians(cfg.camera.fovy_deg) / 2.0
    camera_distance = margin * (cfg.camera.marker_size / 2.0) / math.tan(half_fov)
    return camera_distance + camera_forward_offset(cfg) + MARKER_FACE_OFFSET

# Update a marker measurement after the robot moved
def propagate_marker(rng: float, bearing: float, view_angle: float,
                    ds: float, dtheta: float) -> tuple[float, float, float]:
    px, py = rng * math.cos(bearing), rng * math.sin(bearing)
    tx, ty = ds * math.cos(dtheta / 2.0), ds * math.sin(dtheta / 2.0)
    c, s = math.cos(dtheta), math.sin(dtheta)
    qx = c * (px - tx) + s * (py - ty)
    qy = -s * (px - tx) + c * (py - ty)
    return math.hypot(qx, qy), math.atan2(qy, qx), float(wrap_angle(view_angle - dtheta))

# Marker range, bearing and view angle predicted from an ESTIMATED robot pose (None if unusable).
# landmark / dock_yaw come from the learned tag map; None = dock not learned yet -> no prediction.
def marker_from_pose(x_est, cfg: Config = CONFIG, landmark=None, dock_yaw=None) -> Optional[tuple[float, float, float]]:
    x = np.asarray(x_est, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)) or landmark is None or dock_yaw is None:
        return None
    rng, bearing = measurement_model(x, landmark)
    if rng < 1e-6:
        return None
    return float(rng), float(bearing), float(view_angle_model(x, dock_yaw))

def is_docked(rng: float, bearing: float, view_angle: float, cfg: Config = CONFIG) -> bool:
    d = cfg.docking
    axis = wrap_angle(bearing - view_angle)
    return (rng <= d.target_distance + d.distance_tolerance
            and abs(bearing) <= d.bearing_tolerance
            and abs(axis) <= AXIS_TOLERANCE)

# Visual-servoing law
def docking_command(rng: float, bearing: float, view_angle: float,
                    cfg: Config = CONFIG) -> tuple[float, float]:
    d = cfg.docking
    if rng <= d.target_distance + d.distance_tolerance:
        return limit_command(0.0, K_CENTER * bearing, cfg)       # arrived: only centre

    speed = float(np.clip(K_RANGE * (rng - d.target_distance), DOCK_MIN_SPEED, DOCK_MAX_SPEED))
    speed *= float(np.clip(1.0 - abs(bearing) / BETA_MAX, 0.0, 1.0))   # turn first if off-centre
    axis = float(wrap_angle(bearing - view_angle))
    speed *= float(np.clip(1.0 - abs(axis) / AXIS_SLOW, 0.3, 1.0))     # slow down while off the marker axis
    w = (K_BEARING + K_AXIS) * bearing - K_AXIS * float(np.clip(view_angle, -0.5, 0.5))
    if abs(bearing) >= BETA_MAX:                                        # not moving: just turn towards the marker
        w = (K_BEARING + K_AXIS) * bearing                              # (the view term could cancel it: dead-lock)
    w = float(np.clip(w, -DOCK_MAX_TURN, DOCK_MAX_TURN))
    return limit_command(speed, w, cfg)

def _is_valid(marker: Optional[MarkerMeasurement]) -> bool:
    return (marker is not None and marker.range > 0.0
            and all(math.isfinite(v) for v in (marker.range, marker.bearing, marker.view_angle)))

class DockingController:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self._lm = None               # dock position / yaw from the learned tag map (None = config)
        self._yaw = None
        self.reset()

    def reset(self) -> None:
        self.phase = DockPhase.SEARCH
        self._meas: Optional[_Measurement] = None
        self._search_sign = 1.0       # direction to rotate while searching
        self._search_time = 0.0
        self._arrived_time = 0.0
        self._t_start: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_seen: Optional[float] = None   # time of the last REAL marker detection
        self._unseen_time = 0.0                   # time since the last real detection [s]
        self._real: Optional[_Measurement] = None # last REAL measurement, propagated with odometry
        self._ok_count = 0                        # consecutive real detections that satisfy is_docked
        self._retreat = False                     # True while backing off to re-approach on the marker axis

    def begin(self, x_est=None, landmark=None, dock_yaw=None) -> None:
        self.reset()
        self._lm, self._yaw = landmark, dock_yaw
        if x_est is None or landmark is None:
            return
        x = np.asarray(x_est, dtype=float)
        if x.shape == (3,) and np.all(np.isfinite(x)):
            self._search_sign = 1.0 if measurement_model(x, landmark)[1] >= 0.0 else -1.0

    # Dock pose learned later (while navigating): replace the fallback pose
    def set_landmark(self, landmark, dock_yaw) -> None:
        self._lm, self._yaw = landmark, dock_yaw

    @property
    def finished(self) -> bool:
        return self.phase in (DockPhase.DOCKED, DockPhase.FAILED)

    def _command(self, v: float, w: float, meas: Optional[_Measurement] = None) -> DockCommand:
        if meas is None:
            rng = bearing = axis = float("nan")
        else:
            rng, bearing = meas.range, meas.bearing
            axis = float(wrap_angle(meas.bearing - meas.view_angle))
        return DockCommand(float(v), float(w), self.phase, rng, bearing, axis, self.finished)
    
    # Arrived at the right range but off the marker axis: back off, then the approach law re-aligns
    def _retreating(self, rng: float, bearing: float, view: float) -> bool:
        d = self.cfg.docking
        if self._retreat:
            if rng >= d.target_distance + d.distance_tolerance + RETREAT_EXTRA:
                self._retreat = False
            return self._retreat
        off_axis = abs(float(wrap_angle(bearing - view))) > AXIS_TOLERANCE
        if (rng <= d.target_distance + d.distance_tolerance and off_axis
                and self._arrived_time > RETREAT_DELAY):
            self._retreat = True
        return self._retreat

    def _fail(self) -> DockCommand:
        self.phase = DockPhase.FAILED
        return self._command(0.0, 0.0)
    
    # Docking command for one control step.
    # With x_est=None the raw marker measurement drives the controller (propagated with odometry
    # for up to MEASUREMENT_TIMEOUT). With x_est (EKF pose) the controller works on the
    # range / bearing / view angle predicted from the estimate, so it keeps going through short
    # marker losses. Real detections still matter: they feed the EKF, reset the blind timer and
    # are required (within CONFIRM_WINDOW) before DOCKED is declared.
    def update(self, marker: Optional[MarkerMeasurement],
            odometry: OdometryIncrement, x_est=None) -> DockCommand:
        t = float(odometry.time)
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if self._t_start is None:
            self._t_start = t
        if self.finished:
            return self._command(0.0, 0.0)

        # Bookkeeping of real detections (used by both modes)
        if _is_valid(marker):
            self._last_seen = t
            self._unseen_time = 0.0
        else:
            self._unseen_time += dt

        if x_est is not None:
            est = marker_from_pose(x_est, self.cfg, self._lm, self._yaw)
            if est is not None:
                return self._update_estimated(est, marker, t, dt, odometry)

        # Keep the stored measurement up to date.
        if _is_valid(marker):
            self._meas = _Measurement(marker.range, marker.bearing, marker.view_angle, t)
            self._search_sign = 1.0 if marker.bearing >= 0.0 else -1.0
        elif self._meas is not None:
            m = self._meas
            m.range, m.bearing, m.view_angle = propagate_marker(
                m.range, m.bearing, m.view_angle, odometry.ds, odometry.dtheta)

        if t - self._t_start > MAX_DOCK_TIME:
            return self._fail()

        fresh = self._meas is not None and (t - self._meas.time) <= MEASUREMENT_TIMEOUT + 1e-9
        if not fresh:
            last = self._meas
            self._meas = None
            self.phase = DockPhase.SEARCH
            self._arrived_time = 0.0
            self._search_time += dt
            if self._search_time > self.cfg.docking.max_search_time:
                return self._fail()
            if last is not None and last.range < 1.5:
                # Đang gần dock: lùi chậm và nghiêng nhẹ về phía marker biến mất, không quay tròn
                return self._command(BACKOFF_SPEED, self._search_sign * 0.15)
            return self._command(0.0, self._search_sign * SEARCH_OMEGA)

        self._search_time = 0.0
        m = self._meas
        if is_docked(m.range, m.bearing, m.view_angle, self.cfg):
            self.phase = DockPhase.DOCKED
            return self._command(0.0, 0.0, m)

        d = self.cfg.docking
        if m.range <= d.target_distance + d.distance_tolerance:
            self.phase = DockPhase.ALIGN
            self._arrived_time += dt
            if self._arrived_time > ALIGN_TIMEOUT:
                return self._fail()
        else:
            self.phase = DockPhase.APPROACH
            self._arrived_time = 0.0

        if self._retreating(m.range, m.bearing, m.view_angle):
            return self._command(RETREAT_SPEED, 0.0, m)
        v, w = docking_command(m.range, m.bearing, m.view_angle, self.cfg)
        return self._command(v, w, m)

    # Docking step driven by the estimated pose. `est` = (range, bearing, view angle) from the EKF.
    # A REAL marker detection always wins over `est` (the dock pose in the map may be off).
    def _update_estimated(self, est, marker: Optional[MarkerMeasurement],
                        t: float, dt: float, odometry: OdometryIncrement) -> DockCommand:
        rng, bearing, view = est
        if t - self._t_start > MAX_DOCK_TIME:
            return self._fail()
        d = self.cfg.docking

        # Real detection beats the map-based estimate; between camera frames propagate it with odometry
        seen = _is_valid(marker)
        if seen:
            self._real = _Measurement(marker.range, marker.bearing, marker.view_angle, t)
            self._search_sign = 1.0 if marker.bearing >= 0.0 else -1.0
        elif self._real is not None:
            m = self._real
            m.range, m.bearing, m.view_angle = propagate_marker(
                m.range, m.bearing, m.view_angle, odometry.ds, odometry.dtheta)
        fresh = self._real is not None and (t - self._real.time) <= REAL_HOLD_TIME + 1e-9
        if fresh:
            rng, bearing, view = self._real.range, self._real.bearing, self._real.view_angle
        arrived = rng <= d.target_distance + d.distance_tolerance

        # DOCKED: several consecutive REAL detections inside the tolerances (filters single bad frames)
        if seen:
            self._ok_count = self._ok_count + 1 if is_docked(rng, bearing, view, self.cfg) else 0
            if self._ok_count >= DOCK_CONFIRM_FRAMES:
                self.phase = DockPhase.DOCKED
                return self._command(0.0, 0.0, _Measurement(rng, bearing, view, t))

        # Blind for too long. Close to the dock and seen before: too close, reverse slowly.
        # Otherwise: the estimate may be wrong, so search by spinning.
        if not fresh and self._unseen_time > BLIND_TIMEOUT:
            if arrived and self._last_seen is not None:
                self.phase = DockPhase.ALIGN
                self._arrived_time += dt
                if self._arrived_time > ALIGN_TIMEOUT:
                    return self._fail()
                return self._command(BACKOFF_SPEED, 0.0, _Measurement(rng, bearing, view, t))
            self.phase = DockPhase.SEARCH
            self._arrived_time = 0.0
            self._search_time += dt
            if self._search_time > self.cfg.docking.max_search_time:
                return self._fail()
            return self._command(0.0, self._search_sign * SEARCH_OMEGA)
        self._search_time = 0.0

        meas = _Measurement(rng, bearing, view, t)
        self._meas = meas
        if arrived:
            self.phase = DockPhase.ALIGN
            self._arrived_time += dt
            if self._arrived_time > ALIGN_TIMEOUT:
                return self._fail()
        else:
            self.phase = DockPhase.APPROACH
            self._arrived_time = 0.0

        if self._retreating(rng, bearing, view):
            return self._command(RETREAT_SPEED, 0.0, meas)
        v, w = docking_command(rng, bearing, view, self.cfg)
        return self._command(v, w, meas)

# Re-learn the dock pose from CLOSE observations only (accurate), then the EKF may use it.
# update() returns None, or (dock_lm, dock_yaw, first_time) when the dock pose was (re)learned.
class DockRelearner:
    def __init__(self, dock_id: int):
        self.dock_id = int(dock_id)
        self.near = TagMap(min_obs=3)      # dock observations made from close range only
        self.refined = False               # True once the dock pose was re-learned from a close view

    def update(self, x_est, marker, tag_map):
        if marker.range >= 1.5 or abs(marker.view_angle) >= 0.6:
            return None
        self.near.add(x_est, marker, max_range=1.5)
        if self.near.count(self.dock_id) < 3:
            return None
        tag_map._obs[self.dock_id] = list(self.near._obs[self.dock_id])
        lm, yaw = self.near.table()[self.dock_id]
        first = not self.refined
        self.refined = True
        return lm, yaw, first

if __name__ == "__main__":
    from simulation import Simulation
    with Simulation() as sim:
        wx, wy = CONFIG.env.dock_pose[0] - 1.0, CONFIG.env.dock_pose[1]
        start = (wx, wy, math.pi / 2)          # where navigation hands over
        sim.reset(start)
        sensing, dock = Sensing(sim), DockingController()
        dock.begin(start)
        for step in range(int(60.0 / CONFIG.sim.control_dt)):
            frame = sensing.read()
            cmd = dock.update(frame.marker, frame.odometry)
            if step % 25 == 0 or cmd.finished:
                print(f"t={frame.time:5.1f}s  {cmd.phase.value:8s}  v={cmd.v:5.2f}  w={cmd.w:5.2f}"
                    f"  range={cmd.marker_range:5.2f}  axis={cmd.axis_angle:5.2f}")
            if cmd.finished:
                break
            sim.step(cmd.v, cmd.w)
        print("Result:", cmd.phase.value)
        print("True pose (evaluation only):", sim.ground_truth().round(3))