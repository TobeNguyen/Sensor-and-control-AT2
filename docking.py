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
from robot_model import default_landmark, measurement_model, view_angle_model, wrap_angle
from sensing import MarkerMeasurement, OdometryIncrement, Sensing

DOCK_MAX_SPEED = 0.12        # [m/s]
DOCK_MIN_SPEED = 0.03        # [m/s] never creep slower than this while approaching
K_RANGE = 1.0                # [1/s] speed per metre of range error
K_BEARING = 1.5              # [1/s] centre the marker in the image
K_AXIS = 4.0                 # [1/s] converge onto the marker axis
K_CENTER = 2.0               # [1/s] in-place centring during ALIGN
BETA_MAX = 0.6               # [rad] forward speed is zero when |bearing| >= this
AXIS_TOLERANCE = 0.10        # [rad] max |delta| to count as docked
SEARCH_OMEGA = 0.7           # [rad/s] rotation speed while searching
MEASUREMENT_TIMEOUT = 0.5    # [s] how long a propagated measurement stays usable
ALIGN_TIMEOUT = 5.0          # [s] give up if close but never aligned
MAX_DOCK_TIME = 90.0         # [s] total time limit for docking
MARKER_FACE_OFFSET = 0.005   # [m] half thickness of the marker panel (see simulation.py)
VISIBILITY_MARGIN = 1.2      # marker must fit in the image with this safety factor
AXIS_SLOW = 0.6              # [rad] forward speed is reduced as |axis| grows towards this
BLIND_TIMEOUT = 4.0          # [s] estimate-driven docking: longest time without a real detection
CONFIRM_WINDOW = 1.0         # [s] DOCKED needs a real marker detection at least this recent

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

# Marker range, bearing and view angle predicted from an ESTIMATED robot pose (None if unusable)
def marker_from_pose(x_est, cfg: Config = CONFIG, landmark=None) -> Optional[tuple[float, float, float]]:
    x = np.asarray(x_est, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        return None
    lm = default_landmark(cfg) if landmark is None else landmark
    rng, bearing = measurement_model(x, lm)
    if rng < 1e-6:
        return None
    return float(rng), float(bearing), float(view_angle_model(x, cfg.env.dock_pose[2]))

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
    w = (K_BEARING + K_AXIS) * bearing - K_AXIS * view_angle
    if abs(bearing) >= BETA_MAX:                                        # not moving: just turn towards the marker
        w = (K_BEARING + K_AXIS) * bearing                              # (the view term could cancel it: dead-lock)
    return limit_command(speed, w, cfg)

def _is_valid(marker: Optional[MarkerMeasurement]) -> bool:
    return (marker is not None and marker.range > 0.0
            and all(math.isfinite(v) for v in (marker.range, marker.bearing, marker.view_angle)))

class DockingController:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
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

    def begin(self, x_est=None, landmark=None) -> None:
        self.reset()
        if x_est is None:
            return
        x = np.asarray(x_est, dtype=float)
        if x.shape == (3,) and np.all(np.isfinite(x)):
            lm = default_landmark(self.cfg) if landmark is None else landmark
            self._search_sign = 1.0 if measurement_model(x, lm)[1] >= 0.0 else -1.0

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
            est = marker_from_pose(x_est, self.cfg)
            if est is not None:
                return self._update_estimated(est, marker, t, dt)

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
            self._meas = None
            self.phase = DockPhase.SEARCH
            self._arrived_time = 0.0
            self._search_time += dt
            if self._search_time > self.cfg.docking.max_search_time:
                return self._fail()
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

        v, w = docking_command(m.range, m.bearing, m.view_angle, self.cfg)
        return self._command(v, w, m)

    # Docking step driven by the estimated pose. `est` = (range, bearing, view angle) from the EKF.
    def _update_estimated(self, est, marker: Optional[MarkerMeasurement],
                        t: float, dt: float) -> DockCommand:
        rng, bearing, view = est
        if t - self._t_start > MAX_DOCK_TIME:
            return self._fail()

        # Blind for too long: the estimate may be wrong, so stop trusting it and search instead.
        if self._unseen_time > BLIND_TIMEOUT:
            self.phase = DockPhase.SEARCH
            self._arrived_time = 0.0
            self._search_time += dt
            if self._search_time > self.cfg.docking.max_search_time:
                return self._fail()
            return self._command(0.0, self._search_sign * SEARCH_OMEGA)
        self._search_time = 0.0

        if _is_valid(marker):
            self._search_sign = 1.0 if marker.bearing >= 0.0 else -1.0
        meas = _Measurement(rng, bearing, view, t)
        self._meas = meas

        seen_recently = (self._last_seen is not None
                        and (t - self._last_seen) <= CONFIRM_WINDOW + 1e-9)
        if seen_recently and is_docked(rng, bearing, view, self.cfg):
            self.phase = DockPhase.DOCKED
            return self._command(0.0, 0.0, meas)

        d = self.cfg.docking
        if rng <= d.target_distance + d.distance_tolerance:
            self.phase = DockPhase.ALIGN
            self._arrived_time += dt
            if self._arrived_time > ALIGN_TIMEOUT:
                return self._fail()
        else:
            self.phase = DockPhase.APPROACH
            self._arrived_time = 0.0

        v, w = docking_command(rng, bearing, view, self.cfg)
        return self._command(v, w, meas)
    
if __name__ == "__main__":
    from simulation import Simulation
    with Simulation() as sim:
        wx, wy = CONFIG.env.waypoints[-1]
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