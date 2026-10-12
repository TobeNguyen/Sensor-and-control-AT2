# Including libraries
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from config import CONFIG, Config
from controller import DEFAULT_CRUISE_SPEED, LQRTracker, Reference, segment_reference
from sensing import LidarScan, lidar_points, sector_min_range, make_lidar_scan
from robot_model import motion_model

LOOK_DISTANCE = 0.6                  # [m] corridor length used for the free-path test
ENTER_MARGIN = 0.05                  # [m] extra half-width to start avoiding
EXIT_MARGIN = 0.08                   # [m] extra half-width required to stop avoiding
AVOID_SPEED = 0.08                   # [m/s] forward speed while avoiding
FRONT_HALF_WIDTH = 0.4               # [rad] front sector passed to the speed limiter
HEADING_STEP = math.radians(15.0)    # candidate heading resolution
HYSTERESIS = 0.8                     # weight of the "keep previous heading" cost
EXIT_CLEAR_STEPS = 25                # steps of clear path (0.5 s) before leaving AVOID
HANDOFF_RANGE = 1.0                  # [m] near the last waypoint ...
HANDOFF_AVOID_TIME = 8.0             # [s] ... stuck in AVOID this long -> hand over to docking

class NavPhase(Enum):
    TRACK = "track"
    AVOID = "avoid"
    DONE = "done"

@dataclass(frozen=True)
class NavCommand:
    v: float                 # [m/s]
    w: float                 # [rad/s]
    phase: NavPhase
    waypoint_index: int
    finished: bool           # True once the last waypoint is reached -> start docking
    goal_distance: float     # [m] distance to the current waypoint (nan if unknown)

# Wrap angle to (-pi, pi)
def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

# Distance to the first LIDAR point
def free_distance(points: np.ndarray, heading: float, look: float, half_width: float) -> float:
    if len(points) == 0:
        return float(look)
    c, s = math.cos(heading), math.sin(heading)
    ahead = points[:, 0] * c + points[:, 1] * s
    lateral = -points[:, 0] * s + points[:, 1] * c
    blocked = (ahead > 0.0) & (ahead < look) & (np.abs(lateral) < half_width)
    return float(ahead[blocked].min()) if blocked.any() else float(look)

# Full-circle grid of headings plus the exact goal direction
def heading_candidates(goal_rel: float, step: float = HEADING_STEP) -> np.ndarray:
    return np.append(np.arange(-math.pi, math.pi, step), _wrap(goal_rel))

# Pick the free heading closest to the goal direction
def choose_heading(points: np.ndarray, goal_rel: float, half_width: float,
                look: float = LOOK_DISTANCE, prev_rel: Optional[float] = None,
                step: float = HEADING_STEP) -> tuple[float, float, bool]:
    cands = heading_candidates(goal_rel, step)
    free_d = np.array([free_distance(points, h, look, half_width) for h in cands])
    free = free_d >= look - 1e-9
    if free.any():
        cost = np.abs(_wrap(cands - goal_rel))
        if prev_rel is not None:
            cost = cost + HYSTERESIS * np.abs(_wrap(cands - prev_rel))
        cost[~free] = np.inf
        i = int(np.argmin(cost))
        return float(cands[i]), float(free_d[i]), True
    i = int(np.argmax(free_d))
    return float(cands[i]), float(free_d[i]), False

class Navigator:
    def __init__(self, cfg: Config = CONFIG, tracker: Optional[LQRTracker] = None,
                waypoints: Optional[Sequence[Sequence[float]]] = None,
                v_cruise: float = DEFAULT_CRUISE_SPEED):
        self.cfg = cfg
        self.tracker = LQRTracker(cfg) if tracker is None else tracker
        wps = cfg.env.waypoints if waypoints is None else waypoints
        self.waypoints = [np.asarray(p, dtype=float) for p in wps]
        self.v_cruise = v_cruise
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.phase = NavPhase.TRACK
        self.finished = False
        self._anchor: Optional[np.ndarray] = None   # start point of the current segment
        self._prev_world: Optional[float] = None    # last avoidance heading (world frame)
        self._clear = 0                             # consecutive steps with a free goal path
        self._avoid_t = 0.0                         # time spent in AVOID [s]

    @property
    def target(self) -> Optional[np.ndarray]:
        return None if self.finished else self.waypoints[self.index]

    def _stop(self, goal_distance: float = float("nan")) -> NavCommand:
        return NavCommand(0.0, 0.0, self.phase, self.index, self.finished, goal_distance)

    def update(self, x_est, scan: LidarScan) -> NavCommand:
        x = np.asarray(x_est, dtype=float)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            return self._stop()                      # invalid estimate -> stop
        if self.finished:
            return self._stop(0.0)
        if self._anchor is None:
            self._anchor = x[:2].copy()

        # Advance past every waypoint that is already reached.
        tol = self.cfg.controller.waypoint_tolerance
        while self.index < len(self.waypoints):
            wp = self.waypoints[self.index]
            if float(np.hypot(*(wp - x[:2]))) >= tol:
                break
            self._anchor = wp.copy()
            self.index += 1
            self.phase = NavPhase.TRACK
            self._prev_world = None
            self._clear = 0
            self._avoid_t = 0.0
        if self.index >= len(self.waypoints):
            self.finished = True
            self.phase = NavPhase.DONE
            return self._stop(0.0)

        goal = self.waypoints[self.index]
        delta = goal - x[:2]
        goal_dist = float(np.hypot(*delta))
        goal_rel = float(_wrap(math.atan2(delta[1], delta[0]) - x[2]))

        # Safety net: stuck in AVOID near the last waypoint -> let the camera take over.
        self._avoid_t = self._avoid_t + self.cfg.sim.control_dt if self.phase is NavPhase.AVOID else 0.0
        if (self.index == len(self.waypoints) - 1 and goal_dist < HANDOFF_RANGE
                and self._avoid_t > HANDOFF_AVOID_TIME):
            self.finished = True
            self.phase = NavPhase.DONE
            return self._stop(0.0)

        points = lidar_points(scan)
        margin = EXIT_MARGIN if self.phase is NavPhase.AVOID else ENTER_MARGIN
        half_width = self.cfg.robot.body_radius + margin
        look = min(LOOK_DISTANCE, goal_dist)
        goal_free = free_distance(points, goal_rel, look, half_width) >= look - 1e-9
        front = sector_min_range(scan, 0.0, FRONT_HALF_WIDTH)

        # Leave AVOID only after the goal path has been clear for EXIT_CLEAR_STEPS in a row.
        self._clear = self._clear + 1 if goal_free else 0
        if goal_free and (self.phase is NavPhase.TRACK or self._clear >= EXIT_CLEAR_STEPS):
            self.phase = NavPhase.TRACK
            self._prev_world = None
            ref = segment_reference(x, self._anchor, goal, self.v_cruise)
        else:
            self.phase = NavPhase.AVOID
            self._anchor = x[:2].copy()              # next TRACK segment starts here
            prev_rel = None if self._prev_world is None else float(_wrap(self._prev_world - x[2]))
            heading_rel, _, found = choose_heading(points, goal_rel, half_width,
                                                LOOK_DISTANCE, prev_rel)
            self._prev_world = float(_wrap(x[2] + heading_rel))
            v_ref = AVOID_SPEED * max(0.0, math.cos(heading_rel)) if found else 0.0
            ref = Reference(x[0], x[1], self._prev_world, v_ref, 0.0)

        v, w = self.tracker.compute(x, ref, front_range=front)
        return NavCommand(v, w, self.phase, self.index, False, goal_dist)
    
if __name__ == "__main__":
    # Quick check on a free map: the robot should drive to the first waypoint.
    angles = np.linspace(0.0, 2.0 * math.pi, CONFIG.lidar.num_beams, endpoint=False)
    free_scan = make_lidar_scan(np.full(len(angles), CONFIG.lidar.range_max), angles)
    nav, pose, dt = Navigator(waypoints=[(1.0, 1.0)]), np.array(CONFIG.env.start_pose), CONFIG.sim.control_dt
    for _ in range(3000):
        cmd = nav.update(pose, free_scan)
        if cmd.finished:
            break
        pose = motion_model(pose, cmd.v * dt, cmd.w * dt)
    print("Final pose:", np.round(pose, 3), "| finished:", cmd.finished)