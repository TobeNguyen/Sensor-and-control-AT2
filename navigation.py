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
from robot_model import motion_model, wrap_angle

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
        wps = [] if waypoints is None else waypoints   # no waypoints from the config: they come from the learned dock
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
        # After an avoidance detour, also skip the current waypoint if the next one is clearly closer.
        tol = self.cfg.controller.waypoint_tolerance
        while self.index < len(self.waypoints):
            wp = self.waypoints[self.index]
            d_cur = float(np.hypot(*(wp - x[:2])))
            reached = d_cur < tol
            skipped = False
            if (not reached and self.phase is NavPhase.AVOID
                    and self.index < len(self.waypoints) - 1):
                d_next = float(np.hypot(*(self.waypoints[self.index + 1] - x[:2])))
                skipped = d_next + 0.2 < d_cur
            if not (reached or skipped):
                break
            self._anchor = wp.copy() if reached else x[:2].copy()
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

# Direction (robot frame) with the most free space, with a small preference for straight ahead
def freest_direction(lidar, half_width: float = 0.4) -> float:
    best, best_score = 0.0, -1e9
    for a in np.linspace(-math.pi, math.pi, 36, endpoint=False):
        score = min(sector_min_range(lidar, float(a), half_width), 1.5) - 0.6 * abs(float(a))
        if score > best_score:
            best, best_score = float(a), score
    return best

# New navigator after a recovery. Lost while navigating: continue with the remaining waypoints.
# Docking failed: drive to the approach point in front of the dock (not the wall-side last waypoint).
def resume_navigator(nav: Navigator, cfg: Config, after_dock: bool, approach=None) -> Navigator:
    if after_dock and approach is not None:
        remaining = [approach]
    elif after_dock:
        remaining = [nav.waypoints[-1]]
    else:
        remaining = nav.waypoints[nav.index:]
    if not remaining:
        remaining = [nav.waypoints[-1]]
    return Navigator(cfg, tracker=nav.tracker, waypoints=remaining)

# Point in front of the dock (learned pose only)
def dock_approach_point(cfg: Config, lm, yaw) -> np.ndarray:
    return np.array([lm[0] + cfg.tags.approach_dist * math.cos(yaw),
                    lm[1] + cfg.tags.approach_dist * math.sin(yaw)])

def go_to(nav: Navigator, cfg: Config, x_est, tx: float, ty: float, max_step: float = 1.5) -> Navigator:
    d = math.hypot(tx - x_est[0], ty - x_est[1])
    k = min(d, max_step) / max(d, 1e-6)
    wp = (x_est[0] + k * (tx - x_est[0]), x_est[1] + k * (ty - x_est[1]))
    return Navigator(cfg, tracker=nav.tracker, waypoints=[wp])

# AIM: look at the dock while standing still, turn towards it, drive a short step, repeat.
# Dock not seen -> fallback chain: dock approach point, then the tags (nearest to the dock first).
# update() returns (v, w, nav, event); event = None | "DOCK" (start docking) | "FAILED" (dock not found)
class AimController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.goal_timeout = 25.0     # [s] give up on a fallback point after this long
        self.state = "LOOK"          # LOOK -> TURN -> MOVE -> LOOK ... (SEARCH / PLAN when the dock is not seen)
        self.t0 = None               # time when the current LOOK started
        self.obs = []                # (range, bearing, view_angle) of the dock seen while standing still
        self.target = 0.0            # heading to turn to [rad]
        self.goal = 0.0              # distance to drive in the current MOVE [m]
        self.moved = 0.0             # distance driven so far in the current MOVE [m]
        self.idle = 0.0
        self.stall = 0.0             # time in MOVE without actually moving
        self.stuck_n = 0             # consecutive "stuck" events
        self.stuck_t = -99.0         # time of the last "stuck" event
        self.tried: set = set()      # fallback points already reached and scanned (-1 = dock approach point)
        self.goal_pt = None          # fallback point the robot is driving to
        self.goal_id = None          # id of that point (-1 = dock approach point, else tag id)
        self.goal_t0 = 0.0           # time when the current fallback point was chosen
        self.go_t0 = 0.0             # time when the current GO (Navigator) segment started
        self.seen_pt = None          # last observed dock position (estimator frame)

    # Used by main.py wherever it used to write: ap_state, ap_t0 = "LOOK", None
    def restart_look(self) -> None:
        self.state, self.t0 = "LOOK", None

    def update(self, frame, x_est, nav: Navigator, scan, dock_lm, dock_yaw, tag_map, dock_id: int,
            dt: float, note):
        cfg = self.cfg
        v = w = 0.0
        event = None
        if self.state == "LOOK":
            if self.t0 is None:
                self.t0, self.obs = frame.time, []
            if frame.time - self.t0 >= 0.4 and frame.marker is not None:
                self.obs.append((frame.marker.range, frame.marker.bearing, frame.marker.view_angle))
            if frame.time - self.t0 >= 1.2:
                self.t0 = None
                if len(self.obs) >= 2:
                    self.goal_pt, self.goal_id = None, None      # dock seen, fallback no longer needed
                    rng_d = float(np.median([o[0] for o in self.obs]))
                    brg_d = float(np.median([o[1] for o in self.obs]))
                    view_d = float(np.median([o[2] for o in self.obs]))
                    yaw_d = wrap_angle(x_est[2] + view_d + math.pi) if rng_d < 2.5 else dock_yaw
                    a_dock = x_est[2] + brg_d
                    self.seen_pt = np.array([x_est[0] + rng_d * math.cos(a_dock), x_est[1] + rng_d * math.sin(a_dock)])
                    if rng_d > 2.0:
                        gx = x_est[0] + (rng_d - 1.8) * math.cos(a_dock)
                        gy = x_est[1] + (rng_d - 1.8) * math.sin(a_dock)
                    else:
                        gx = x_est[0] + rng_d * math.cos(a_dock) + cfg.tags.approach_dist * math.cos(yaw_d)
                        gy = x_est[1] + rng_d * math.sin(a_dock) + cfg.tags.approach_dist * math.sin(yaw_d)
                    gd = math.hypot(gx - x_est[0], gy - x_est[1])
                    close_enough = rng_d <= cfg.tags.approach_dist + 0.4
                    on_axis = abs(wrap_angle(brg_d - view_d)) < 0.25 and abs(brg_d) < 0.5
                    too_close = rng_d < 0.9
                    if (gd < 0.25 and close_enough and on_axis) or (rng_d <= 1.5 and on_axis):
                        event = "DOCK"
                        note(frame.time, f"AIM done: in front of the dock ({rng_d:.2f} m) -> docking")
                    elif gd < 0.25:
                        L = max(0.3, min(1.0, rng_d - cfg.tags.approach_dist))
                        nav = go_to(nav, cfg, x_est, x_est[0] + L * math.cos(a_dock), x_est[1] + L * math.sin(a_dock))
                        self.state, self.go_t0 = "GO", frame.time
                        note(frame.time, f"AIM: dock still {rng_d:.2f} m away -> driving straight at it")
                    else:
                        nav = go_to(nav, cfg, x_est, gx, gy)
                        self.state, self.go_t0 = "GO", frame.time
                        note(frame.time, f"AIM: dock {rng_d:.2f} m -> approach point {gd:.2f} m away")
                else:
                    # dock not seen: keep driving to the pending fallback point; scan only when we are there;
                    # plan a new fallback point when there is none
                    pending_fb = self.goal_pt is not None
                    arrived_fb, gd_fb = False, 0.0
                    if pending_fb:
                        gd_fb = math.hypot(self.goal_pt[0] - x_est[0], self.goal_pt[1] - x_est[1])
                        arrived_fb = gd_fb <= 0.3 or frame.time - self.goal_t0 > self.goal_timeout
                    if pending_fb and not arrived_fb:
                        nav = go_to(nav, cfg, x_est, self.goal_pt[0], self.goal_pt[1])
                        self.state, self.go_t0 = "GO", frame.time
                    elif pending_fb:
                        self.state = "FACE"
                        note(frame.time, "AIM: dock not seen at the fallback point -> facing the dock, then scanning (max 180 deg)")
                    else:
                        self.state = "PLAN"
                        note(frame.time, "AIM: dock not seen -> choosing a fallback point")
        elif self.state == "FACE":
            if dock_lm is None:
                self.state = "SEARCH"
                scan.reset()
            else:
                err = wrap_angle(math.atan2(dock_lm[1] - x_est[1], dock_lm[0] - x_est[0]) - x_est[2] - 0.6)
                if abs(err) < 0.1:
                    self.state = "SEARCH"
                    scan.reset()
                else:
                    w = float(np.clip(2.0 * err, -0.6, 0.6))
        elif self.state == "SEARCH":               # limited stop-go scan at a fallback point
            w, observe = scan.update(frame.time, frame.odometry.dtheta)
            if observe and frame.marker is not None:
                self.state = "LOOK"
            elif scan.spun >= math.pi:             # half a turn is enough, not 2*pi
                if self.goal_id is not None:
                    self.tried.add(self.goal_id)   # reached and scanned: do not come back here
                self.goal_pt, self.goal_id = None, None
                self.state = "PLAN"
                w = 0.0
        elif self.state == "PLAN":                 # choose the next fallback point
            goal_pt, goal_id = None, None
            if dock_lm is not None and -1 not in self.tried:
                goal_pt, goal_id = dock_approach_point(cfg, dock_lm, dock_yaw), -1
            else:
                ref_pt = np.asarray(dock_lm[:2], dtype=float) if dock_lm is not None else x_est[:2]
                tags_now = {int(i): pose for i, pose in tag_map.table().items()
                            if int(i) != dock_id and int(i) not in self.tried}
                if tags_now:
                    goal_id = min(tags_now, key=lambda i: float(np.hypot(*(tags_now[i][0] - ref_pt))))
                    p_tag, yaw_tag = tags_now[goal_id]
                    goal_pt = p_tag + cfg.tags.standoff * np.array([math.cos(yaw_tag), math.sin(yaw_tag)])
            if goal_pt is None:                    # nothing left to try
                if dock_lm is not None:
                    event = "DOCK"
                    note(frame.time, "AIM: dock not seen anywhere -> blind docking")
                else:
                    event = "FAILED"
                    note(frame.time, "AIM FAILED: dock not seen and no fallback point left")
            else:
                self.goal_pt, self.goal_id, self.goal_t0 = goal_pt, goal_id, frame.time
                gd = math.hypot(goal_pt[0] - x_est[0], goal_pt[1] - x_est[1])
                label = "dock approach point" if goal_id == -1 else f"tag {goal_id} stand-off"
                note(frame.time, f"AIM: fallback -> {label} ({goal_pt[0]:+.2f}, {goal_pt[1]:+.2f}), {gd:.2f} m away")
                if gd < 0.3:                       # already there: scan right here
                    self.state = "FACE"
                else:
                    nav = go_to(nav, cfg, x_est, goal_pt[0], goal_pt[1])
                    self.state, self.go_t0 = "GO", frame.time
        elif self.state == "GO":                   # LQR + LiDAR avoidance via Navigator
            ncmd = nav.update(x_est, frame.lidar)
            v, w = ncmd.v, ncmd.w
            if ncmd.finished or frame.time - self.go_t0 > 8.0:
                if self.seen_pt is not None:       # arrive FACING the dock so the marker stays in view
                    self.target = wrap_angle(math.atan2(self.seen_pt[1] - x_est[1], self.seen_pt[0] - x_est[0]))
                    self.goal, self.state = 0.0, "TURN"
                else:
                    self.state = "LOOK"
        elif self.state == "TURN":
            err = wrap_angle(self.target - x_est[2])
            if abs(err) < 0.08:
                self.state, self.moved, self.stall = "MOVE", 0.0, 0.0
            else:
                w = float(np.clip(2.0 * err, -0.6, 0.6))
        elif self.state == "MOVE":
            self.moved += frame.odometry.ds
            self.stall = self.stall + dt if abs(frame.odometry.ds) < 1e-4 else 0.0
            if self.stall > 1.0:
                self.stall, self.goal_pt, self.state = 0.0, None, "PLAN"
                note(frame.time, "AIM: MOVE stalled -> replanning with obstacle avoidance")
            elif sector_min_range(frame.lidar, 0.0, 0.5) < 0.6:
                self.target = wrap_angle(x_est[2] + freest_direction(frame.lidar))
                self.goal, self.state = 0.8, "TURN"
            elif self.moved >= self.goal:
                self.state = "LOOK"
            else:
                v = 0.2
        if self.state in ("TURN", "MOVE") and abs(v) < 1e-6 and abs(w) < 1e-6:
            self.idle += dt
            if self.idle > 0.5:
                self.stuck_n = self.stuck_n + 1 if frame.time - self.stuck_t < 15.0 else 1
                self.stuck_t = frame.time
                if self.stuck_n >= 2:
                    self.stuck_n, self.idle, self.goal_pt, self.state = 0, 0.0, None, "PLAN"
                    note(frame.time, "AIM: repeated stuck -> replanning with obstacle avoidance")
                else:
                    left = sector_min_range(frame.lidar, math.pi / 2, 0.5)
                    right = sector_min_range(frame.lidar, -math.pi / 2, 0.5)
                    side = math.pi / 2 if left >= right else -math.pi / 2
                    self.target = wrap_angle(x_est[2] + side)
                    self.goal, self.state, self.idle = 0.5, "TURN", 0.0
                    note(frame.time, "AIM: stuck -> forced 90 deg turn")
        else:
            self.idle = 0.0
        return v, w, nav, event

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