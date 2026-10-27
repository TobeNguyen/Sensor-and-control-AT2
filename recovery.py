# Including libraries
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from config import CONFIG, Config
from navigation import Navigator, freest_direction
from robot_model import landmark_table, wrap_angle
from sensing import LidarScan, sector_min_range
MAX_TAGS = 1   # tags to visit before following the map
EXPLORE_SCAN_ANGLE = math.pi   # EXPLORE: rotation at every stop [rad]
EXPLORE_STEP = 1.5             # EXPLORE: distance driven between two scans [m]
class RecoveryPhase(Enum):
    IDLE = "idle"
    SPIN = "spin"        # rotate in place looking for any known tag
    GOTO = "goto"        # drive to the stand-off point in front of the nearest tag
    MAP_GOTO = "map"     # no tag could be seen: follow the learned map to the dock approach point
    DONE = "done"        # position uncertainty is small again, or the approach point was reached
    FAILED = "failed"    # nothing left to try (no tags and no dock in the map)

class TagRecovery:
    def __init__(self, cfg: Config = CONFIG, tracker=None, tag_map=None):
        self.cfg = cfg
        self.tracker = tracker
        dock_id = int(cfg.camera.marker_id)
        table = {} if tag_map is None else landmark_table(cfg, tag_map)   # learned map only
        # Extra tags only: the dock is the place we are trying to get back to.
        self.tags = {i: p for i, p in table.items() if i != dock_id}
        self.dock = table.get(dock_id)                   # None until the dock is learned
        self.reset()

    def reset(self) -> None:
        self.phase = RecoveryPhase.IDLE
        self.target_id: Optional[int] = None
        self.arrived_by_map = False           # True: finished by reaching the map approach point (not relocalised)
        self._visited: set = set()
        self._nav: Optional[Navigator] = None
        self._spun = 0.0                      # accumulated rotation during SPIN [rad]
        self._t = 0.0

    @property
    def available(self) -> bool:
        return bool(self.tags) or self.dock is not None

    @property
    def active(self) -> bool:
        return self.phase in (RecoveryPhase.SPIN, RecoveryPhase.GOTO, RecoveryPhase.MAP_GOTO)

    @property
    def finished(self) -> bool:
        return self.phase in (RecoveryPhase.DONE, RecoveryPhase.FAILED)

    # Begin recovery: first look around from the current spot (full turn), then visit the tags one by one.
    def start(self, x_est=None) -> None:
        self.reset()
        self.phase = RecoveryPhase.SPIN if self.available else RecoveryPhase.FAILED

    # Point in front of the tag (along its normal) from where the camera can see it
    def standoff_point(self, tag_id: int) -> np.ndarray:
        pos, yaw = self.tags[tag_id]
        return pos + self.cfg.tags.standoff * np.array([math.cos(yaw), math.sin(yaw)])

    # Point in front of the dock (from the learned map) where docking can take over
    def approach_point(self) -> np.ndarray:
        pos, yaw = self.dock
        return pos + self.cfg.tags.approach_dist * np.array([math.cos(yaw), math.sin(yaw)])

    def _start_goto(self, x_est: np.ndarray) -> None:
        todo = [i for i in self.tags if i not in self._visited]
        if not todo or len(self._visited) >= MAX_TAGS:
            self._start_map_follow()
            return
        self.target_id = min(todo, key=lambda i: float(np.hypot(*(self.standoff_point(i) - x_est[:2]))))
        self._visited.add(self.target_id)
        self._nav = Navigator(self.cfg, tracker=self.tracker,
                            waypoints=[self.standoff_point(self.target_id)])
        self.phase = RecoveryPhase.GOTO

    # Every tag was tried without localising: drive to the dock approach point using the saved map
    def _start_map_follow(self) -> None:
        if self.dock is None:
            self.phase = RecoveryPhase.FAILED
            return
        self.target_id = int(self.cfg.camera.marker_id)
        self._nav = Navigator(self.cfg, tracker=self.tracker, waypoints=[self.approach_point()])
        self.phase = RecoveryPhase.MAP_GOTO

    # One control step. `localized` = not estimator.is_lost(); `dtheta` = odometry heading change.
    # Returns (v, w). Check `phase` afterwards: DONE or FAILED means the main loop takes over
    # (DONE with arrived_by_map = True means: go straight to docking).
    def update(self, x_est, scan: LidarScan, localized: bool, dtheta: float) -> tuple[float, float]:
        if self.phase in (RecoveryPhase.IDLE, RecoveryPhase.DONE, RecoveryPhase.FAILED):
            return 0.0, 0.0
        if localized:                                  # a tag (or the dock) corrected the estimate
            self.phase = RecoveryPhase.DONE
            return 0.0, 0.0
        self._t += self.cfg.sim.control_dt
        if self._t > 25.0 and self.phase is not RecoveryPhase.MAP_GOTO:
            self._start_map_follow()          # stop searching, go straight to the dock approach point
            if self.phase is RecoveryPhase.FAILED:
                return 0.0, 0.0
        x = np.asarray(x_est, dtype=float)

        if self.phase is RecoveryPhase.SPIN:
            self._spun += abs(dtheta)
            if self._spun < math.pi:
                return 0.0, self.cfg.tags.spin_omega
            self._start_goto(x)                        # full turn without a tag: next tag, or follow the map
            if self.phase is RecoveryPhase.FAILED:
                return 0.0, 0.0

        cmd = self._nav.update(x, scan)
        if not cmd.finished:
            return cmd.v, cmd.w
        if self.phase is RecoveryPhase.MAP_GOTO:       # reached the dock approach point
            self.arrived_by_map = True
            self.phase = RecoveryPhase.DONE
            return 0.0, 0.0
        self.phase = RecoveryPhase.SPIN                # reached a tag stand-off: look around again
        self._spun = 0.0
        return 0.0, self.cfg.tags.spin_omega
    
# Rotate a little, stand still, observe, repeat: the camera image is sharp and not delayed while still
class StopGoScan:
    def __init__(self, omega: float = 0.8, step: float = 0.5, hold: float = 0.7, settle: float = 0.3):
        self.omega, self.step, self.hold, self.settle = omega, step, hold, settle
        self.sign = 1
        self.reset()

    def reset(self) -> None:
        self.spun = 0.0          # total rotation since reset [rad]
        self.step_spun = 0.0     # rotation since the last stop [rad]
        self.hold_t0 = None      # time when the robot stopped
        self.started = False     # the first stop happens right after reset (look before turning)

    # Returns (w, observe). observe is True only when the robot has been still long enough
    def update(self, t: float, dtheta: float):
        self.spun += abs(dtheta)
        if not self.started:
            self.started = True
            self.hold_t0 = t
        if self.hold_t0 is not None:
            if t - self.hold_t0 >= self.hold:
                self.hold_t0, self.step_spun = None, 0.0
                return self.sign * self.omega, False
            return 0.0, t - self.hold_t0 >= self.settle
        self.step_spun += abs(dtheta)
        if self.step_spun >= self.step:
            self.hold_t0 = t
            return 0.0, False
        return self.sign * self.omega, False

# Map seen by the EKF: the dock stays hidden until it has been re-learned from a close view
class MapView:
    def __init__(self, tag_map, dock_id: int):
        self.tag_map, self.dock_id, self.show_dock = tag_map, int(dock_id), False

    def table(self) -> dict:
        t = self.tag_map.table()
        if not self.show_dock:
            t.pop(self.dock_id, None)
        return t

# EXPLORE: PICK -> TURN -> MOVE -> SCAN -> PICK ... (same logic as before, state kept in the object)
class Explorer:
    def __init__(self, cfg: Config, run_cfg: Config, tag_map, dock_id: int, scan: StopGoScan, dt: float):
        self.cfg, self.run_cfg = cfg, run_cfg
        self.tag_map, self.dock_id, self.scan, self.dt = tag_map, int(dock_id), scan, dt
        self.state = "PICK"
        self.target = 0.0
        self.moved = 0.0
        self.step = EXPLORE_STEP
        self.visited: set = set()
        self.scanned: list = [np.zeros(2)]
        self.t0 = 0.0
        self.stall = 0.0
        self.bad: list = []
        self.dirs: list = []

    # Returns (v, w)
    def update(self, frame, x_est, sigma_xy: float) -> tuple[float, float]:
        run_cfg, tag_map, dock_id, scan, dt = self.run_cfg, self.tag_map, self.dock_id, self.scan, self.dt
        v = w = 0.0
        observe = False
        if self.state == "SCAN":                   # rotate-stop-rotate, observe while still
            w, observe = scan.update(frame.time, frame.odometry.dtheta)
            if frame.marker is not None and tag_map.count(dock_id) < run_cfg.tags.survey_min_obs:
                w = 0.0
            if scan.spun >= EXPLORE_SCAN_ANGLE:
                self.dirs.append((float(x_est[0]), float(x_est[1]), float(x_est[2])))
                scan.sign *= -1
                self.state = "PICK"
        elif self.state == "PICK":
            if sigma_xy > 0.1:
                self.state = "SCAN"
                scan.reset()
                return 0.0, 0.0
            best_a, best_s = None, -1e9
            for a in np.linspace(-math.pi, math.pi, 36, endpoint=False):
                free = sector_min_range(frame.lidar, float(a), 0.5)
                if free < 0.7:
                    continue
                tgt = x_est[:2] + min(free, 1.5) * np.array(
                    [math.cos(x_est[2] + a), math.sin(x_est[2] + a)])
                if any(float(np.hypot(*(tgt - b))) < 0.6 for b in self.bad):
                    continue
                near = min(float(np.hypot(*(tgt - p))) for p in self.scanned)
                s = 1.5 * min(near, 1.0) + min(free, 1.5) - 0.5 * abs(float(a)) + 0.5
                if near < 0.8:
                    s -= 4.0
                for (qx, qy, _qh) in self.dirs[-3:]:
                    back = math.atan2(qy - x_est[1], qx - x_est[0])
                    if math.hypot(qx - x_est[0], qy - x_est[1]) < 3.0 and abs(wrap_angle(x_est[2] + a - back)) < 0.5:
                        s -= 2.0
                if s > best_s:
                    best_a, best_s = float(a), s
            if best_a is None:
                best_a = freest_direction(frame.lidar)
            self.target = wrap_angle(x_est[2] + best_a)
            self.step = EXPLORE_STEP
            n_extra = len(self.cfg.tag_table()) - 1
            known_extra = sum(1 for i in tag_map.table() if int(i) != dock_id)
            all_known = known_extra >= n_extra
            pending = {}
            if all_known:
                self.step = 1.5
            for i, (p_tag, yaw_tag) in tag_map.table().items():
                if int(i) != dock_id and int(i) not in self.visited:
                    pending[int(i)] = p_tag + run_cfg.tags.standoff * np.array(
                        [math.cos(yaw_tag), math.sin(yaw_tag)])
            if pending and not all_known:
                tid = min(pending, key=lambda i: float(np.hypot(*(pending[i] - x_est[:2]))))
                spt = pending[tid]
                gvec = spt - x_est[:2]
                gdist = float(np.hypot(*gvec))
                self.visited.add(tid)              # xét một lần rồi thôi
                seen_already = min(float(np.hypot(*(spt - p))) for p in self.scanned) < 0.8
                if gdist >= 0.4 and not seen_already:
                    rel = wrap_angle(math.atan2(gvec[1], gvec[0]) - x_est[2])
                    if sector_min_range(frame.lidar, rel, 0.4) > min(1.0, gdist) + 0.3:
                        self.target = wrap_angle(x_est[2] + rel)
                        self.step = max(0.5, min(2.0, gdist))
            self.state, self.t0 = "TURN", frame.time
        elif self.state == "TURN":
            err = wrap_angle(self.target - x_est[2])
            if abs(err) < 0.1:
                self.state, self.moved, self.stall = "MOVE", 0.0, 0.0
            elif frame.time - self.t0 > 8.0:
                self.bad.append(x_est[:2] + 0.8 * np.array([math.cos(self.target), math.sin(self.target)]))
                self.state = "PICK"
            else:
                w = float(np.clip(2.0 * err, -0.6, 0.6))
        elif self.state == "MOVE":
            self.moved += frame.odometry.ds
            self.stall = self.stall + dt if abs(frame.odometry.ds) < 1e-4 else 0.0
            blocked = self.stall > 2.0 or sector_min_range(frame.lidar, 0.0, 0.5) < 0.6 or sigma_xy > 0.1
            if blocked and self.moved < 0.3:
                self.bad.append(x_est[:2] + 0.8 * np.array([math.cos(x_est[2]), math.sin(x_est[2])]))
                self.state = "PICK"
            elif self.moved >= self.step or blocked:
                if sum(1 for p in self.scanned if float(np.hypot(*(x_est[:2] - p))) < 0.6) >= 2:
                    self.state = "PICK"
                else:
                    self.state = "SCAN"
                    self.scanned.append(x_est[:2].copy())
                    scan.reset()
            else:
                v = 0.2
        if observe:
            for marker in frame.markers:
                is_dock = int(marker.marker_id) == dock_id
                limit = run_cfg.tags.dock_max_range if is_dock else run_cfg.tags.survey_max_range
                if sigma_xy < 0.08:
                    tag_map.add(x_est, marker, max_range=limit if is_dock else min(limit, 2.0))
        return v, w