# Including libraries
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from config import CONFIG, Config
from navigation import Navigator
from robot_model import landmark_table
from sensing import LidarScan

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

    @property
    def available(self) -> bool:
        return bool(self.tags) or self.dock is not None

    @property
    def active(self) -> bool:
        return self.phase in (RecoveryPhase.SPIN, RecoveryPhase.GOTO, RecoveryPhase.MAP_GOTO)

    @property
    def finished(self) -> bool:
        return self.phase in (RecoveryPhase.DONE, RecoveryPhase.FAILED)

    # Begin recovery: first look around from the current spot, then visit the tags one by one.
    def start(self, x_est=None) -> None:
        self.reset()
        self._final_spin = False
        if not self.available:
            self.phase = RecoveryPhase.FAILED
        elif self.dock is not None:
            self._start_map_follow()
        else:
            self._start_goto(np.asarray(x_est, dtype=float))

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
        if not todo:
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
        x = np.asarray(x_est, dtype=float)
        omega = self.cfg.tags.spin_omega

        if self.phase is RecoveryPhase.SPIN:
            self._spun += abs(dtheta)
            if self._spun < math.pi:                     # quay nửa vòng thôi
                return 0.0, omega
            if self._final_spin:                         # spin cuối cùng cũng không thấy: bỏ cuộc
                self.phase = RecoveryPhase.FAILED
                return 0.0, 0.0
        self._start_goto(x)
        if self.phase is RecoveryPhase.FAILED:
            return 0.0, 0.0

        cmd = self._nav.update(x, scan)
        if cmd.finished:
            if self.phase is RecoveryPhase.MAP_GOTO:
                self.arrived_by_map = True
                self.phase = RecoveryPhase.DONE
                return 0.0, 0.0
            self._start_goto(x)                          # hết tag thì tự sang map-follow
            if self.phase is RecoveryPhase.FAILED:
                self._final_spin = True
                self._spun = 0.0
                self.phase = RecoveryPhase.SPIN          # spin là lựa chọn cuối
                return 0.0, omega
        return cmd.v, cmd.w