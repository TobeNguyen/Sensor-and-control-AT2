# Including libraries
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from config import CONFIG, Config
from navigation import Navigator
from sensing import LidarScan

class RecoveryPhase(Enum):
    IDLE = "idle"
    SPIN = "spin"        # rotate in place looking for any known tag
    GOTO = "goto"        # drive to the stand-off point in front of the nearest tag
    DONE = "done"        # position uncertainty is small again
    FAILED = "failed"    # every tag was tried without getting localised

class TagRecovery:
    def __init__(self, cfg: Config = CONFIG, tracker=None):
        self.cfg = cfg
        self.tracker = tracker
        dock_id = int(cfg.camera.marker_id)
        # Extra tags only: the dock is the place we are trying to get back to.
        self.tags = {int(i): (np.array(p[:2], dtype=float), float(p[2]))
                    for i, p in cfg.tag_table().items() if int(i) != dock_id}
        self.reset()

    def reset(self) -> None:
        self.phase = RecoveryPhase.IDLE
        self.target_id: Optional[int] = None
        self._visited: set = set()
        self._nav: Optional[Navigator] = None
        self._spun = 0.0                      # accumulated rotation during SPIN [rad]

    @property
    def active(self) -> bool:
        return self.phase in (RecoveryPhase.SPIN, RecoveryPhase.GOTO)

    @property
    def finished(self) -> bool:
        return self.phase in (RecoveryPhase.DONE, RecoveryPhase.FAILED)

    # Begin recovery: first look around from the current spot, then visit the tags one by one.
    def start(self) -> None:
        self.reset()
        self.phase = RecoveryPhase.SPIN if self.tags else RecoveryPhase.FAILED

    # Point in front of the tag (along its normal) from where the camera can see it
    def standoff_point(self, tag_id: int) -> np.ndarray:
        pos, yaw = self.tags[tag_id]
        return pos + self.cfg.tags.standoff * np.array([math.cos(yaw), math.sin(yaw)])

    def _start_goto(self, x_est: np.ndarray) -> None:
        todo = [i for i in self.tags if i not in self._visited]
        if not todo:
            self.phase = RecoveryPhase.FAILED
            return
        self.target_id = min(todo, key=lambda i: float(np.hypot(*(self.standoff_point(i) - x_est[:2]))))
        self._visited.add(self.target_id)
        self._nav = Navigator(self.cfg, tracker=self.tracker,
                            waypoints=[self.standoff_point(self.target_id)])
        self.phase = RecoveryPhase.GOTO

    # One control step. `localized` = not estimator.is_lost(); `dtheta` = odometry heading change.
    # Returns (v, w). Check `phase` afterwards: DONE or FAILED means the main loop takes over.
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
            if self._spun < 2.0 * math.pi:
                return 0.0, omega
            self._start_goto(x)                        # full turn without a tag: go to the next one
            if self.phase is RecoveryPhase.FAILED:
                return 0.0, 0.0

        cmd = self._nav.update(x, scan)                # LiDAR obstacle avoidance included
        if cmd.finished:                               # arrived at the stand-off point: look around
            self.phase = RecoveryPhase.SPIN
            self._spun = 0.0
            return 0.0, omega
        return cmd.v, cmd.w