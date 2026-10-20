# Common robot interface: the same methods for the simulator and the real robot
from __future__ import annotations

import math
import time as _time
from abc import ABC, abstractmethod

import numpy as np

from config import CONFIG, Config


class RobotEnvironment(ABC):
    """Navigation, EKF and controllers must only use these methods (no MuJoCo calls)."""

    @property
    @abstractmethod
    def time(self) -> float:
        """Current time [s] on the clock used by all sensors."""

    @property
    @abstractmethod
    def lidar_angles(self) -> np.ndarray:
        """Beam angles [rad], robot frame, counter-clockwise from +x."""

    @abstractmethod
    def read_encoders(self) -> np.ndarray:
        """Cumulative [left, right] wheel angle [rad]."""

    @abstractmethod
    def read_lidar(self) -> np.ndarray:
        """One range per beam [m]."""

    @abstractmethod
    def read_camera(self) -> np.ndarray:
        """RGB uint8 image (H, W, 3)."""

    @abstractmethod
    def step(self, v: float, w: float) -> None:
        """Apply a body velocity command (m/s, rad/s) for one control period."""

    def sensor_ages(self) -> tuple[float, float]:
        """(lidar_age, camera_age) [s]. The simulator delivers fresh data, so 0."""
        return 0.0, 0.0

    def emergency_stop(self) -> None:
        self.step(0.0, 0.0)

    def close(self) -> None:
        pass

    def __enter__(self) -> "RobotEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MuJoCoEnvironment(RobotEnvironment):
    """Adapter around Simulation. Ground truth stays on the Simulation (evaluation only)."""

    def __init__(self, sim):
        self._sim = sim

    @property
    def time(self) -> float:
        return self._sim.time

    @property
    def lidar_angles(self) -> np.ndarray:
        return self._sim.lidar_angles

    def read_encoders(self) -> np.ndarray:
        return self._sim.read_encoders()

    def read_lidar(self) -> np.ndarray:
        return self._sim.read_lidar()

    def read_camera(self) -> np.ndarray:
        return self._sim.read_camera()

    def step(self, v: float, w: float) -> None:
        self._sim.step(v, w)


class HardwareEnvironment(RobotEnvironment):
    """Real robot. Implement the four _driver_* hooks for your hardware.

    cfg must hold MEASURED wheel_radius / wheel_base (see calibrate_odometry.py).
    All timestamps returned by the hooks must use time.monotonic().
    """

    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self._t0 = _time.monotonic()
        self._lidar_stamp = self._camera_stamp = self._t0
        self._last_step = self._t0
        self._angles = np.linspace(0.0, 2.0 * math.pi, cfg.lidar.num_beams, endpoint=False)

    # ---- driver hooks (to be implemented for the real robot) ----
    def _driver_encoders(self) -> np.ndarray:
        raise NotImplementedError("return cumulative [left, right] wheel angle [rad]")

    def _driver_lidar(self) -> tuple[np.ndarray, float]:
        raise NotImplementedError("return (ranges [m], monotonic timestamp)")

    def _driver_camera(self) -> tuple[np.ndarray, float]:
        raise NotImplementedError("return (RGB uint8 image, monotonic timestamp)")

    def _driver_wheels(self, left: float, right: float) -> None:
        raise NotImplementedError("send wheel angular speeds [rad/s]; the firmware needs a command watchdog")

    # ---- RobotEnvironment ----
    @property
    def time(self) -> float:
        return _time.monotonic() - self._t0

    @property
    def lidar_angles(self) -> np.ndarray:
        return self._angles

    def read_encoders(self) -> np.ndarray:
        return np.asarray(self._driver_encoders(), dtype=float)

    def read_lidar(self) -> np.ndarray:
        ranges, stamp = self._driver_lidar()
        self._lidar_stamp = stamp
        return np.asarray(ranges, dtype=float)

    def read_camera(self) -> np.ndarray:
        image, stamp = self._driver_camera()
        self._camera_stamp = stamp
        return np.asarray(image)

    def sensor_ages(self) -> tuple[float, float]:
        now = _time.monotonic()
        return now - self._lidar_stamp, now - self._camera_stamp

    def step(self, v: float, w: float) -> None:
        rob = self.cfg.robot
        v = float(np.clip(v, -rob.max_linear_vel, rob.max_linear_vel))
        w = float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel))
        r, b = rob.wheel_radius, rob.wheel_base
        self._driver_wheels((v - w * b / 2.0) / r, (v + w * b / 2.0) / r)
        remaining = self.cfg.sim.control_dt - (_time.monotonic() - self._last_step)
        if remaining > 0.0:
            _time.sleep(remaining)                 # hold the control period
        self._last_step = _time.monotonic()

    def emergency_stop(self) -> None:
        self._driver_wheels(0.0, 0.0)              # no pacing: stop immediately

    def close(self) -> None:
        try:
            self.emergency_stop()
        except NotImplementedError:
            pass