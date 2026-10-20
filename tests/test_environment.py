import time
from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG, RealismConfig
from environment import HardwareEnvironment, MuJoCoEnvironment, RobotEnvironment
from sensing import Sensing

N = CONFIG.lidar.num_beams
R, B = CONFIG.robot.wheel_radius, CONFIG.robot.wheel_base


class FakeHardware(HardwareEnvironment):
    def __init__(self, lidar_age=0.0):
        super().__init__(CONFIG)
        self.wheels, self.lidar_age = [], lidar_age

    def _driver_encoders(self):
        return np.zeros(2)

    def _driver_lidar(self):
        return np.full(N, 2.0), time.monotonic() - self.lidar_age

    def _driver_camera(self):
        img = np.full((CONFIG.camera.height, CONFIG.camera.width, 3), 255, np.uint8)
        return img, time.monotonic()

    def _driver_wheels(self, left, right):
        self.wheels.append((left, right))


def test_interface_is_abstract():
    with pytest.raises(TypeError):
        RobotEnvironment()


def test_unimplemented_driver_hooks_fail_loudly_and_close_is_safe():
    env = HardwareEnvironment(CONFIG)
    with pytest.raises(NotImplementedError):
        env.read_encoders()
    env.close()


def test_step_converts_body_velocity_to_wheel_speeds():
    env = FakeHardware()
    env.step(0.1, 0.0)
    assert env.wheels[-1] == pytest.approx((0.1 / R, 0.1 / R))
    env.step(0.1, 1.0)
    assert env.wheels[-1] == pytest.approx(((0.1 - B / 2) / R, (0.1 + B / 2) / R))


def test_step_clips_to_robot_limits():
    env = FakeHardware()
    env.step(10.0, 0.0)
    assert env.wheels[-1][0] == pytest.approx(CONFIG.robot.max_linear_vel / R)


def test_step_holds_the_control_period():
    env = FakeHardware()
    env.step(0.0, 0.0)
    t0 = time.monotonic()
    env.step(0.0, 0.0)
    assert time.monotonic() - t0 >= 0.9 * CONFIG.sim.control_dt


def test_emergency_stop_sends_zero_wheel_speeds():
    env = FakeHardware()
    env.step(0.1, 0.0)
    env.emergency_stop()
    assert env.wheels[-1] == (0.0, 0.0)


def test_sensor_age_reaches_the_sensor_frame():
    env = FakeHardware(lidar_age=0.3)
    frame = Sensing(env, camera_period=None).read()
    assert 0.29 <= frame.lidar_age < 1.0


# ------------------------------------------------------------- MuJoCo adapter
@pytest.fixture
def sim(tmp_path):
    pytest.importorskip("mujoco")
    from simulation import Simulation
    s = Simulation(replace(CONFIG, realism=RealismConfig(enabled=False)), scene_dir=tmp_path, seed=0)
    yield s
    s.close()


def test_mujoco_environment_exposes_no_ground_truth(sim):
    env = MuJoCoEnvironment(sim)
    assert not hasattr(env, "ground_truth") and not hasattr(env, "collided")


def test_mujoco_environment_forwards_commands_and_sensors(sim):
    env = MuJoCoEnvironment(sim)
    x0 = sim.ground_truth()[0]
    for _ in range(50):
        env.step(0.2, 0.0)
    assert sim.ground_truth()[0] - x0 == pytest.approx(0.2, abs=0.01)
    assert env.time == pytest.approx(sim.time)
    assert env.read_encoders().shape == (2,) and env.read_lidar().shape == (N,)
    assert env.sensor_ages() == (0.0, 0.0)