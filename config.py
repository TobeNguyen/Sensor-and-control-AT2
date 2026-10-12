# Including libraries
from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"

@dataclass(frozen=True)
class SimConfig:
    model_path: Path = ASSETS_DIR / "turtlebot3_scene.xml"
    dt: float = 0.002           # physics step [s]
    control_dt: float = 0.02    # control / EKF period [s] (50 Hz)
    max_time: float = 120.0     # time limit per trial [s]
    seed: int = 42              # for reproducible noise

@dataclass(frozen=True)
class RobotConfig:
    wheel_radius: float = 0.033      # [m]
    wheel_base: float = 0.287        # distance between wheels [m]
    body_radius: float = 0.22        # used for obstacle clearance [m]
    max_linear_vel: float = 0.26     # [m/s]
    max_angular_vel: float = 1.82    # [rad/s]
    encoder_noise_std: float = 0.002  # wheel angle noise per step [rad]

@dataclass(frozen=True)
class LidarConfig:
    num_beams: int = 360
    range_min: float = 0.12    # [m]
    range_max: float = 3.5     # [m]
    noise_std: float = 0.01    # [m]
    safe_distance: float = 0.35  # start avoiding obstacles below this [m]

@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fovy_deg: float = 60.0
    aruco_dict: str = "DICT_4X4_50"
    marker_id: int = 0
    marker_size: float = 0.15      # side length [m]
    range_noise_std: float = 0.02  # [m]
    bearing_noise_std: float = 0.01  # [rad]

@dataclass(frozen=True)
class EKFConfig:
    # State: [x, y, theta]
    init_pose_std: tuple = (0.05, 0.05, 0.05)   # [m, m, rad]
    process_noise_std: tuple = (0.01, 0.01, 0.01)  # per step [m, m, rad]

    def P0(self) -> np.ndarray:
        return np.diag(np.square(self.init_pose_std))

    def Q(self) -> np.ndarray:
        return np.diag(np.square(self.process_noise_std))
    
@dataclass(frozen=True)
class ControllerConfig:
    # LQR weights: state error [x, y, theta], input [v, w]
    q_diag: tuple = (10.0, 10.0, 1.0)
    r_diag: tuple = (1.0, 1.0)
    waypoint_tolerance: float = 0.10   # [m]

    def Q(self) -> np.ndarray:
        return np.diag(self.q_diag)

    def R(self) -> np.ndarray:
        return np.diag(self.r_diag)
    
@dataclass(frozen=True)
class DockingConfig:
    target_distance: float = 0.40   # stop this far from marker [m]
    distance_tolerance: float = 0.05  # [m]
    bearing_tolerance: float = 0.05   # [rad]
    max_search_time: float = 10.0     # give up if marker not seen [s]

@dataclass(frozen=True)
class EnvConfig:
    room_half_size: tuple = (2.0, 2.0)  # room spans [-2, 2] in x and y [m]
    start_pose: tuple = (-1.5, -1.5, 0.0)  # x, y, theta
    # Circular obstacles: (x, y, radius)
    obstacles: tuple = (
        (0.0, -0.5, 0.2),
        (0.8, 0.6, 0.2),
        (-0.8, 0.5, 0.2),
    )
    waypoints: tuple = (
        (-1.0, -1.2),
        (0.6, -1.2),
        (1.5, -0.1),
        (1.2, 1.0),
    )
    # Charging station marker: x, y, yaw (marker faces -x direction)
    dock_pose: tuple = (1.9, 1.0, math.pi)

@dataclass(frozen=True)
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    # EKF measurement noise for [range, bearing] from the camera
    def R_camera(self) -> np.ndarray:
        return np.diag([self.camera.range_noise_std ** 2,
                        self.camera.bearing_noise_std ** 2])
    
CONFIG = Config()