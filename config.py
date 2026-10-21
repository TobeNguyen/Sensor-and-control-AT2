# Including libraries
from dataclasses import dataclass, field
from pathlib import Path
import math
import numpy as np
from typing import Optional
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"

@dataclass(frozen=True)
class SimConfig:
    model_path: Path = ASSETS_DIR / "turtlebot3_scene.xml"
    dt: float = 0.002           # physics step [s]
    control_dt: float = 0.02    # control / EKF period [s] (50 Hz)
    max_time: float = 180.0     # time limit per trial [s]
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
    view_noise_std: float = 0.03     # [rad] std of the marker view angle (third camera measurement)
    forward_offset: Optional[float] = None  # measured camera position ahead of the robot centre [m]; None = nominal
    lateral_offset: Optional[float] = None  # measured camera offset to the robot's left [m]; None = nominal (0)
    yaw: Optional[float] = None             # measured camera yaw relative to the robot heading [rad]; None = nominal (0)

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
    target_distance: float = 0.55   # stop this far from marker [m]
    distance_tolerance: float = 0.05  # [m]
    bearing_tolerance: float = 0.08   # [rad]
    max_search_time: float = 10.0     # give up if marker not seen [s]

@dataclass(frozen=True)
class EnvConfig:
    room_half_size: tuple = (2.0, 2.0)  # room spans [-2, 2] in x and y [m]
    start_pose: tuple = (0.0, 0.0, 0.0)  # x, y, theta
    # Circular obstacles: (x, y, radius)
    obstacles: tuple = (
        (0.8, 0.42, 0.2),
        (-0.9, 0.6, 0.2),
        (-0.6, -0.9, 0.2),
    )
    waypoints: tuple = (
        (-1.0, -1.2),
        (0.6, -1.2),
    )
    # Charging station marker: x, y, yaw (marker faces -x direction)
    dock_pose: tuple = (1.9, 1.0, math.pi)

@dataclass(frozen=True)
class TagConfig:
    fixed_tags: tuple = (
        (-1.95, -1.0, 0.0),
        (0.0, 1.95, -math.pi / 2.0),
        (1.95, -1.0, math.pi),
    )
    standoff: float = 0.9    # [m] recovery waypoint distance in front of a tag
    lost_std: float = 0.15   # [m] sqrt(trace(P_xy)) above this = robot is lost
    spin_omega: float = 0.7  # [rad/s] in-place rotation while looking for a tag
    use_survey: bool = True       # learn the tag poses at the start pose instead of reading them from the config
    survey_omega: float = 0.4     # [rad/s] in-place rotation while scanning
    survey_angle: float = 2.0 * math.pi # [rad] total rotation of the survey (half a turn from the start heading)
    survey_min_obs: int = 3       # observations needed before a tag is added to the learned map
    survey_max_range: float = 2.5 # [m] ignore tag observations farther than this (too noisy for the map)
    survey_max_time: float = 60.0 # [s] stop scanning after this
    approach_dist: float = 1.0    # [m] map-follow target in front of the dock
    dock_first: bool = True      # True: goal = learned dock approach point (no fixed waypoints)
    dock_max_range: float = 3.0   # [m] max range accepted for the dock observation during the scan

@dataclass(frozen=True)
class RealismConfig:
    enabled: bool = True
    # World differs from what the robot believes (sampled per seed, uniform +-)
    dock_shift_xy: tuple = (0.03, 0.25)     # [m] dock moved along x / along the wall (y)
    dock_yaw_err: float = math.radians(5.0) # [rad]
    cam_fovy_err: float = 0.03              # relative error of the true field of view
    cam_offset_err: float = 0.015           # [m] true camera position vs believed
    wheel_radius_err: float = 0.02          # relative odometry scale error
    wheel_base_err: float = 0.02            # relative
    slip_v: float = 0.02                    # max fraction the body moves less than the wheels
    slip_w: float = 0.05                    # rotation slips more
    # Actuators
    max_lin_acc: float = 0.5                # [m/s^2]
    max_ang_acc: float = 3.0                # [rad/s^2]
    deadband_v: float = 0.01                # [m/s] smaller commands do not move the robot
    deadband_w: float = 0.03                # [rad/s]
    # Sensors
    camera_delay: float = 0.15              # [s] (quantised to the 0.1 s camera period)
    lidar_delay_steps: int = 2              # control steps
    frame_drop: float = 0.05                # fraction of camera frames lost
    exposure: float = 0.02                  # [s] motion blur when rotating
    image_noise: float = 3.0                # gray levels
    brightness: tuple = (0.7, 1.2)          # image gain per run
    lidar_dropout: float = 0.03             # fraction of beams lost
    lidar_noise_range_gain: float = 0.5     # sigma = noise_std * (1 + gain * range)

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
    tags: TagConfig = field(default_factory=TagConfig)
    realism: RealismConfig = field(default_factory=RealismConfig)

    # EKF measurement noise for [range, bearing] from the camera
    def R_camera(self) -> np.ndarray:
        return np.diag([self.camera.range_noise_std ** 2,
                        self.camera.bearing_noise_std ** 2])
    
    # Known markers {marker_id: (x, y, yaw)}: dock (marker_id) + extra tags
    def tag_table(self) -> dict:
        mid = int(self.camera.marker_id)
        table = {mid: tuple(float(v) for v in self.env.dock_pose)}
        for k, (x, y, yaw) in enumerate(self.tags.fixed_tags):
            table[mid + 1 + k] = (float(x), float(y), float(yaw))
        return table
    
CONFIG = Config()