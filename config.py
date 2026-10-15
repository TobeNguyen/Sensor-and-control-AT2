# Including libraries
from dataclasses import dataclass, field
from pathlib import Path
import math
from functools import lru_cache
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
    view_noise_std: float = 0.03     # [rad] std of the marker view angle (third camera measurement)

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
class TagConfig:
    n_tags: int = 4          # number of extra checkpoint tags (ids marker_id+1 ... marker_id+n_tags); 0 = dock only
    inset: float = 0.05      # [m] distance of each tag panel from the wall
    standoff: float = 0.9    # [m] recovery waypoint distance in front of a tag
    lost_std: float = 0.15   # [m] sqrt(trace(P_xy)) above this = robot is lost
    spin_omega: float = 0.7  # [rad/s] in-place rotation while looking for a tag

# {marker_id: (x, y, yaw)}, yaw = direction of the marker normal (pointing into the room).
# The dock keeps marker_id. Extra tags are spread over the four walls, one per wall first:
#   tag 1: wall opposite to the dock, staggered (dock top right -> tag bottom left)
#   tag 2: side wall (middle)
#   tag 3: the dock wall itself, mirrored along the wall so it never overlaps the dock
#   tag 4: other side wall (middle)
# Tags beyond 4 go round the walls again, shifted along the wall.
@lru_cache(maxsize=None)
def _tag_table(marker_id: int, dock_pose: tuple, room_half: tuple, n_tags: int, inset: float) -> dict:
    table = {marker_id: dock_pose}
    if n_tags <= 0:
        return table
    if marker_id + n_tags > 49:                                  # DICT_4X4_50 has ids 0..49
        raise ValueError("n_tags too large for the ArUco dictionary (ids 0..49)")
    ax, ay = room_half[0] - inset, room_half[1] - inset
    corners = [(ax, -ay), (ax, ay), (-ax, ay), (-ax, -ay)]       # counter-clockwise
    yaws = (math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0)         # east, north, west, south wall
    sides = []                                                   # (start, unit direction, length, yaw)
    for i in range(4):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
        length = math.hypot(x1 - x0, y1 - y0)
        sides.append(((x0, y0), ((x1 - x0) / length, (y1 - y0) / length), length, yaws[i]))

    # Wall that holds the dock (nearest side) and the dock position along it
    dock_side, best, t_dock = 0, float("inf"), 0.0
    for i, ((x0, y0), (ux, uy), length, _) in enumerate(sides):
        t = min(max((dock_pose[0] - x0) * ux + (dock_pose[1] - y0) * uy, 0.0), length)
        dist = math.hypot(x0 + t * ux - dock_pose[0], y0 + t * uy - dock_pose[1])
        if dist < best:
            best, dock_side, t_dock = dist, i, t

    corner_margin = 0.3                                          # [m] keep tags away from corners
    dock_gap = 0.6                                               # [m] min distance to the dock panel
    wall_order = [(dock_side + 2) % 4, (dock_side + 1) % 4, dock_side, (dock_side + 3) % 4]
    rounds = (n_tags + 3) // 4                                   # tags per wall (at most)

    for k in range(n_tags):
        wall, r = wall_order[k % 4], k // 4
        (x0, y0), (ux, uy), length, yaw = sides[wall]
        lo, hi = corner_margin, length - corner_margin
        usable = hi - lo

        # Base position of the first tag on this wall
        if k % 4 == 0:                                           # opposite wall: dock mirrored through the centre
            t = (-dock_pose[0] - x0) * ux + (-dock_pose[1] - y0) * uy
        elif k % 4 == 2:                                         # dock wall: dock mirrored along the wall
            t = length - t_dock
        else:                                                    # side walls: middle
            t = length / 2.0
        t = min(max(t, lo), hi)
        if r > 0:                                                # later rounds: shift along the wall
            t = lo + (t - lo + r * usable / rounds) % usable
        if wall == dock_side and abs(t - t_dock) < dock_gap:     # never overlap the dock panel
            t = t_dock + dock_gap if t_dock + dock_gap <= hi else t_dock - dock_gap
        table[marker_id + 1 + k] = (x0 + t * ux, y0 + t * uy, yaw)
    return table

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

    # EKF measurement noise for [range, bearing] from the camera
    def R_camera(self) -> np.ndarray:
        return np.diag([self.camera.range_noise_std ** 2,
                        self.camera.bearing_noise_std ** 2])
    
    # Known markers {marker_id: (x, y, yaw)}: dock (marker_id) + extra tags
    def tag_table(self) -> dict:
        return dict(_tag_table(int(self.camera.marker_id),
                            tuple(float(v) for v in self.env.dock_pose),
                            tuple(float(v) for v in self.env.room_half_size),
                            int(self.tags.n_tags), float(self.tags.inset)))
    
CONFIG = Config()