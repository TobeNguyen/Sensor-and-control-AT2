# Including libraries
from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np
import cv2

from config import CONFIG, Config

LIDAR_HEIGHT = 0.10    # [m] height of the LiDAR plane
CAMERA_HEIGHT = 0.15   # [m] height of the camera and the marker centre
WALL_THICKNESS = 0.05  # [m] half thickness of the walls
WALL_HEIGHT = 0.5      # [m]
MARKER_FILE = "aruco_marker.png"

# TurtleBot3 Waffle visual meshes (STL in millimetres), placed as in the official URDF.
MESH_SUBDIR = "turtlebot3"
MESH_FILES = {
    "tb3_base": "waffle_base.stl",
    "tb3_left_tire": "left_tire.stl",
    "tb3_right_tire": "right_tire.stl",
    "tb3_lds": "lds.stl",
}
MESH_SCALE = "0.001 0.001 0.001"
BASE_MESH_POS = "-0.064 0 0.010"   # base_link is 0.010 above the footprint, mesh x offset -0.064
LDS_MESH_POS = "-0.064 0 0.132"    # base_scan is at (-0.064, 0, 0.122) above base_link

# Wrap an angle to [-pi, pi]
def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

# Number of data modules per side
def _aruco_modules(dict_name: str) -> int:
    match = re.match(r"DICT_(\d)X\d_", dict_name)
    if match is None:
        raise ValueError(f"Unsupported ArUco dictionary: {dict_name}")
    return int(match.group(1))

# Marker image with a 1-module white margin
def generate_marker_image(cfg: Config = CONFIG, px_per_module: int = 20) -> np.ndarray:
    n = _aruco_modules(cfg.camera.aruco_dict)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.camera.aruco_dict))
    marker = cv2.aruco.generateImageMarker(
        dictionary, cfg.camera.marker_id, (n + 2) * px_per_module)
    pad = px_per_module
    return cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

def _scene_directory(cfg: Config, directory: Path | None) -> Path:
    return Path(directory) if directory is not None else cfg.sim.model_path.parent

def meshes_available(directory: Path) -> bool:
    return all((Path(directory) / MESH_SUBDIR / f).is_file() for f in MESH_FILES.values())

# Visual mesh geoms attached to the robot body
def _waffle_visual_geoms(cfg: Config) -> str:
    wheel_y, wheel_z = cfg.robot.wheel_base / 2.0, cfg.robot.wheel_radius
    flags = 'contype="0" conaffinity="0" density="1"'
    return "\n".join([
        f'      <geom name="tb3_base_vis" type="mesh" mesh="tb3_base" pos="{BASE_MESH_POS}" '
        f'rgba="0.45 0.45 0.45 1" {flags}/>',
        f'      <geom name="tb3_wheel_left" type="mesh" mesh="tb3_left_tire" '
        f'pos="0 {wheel_y} {wheel_z}" rgba="0.15 0.15 0.15 1" {flags}/>',
        f'      <geom name="tb3_wheel_right" type="mesh" mesh="tb3_right_tire" '
        f'pos="0 {-wheel_y} {wheel_z}" rgba="0.15 0.15 0.15 1" {flags}/>',
        # LiDAR: bright red so it stands out
        f'      <geom name="tb3_lds_vis" type="mesh" mesh="tb3_lds" pos="{LDS_MESH_POS}" '
        f'rgba="0.9 0.1 0.1 1" {flags}/>',
        # Camera: body box and a blue lens, both BEHIND the camera origin (x = body_radius + 0.01)
        f'      <geom name="cam_body_vis" type="box" size="0.01 0.03 0.015" '
        f'pos="{cfg.robot.body_radius - 0.019} 0 {CAMERA_HEIGHT}" '
        f'rgba="0.05 0.05 0.05 1" {flags}/>',
        f'      <geom name="cam_lens_vis" type="cylinder" size="0.008 0.004" '
        f'pos="{cfg.robot.body_radius - 0.013} 0 {CAMERA_HEIGHT}" euler="0 1.5708 0" '
        f'rgba="0.1 0.3 1.0 1" {flags}/>',
        # Camera mount: horizontal arm from the robot towards the camera, plus a vertical post
        f'      <geom name="cam_arm_vis" type="box" size="0.0575 0.008 0.006" '
        f'pos="0.1375 0 {CAMERA_HEIGHT}" rgba="0.3 0.3 0.3 1" {flags}/>',
        f'      <geom name="cam_post_vis" type="box" size="0.008 0.012 0.05" '
        f'pos="0.10 0 {CAMERA_HEIGHT - 0.05}" rgba="0.3 0.3 0.3 1" {flags}/>',
    ])

# Build the MJCF scene string from the configuration
def build_scene_xml(cfg: Config = CONFIG, use_meshes: bool = False) -> str:
    env, rob, cam, sim = cfg.env, cfg.robot, cfg.camera, cfg.sim
    hx, hy = env.room_half_size
    t, h = WALL_THICKNESS, WALL_HEIGHT

    n = _aruco_modules(cam.aruco_dict)
    # Black marker is marker_size wide; the white margin adds one module per side.
    panel_half = cam.marker_size * (n + 2) / (n + 2) * (n + 4) / (n + 2) / 2.0
    dx, dy, dyaw = env.dock_pose

    obstacles = "\n".join(
        f'    <geom name="obstacle_{i}" type="cylinder" size="{r} {h / 2}" '
        f'pos="{ox} {oy} {h / 2}" rgba="0.8 0.3 0.2 1"/>'
        for i, (ox, oy, r) in enumerate(env.obstacles))

    if use_meshes:
        compiler = f'<compiler angle="radian" texturedir="." meshdir="{MESH_SUBDIR}"/>'
        mesh_assets = "\n".join(
            f'    <mesh name="{name}" file="{file}" scale="{MESH_SCALE}"/>'
            for name, file in MESH_FILES.items())
        # Explicit inertia (same as the cylinder), so the visual meshes never change the physics.
        body_inertial = '<inertial pos="0 0 0.1" mass="1" diaginertia="0.0142 0.0142 0.0242"/>'
        chassis_rgba = "0.2 0.4 0.8 0"          # collision cylinder stays, but invisible
        visual_geoms = _waffle_visual_geoms(cfg)
    else:
        compiler = '<compiler angle="radian" texturedir="."/>'
        mesh_assets = body_inertial = visual_geoms = ""
        chassis_rgba = "0.2 0.4 0.8 1"

    return f"""<mujoco model="track_a_scene">
    {compiler}
    <option timestep="{sim.dt}" gravity="0 0 -9.81"/>
    <visual>
    <global offwidth="{cam.width}" offheight="{cam.height}"/>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    </visual>
    <asset>
    <texture name="marker_tex" type="2d" file="{MARKER_FILE}"/>
    <material name="marker_mat" texture="marker_tex" specular="0" shininess="0"/>
{mesh_assets}
    </asset>
    <worldbody>
    <geom name="floor" type="plane" size="{hx + 1} {hy + 1} 0.1" rgba="0.85 0.85 0.85 1"/>
    <geom name="wall_east" type="box" size="{t} {hy + 2 * t} {h / 2}" pos="{hx + t} 0 {h / 2}" rgba="0.5 0.5 0.6 1"/>
    <geom name="wall_west" type="box" size="{t} {hy + 2 * t} {h / 2}" pos="{-(hx + t)} 0 {h / 2}" rgba="0.5 0.5 0.6 1"/>
    <geom name="wall_north" type="box" size="{hx} {t} {h / 2}" pos="0 {hy + t} {h / 2}" rgba="0.5 0.5 0.6 1"/>
    <geom name="wall_south" type="box" size="{hx} {t} {h / 2}" pos="0 {-(hy + t)} {h / 2}" rgba="0.5 0.5 0.6 1"/>
{obstacles}
    <geom name="dock_marker" type="box" size="0.005 {panel_half} {panel_half}"
        pos="{dx} {dy} {CAMERA_HEIGHT}" euler="0 0 {dyaw}" material="marker_mat"/>
    <body name="base" pos="0 0 0">
        {body_inertial}
        <joint name="x" type="slide" axis="1 0 0"/>
        <joint name="y" type="slide" axis="0 1 0"/>
        <joint name="yaw" type="hinge" axis="0 0 1"/>
        <geom name="chassis" type="cylinder" size="{rob.body_radius} 0.08" pos="0 0 0.1"
            mass="1" rgba="{chassis_rgba}"/>
{visual_geoms}
        <camera name="front_cam" pos="{rob.body_radius + 0.01} 0 {CAMERA_HEIGHT}"
            xyaxes="0 -1 0 0 0 1" fovy="{cam.fovy_deg}"/>
    </body>
    </worldbody>
</mujoco>
"""

# Write the scene XML and marker texture
def write_scene_files(cfg: Config = CONFIG, directory: Path | None = None,
                    use_meshes: bool = False) -> Path:
    directory = _scene_directory(cfg, directory)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / MARKER_FILE), generate_marker_image(cfg))
    xml_path = directory / cfg.sim.model_path.name
    xml_path.write_text(build_scene_xml(cfg, use_meshes))
    return xml_path

class Simulation:
    def __init__(self, cfg: Config = CONFIG, seed: int | None = None,
                scene_dir: Path | None = None):
        self.cfg = cfg

        # Use the TurtleBot3 meshes (visual only) when available; fall back to the cylinder.
        use_meshes = meshes_available(_scene_directory(cfg, scene_dir))
        self.model_path = write_scene_files(cfg, scene_dir, use_meshes)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        except ValueError as exc:
            if not use_meshes:
                raise
            print(f"Warning: TurtleBot3 meshes could not be loaded ({exc}); using the cylinder.")
            use_meshes = False
            self.model_path = write_scene_files(cfg, scene_dir, use_meshes=False)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.uses_meshes = use_meshes

        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)

        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._substeps = int(round(cfg.sim.control_dt / cfg.sim.dt))
        self._renderer = None
        self._encoders = np.zeros(2)  # cumulative [left, right] wheel angle [rad]
        self.lidar_angles = np.linspace(
            0.0, 2.0 * math.pi, cfg.lidar.num_beams, endpoint=False)
        self.reset()

    # Reset time, encoders and robot pose
    def reset(self, pose: tuple | None = None) -> None:
        x, y, theta = self.cfg.env.start_pose if pose is None else pose
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [x, y, theta]
        self.data.qvel[:] = 0.0
        self._encoders[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def time_exceeded(self) -> bool:
        return self.time >= self.cfg.sim.max_time

    @property
    def collided(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._base_id in (self.model.geom_bodyid[c.geom1],
                                self.model.geom_bodyid[c.geom2]):
                return True
        return False
    # True pose
    def ground_truth(self) -> np.ndarray:
        x, y, theta = self.data.qpos[:3]
        return np.array([x, y, wrap_angle(theta)])
    
    # Apply v (m/s) and w (rad/s) for one control period
    def step(self, v: float, w: float) -> None:
        rob = self.cfg.robot
        v = float(np.clip(v, -rob.max_linear_vel, rob.max_linear_vel))
        w = float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel))
        dt = self.cfg.sim.dt
        dphi = np.zeros(2)

        for _ in range(self._substeps):
            theta = self.data.qpos[2]
            self.data.qvel[0] = v * math.cos(theta)
            self.data.qvel[1] = v * math.sin(theta)
            self.data.qvel[2] = w
            mujoco.mj_step(self.model, self.data)

            # Wheel increments from the actual body velocity after the step
            theta = self.data.qpos[2]
            v_act = self.data.qvel[0] * math.cos(theta) + self.data.qvel[1] * math.sin(theta)
            w_act = self.data.qvel[2]
            dphi[0] += (v_act - w_act * rob.wheel_base / 2.0) / rob.wheel_radius * dt
            dphi[1] += (v_act + w_act * rob.wheel_base / 2.0) / rob.wheel_radius * dt

        dphi += self.rng.normal(0.0, rob.encoder_noise_std, 2)
        self._encoders += dphi
        mujoco.mj_forward(self.model, self.data)

    # Cumulative wheal angles
    def read_encoders(self) -> np.ndarray:
        return self._encoders.copy()
    
    # Ranges for each beam
    def read_lidar(self) -> np.ndarray:
        lid = self.cfg.lidar
        x, y, theta = self.data.qpos[:3]
        origin = np.array([x, y, LIDAR_HEIGHT])
        ranges = np.full(lid.num_beams, lid.range_max)
        geomid = np.zeros(1, dtype=np.int32)

        for i, a in enumerate(self.lidar_angles):
            direction = np.array([math.cos(theta + a), math.sin(theta + a), 0.0])
            d = mujoco.mj_ray(self.model, self.data, origin, direction,
                            None, 1, self._base_id, geomid)
            if 0.0 <= d < lid.range_max:
                ranges[i] = d

        hit = ranges < lid.range_max
        ranges[hit] += self.rng.normal(0.0, lid.noise_std, int(hit.sum()))
        return np.clip(ranges, lid.range_min, lid.range_max)
    
    # RGB image from the front camera
    def read_camera(self) -> np.ndarray:
        if self._renderer is None:
            cam = self.cfg.camera
            self._renderer = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
        self._renderer.update_scene(self.data, camera="front_cam")
        return self._renderer.render().copy()
    
    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "Simulation":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

if __name__ == "__main__":
    # Quick manual check: drive in a circle and print the true pose.
    with Simulation() as sim:
        print("Scene written to:", sim.model_path, "| TurtleBot3 meshes:", sim.uses_meshes)
        for step in range(250):  # 5 s at 50 Hz
            sim.step(0.15, 0.4)
            if step % 50 == 0:
                x, y, theta = sim.ground_truth()
                print(f"t={sim.time:4.1f}s  x={x:6.2f}  y={y:6.2f}  theta={theta:6.2f}")
        print("Collided:", sim.collided)
        print("LiDAR min range:", sim.read_lidar().min())