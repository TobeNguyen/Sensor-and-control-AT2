# Including libraries
from __future__ import annotations

import math
import re
from pathlib import Path
from collections import deque
from dataclasses import replace
import mujoco
import numpy as np
import cv2
import os
import sys
from contextlib import contextmanager
from typing import Optional

from config import CONFIG, Config

LIDAR_HEIGHT = 0.10    # [m] height of the LiDAR plane
CAMERA_HEIGHT = 0.15   # [m] height of the camera and the marker centre
WALL_THICKNESS = 0.05  # [m] half thickness of the walls
WALL_HEIGHT = 0.5      # [m]
MARKER_FILE = "aruco_marker.png"
CHESSBOARD_FILE = "chessboard.png"
CHESSBOARD_PATTERN = (9, 6)   # inner corners (cols, rows)
CHESSBOARD_SQUARE = 0.04      # [m]
FLOOR_SQUARE = 0.25   # [m] side of one floor square

# Board position on the floor: near the south-west corner, away from the start and the dock
def chessboard_pose(cfg: Config = CONFIG) -> tuple[float, float]:
    hx, hy = cfg.env.room_half_size
    return (-hx + 0.5, -hy + 0.4)

# Chessboard texture with a 1-square white margin (same layout as render_chessboard_views)
def generate_chessboard_image(pattern: tuple = CHESSBOARD_PATTERN, px: int = 60) -> np.ndarray:
    cols, rows = pattern
    board = np.full(((rows + 3) * px, (cols + 3) * px), 255, dtype=np.uint8)
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2 == 0:
                board[(j + 1) * px:(j + 2) * px, (i + 1) * px:(i + 2) * px] = 0
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

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

# Marker image with a 1-module white margin (marker_id=None -> the dock marker)
def generate_marker_image(cfg: Config = CONFIG, px_per_module: int = 20,
                        marker_id: int | None = None) -> np.ndarray:
    n = _aruco_modules(cfg.camera.aruco_dict)
    marker_id = cfg.camera.marker_id if marker_id is None else marker_id
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.camera.aruco_dict))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, (n + 2) * px_per_module)
    pad = px_per_module
    return cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

# Texture file of a marker: the dock keeps MARKER_FILE, extra tags get one file per id
def marker_filename(marker_id: int, cfg: Config = CONFIG) -> str:
    return MARKER_FILE if marker_id == cfg.camera.marker_id else f"aruco_marker_{marker_id}.png"

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
def build_scene_xml(cfg: Config = CONFIG, use_meshes: bool = False,
                    obstacles: list[tuple] | None = None,
                    cam_offset: float | None = None) -> str:
    env, rob, cam, sim = cfg.env, cfg.robot, cfg.camera, cfg.sim
    cam_x = rob.body_radius + 0.01 if cam_offset is None else cam_offset
    hx, hy = env.room_half_size
    t, h = WALL_THICKNESS, WALL_HEIGHT

    n = _aruco_modules(cam.aruco_dict)
    # Black marker is marker_size wide; the white margin adds one module per side.
    panel_half = cam.marker_size * (n + 2) / (n + 2) * (n + 4) / (n + 2) / 2.0
    dx, dy, dyaw = env.dock_pose

    obstacle_list = cfg.env.obstacles if obstacles is None else obstacles
    obstacles_xml = "\n".join(
        f'    <geom name="obstacle_{i}" type="cylinder" size="{r} {h / 2}" '
        f'pos="{ox} {oy} {h / 2}" rgba="0.8 0.3 0.2 1"/>'
        for i, (ox, oy, r) in enumerate(obstacle_list))

    # Extra checkpoint tags (every id in the tag table except the dock): one texture, one material
    # and one thin panel each, placed like the dock panel (box normal = tag yaw).
    tag_assets, tag_geoms = [], []
    for tid, (tx, ty, tyaw) in sorted(cfg.tag_table().items()):
        if tid == cam.marker_id:
            continue
        tag_assets.append(
            f'    <texture name="tag_tex_{tid}" type="2d" file="{marker_filename(tid, cfg)}"/>\n'
            f'    <material name="tag_mat_{tid}" texture="tag_tex_{tid}" specular="0" shininess="0"/>')
        tag_geoms.append(
            f'    <geom name="tag_marker_{tid}" type="box" size="0.005 {panel_half} {panel_half}"\n'
            f'        pos="{tx} {ty} {CAMERA_HEIGHT}" euler="0 0 {tyaw}" material="tag_mat_{tid}"/>')
    tag_assets_xml = "\n".join(tag_assets)
    tag_geoms_xml = "\n".join(tag_geoms)

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

    # Checkerboard floor: one texture tile holds 2x2 squares, so repeat = 1 / (2 * square) per metre
    floor_repeat = 1.0 / (2.0 * FLOOR_SQUARE)

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
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.35 0.35 0.35" rgb2="0.8 0.8 0.8"
        width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true"
        texrepeat="{floor_repeat} {floor_repeat}" specular="0" shininess="0" reflectance="0"/>
{tag_assets_xml}
{mesh_assets}
    </asset>
    <worldbody>
    <geom name="floor" type="plane" size="{hx + 1} {hy + 1} 0.1" material="floor_mat"/>
    <geom name="wall_east" type="box" size="{t} {hy + 2 * t} {h / 2}" pos="{hx + t} 0 {h / 2}" rgba="0.5 0.5 0.6 1"/>
    <geom name="wall_west" type="box" size="{t} {hy + 2 * t} {h / 2}" pos="{-(hx + t)} 0 {h / 2}" rgba="0.5 0.5 0.6 1"/>
    <geom name="wall_north" type="box" size="{hx} {t} {h / 2}" pos="0 {hy + t} {h / 2}" rgba="0.5 0.5 0.6 1"/>
    <geom name="wall_south" type="box" size="{hx} {t} {h / 2}" pos="0 {-(hy + t)} {h / 2}" rgba="0.5 0.5 0.6 1"/>
{obstacles_xml}
    <geom name="dock_marker" type="box" size="0.005 {panel_half} {panel_half}"
        pos="{dx} {dy} {CAMERA_HEIGHT}" euler="0 0 {dyaw}" material="marker_mat"/>
{tag_geoms_xml}
    <body name="base" pos="0 0 0">
        {body_inertial}
        <joint name="x" type="slide" axis="1 0 0"/>
        <joint name="y" type="slide" axis="0 1 0"/>
        <joint name="yaw" type="hinge" axis="0 0 1"/>
        <geom name="chassis" type="cylinder" size="{rob.body_radius} 0.08" pos="0 0 0.1"
            mass="1" rgba="{chassis_rgba}"/>
{visual_geoms}
        <camera name="front_cam" pos="{cam_x} 0 {CAMERA_HEIGHT}"
            xyaxes="0 -1 0 0 0 1" fovy="{cam.fovy_deg}"/>
    </body>
    </worldbody>
</mujoco>
"""

# Write the scene XML and the marker textures (dock + extra tags)
def write_scene_files(cfg: Config = CONFIG, directory: Path | None = None,
                    use_meshes: bool = False,
                    obstacles: list[tuple] | None = None,
                    cam_offset: float | None = None) -> Path:
    directory = _scene_directory(cfg, directory)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / MARKER_FILE), generate_marker_image(cfg))
    cv2.imwrite(str(directory / CHESSBOARD_FILE), generate_chessboard_image())
    for tid in cfg.tag_table():
        if tid != cfg.camera.marker_id:
            cv2.imwrite(str(directory / marker_filename(tid, cfg)),
                        generate_marker_image(cfg, marker_id=tid))
    xml_path = directory / cfg.sim.model_path.name
    xml_path.write_text(build_scene_xml(cfg, use_meshes, obstacles, cam_offset))
    return xml_path

class Simulation:
    def __init__(self, cfg: Config = CONFIG, seed: int | None = None,
                scene_dir: Path | None = None):
        self.cfg = cfg
        rl = cfg.realism
        self.real = rl.enabled
        base_seed = cfg.sim.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(base_seed)

        # Hidden world (what is REALLY there) vs the config (what the robot believes)
        self.world_cfg, self.cam_offset = self._sample_world(np.random.default_rng(base_seed + 7919))

        # Obstacles: the 3 fixed obstacles from the config (the robot only sees them through the LiDAR)
        self.obstacles = list(cfg.env.obstacles)

        # Use the TurtleBot3 meshes (visual only) when available; fall back to the cylinder.
        use_meshes = meshes_available(_scene_directory(cfg, scene_dir))
        self.model_path = write_scene_files(self.world_cfg, scene_dir, use_meshes,
                                            self.obstacles, self.cam_offset)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        except ValueError as exc:
            if not use_meshes:
                raise
            print(f"Warning: TurtleBot3 meshes could not be loaded ({exc}); using the cylinder.")
            use_meshes = False
            self.model_path = write_scene_files(self.world_cfg, scene_dir, False,
                                                self.obstacles, self.cam_offset)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.uses_meshes = use_meshes

        self.data = mujoco.MjData(self.model)

        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._substeps = int(round(cfg.sim.control_dt / cfg.sim.dt))
        self._renderer = None
        self._encoders = np.zeros(2)  # cumulative [left, right] wheel angle [rad]
        self._v = self._w = 0.0       # wheel speeds actually reached (acceleration limit)
        self._cam_buf: deque = deque(maxlen=8)                          # (time, image)
        self._lidar_buf: deque = deque(maxlen=max(1, rl.lidar_delay_steps + 1))
        self.lidar_angles = np.linspace(
            0.0, 2.0 * math.pi, cfg.lidar.num_beams, endpoint=False)
        self.reset()

    # Sample the hidden differences between the real world and the config (per seed)
    def _sample_world(self, wr: np.random.Generator):
        cfg, rl = self.cfg, self.cfg.realism
        self.r_true, self.b_true = cfg.robot.wheel_radius, cfg.robot.wheel_base
        self.slip_v = self.slip_w = 0.0
        self.gain = 1.0
        if not rl.enabled:
            return cfg, None
        u = lambda a: float(wr.uniform(-a, a))
        dx, dy, dyaw = cfg.env.dock_pose
        hy = cfg.env.room_half_size[1]
        dock = (dx + u(rl.dock_shift_xy[0]),
                float(np.clip(dy + u(rl.dock_shift_xy[1]), -hy + 0.6, hy - 0.6)),
                dyaw + u(rl.dock_yaw_err))
        cam = replace(cfg.camera, fovy_deg=cfg.camera.fovy_deg * (1.0 + u(rl.cam_fovy_err)))
        world = replace(cfg, camera=cam, env=replace(cfg.env, dock_pose=dock))
        self.r_true = cfg.robot.wheel_radius * (1.0 + u(rl.wheel_radius_err))
        self.b_true = cfg.robot.wheel_base * (1.0 + u(rl.wheel_base_err))
        self.slip_v = float(wr.uniform(0.0, rl.slip_v))
        self.slip_w = float(wr.uniform(0.0, rl.slip_w))
        self.gain = float(wr.uniform(*rl.brightness))
        return world, cfg.robot.body_radius + 0.01 + u(rl.cam_offset_err)

    # Reset time, encoders and robot pose
    def reset(self, pose: tuple | None = None) -> None:
        x, y, theta = self.cfg.env.start_pose if pose is None else pose
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [x, y, theta]
        self.data.qvel[:] = 0.0
        self._encoders[:] = 0.0
        self._v = self._w = 0.0
        self._cam_buf.clear()
        self._lidar_buf.clear()
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

    # True pose (EVALUATION ONLY)
    def ground_truth(self) -> np.ndarray:
        x, y, theta = self.data.qpos[:3]
        return np.array([x, y, wrap_angle(theta)])

    # Cumulative wheel angles [left, right] in rad
    def read_encoders(self) -> np.ndarray:
        return self._encoders.copy()

    @staticmethod
    def _ramp(cur: float, target: float, max_step: float) -> float:
        return cur + float(np.clip(target - cur, -max_step, max_step))

    # Apply v (m/s) and w (rad/s) for one control period
    def step(self, v: float, w: float) -> None:
        rob = self.cfg.robot
        rl = self.cfg.realism
        v = float(np.clip(v, -rob.max_linear_vel, rob.max_linear_vel))
        w = float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel))
        r, b = rob.wheel_radius, rob.wheel_base
        v_body, w_body = v, w
        if self.real:
            dtc = self.cfg.sim.control_dt
            v = 0.0 if abs(v) < rl.deadband_v else v           # motor deadband
            w = 0.0 if abs(w) < rl.deadband_w else w
            v = self._ramp(self._v, v, rl.max_lin_acc * dtc)   # acceleration limit
            w = self._ramp(self._w, w, rl.max_ang_acc * dtc)
            self._v, self._w = v, w
            v_body, w_body = v * (1.0 - self.slip_v), w * (1.0 - self.slip_w)   # wheel slip
            r, b = self.r_true, self.b_true                    # true wheel geometry
        dt = self.cfg.sim.dt
        dphi = np.zeros(2)

        for _ in range(self._substeps):
            theta = self.data.qpos[2]
            self.data.qvel[0] = v_body * math.cos(theta)
            self.data.qvel[1] = v_body * math.sin(theta)
            self.data.qvel[2] = w_body
            mujoco.mj_step(self.model, self.data)
            # Encoders measure the WHEELS (true geometry), not the body: slip is invisible to them
            dphi[0] += (v - w * b / 2.0) / r * dt
            dphi[1] += (v + w * b / 2.0) / r * dt

        dphi += self.rng.normal(0.0, rob.encoder_noise_std, 2)
        self._encoders += dphi
        mujoco.mj_forward(self.model, self.data)

    # Ranges for each beam
    def read_lidar(self) -> np.ndarray:
        lid = self.cfg.lidar
        rl = self.cfg.realism
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
        gain = rl.lidar_noise_range_gain if self.real else 0.0
        ranges[hit] += (self.rng.normal(0.0, 1.0, int(hit.sum()))
                        * lid.noise_std * (1.0 + gain * ranges[hit]))
        if self.real:
            ranges[self.rng.random(lid.num_beams) < rl.lidar_dropout] = lid.range_max
        out = np.clip(ranges, lid.range_min, lid.range_max)
        if not self.real:
            return out
        self._lidar_buf.append(out)               # the scan arrives a few steps late
        return self._lidar_buf[0].copy()

    # Motion blur + brightness + sensor noise
    def _degrade(self, img: np.ndarray) -> np.ndarray:
        rl, cam = self.cfg.realism, self.cfg.camera
        out = img.astype(np.float32) * self.gain
        f = (cam.height / 2.0) / math.tan(math.radians(cam.fovy_deg) / 2.0)
        n = int(round(abs(self._w) * f * rl.exposure))          # blur length [px] while rotating
        if n >= 2:
            out = cv2.filter2D(out, -1, np.ones((1, n), np.float32) / n)
        out += self.rng.normal(0.0, rl.image_noise, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    # RGB image from the front camera
    def read_camera(self) -> np.ndarray:
        if self._renderer is None:
            cam = self.cfg.camera
            self._renderer = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
        self._renderer.update_scene(self.data, camera="front_cam")
        img = self._renderer.render().copy()
        if not self.real:
            return img
        rl = self.cfg.realism
        self._cam_buf.append((self.time, self._degrade(img)))
        if self.rng.random() < rl.frame_drop:                   # lost frame -> nothing detected
            return np.zeros_like(img)
        ready = [im for t, im in self._cam_buf if t <= self.time - rl.camera_delay + 1e-9]
        return (ready[-1] if ready else self._cam_buf[0][1]).copy()   # latency
    
    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "Simulation":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

# Render views of a chessboard with the simulated camera: the same procedure as on the real
# robot (hold a board at different distances and tilts in front of the camera).
def render_chessboard_views(cfg: Config = CONFIG, n_views: int = 20, pattern: tuple = (9, 6),
                            square: float = 0.04, seed: int = 0) -> list[np.ndarray]:
    import tempfile
    cols, rows = pattern
    nx, ny = cols + 3, rows + 3                  # (cols+1) x (rows+1) squares + 1 white square margin each side
    px = 60                                      # texture pixels per square
    board = np.full((ny * px, nx * px), 255, dtype=np.uint8)
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2 == 0:
                board[(j + 1) * px:(j + 2) * px, (i + 1) * px:(i + 2) * px] = 0
    hx, hy = nx * square / 2.0, ny * square / 2.0
    xml = f"""<mujoco model="chessboard">
  <compiler texturedir="."/>
  <visual>
    <global offwidth="{cfg.camera.width}" offheight="{cfg.camera.height}" fovy="{cfg.camera.fovy_deg}"/>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="board_tex" type="2d" file="board.png"/>
    <material name="board_mat" texture="board_tex" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.01" rgba="0.5 0.5 0.55 1"/>
    <geom name="board" type="box" size="{hx} {hy} 0.002" material="board_mat"/>
  </worldbody>
</mujoco>"""
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cv2.imwrite(str(tmp / "board.png"), cv2.cvtColor(board, cv2.COLOR_GRAY2BGR))
        (tmp / "board.xml").write_text(xml)
        model = mujoco.MjModel.from_xml_path(str(tmp / "board.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=cfg.camera.height, width=cfg.camera.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        try:
            for _ in range(n_views):
                cam.lookat[:] = [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0]
                cam.distance = float(rng.uniform(0.6, 1.0))
                cam.azimuth = float(rng.uniform(0.0, 360.0))
                cam.elevation = float(rng.uniform(-80.0, -40.0))    # tilt of the board up to 50 deg
                renderer.update_scene(data, camera=cam)
                images.append(renderer.render().copy())
        finally:
            renderer.close()
    return images

# Live MuJoCo viewer (or nothing when disabled)
@contextmanager
def _viewer(sim: Simulation, enabled: bool):
    if not enabled:
        yield None
        return
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as handle:
        handle.cam.lookat[:] = [0.0, 0.0, 0.0]
        handle.cam.distance = 6.5
        handle.cam.azimuth = 90.0
        handle.cam.elevation = -75.0
        yield handle

# On macOS the live viewer needs mjpython; re-run this script under it
def _ensure_mjpython() -> None:
    if sys.platform != "darwin" or os.environ.get("_MJ_RELAUNCHED"):
        return
    mjpython = Path(sys.executable).parent / "mjpython"
    if not mjpython.exists():
        return
    os.environ["_MJ_RELAUNCHED"] = "1"
    # Launch via python so a broken shebang (spaces in the path) is bypassed
    os.execv(sys.executable, [sys.executable, str(mjpython), *sys.argv])

# Repeated trials with different noise seeds: success rate and mean +- std of the key metrics.
# run_fn = run_pipeline from main.py
def run_trials(run_fn, n: int, seed0: int = 0, cfg: Config = CONFIG, use_marker_cov: bool = False,
            marker_sigma_px: float = 0.5, scene_dir: Optional[Path] = None,
            use_lidar_map: bool = False, use_recovery: bool = False,
            block_sight: bool = True, calib_odometry: bool = False) -> int:
    from visualization import compute_report
    rows = []
    print(f"marker R: {'per-measurement' if use_marker_cov else 'fixed'} | sigma_px {marker_sigma_px}")
    print(f"{'seed':>5} {'result':>7} {'reason':>15} {'time':>6} {'pos RMSE':>9} "
        f"{'NEES':>6} {'NIS/dof':>8} {'det %':>6} {'rej':>4} {'rec':>4} {'dock lat':>9}")
    for i in range(n):
        res = run_fn(cfg, seed=seed0 + i, scene_dir=scene_dir,
                    dock_use_estimate=True, calib_odometry=calib_odometry,
                    marker_sigma_px=marker_sigma_px, use_marker_cov=use_marker_cov,
                    use_lidar_map=use_lidar_map, use_recovery=use_recovery)
        rep = compute_report(res, cfg)
        n_cam, n_det = res.camera_stats
        row = {"ok": res.success, "reason": res.reason, "time": res.time,
            "rmse": rep.get("position_rmse", float("nan")),
            "nees": rep.get("mean_nees", float("nan")),
            "nis": res.diag.get("nis_per_dof", float("nan")),
            "det": 100.0 * n_det / max(1, n_cam),
            "rej": res.diag.get("n_rejected", 0),
            "rec": res.diag.get("n_tag_recoveries", 0),
            "lat": res.metrics(res.world_cfg or cfg)["dock_lateral"]}
        rows.append(row)
        print(f"{seed0 + i:>5} {'OK' if res.success else 'FAIL':>7} {res.reason:>15} {row['time']:>6.1f} "
            f"{row['rmse']:>9.3f} {row['nees']:>6.2f} {row['nis']:>8.2f} {row['det']:>6.0f} "
            f"{row['rej']:>4} {row['rec']:>4} {row['lat']:>9.3f}", flush=True)

    ok = np.array([r["ok"] for r in rows])
    print(f"\nSuccess rate : {int(ok.sum())}/{n} ({100 * ok.mean():.0f} %)")
    for label, key in (("Position RMSE [m]", "rmse"), ("Mean NEES (~3)", "nees"),
                    ("Mean NIS/dof (~1)", "nis"), ("Marker accepted [%]", "det"),
                    ("Gate rejections", "rej"), ("Tag recoveries", "rec"),
                    ("Dock lateral [m]", "lat")):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"{label:<20}: {np.nanmean(v):.3f} +- {np.nanstd(v):.3f}")
    failures = {}
    for r in rows:
        if not r["ok"]:
            failures[r["reason"]] = failures.get(r["reason"], 0) + 1
    if failures:
        print("Failure causes:", failures)
    return 0 if ok.all() else 1

# Configurations compared by a plain `python main.py --suite` (edit this list to add / remove variants)
SUITE = (
    ("Final system",             dict(calib_odometry=True,  use_recovery=True)),
    ("Ablation: no calibration", dict(calib_odometry=False, use_recovery=True)),
)

# Run every configuration in SUITE for n seeds (headless) and print one comparison table.
# run_fn = run_pipeline from main.py
def run_suite(run_fn, n: int, seed0: int = 0, cfg: Config = CONFIG) -> int:
    from visualization import compute_report
    nan = float("nan")
    results = []
    for name, kw in SUITE:
        print(f"\n=== {name} | {n} trials ===", flush=True)
        rows, fails = [], {}
        for i in range(n):
            res = run_fn(cfg, seed=seed0 + i, dock_use_estimate=True, **kw)
            rep = compute_report(res, cfg)
            lat = res.metrics(res.world_cfg or cfg)["dock_lateral"]
            rows.append((float(res.success), res.time, rep.get("position_rmse", nan),
                        rep.get("mean_nees", nan), lat))
            if not res.success:
                fails[res.reason] = fails.get(res.reason, 0) + 1
            print(f"  seed {seed0 + i:>4}: {'OK  ' if res.success else 'FAIL'} {res.reason:<14} "
                f"t={res.time:6.1f}s  rmse={rows[-1][2]:.3f} m", flush=True)
        results.append((name, np.array(rows, dtype=float), fails))

    print("\n" + "=" * 86)
    print(f"{'configuration':<28}{'success':>9}{'time [s]':>10}{'RMSE [m]':>10}{'NEES (~3)':>11}{'dock lat [m]':>14}")
    print("-" * 86)
    for name, a, fails in results:
        print(f"{name:<28}{100 * a[:, 0].mean():>8.0f}%{np.nanmean(a[:, 1]):>10.1f}"
            f"{np.nanmean(a[:, 2]):>10.3f}{np.nanmean(a[:, 3]):>11.2f}{np.nanmean(a[:, 4]):>14.3f}")
        if fails:
            print(f"{'':<28}failures: {fails}")
    print("=" * 86)
    return 0

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