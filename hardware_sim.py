# hardware_sim.py - test hardware.py (BACKUP runner) in MuJoCo under the REAL robot's conditions.
#   - runs exactly hardware.Mission + hardware.safety (the code that drives the real robot)
#   - simple test arena: walls, ONE box obstacle 20 x 10 x 20 cm partly in front of the dock,
#     dock in front, 2 tags on the side walls
#   - robot drawn with the same TurtleBot3 Waffle meshes as simulation.py (if the STL files are there)
#   - writes its OWN scene (results/hardware_sim/scene/...): simulation.py's XML / textures are never touched
#
# What is made "like the real robot":
#   camera   real K (fx, fy, cx, cy) + real LENS DISTORTION (image is distorted, then undistorted by Sensing
#            with the real calibration, exactly like on the robot), real mounting fwd / lat / yaw,
#            0.2 s delay with jitter, 10 Hz, frame drops (old image re-used), motion blur, noise, JPEG
#   calib    --calib-error: the true camera mounting / focal length differ a little from the calibration
#   odometry firmware converts v, w with the NOMINAL TB3 values, the body moves with the TRUE wheels,
#            odometry uses the CALIBRATED values (odometry_hw.json); wheel slip on turns; encoder noise
#            --calib-odom: run the same 1 m + 2 turns calibration in the sim (tape measure = ground truth)
#   motors   dead band, acceleration limits, 0.1 s command latency (WiFi), control loop jitter
#   LiDAR    5 Hz, mounted 6.4 cm behind the centre, noise, dropouts, returns < 0.30 m ignored
#
#   python hardware_sim.py                              one run (prints the arena to build)
#   python hardware_sim.py --viewer                     with the MuJoCo viewer (macOS: mjpython)
#   python hardware_sim.py --calib-odom --calib-error   everything
#   python hardware_sim.py --trials 10 --calib-error    10 seeds, success table
#   python hardware_sim.py --ideal                      no imperfections (debug the logic only)
#   python hardware_sim.py --video                      also save camera.mp4 (what the robot saw)
#   python hardware_sim.py --no-final                   stop at the camera DOCKED pose (no FINAL approach)
#
# FINAL approach (tested here first, hardware.py unchanged): after the camera says DOCKED (~0.85 m) the
# robot drives straight on (odometry heading hold) and stops by LiDAR at FINAL_RANGE from the dock wall.
# If hardware.py already has its own FINAL state (H.FINAL_APPROACH), that one is used instead.
#
# Live camera window (visualization.LiveCameraView): the image the robot REALLY processes (delayed,
# distorted, noisy), detected markers outlined, state / dock range / bearing, image centre.
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import time
from collections import deque
from dataclasses import replace

import numpy as np
import cv2
import mujoco

import hardware as H                                   # the real-robot code under test
import simulation as S                                 # helpers only (scene text, marker images)
from config import CONFIG, RESULTS_DIR
from sensing import Sensing, sector_min_range
from robot_model import wrap_angle

# ============================================================ test arena (BUILD THIS)
# Frame: origin = centre of the arena, +x towards the dock wall ("east"), +y to the left ("north").
ROOM_X, ROOM_Y = 3.0, 2.4                 # [m] inner size of the arena (x, y)
START = (-1.0, -0.3, math.pi / 2)         # robot centre (x, y, heading): starts facing NORTH, not the dock
DOCK = (ROOM_X / 2 - 0.02, 0.0, math.pi)  # dock marker centre on the east wall, facing the room (-x)
TAGS = ((0.0, ROOM_Y / 2 - 0.02, -math.pi / 2),    # tag on the north wall, facing south
        (0.0, -(ROOM_Y / 2 - 0.02), math.pi / 2))  # tag on the south wall, facing north
BOX_L, BOX_W, BOX_H = 0.10, 0.20, 0.20    # [m] box: depth along the dock axis, width of the front face, height
# Box BESIDE the docking lane: the robot drives past it and stops (0.85 m) closer to the dock than the box
BOX_FRONT = 1.05          # [m] marker -> box centre along the axis (box spans 1.00-1.10 m; AIM point 1.3 m)
BOX_GAP = 0.35            # [m] marker axis -> inner edge (> LAT_TOL 0.15 + half robot 0.15 + 0.05 margin)
BOX = (DOCK[0] + BOX_FRONT * math.cos(DOCK[2]) - (BOX_GAP + BOX_W / 2) * math.sin(DOCK[2]),
       DOCK[1] + BOX_FRONT * math.sin(DOCK[2]) + (BOX_GAP + BOX_W / 2) * math.cos(DOCK[2]),
    DOCK[2])                           # box centre (x, y) and yaw (long side along the marker axis)
CAM_Z = S.CAMERA_HEIGHT                   # [m] camera lens height (measure it on the robot); markers at S.CAMERA_HEIGHT

# ============================================================ real camera (fallback if no calibration files)
CAM_FWD, CAM_LAT, CAM_YAW = 0.234, 0.054, math.radians(-5.08)
CAM_FY = 1304.0                           # [px]

# ============================================================ real-robot imperfections
# camera
CAM_DELAY_TRUE = 0.20        # [s] true image age (hardware.CAMERA_DELAY is what the robot assumes)
CAM_DELAY_JITTER = 0.05      # [s] +- per frame
FRAME_DROP = 0.10            # probability that no new image arrives (old image re-used, like rosbridge)
IMG_NOISE = 4.0              # [grey levels] sensor noise
IMG_GAIN = (0.75, 1.15)      # brightness range (per run)
EXPOSURE = 0.03              # [s] motion blur while turning
JPEG_QUALITY = 80            # compressed image over WiFi
RENDER_MARGIN = 1.25         # render a wider image so the distorted image is filled
# calibration error (--calib-error): true camera = calibration + error (1 sigma)
CAL_ERR_POS = 0.01           # [m] forward / lateral mounting
CAL_ERR_YAW = math.radians(1.0)
CAL_ERR_F = 0.01             # relative focal length
# odometry / motors
WHEEL_TRUE_ERR = 0.01        # true wheel radius / base = best estimate * (1 + U(-err, err)), per run
WHEEL_R_ERR, WHEEL_B_ERR = +0.02, -0.03   # used only if there is NO odometry_hw.json
SLIP_W = (0.0, 0.03)         # rotation lost to wheel slip (per run)
ENC_NOISE = 0.002            # [rad] per control step
DEAD_V, DEAD_W = 0.01, 0.05  # motor dead band [m/s], [rad/s]
ACC_V, ACC_W = 0.5, 2.0      # acceleration limits [m/s^2], [rad/s^2]
CMD_DELAY = 0.10             # [s] WiFi latency of the velocity command
LOOP_JITTER = 0.02           # [s] extra time per control step (0..jitter)
TAPE_ERR = (0.005, math.radians(2.0))     # --calib-odom: tape measure error (distance, angle)
# LiDAR
LIDAR_PERIOD = 0.2           # [s] 5 Hz
LIDAR_X = -0.064             # [m] LDS position behind the robot centre (TB3 Waffle)
LIDAR_NOISE = 0.01           # [m]
LIDAR_DROPOUT = 0.05


# ============================================================ FINAL approach (sim-side test)
FINAL_RANGE = 0.45           # [m] robot centre -> dock wall at the very end (> FRONT_STOP - LIDAR_BEHIND)
FINAL_V = 0.05               # [m/s] slow, blind part
FINAL_ODOM_EXTRA = 0.05      # [m] odometry backup may drive this much further than the camera said
LIDAR_BEHIND = -LIDAR_X      # [m] LDS behind the robot centre


class FinalMission(H.Mission):
    """hardware.Mission + FINAL: after DOCKED drive straight on, stop by LiDAR (backup: odometry / time)."""

    def __init__(self, enabled=True):
        super().__init__()
        self.final_enabled, self.final = enabled, None

    def step(self, f, dt):
        if self.state != "FINAL":
            v, w = super().step(f, dt)
            if self.done and self.success and self.final_enabled and self.final is None:
                r, b, _ = self.last.relative(self.pose)              # camera range at the DOCKED moment
                go_on = r - FINAL_RANGE
                if go_on > 0.02:
                    self.done, self.success = False, False
                    self.final = (self.pose.copy(), go_on, self.pose[2] + b, f.time)
                    self.go(f.time, "FINAL", f"FINAL: driving {go_on:.2f} m closer (stop at {FINAL_RANGE:.2f} m)")
                    return 0.0, 0.0
            return v, w
        self.sense(f)                                                 # keep the odometry pose up to date
        t = f.time
        p0, go_on, head, t0 = self.final
        moved = math.hypot(self.pose[0] - p0[0], self.pose[1] - p0[1])
        front = sector_min_range(f.lidar, 0.0, math.radians(10))     # dock wall straight ahead
        wall = front - LIDAR_BEHIND if front < f.lidar.range_max else float("inf")
        why = ("LiDAR" if wall <= FINAL_RANGE else
               "odometry" if moved >= go_on + FINAL_ODOM_EXTRA else
               "timeout" if t - t0 > go_on / FINAL_V + 5.0 else None)
        if why is not None:
            self.done, self.success, self.reason = True, True, "docked"
            self.note(t, f"FINAL done ({why}): drove {moved:.2f} m, LiDAR wall {wall:.2f} m from the robot centre")
            return 0.0, 0.0
        w = float(np.clip(1.5 * wrap_angle(head - self.pose[2]), -H.DOCK_TURN, H.DOCK_TURN))
        return FINAL_V, w


def make_mission(args):
    if getattr(H, "FINAL_APPROACH", None) is not None:      # hardware.py has its own FINAL state
        H.FINAL_APPROACH = H.FINAL_APPROACH and not args.no_final
        return H.Mission()
    return FinalMission(enabled=not args.no_final)


def final_target(args):
    if args.no_final:
        return H.DOCK_TARGET
    return getattr(H, "FINAL_RANGE", FINAL_RANGE) if getattr(H, "FINAL_APPROACH", True) else H.DOCK_TARGET


# ============================================================ config
def load_calibration(cfg):
    """(Calibration or None, cfg with the calibrated camera mounting, text)."""
    try:
        from calibration import calibrate_hardware
        with contextlib.redirect_stdout(io.StringIO()):
            cal, cfg_c = calibrate_hardware(cfg)
        return cal, cfg_c, "real calibration (results/camera_*.json)"
    except Exception as e:
        c = replace(cfg.camera, forward_offset=CAM_FWD, lateral_offset=CAM_LAT, yaw=CAM_YAW)
        return None, replace(cfg, camera=c), f"fallback constants, no lens distortion ({type(e).__name__})"


def sim_config(speed_scale, nominal_odometry=False):
    """Robot BELIEF: what hardware.py would use on the real robot."""
    cfg = H.hardware_config(speed_scale)
    if nominal_odometry:                                   # --calib-odom starts from the TB3 values
        cfg = replace(cfg, robot=replace(cfg.robot, wheel_radius=CONFIG.robot.wheel_radius,
                                         wheel_base=CONFIG.robot.wheel_base))
    cal, cfg, src = load_calibration(cfg)
    fy = float(cal.K[1, 1]) if cal is not None else CAM_FY
    cfg = replace(
        cfg,
        camera=replace(cfg.camera, fovy_deg=math.degrees(2.0 * math.atan(cfg.camera.height / 2.0 / fy))),
        env=replace(cfg.env, room_half_size=(ROOM_X / 2, ROOM_Y / 2), start_pose=START,
                    dock_pose=DOCK, obstacles=()),
        tags=replace(cfg.tags, fixed_tags=TAGS))
    return cfg, cal, src


class TrueCamera:
    """The camera as it REALLY is (= calibration, + error with --calib-error)."""

    def __init__(self, cfg, cal, rng, calib_error):
        c = cfg.camera
        e = (lambda s: float(rng.normal(0.0, s))) if calib_error else (lambda s: 0.0)
        self.fwd, self.lat, self.yaw = c.forward_offset + e(CAL_ERR_POS), c.lateral_offset + e(CAL_ERR_POS), \
            c.yaw + e(CAL_ERR_YAW)
        fs = 1.0 + e(CAL_ERR_F)
        W, Hh = c.width, c.height
        if cal is not None:
            K = np.array(cal.K, dtype=float).copy()
            K[0, 0] *= fs; K[1, 1] *= fs
            self.K, self.dist = K, np.asarray(cal.dist_coeffs, dtype=float).ravel()
        else:                                           # ideal pinhole with the focal length the robot assumes
            f = (Hh / 2.0) / math.tan(math.radians(c.fovy_deg) / 2.0) * fs
            self.K, self.dist = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]]), np.zeros(5)
        # render a wider ideal pinhole image, then map it onto the real (distorted) pixel grid
        self.f_r = float(self.K[1, 1])                  # same pixel scale, bigger image
        self.RW, self.RH = int(W * RENDER_MARGIN) // 2 * 2, int(Hh * RENDER_MARGIN) // 2 * 2
        self.fovy_r = math.degrees(2.0 * math.atan(self.RH / 2.0 / self.f_r))
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(Hh, dtype=np.float32))
        pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
        n = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)     # ideal normalized rays
        self.map_x = (n[:, 0] * self.f_r + self.RW / 2.0).reshape(Hh, W).astype(np.float32)
        self.map_y = (n[:, 1] * self.f_r + self.RH / 2.0).reshape(Hh, W).astype(np.float32)

    def text(self, cfg):
        c = cfg.camera
        return (f"true camera vs calibration: fwd {100 * (self.fwd - c.forward_offset):+.1f} cm, "
                f"lat {100 * (self.lat - c.lateral_offset):+.1f} cm, yaw {math.degrees(self.yaw - c.yaw):+.2f} deg")


# ============================================================ scene
def mesh_dir(cfg):
    """TurtleBot3 STL folder used by simulation.py (None if the meshes are missing)."""
    d = S._scene_directory(cfg, None)
    return (d / S.MESH_SUBDIR).resolve() if S.meshes_available(d) else None


def write_scene(cfg, directory, cam: TrueCamera, name="hardware_sim_scene.xml", use_meshes=True):
    """Own XML + textures in `directory` (simulation.py's files are not touched).
    The robot uses the same TurtleBot3 meshes as simulation.py (read from its folder, not copied)."""
    directory.mkdir(parents=True, exist_ok=True)
    mdir = mesh_dir(cfg) if use_meshes else None
    cv2.imwrite(str(directory / S.MARKER_FILE), S.generate_marker_image(cfg))
    for tid in cfg.tag_table():
        if tid != cfg.camera.marker_id:
            cv2.imwrite(str(directory / S.marker_filename(tid, cfg)), S.generate_marker_image(cfg, marker_id=tid))
    xml = S.build_scene_xml(cfg, use_meshes=mdir is not None, obstacles=[])
    if mdir is not None:                              # absolute path: our XML lives in another folder
        xml = xml.replace(f'meshdir="{S.MESH_SUBDIR}"', f'meshdir="{mdir.as_posix()}"')
        # camera visuals at the REAL mounting (behind the lens, so the camera never sees them)
        f, l, z = cam.fwd, cam.lat, CAM_Z
        arm_c, arm_h = (0.08 + f - 0.035) / 2, max(0.01, (f - 0.035 - 0.08) / 2)
        for gname, pos, size in (("cam_body_vis", f"{f - 0.029:.4f} {l:.4f} {z}", None),
                                 ("cam_lens_vis", f"{f - 0.023:.4f} {l:.4f} {z}", None),
                                 ("cam_arm_vis", f"{arm_c:.4f} {l:.4f} {z}", f"{arm_h:.4f} 0.008 0.006"),
                                 ("cam_post_vis", f"0.10 {l:.4f} {z - 0.05}", None)):
            xml = re.sub(rf'(name="{gname}"[^>]*?pos=")[^"]*"', rf'\g<1>{pos}"', xml)
            if size:
                xml = re.sub(rf'(name="{gname}"[^>]*?size=")[^"]*"', rf'\g<1>{size}"', xml)
    bx, by, byaw = BOX
    box = (f'    <geom name="box_obstacle" type="box" size="{BOX_L / 2} {BOX_W / 2} {BOX_H / 2}" '
           f'pos="{bx} {by} {BOX_H / 2}" euler="0 0 {byaw}" rgba="0.8 0.3 0.2 1"/>\n')
    xml = xml.replace('    <geom name="dock_marker"', box + '    <geom name="dock_marker"', 1)
    cam_xml = (f'<camera name="front_cam" pos="{cam.fwd:.5f} {cam.lat:.5f} {CAM_Z}" '
               f'xyaxes="{math.sin(cam.yaw):.6f} {-math.cos(cam.yaw):.6f} 0 0 0 1" fovy="{cam.fovy_r:.4f}"/>')
    xml, n = re.subn(r'<camera name="front_cam".*?/>', cam_xml, xml, flags=re.S)
    if n != 1:
        raise RuntimeError("camera line not found in simulation.build_scene_xml output")
    xml = re.sub(r'offwidth="\d+" offheight="\d+"', f'offwidth="{cam.RW}" offheight="{cam.RH}"', xml)
    path = directory / name
    path.write_text(xml)
    return path


def print_arena(cfg, cam_src):
    hx, hy = ROOM_X / 2, ROOM_Y / 2
    corner = lambda x, y: f"{x + hx:.2f} m from the WEST wall, {y + hy:.2f} m from the SOUTH wall"
    face = lambda yaw: {0: "east", 90: "north", 180: "west", -180: "west", -90: "south"}.get(
        int(round(math.degrees(wrap_angle(yaw)))), f"{math.degrees(yaw):.0f} deg")
    c = cfg.camera
    print("=" * 78)
    print("TEST ARENA (the robot does NOT know this - build it like this)")
    print(f"  arena    : {ROOM_X:.2f} m (west-east) x {ROOM_Y:.2f} m (south-north), walls >= 0.25 m high")
    print(f"  origin   : arena centre, x -> east (dock wall), y -> north")
    x, y, th = START
    print(f"  robot    : centre at ({x:+.2f}, {y:+.2f}) = {corner(x, y)}, facing {face(th)}")
    print(f"  markers  : centre {S.CAMERA_HEIGHT:.2f} m above the floor (camera lens in this sim at {CAM_Z:.2f} m)")
    for tid, (tx, ty, tyaw) in sorted(cfg.tag_table().items()):
        name = "DOCK" if tid == c.marker_id else "tag "
        print(f"  {name} id{tid}: ({tx:+.2f}, {ty:+.2f}) = {corner(tx, ty)}, facing {face(tyaw)}")
    bx, by, _ = BOX
    print(f"  box      : front face {BOX_W * 100:.0f} x {BOX_H * 100:.0f} cm facing the robot, "
          f"{BOX_L * 100:.0f} cm deep, centre ({bx:+.2f}, {by:+.2f}) = {corner(bx, by)}")
    print(f"             = {BOX_FRONT:.2f} m in front of the dock marker, inner edge {BOX_GAP:.2f} m beside "
          f"the dock axis (partly in front of the dock)")
    print(f"  marker   : {c.marker_size * 100:.0f} cm black square, ArUco {c.aruco_dict}")
    print(f"  camera   : {c.width}x{c.height}, fwd {c.forward_offset:.3f} m, lat {c.lateral_offset:+.3f} m, "
          f"yaw {math.degrees(c.yaw):+.1f} deg | {cam_src}")
    print("=" * 78)


# ============================================================ simulated real robot
class HwSimEnv:
    """Same interface as the real RosbridgeEnvironment, driven by MuJoCo with real-robot imperfections."""

    def __init__(self, cfg, xml_path, cam: TrueCamera, seed=0, ideal=False):
        self.cfg, self.cam, self.ideal = cfg, cam, ideal
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.lidar_angles = np.linspace(0.0, 2.0 * math.pi, cfg.lidar.num_beams, endpoint=False)
        # wheel geometry: firmware (nominal TB3), true (best estimate +- error)
        self.r_fw, self.b_fw = CONFIG.robot.wheel_radius, CONFIG.robot.wheel_base
        odo = H.load_odometry_hw()
        r0, b0 = odo if odo is not None else (CONFIG.robot.wheel_radius * (1 + WHEEL_R_ERR),
                                              CONFIG.robot.wheel_base * (1 + WHEEL_B_ERR))
        u = (lambda a: 0.0) if ideal else (lambda a: float(self.rng.uniform(-a, a)))
        self.r_true, self.b_true = r0 * (1 + u(WHEEL_TRUE_ERR)), b0 * (1 + u(WHEEL_TRUE_ERR))
        if ideal:
            self.r_true, self.b_true = cfg.robot.wheel_radius, cfg.robot.wheel_base
            self.r_fw, self.b_fw = self.r_true, self.b_true
        self.slip = 0.0 if ideal else float(self.rng.uniform(*SLIP_W))
        self.gain = 1.0 if ideal else float(self.rng.uniform(*IMG_GAIN))
        self._enc = np.zeros(2)
        self._v = self._w = 0.0
        self._cmdq = deque()
        self._renderer = None
        self._last_img = None
        self._ranges = np.full(cfg.lidar.num_beams, cfg.lidar.range_max)
        self._lidar_t = -99.0
        self.collided_any = False
        self.reset_pose(cfg.env.start_pose)

    def reset_pose(self, pose):
        self.data.qpos[:3] = pose
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._hist = [(self.time, self.data.qpos[:3].copy())]
        self._v = self._w = 0.0
        self._cmdq.clear()
        self._v_cmd = (0.0, 0.0)
        self._lidar_t = -99.0

    @property
    def time(self):
        return float(self.data.time)

    def truth(self):
        x, y, th = self.data.qpos[:3]
        return np.array([x, y, wrap_angle(th)])

    def read_encoders(self):
        return self._enc.copy()

    def sensor_ages(self):
        return self.time - self._lidar_t, CAM_DELAY_TRUE

    # ---------------------------------------------------------- LiDAR
    def read_lidar(self):
        if self.time - self._lidar_t < LIDAR_PERIOD - 1e-9:
            return self._ranges.copy()                         # no new scan yet (5 Hz)
        lid = self.cfg.lidar
        x, y, th = self.data.qpos[:3]
        lx = 0.0 if self.ideal else LIDAR_X
        origin = np.array([x + lx * math.cos(th), y + lx * math.sin(th), S.LIDAR_HEIGHT])
        r = np.full(lid.num_beams, lid.range_max)
        gid = np.zeros(1, dtype=np.int32)
        for i, a in enumerate(self.lidar_angles):
            d = mujoco.mj_ray(self.model, self.data, origin,
                              np.array([math.cos(th + a), math.sin(th + a), 0.0]), None, 1, self._base, gid)
            if 0.0 <= d < lid.range_max:
                r[i] = d
        if not self.ideal:
            hit = r < lid.range_max
            r[hit] += self.rng.normal(0.0, LIDAR_NOISE, int(hit.sum()))
            r[self.rng.random(lid.num_beams) < LIDAR_DROPOUT] = lid.range_max
        r = np.clip(r, lid.range_min, lid.range_max)
        r[r < H.IGNORE_NEAR] = lid.range_max                   # same filter as the real driver
        self._ranges, self._lidar_t = r, self.time
        return r.copy()

    # ---------------------------------------------------------- camera
    def _pose_at(self, t):
        past = self._hist[0][1]
        for tt, q in self._hist:
            if tt <= t:
                past = q
        return past

    def read_camera(self):
        cam = self.cam
        if self._last_img is not None and not self.ideal and self.rng.random() < FRAME_DROP:
            return self._last_img.copy()                       # WiFi: no new image, the old one again
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=cam.RH, width=cam.RW)
        delay = 0.0 if self.ideal else CAM_DELAY_TRUE + float(self.rng.uniform(-CAM_DELAY_JITTER, CAM_DELAY_JITTER))
        now = self.data.qpos[:3].copy()
        self.data.qpos[:3] = self._pose_at(self.time - delay)  # render the pose of `delay` ago
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera="front_cam")
        ideal_img = self._renderer.render().copy()
        self.data.qpos[:3] = now
        mujoco.mj_forward(self.model, self.data)
        img = cv2.remap(ideal_img, cam.map_x, cam.map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))   # real lens + real K
        if not self.ideal:
            img = self._degrade(img)
        self._last_img = img
        return img.copy()

    def _degrade(self, img):
        out = img.astype(np.float32) * self.gain
        n = int(round(abs(self._w) * float(self.cam.K[0, 0]) * EXPOSURE))   # blur length [px] while turning
        if n >= 2:
            out = cv2.filter2D(out, -1, np.ones((1, n), np.float32) / n)
        out += self.rng.normal(0.0, IMG_NOISE, out.shape).astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else out

    # ---------------------------------------------------------- motion
    def step(self, v, w):
        rob, dt, tc = self.cfg.robot, self.cfg.sim.dt, self.cfg.sim.control_dt
        v = float(np.clip(v, -rob.max_linear_vel, rob.max_linear_vel))
        w = float(np.clip(w, -rob.max_angular_vel, rob.max_angular_vel))
        if not self.ideal:                                    # command arrives CMD_DELAY later
            self._cmdq.append((self.time + CMD_DELAY, v, w))
            v = w = None
            while self._cmdq and self._cmdq[0][0] <= self.time + 1e-9:
                _, v, w = self._cmdq.popleft()
            if v is None:
                v, w = self._v_cmd
            self._v_cmd = (v, w)
            v = 0.0 if abs(v) < DEAD_V else v
            w = 0.0 if abs(w) < DEAD_W else w
            self._v += float(np.clip(v - self._v, -ACC_V * tc, ACC_V * tc))
            self._w += float(np.clip(w - self._w, -ACC_W * tc, ACC_W * tc))
        else:
            self._v, self._w = v, w
        wl = (self._v - self._w * self.b_fw / 2) / self.r_fw   # wheel speeds set by the firmware
        wr = (self._v + self._w * self.b_fw / 2) / self.r_fw
        v_b = self.r_true * (wl + wr) / 2                      # what the body really does
        w_b = self.r_true * (wr - wl) / self.b_true * (1.0 - self.slip)
        period = tc + (0.0 if self.ideal else float(self.rng.uniform(0.0, LOOP_JITTER)))
        n = max(1, int(round(period / dt)))
        for _ in range(n):
            th = self.data.qpos[2]
            self.data.qvel[:3] = [v_b * math.cos(th), v_b * math.sin(th), w_b]
            mujoco.mj_step(self.model, self.data)
            self.collided_any = self.collided_any or self._contact()
        self._enc += np.array([wl, wr]) * n * dt
        if not self.ideal:
            self._enc += self.rng.normal(0.0, ENC_NOISE, 2)
        mujoco.mj_forward(self.model, self.data)
        self._hist.append((self.time, self.data.qpos[:3].copy()))
        self._hist = [h for h in self._hist if self.time - h[0] < 1.0]

    def _contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._base in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]):
                return True
        return False

    def emergency_stop(self):
        self._v = self._w = 0.0
        self._cmdq.clear()
        self._v_cmd = (0.0, 0.0)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ============================================================ odometry calibration in the sim
def calibrate_odometry_sim(env: HwSimEnv, cfg, rng):
    """Same procedure as hardware.calibrate_odometry_hw: drive 1 m, spin 2 turns.
    The tape measure is the ground truth + a small measuring error."""
    dt = cfg.sim.control_dt
    r0, b0 = cfg.robot.wheel_radius, cfg.robot.wheel_base

    def drive(v, w, duration):
        turned, last = 0.0, env.truth()[2]
        t0 = env.time
        while env.time - t0 < duration + 1.0:                 # + 1 s settle
            env.step(v, w) if env.time - t0 < duration else env.step(0.0, 0.0)
            th = env.truth()[2]
            turned += wrap_angle(th - last)
            last = th
        return turned

    e0, g0 = env.read_encoders(), env.truth()
    drive(0.10, 0.0, 10.0)
    dphi = env.read_encoders() - e0
    d_true = math.dist(env.truth()[:2], g0[:2]) + float(rng.normal(0.0, TAPE_ERR[0]))
    r = d_true / float(dphi.mean())
    e0 = env.read_encoders()
    turned = drive(0.0, 0.3, 4.0 * math.pi / 0.3)
    dphi = env.read_encoders() - e0
    theta = turned + float(rng.normal(0.0, TAPE_ERR[1]))     # what you read on the tape mark
    b = r * float(dphi[1] - dphi[0]) / theta
    print(f"[sim calib-odom] wheel_radius {r0:.4f} -> {r:.4f} m (true {env.r_true:.4f}) | "
          f"wheel_base {b0:.4f} -> {b:.4f} m (true {env.b_true:.4f}, slip {100 * env.slip:.1f} %)")
    env.reset_pose(cfg.env.start_pose)                        # put the robot back on the start mark
    return replace(cfg, robot=replace(cfg.robot, wheel_radius=r, wheel_base=b))


# ============================================================ camera view (visualization.py)
class CameraMonitor:
    """What the robot camera sees: live window (visualization.LiveCameraView) + optional video file."""

    def __init__(self, cfg, live=True, video_path=None):
        from visualization import LiveCameraView
        self.live = LiveCameraView(window="hardware_sim camera (what the robot sees)",
                                   period=H.CAMERA_PERIOD, scale=1.0, enabled=live)
        self.video_path, self.writer = video_path, None
        self.last_t, self.last_img = -math.inf, None
        self.detector = None
        try:                                              # only to DRAW the markers (robot uses Sensing)
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.camera.aruco_dict))
            self.detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()) \
                if hasattr(cv2.aruco, "ArucoDetector") else (d, cv2.aruco.DetectorParameters_create())
        except Exception:
            pass

    def _outline(self, rgb):
        if self.detector is None:
            return
        try:
            if isinstance(self.detector, tuple):
                corners, ids, _ = cv2.aruco.detectMarkers(rgb, *self.detector)
            else:
                corners, ids, _ = self.detector.detectMarkers(rgb)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(rgb, corners, ids)
        except Exception:
            pass

    def update(self, f, state, v, w, truth=None):
        """Call only on frames where Sensing processed a new camera image (markers are fresh)."""
        if f.image is None:
            return
        self.last_t = f.time
        rgb = np.ascontiguousarray(f.image).copy()
        h, wd = rgb.shape[:2]
        self._outline(rgb)
        cv2.drawMarker(rgb, (wd // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)   # image centre
        line = f"t={f.time:5.1f}s  v={v:+.2f} w={w:+.2f}"
        if truth is not None:
            line += f"  true ({truth[0]:+.2f},{truth[1]:+.2f})"
        cv2.putText(rgb, line, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.live.update(f.time, rgb, mode=state, marker=f.marker, markers=f.markers)
        self.last_img = rgb
        if self.video_path is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            top = state + (f" | dock {f.marker.range:.2f} m {math.degrees(f.marker.bearing):+.0f} deg"
                           if f.marker is not None else " | no dock")
            cv2.putText(bgr, top, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                              round(1.0 / H.CAMERA_PERIOD), (wd, h))
            self.writer.write(bgr)

    def close(self, out=None):
        self.live.close()
        if self.writer is not None:
            self.writer.release()
        if out is not None and self.last_img is not None:      # last view (e.g. the docked view)
            cv2.imwrite(str(out / "camera_final.jpg"), cv2.cvtColor(self.last_img, cv2.COLOR_RGB2BGR))


# ============================================================ evaluation (ground truth: sim only)
def dock_error(truth):
    dx, dy, dyaw = DOCK                                   # dyaw = direction the marker faces
    x, y, th = truth
    rng = math.hypot(x - dx, y - dy)
    lat = -math.sin(dyaw) * (x - dx) + math.cos(dyaw) * (y - dy)   # signed offset from the marker axis
    head = wrap_angle(th - (dyaw + math.pi))              # 0 = robot faces the marker head-on
    return rng, lat, head


def save_plot(out, cfg, true_path, odom_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    hx, hy = ROOM_X / 2, ROOM_Y / 2
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_patch(Rectangle((-hx, -hy), ROOM_X, ROOM_Y, fill=False, lw=2))
    bx, by, _ = BOX
    byaw = BOX[2]
    corner0 = (bx - BOX_L / 2 * math.cos(byaw) + BOX_W / 2 * math.sin(byaw),
               by - BOX_L / 2 * math.sin(byaw) - BOX_W / 2 * math.cos(byaw))
    ax.add_patch(Rectangle(corner0, BOX_L, BOX_W, angle=math.degrees(byaw), color="tab:red", alpha=0.7, label="box"))
    for tid, (tx, ty, tyaw) in sorted(cfg.tag_table().items()):
        dock = tid == cfg.camera.marker_id
        ax.plot(tx, ty, "ks" if dock else "bs", ms=9)
        ax.annotate("", (tx + 0.25 * math.cos(tyaw), ty + 0.25 * math.sin(tyaw)), (tx, ty),
                    arrowprops=dict(arrowstyle="->"))
        ax.text(tx, ty + 0.08, f"{'DOCK ' if dock else 'tag '}{tid}", ha="center")
    tp, op = np.asarray(true_path), np.asarray(odom_path)
    ax.plot(tp[:, 0], tp[:, 1], "g-", lw=2, label="true path")
    ax.plot(op[:, 0], op[:, 1], "m--", lw=1, label="odometry (robot belief)")
    ax.plot(*START[:2], "go")
    ax.set_xlim(-hx - 0.2, hx + 0.2); ax.set_ylim(-hy - 0.2, hy + 0.2)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    fig.savefig(out / "route.png", dpi=120)
    plt.close(fig)


# ============================================================ one run
def run_once(cfg, cal, seed, args, out=None, verbose=True):
    rng = np.random.default_rng(seed + 1000)
    cam = TrueCamera(cfg, cal, rng, calib_error=args.calib_error and not args.ideal)
    scene_dir = RESULTS_DIR / "hardware_sim" / "scene"
    xml = write_scene(cfg, scene_dir, cam, name=f"hardware_sim_scene_{seed}.xml", use_meshes=not args.no_mesh)
    try:
        env = HwSimEnv(cfg, xml, cam, seed, ideal=args.ideal)
    except ValueError as e:                            # meshes could not be loaded -> cylinder robot
        print(f"Warning: TurtleBot3 meshes not loaded ({e}); using the cylinder.")
        xml = write_scene(cfg, scene_dir, cam, name=f"hardware_sim_scene_{seed}.xml", use_meshes=False)
        env = HwSimEnv(cfg, xml, cam, seed, ideal=args.ideal)
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    true_path, odom_path = [], []
    handle = None
    with sink:
        if args.calib_error and not args.ideal:
            print("[sim]", cam.text(cfg))
        if args.calib_odom:
            cfg = calibrate_odometry_sim(env, cfg, rng)
        # real robot: Sensing undistorts with the real calibration; without one: ideal pinhole
        sensing = Sensing(env, cfg, camera_period=H.CAMERA_PERIOD, calibration=cal)
        mission = make_mission(args)
        dt = cfg.sim.control_dt
        sx, sy, sth = START
        last_print, last_guard = -99.0, ""
        if args.viewer:
            import mujoco.viewer
            handle = mujoco.viewer.launch_passive(env.model, env.data)
            handle.cam.lookat[:] = [0, 0, 0]; handle.cam.distance = 4.5; handle.cam.elevation = -70
        monitor, n_cam_seen = None, -1
        if out is not None and (not args.no_camera or args.video):
            monitor = CameraMonitor(cfg, live=not args.no_camera,
                                    video_path=(out / "camera.mp4") if args.video else None)
        t0 = env.time
        if verbose:
            print(f"{'t[s]':>6} {'state':<9} {'v':>5} {'w':>6} {'odom x,y':>12} {'true x,y':>12} dock")
        try:
            while True:
                f = sensing.read()
                v, w = mission.step(f, dt)
                v, w, guard = H.safety(v, w, f.lidar, f.lidar_age)
                if guard != last_guard:
                    last_guard = guard
                    if guard:
                        mission.note(f.time, f"SAFETY: {guard}")
                gt = env.truth()
                p = mission.pose                              # odometry frame -> world (start pose)
                odom_path.append((sx + p[0] * math.cos(sth) - p[1] * math.sin(sth),
                                  sy + p[0] * math.sin(sth) + p[1] * math.cos(sth)))
                true_path.append((gt[0], gt[1]))
                if monitor is not None and sensing.n_camera != n_cam_seen:   # new camera frame
                    n_cam_seen = sensing.n_camera
                    monitor.update(f, mission.state, v, w, gt)
                if verbose and f.time - last_print >= 1.0:
                    last_print = f.time
                    dk = f"{f.marker.range:.2f} m {math.degrees(f.marker.bearing):+.0f} deg" if f.marker else "-"
                    print(f"{f.time - t0:6.1f} {mission.state:<9} {v:5.2f} {w:6.2f} "
                          f"{odom_path[-1][0]:+5.2f},{odom_path[-1][1]:+5.2f} {gt[0]:+5.2f},{gt[1]:+5.2f} {dk}",
                          flush=True)
                if mission.done:
                    break
                if env.collided_any:
                    mission.reason = "collision"
                    mission.note(f.time, "COLLISION")
                    break
                if f.time - t0 > H.RUN_TIMEOUT:
                    mission.reason = "timeout"
                    mission.note(f.time, "TIMEOUT")
                    break
                env.step(v, w)
                if handle is not None:
                    if not handle.is_running():
                        mission.reason = "viewer closed"
                        break
                    handle.sync()
                    time.sleep(dt / 4)
        finally:
            env.close()
            if monitor is not None:
                monitor.close(out)
            if handle is not None:
                handle.close()
    rng_d, lat, head = dock_error(env.truth())
    ok = mission.success and not env.collided_any
    res = dict(seed=seed, success=ok, reason=mission.reason, time=round(env.time - t0, 1),
               dock_range=round(rng_d, 3), dock_lateral=round(lat, 3), dock_heading_deg=round(math.degrees(head), 1),
               odom_drift=round(math.dist(true_path[-1], odom_path[-1]), 3),
               detections=f"{sensing.n_detect}/{sensing.n_camera}",
               odometry=f"r {cfg.robot.wheel_radius:.4f} b {cfg.robot.wheel_base:.4f} "
                        f"(true r {env.r_true:.4f} b {env.b_true:.4f})",
               events=mission.events)
    if out is not None:
        save_plot(out, cfg, true_path, odom_path,
                  f"seed {seed}: {'SUCCESS' if ok else 'FAIL'} ({mission.reason}), lat {lat:+.2f} m")
    return res


def main_cli() -> int:
    p = argparse.ArgumentParser(description="hardware.py in MuJoCo under real-robot conditions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (single run)")
    p.add_argument("--speed-scale", type=float, default=H.SPEED_SCALE)
    p.add_argument("--calib-odom", action="store_true", help="calibrate the odometry in the sim first (1 m + 2 turns)")
    p.add_argument("--calib-error", action="store_true", help="true camera differs a little from the calibration")
    p.add_argument("--ideal", action="store_true", help="no imperfections at all (debug the logic)")
    p.add_argument("--no-mesh", action="store_true", help="draw the robot as a cylinder")
    p.add_argument("--no-camera", action="store_true", help="no live camera window")
    p.add_argument("--video", action="store_true", help="save camera.mp4 (what the robot saw)")
    p.add_argument("--no-final", action="store_true", help="stop at the camera DOCKED pose (no FINAL approach)")
    args = p.parse_args()
    if args.viewer:
        S._ensure_mjpython()                              # macOS: the viewer needs mjpython

    cfg, cal, cam_src = sim_config(args.speed_scale, nominal_odometry=args.calib_odom)
    if args.ideal:
        cal = None                                        # ideal pinhole, no distortion
    print_arena(cfg, cam_src)
    print(f"Mode: {'IDEAL' if args.ideal else 'REAL-ROBOT conditions'}"
          f"{' + calib-odom' if args.calib_odom else ''}{' + calib-error' if args.calib_error else ''} | "
          f"lens distortion: {'yes' if cal is not None else 'no'} | robot: "
          f"{'TurtleBot3 meshes' if mesh_dir(cfg) is not None and not args.no_mesh else 'cylinder'} | odometry belief "
          f"r {cfg.robot.wheel_radius:.4f} b {cfg.robot.wheel_base:.4f}")
    print(f"FINAL approach: {'off' if args.no_final else f'on, stop {final_target(args):.2f} m from the dock wall'}"
          f" ({'hardware.py' if getattr(H, 'FINAL_APPROACH', None) is not None else 'hardware_sim.py'})")
    print("Scenes go to:", RESULTS_DIR / "hardware_sim" / "scene", "(simulation.py's scene is untouched)\n")
    base = RESULTS_DIR / "hardware_sim"

    if args.trials <= 1:
        out = base / time.strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        r = run_once(cfg, cal, args.seed, args, out)
        print("\n=== RESULT (ground truth, only possible in simulation) ===")
        print(f"{'SUCCESS' if r['success'] else 'FAILURE'} ({r['reason']}) in {r['time']} s | "
              f"dock range {r['dock_range']:.2f} m (target {final_target(args):.2f}), lateral {r['dock_lateral']:+.3f} m, "
              f"heading {r['dock_heading_deg']:+.1f} deg | odometry drift {r['odom_drift']:.3f} m | "
              f"dock detections {r['detections']} | {r['odometry']}")
        for e in r["events"]:
            print("  " + e)
        (out / "summary.json").write_text(json.dumps(r, indent=2))
        print("Saved to", out)
        return 0 if r["success"] else 1

    rows = []
    print(f"{'seed':>4} {'result':>7} {'reason':>16} {'time':>6} {'range':>6} {'lat':>7} {'head':>6} {'drift':>6}")
    for s in range(args.seed, args.seed + args.trials):
        r = run_once(cfg, cal, s, args, verbose=False)
        rows.append(r)
        print(f"{s:>4} {'OK' if r['success'] else 'FAIL':>7} {r['reason']:>16} {r['time']:>6.1f} "
              f"{r['dock_range']:>6.2f} {r['dock_lateral']:>+7.3f} {r['dock_heading_deg']:>+6.1f} "
              f"{r['odom_drift']:>6.3f}", flush=True)
    ok = [r for r in rows if r["success"]]
    print(f"\nSuccess {len(ok)}/{len(rows)}")
    if ok:
        print(f"mean |lateral| {np.mean([abs(r['dock_lateral']) for r in ok]):.3f} m | "
              f"mean time {np.mean([r['time'] for r in ok]):.1f} s")
    out = base / time.strftime("%Y%m%d_%H%M%S_trials")
    out.mkdir(parents=True, exist_ok=True)
    (out / "trials.json").write_text(json.dumps(rows, indent=2))
    print("Saved to", out)
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())