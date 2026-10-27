# calibrate_extrinsics.py - camera -> robot extrinsics with a chessboard at MEASURED poses (Zhang + PnP).
# Each placement: board centre + normal in the camera frame (PnP) vs in the robot frame (tape / jig).
from __future__ import annotations
import argparse
import csv
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from config import CONFIG, RESULTS_DIR, Config
from calibration import chessboard_object_points, find_chessboard_corners, camera_matrix, load_intrinsics
from robot_model import wrap_angle

OUT_FILE = RESULTS_DIR / 'camera_extrinsics.json'
SIM_BOARD = (1.5, -0.3, math.pi)      # sim only: board centre (x, y) and normal yaw in the world


def board_in_camera(rgb, K, dist, pattern=(9, 6), square=0.025):
    """Board centre as a camera planar point [Z, -X] and the heading of its normal, or None."""
    corners = find_chessboard_corners(rgb, pattern)
    if corners is None:
        return None
    obj = chessboard_object_points(pattern, square)
    ok, rv, tv = cv2.solvePnP(obj, corners.astype(np.float64), K, dist)
    if not ok:
        return None
    R = cv2.Rodrigues(rv)[0]
    centre = R @ obj.mean(axis=0) + tv.ravel()           # (X right, Y down, Z forward)
    n = R[:, 2]
    if n @ centre > 0.0:                                  # normal must point back at the camera
        n = -n
    return np.array([centre[2], -centre[0]]), math.atan2(-n[0], n[2])


def solve_extrinsics(samples):
    """samples: (q_cam, a_cam, p_robot, a_robot) per placement -> (fwd, lat, yaw, report)."""
    if len(samples) < 3:
        raise ValueError(f'only {len(samples)} usable board placements (need >= 3, better >= 5)')
    yaws = np.array([wrap_angle(ar - ac) for _, ac, _, ar in samples])
    yaw = math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean())
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    t = np.array([p - R @ q for q, _, p, _ in samples])
    fwd, lat = t.mean(axis=0)
    err = np.linalg.norm(t - t.mean(axis=0), axis=1)
    report = {'n': len(samples),
              'pos_rms_mm': float(1000.0 * np.sqrt(np.mean(err ** 2))),
              'yaw_std_deg': float(np.degrees(np.std([wrap_angle(y - yaw) for y in yaws])))}
    return float(fwd), float(lat), float(yaw), report


def apply_extrinsics(cfg: Config, fwd: float, lat: float, yaw: float) -> Config:
    """Robot BELIEF uses the calibrated camera mounting (the world is unchanged)."""
    return replace(cfg, camera=replace(cfg.camera, forward_offset=fwd, lateral_offset=lat, yaw=yaw))


def calibrate_in_sim(cfg: Config = CONFIG, seed=None, calibration=None):
    """Same procedure in MuJoCo: the robot is placed at known poses in front of a board (the 'jig')."""
    from simulation import Simulation, CALIB_BOARD_PATTERN, CHESSBOARD_SQUARE
    K = camera_matrix(cfg) if calibration is None else calibration.K
    dist = np.zeros(5) if calibration is None else calibration.dist_coeffs
    bx, by, byaw = SIM_BOARD
    samples = []
    with tempfile.TemporaryDirectory() as tmp, Simulation(cfg, seed=seed, scene_dir=Path(tmp),
                                                          calib_board=SIM_BOARD) as sim:
        for d in (0.8, 1.0, 1.2):
            for l in (-0.1, 0.0, 0.1):
                for th in (-0.15, 0.0, 0.15):
                    pose = (bx - d, by + l, th)
                    sim.reset(pose)
                    img = next((im for im in (sim.read_camera() for _ in range(5)) if im.any()), None)
                    obs = None if img is None else board_in_camera(img, K, dist, CALIB_BOARD_PATTERN,
                                                                   CHESSBOARD_SQUARE)
                    if obs is None:
                        continue
                    c, s = math.cos(th), math.sin(th)
                    dx, dy = bx - pose[0], by - pose[1]
                    samples.append((obs[0], obs[1], np.array([c * dx + s * dy, -s * dx + c * dy]),
                                    wrap_angle(byaw - th)))
        hidden = sim.cam_offset                                    # EVALUATION ONLY
    fwd, lat, yaw, rep = solve_extrinsics(samples)
    rep['hidden_forward'] = hidden
    return fwd, lat, yaw, rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Camera -> robot extrinsic calibration with a chessboard')
    ap.add_argument('--sim', action='store_true', help='run the procedure in MuJoCo')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--poses', help='CSV with columns file,x,y,normal_deg (board in the ROBOT frame)')
    ap.add_argument('--intrinsics', default='results/camera_intrinsics.json')
    ap.add_argument('--pattern', type=int, nargs=2, default=(9, 6), help='inner corners (cols rows)')
    ap.add_argument('--square', type=float, default=0.025, help='square size [m]')
    a = ap.parse_args()

    if a.sim:
        from simulation import Simulation
        from calibration import calibrate_camera_in_sim
        with Simulation(CONFIG, seed=a.seed) as s:
            world = s.world_cfg
        cal = calibrate_camera_in_sim(world, CONFIG, seed=a.seed or 0)     # Zhang (intrinsic) first
        print(f"intrinsic: RMS {cal.intrinsics.reprojection_error:.3f} px | fx {cal.K[0, 0]:.1f}")
        fwd, lat, yaw, rep = calibrate_in_sim(CONFIG, a.seed, calibration=cal)
    else:
        intr = load_intrinsics(a.intrinsics)
        folder, samples = Path(a.poses).parent, []
        with open(a.poses, newline='') as f:
            for row in csv.DictReader(f):
                img = cv2.cvtColor(cv2.imread(str(folder / row['file'])), cv2.COLOR_BGR2RGB)
                obs = board_in_camera(img, intr.K, intr.dist_coeffs, tuple(a.pattern), a.square)
                if obs is None:
                    print('board not found in', row['file'])
                    continue
                samples.append((obs[0], obs[1], np.array([float(row['x']), float(row['y'])]),
                                math.radians(float(row['normal_deg']))))
        fwd, lat, yaw, rep = solve_extrinsics(samples)

    print(f'forward {fwd:.4f} m | lateral {lat:+.4f} m | yaw {math.degrees(yaw):+.2f} deg')
    print(f"{rep['n']} placements | position RMS {rep['pos_rms_mm']:.1f} mm | yaw std {rep['yaw_std_deg']:.2f} deg")
    if rep.get('hidden_forward') is not None:
        print(f"hidden true forward offset: {rep['hidden_forward']:.4f} m (evaluation only)")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({'forward_offset': fwd, 'lateral_offset': lat, 'yaw': yaw, **rep}, indent=2))
    print('saved:', OUT_FILE)