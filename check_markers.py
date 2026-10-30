#!/usr/bin/env python3
"""Check the ArUco markers on the real robot: dictionary, IDs, marker size, and the
conditions the pipeline needs before it learns the dock / switches to AIM -> DOCK.
The robot does NOT move (no velocity command is sent).

    python check_markers.py                       # live, robot IP from hardware.py
    python check_markers.py 192.168.1.213
    python check_markers.py --distance 1.00       # live, 'c' uses this tape distance
    python check_markers.py --image extr_images/extr_00.jpg --distance 1.00

Live keys:  c = measure 20 frames, then type the tape distance  |  q = quit
Tape distance = camera LENS -> marker face, robot placed HEAD-ON in front of the marker.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import RESULTS_DIR, Config
from calibration import (apply_extrinsics, get_calibration, load_extrinsics, load_intrinsics,
                         marker_pose_from_pnp, nominal_intrinsics, solve_marker_pose)
from hardware import ROBOT_IP, hardware_config, with_fov
from robot_model import wrap_angle
from sensing import detect_markers

# Other families tried, to find out what is really printed
OTHER_DICTS = ("DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000", "DICT_7X7_1000",
               "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h11")
N_COLLECT = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Setup:
    cfg: Config
    K: np.ndarray
    dist: np.ndarray
    calibrated: bool


@dataclass
class Det:
    dict_name: str
    marker_id: int
    corners: np.ndarray                 # (4, 2) [px]
    side_px: float                      # mean edge length [px]
    pose: Optional[tuple] = None        # (range, bearing, view, cam_Z, reproj_px) with the CONFIGURED size


# ------------------------------------------------------------------ setup (same as run_hardware)
def load_setup() -> Setup:
    cfg = hardware_config()
    try:
        intr = load_intrinsics(RESULTS_DIR / "camera_intrinsics.json")
        extr = RESULTS_DIR / "camera_extrinsics.json"
        if extr.exists():
            cfg = apply_extrinsics(cfg, *load_extrinsics(extr))
        else:
            print("!! no camera_extrinsics.json: nominal camera mounting is used")
        cal = get_calibration(cfg, intr)
        cfg = with_fov(cfg, cal)
        rms = intr.reprojection_error if intr.reprojection_error is not None else float("nan")
        print(f"Calibration loaded: RMS {rms:.3f} px | fx {cal.K[0, 0]:.1f} fy {cal.K[1, 1]:.1f} | "
              f"cx {cal.K[0, 2]:.1f} cy {cal.K[1, 2]:.1f} | image {tuple(intr.image_size)}")
        print(f"Extrinsics: fwd {cal.forward_offset:.3f} m, lat {cal.lateral_offset:+.3f} m, "
              f"yaw {math.degrees(cal.yaw):+.2f} deg")
        return Setup(cfg, cal.K, cal.dist_coeffs, True)
    except Exception as exc:
        print(f"!! calibration not loaded ({exc})")
        print("   -> NOMINAL simulator K is used: ranges are NOT trustworthy, the size check is meaningless")
        return Setup(cfg, nominal_intrinsics(cfg).K, np.zeros(5), False)


# ------------------------------------------------------------------ detection
def _detect(gray: np.ndarray, name: str) -> list:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if hasattr(cv2.aruco, "ArucoDetector"):                    # OpenCV >= 4.7
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None:
        return []
    return [(int(i), np.asarray(c, dtype=float).reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]


def analyse(rgb: np.ndarray, s: Setup):
    if s.calibrated:
        rgb = cv2.undistort(rgb, s.K, s.dist)                  # same as Sensing._detect
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cam = s.cfg.camera
    dets: list[Det] = []
    names = (cam.aruco_dict,) + tuple(n for n in OTHER_DICTS if n != cam.aruco_dict)
    for name in names:
        if not hasattr(cv2.aruco, name):
            continue
        for mid, c in _detect(gray, name):
            if any(np.allclose(c, d.corners, atol=3.0) for d in dets):
                continue                                       # same physical marker already found
            side = float(np.mean(np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)))
            det = Det(name, mid, c, side)
            if name == cam.aruco_dict:
                best = solve_marker_pose(c, s.K, np.zeros(5), cam.marker_size)
                if best is not None:
                    rv, tv, err = best
                    r, b, v = marker_pose_from_pnp(rv, tv, s.cfg)
                    det.pose = (r, b, v, float(np.ravel(tv)[2]), float(err))
            dets.append(det)
    try:
        pipe = detect_markers(rgb, s.cfg, 0.0, K=s.K)          # EXACTLY what the pipeline accepts
    except Exception as exc:
        print(f"!! detect_markers() raised {exc!r}")
        pipe = []
    return rgb, dets, pipe


# ------------------------------------------------------------------ pipeline conditions for the dock
def gates(cfg: Config, r: float, b: float, v: float) -> list:
    axis = abs(wrap_angle(b - v))
    t = cfg.tags
    return [
        (f"SURVEY can learn it: range <= {t.dock_max_range:.1f} m (2.0 m if sigma_xy > 0.12), "
         f"robot standing still", r <= t.dock_max_range),
        ("NAVIGATE/DOCK can learn it: range <= 2.0 m", r <= 2.0),
        ("DockRelearner (close view): range < 1.5 m and |view| < 34 deg", r < 1.5 and abs(v) < 0.6),
        ("AIM -> DOCK trigger: 0.9 < range < 2.2 m, |bearing| < 17 deg, |bearing - view| < 20 deg",
         0.9 < r < 2.2 and abs(b) < 0.3 and axis < 0.35),
    ]


def report(dets: list, pipe: list, s: Setup) -> None:
    cfg, cam = s.cfg, s.cfg.camera
    known = sorted(cfg.tag_table())
    dock = int(cam.marker_id)
    print("-" * 90)
    print(f"Config: dict {cam.aruco_dict} | marker_size {cam.marker_size:.3f} m | dock id {dock} | "
          f"ids the pipeline accepts {known}")
    if not dets:
        print("  OpenCV sees NO marker in any dictionary -> not in view, too far/small, blur or glare")
    for d in dets:
        line = f"  OpenCV: {d.dict_name:<20} id {d.marker_id:<4} side {d.side_px:6.1f} px"
        if d.dict_name != cam.aruco_dict:
            line += "   <-- WRONG DICTIONARY (pipeline ignores it)"
        elif d.marker_id not in known:
            line += f"   <-- ID NOT IN {known} (pipeline ignores it)"
        elif d.marker_id == dock:
            line += "   = DOCK"
        else:
            line += "   = extra tag"
        if d.pose is not None:
            r, b, v, z, e = d.pose
            line += (f"\n          with {cam.marker_size:.3f} m: range {r:.3f} m (camera Z {z:.3f} m), "
                     f"bearing {math.degrees(b):+.1f} deg, view {math.degrees(v):+.1f} deg, reproj {e:.2f} px")
        print(line)
    print(f"  Pipeline detect_markers() accepts ids: {[m.marker_id for m in pipe] or 'none'}")
    dock_m = next((m for m in pipe if m.marker_id == dock), None)
    if dock_m is None:
        if any(d.marker_id == dock and d.dict_name == cam.aruco_dict for d in dets):
            print("  !! OpenCV sees the dock but the pipeline rejects it (bad pose or |view| > 70 deg)")
        return
    print("  Dock conditions:")
    for text, ok in gates(cfg, dock_m.range, dock_m.bearing, dock_m.view_angle):
        print(f"    [{'OK' if ok else '--'}] {text}")


# ------------------------------------------------------------------ marker size from a tape measurement
def size_check(samples: list, s: Setup, distance: Optional[float]) -> None:
    if not samples:
        print("No pose of a marker in the configured dictionary was collected.")
        return
    mid = samples[0].marker_id
    P = np.array([d.pose for d in samples])
    side = float(np.median([d.side_px for d in samples]))
    z = float(np.median(P[:, 3]))
    print("=" * 90)
    print(f"Marker id {mid}: {len(samples)} frames | camera Z {z:.3f} +- {np.std(P[:, 3]):.3f} m | "
          f"range {np.median(P[:, 0]):.3f} +- {np.std(P[:, 0]):.3f} m")
    print(f"  bearing {math.degrees(np.median(P[:, 1])):+.1f} deg (std {math.degrees(np.std(P[:, 1])):.2f}) | "
          f"view {math.degrees(np.median(P[:, 2])):+.1f} deg | side {side:.1f} px | "
          f"reproj {np.median(P[:, 4]):.2f} px")
    if abs(np.median(P[:, 2])) > math.radians(10):
        print("  !! not head-on (|view| > 10 deg): put the robot straight in front of the marker for the size check")
    if distance is None:
        txt = input("Tape distance camera LENS -> marker face [m] (empty = skip): ").strip()
        if not txt:
            return
        try:
            distance = float(txt)
        except ValueError:
            print("not a number")
            return
    if not s.calibrated:
        print("  !! no real calibration: the result below mixes a wrong K with the marker size")
    s_cfg = s.cfg.camera.marker_size
    s_pnp = s_cfg * distance / z                     # PnP depth scales linearly with the assumed size
    s_pix = side * distance / float(s.K[0, 0])       # pinhole, head-on
    print(f"  configured MARKER_SIZE : {s_cfg:.3f} m")
    print(f"  implied by PnP depth   : {s_pnp:.3f} m")
    print(f"  implied by pixel size  : {s_pix:.3f} m")
    err = abs(s_pnp - s_cfg) / s_cfg
    if err > 0.05:
        print(f"  -> MISMATCH {100 * err:.0f} %. Measure the BLACK square with a ruler; if it agrees, set "
              f"MARKER_SIZE = {s_pnp:.3f} in hardware.py.")
        print("     If the ruler says the config is right, the calibration (fx) is wrong instead.")
    else:
        print(f"  -> OK (within {100 * err:.1f} %)")


# ------------------------------------------------------------------ display
def show(rgb: np.ndarray, dets: list, pipe: list, s: Setup) -> str:
    view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    known = s.cfg.tag_table()
    for d in dets:
        ok = d.dict_name == s.cfg.camera.aruco_dict and d.marker_id in known
        col = (0, 220, 0) if ok else (0, 0, 255)
        cv2.polylines(view, [d.corners.astype(np.int32)], True, col, 2)
        label = f"{d.dict_name.replace('DICT_', '')} id{d.marker_id}"
        if d.pose is not None:
            label += f" r={d.pose[0]:.2f}"
        x, y = d.corners[0].astype(int)
        cv2.putText(view, label, (int(x), max(15, int(y) - 8)), FONT, 0.5, col, 1, cv2.LINE_AA)
    ids = ",".join(str(m.marker_id) for m in pipe) or "-"
    cv2.putText(view, f"pipeline ids: {ids} | c=measure q=quit", (8, 20), FONT, 0.55,
                (255, 255, 0), 1, cv2.LINE_AA)
    try:
        cv2.imshow("check_markers", view)
        k = cv2.waitKey(1) & 0xFF
        return chr(k).lower() if k != 255 else ""
    except cv2.error:
        return ""                                   # headless


def _pick(dets: list, s: Setup) -> Optional[Det]:
    cand = [d for d in dets if d.pose is not None]
    dock = [d for d in cand if d.marker_id == int(s.cfg.camera.marker_id)]
    return (dock or cand or [None])[0]


# ------------------------------------------------------------------ modes
def run_image(path: str, s: Setup, distance: Optional[float]) -> None:
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    rgb, dets, pipe = analyse(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), s)
    report(dets, pipe, s)
    d = _pick(dets, s)
    if d is not None and distance is not None:
        size_check([d], s, distance)
    show(rgb, dets, pipe, s)
    try:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def run_live(ip: str, s: Setup, distance: Optional[float]) -> None:
    from rosbridge_api import RosbridgeRobot
    cam = s.cfg.camera
    with RosbridgeRobot(ip) as robot:
        print(f"Connected to {ip}. The robot will NOT move. Keys: c = measure, q = quit")
        last_print, warned = 0.0, False

        def grab():
            bgr = robot.query_image(timeout=1.0, wait_for_new=True)
            return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        while True:
            rgb = grab()
            if rgb is None:
                print("no camera image (check /camera/image_raw/compressed)")
                continue
            h, w = rgb.shape[:2]
            if (w, h) != (cam.width, cam.height) and not warned:
                print(f"!! image is {w}x{h} but config/calibration expect {cam.width}x{cam.height}")
                warned = True
            rgb, dets, pipe = analyse(rgb, s)
            if time.monotonic() - last_print > 1.0:
                report(dets, pipe, s)
                last_print = time.monotonic()
            key = show(rgb, dets, pipe, s)
            if key == "q":
                break
            if key == "c":
                samples, target = [], None
                print(f"Collecting {N_COLLECT} frames, keep the robot and the marker still...")
                for _ in range(N_COLLECT * 3):
                    img = grab()
                    if img is None:
                        continue
                    img, dd, pp = analyse(img, s)
                    show(img, dd, pp, s)
                    d = _pick(dd, s)
                    if d is not None and (target is None or d.marker_id == target):
                        target = d.marker_id
                        samples.append(d)
                    if len(samples) >= N_COLLECT:
                        break
                size_check(samples, s, distance)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Check ArUco dictionary / IDs / marker size on the real robot")
    p.add_argument("ip", nargs="?", default=ROBOT_IP)
    p.add_argument("--image", default=None, help="analyse a saved image instead of the live camera")
    p.add_argument("--distance", type=float, default=None,
                   help="tape distance camera lens -> marker face [m] (head-on)")
    a = p.parse_args()
    s = load_setup()
    if a.image:
        run_image(a.image, s, a.distance)
    else:
        run_live(a.ip, s, a.distance)


if __name__ == "__main__":
    main()
