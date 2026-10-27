# Including libraries
from __future__ import annotations
import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from config import CONFIG, RESULTS_DIR, Config, RealismConfig
from calibration import chessboard_demo
from docking import DockingController, DockRelearner
from estimator import EKFEstimator
from environment import MuJoCoEnvironment
from safety import SafetyGuard
from navigation import Navigator, AimController, resume_navigator, dock_approach_point
from recovery import RecoveryPhase, TagRecovery, StopGoScan, MapView, Explorer
from robot_model import Phase, RunResult, TagMap, est_to_world, table_to_world, wrap_angle
from sensing import Sensing
from simulation import Simulation, run_trials, run_suite, _viewer, _ensure_mjpython
from visualization import (RouteVisualizer, MAX_RECOVERIES, TrialLog, table_header,
                        table_footer, table_row, mode_label, obstacles_text,
                        compute_report, print_report, plot_run)

def run_pipeline(cfg: Config = CONFIG, seed: Optional[int] = None, estimator=None,
                scene_dir: Optional[Path] = None, viewer: bool = False,
                stop_on_collision: bool = True, verbose: bool = False,
                dock_use_estimate: bool = False, marker_sigma_px: float = 0.5,
                use_marker_cov: bool = False,
                visualizer: Optional[RouteVisualizer] = None,
                use_recovery: bool = False,
                use_survey: Optional[bool] = None,
                use_lidar_map: bool = False, calibration=None, lidar_fault_at: Optional[float] = None,
                calib_odometry: bool = False,
                calib_camera: bool = False, camera_delay: float = 0.0) -> RunResult:
    from calibrate_odometry import calibrate_odometry, apply_odometry
    log = TrialLog()
    phase, reason = Phase.NAVIGATE, "timeout"
    nav_time: Optional[float] = None
    collided_any = False
    cam_stats = (0, 0)
    n_detect_any = 0
    viz_outputs = None
    n_tag_recoveries = 0                 # how many times the tag recovery was started
    recovery_after_dock = False          # True when the recovery was started by a failed docking
    events: list = []                    # timeline strings, also printed live when verbose
    last_target: Optional[int] = None    # tag the recovery is currently driving to
    last_tags: tuple = ()                # ids seen in the latest processed camera frame
    last_n_cam = -1
    tag_map: Optional[TagMap] = None     # tag poses learned during SURVEY
    dock_id = int(cfg.camera.marker_id)
    dock_lm = dock_yaw = None            # dock pose from the learned map (None = config pose)
    survey_t0: Optional[float] = None
    explore_t0: Optional[float] = None   # start time of the EXPLORE phase
    spun = 0.0                           # rotation accumulated during SURVEY [rad]
    scan = StopGoScan(step=0.8)          # SURVEY / EXPLORE: rotate a little, stop, observe
    relearner = DockRelearner(dock_id)   # dock observations made from close range only
    dock_refined = False                 # True once the dock pose was re-learned from a close view
    aim_mode = False                     # NAVIGATE driven by "look - turn - move" instead of a fixed waypoint
    recover_from_aim = False             # True when the recovery interrupted AIM
    dock_retries = 0
    dock_t0 = None
    dock_fail_t = -99.0
    obstacles = list(cfg.env.obstacles)
    world_cfg = cfg
    start = np.array(cfg.env.start_pose, dtype=float)
    dt = cfg.sim.control_dt
    report_every = max(1, int(round(1.0 / dt)))      # one table row per simulated second

    # Record an event and print it right away (verbose runs only)
    def note(t: float, text: str) -> None:
        events.append(f"[t={t:.1f}s] {text}")
        if verbose:
            print(f"  >> [t={t:5.1f}s] {text}", flush=True)

    soft_cfg = cfg
    if calib_odometry:
        r_cal, b_cal = calibrate_odometry(cfg, seed=seed)
        soft_cfg = apply_odometry(cfg, r_cal, b_cal)
        note(0.0, f"odometry calibrated: wheel_radius {r_cal:.4f} m, wheel_base {b_cal:.4f} m")
    with Simulation(cfg, seed=seed, scene_dir=scene_dir) as sim:
        # Effective config: the obstacles that are really in the scene (random or fixed)
        obstacles = list(sim.obstacles)
        world_cfg = sim.world_cfg
        run_cfg = replace(soft_cfg, env=replace(cfg.env, obstacles=()))
        nav, dock = Navigator(run_cfg), DockingController(run_cfg)
        recovery = TagRecovery(run_cfg, tracker=nav.tracker)

        # The EKF LiDAR map must contain the obstacles that are really in the scene
        if estimator is None:
            ekf_env = replace(run_cfg.env, obstacles=())
            estimator = EKFEstimator(replace(run_cfg, env=ekf_env), gate=9.21, camera_delay=camera_delay,
                                    use_marker_cov=use_marker_cov, use_lidar=use_lidar_map,
                                    R=np.diag([0.15 ** 2, 0.05 ** 2]), R_view=0.15 ** 2)
        if verbose:
            print(obstacles_text(obstacles))
            print(table_header())

        sim.reset(tuple(start))              # `start` places the robot in the hidden world (simulation + evaluation only)
        env = MuJoCoEnvironment(sim)          # the ONLY handle the robot software uses (no ground truth)
        if calib_camera and calibration is None:
            from calibration import calibrate_camera_in_sim
            try:
                calibration = calibrate_camera_in_sim(world_cfg, soft_cfg, seed=seed or 0)
                note(0.0, f"camera calibrated: RMS {calibration.intrinsics.reprojection_error:.2f} px")
            except ValueError as exc:
                note(0.0, f"CAMERA CALIBRATION FAILED ({exc}) -> using nominal intrinsics")
                calibration = None
        if calib_camera:
            from calibration_extrinsic import calibrate_in_sim, apply_extrinsics
            try:
                fwd, lat, yaw, rep = calibrate_in_sim(cfg, seed=seed, calibration=calibration)
                soft_cfg = apply_extrinsics(soft_cfg, fwd, lat, yaw)
                note(0.0, f"camera extrinsics: fwd {fwd:.3f} m, lat {lat:+.3f} m, yaw {math.degrees(yaw):+.2f} deg "
                          f"({rep['n']} placements, RMS {rep['pos_rms_mm']:.1f} mm)")
            except ValueError as exc:
                note(0.0, f"CAMERA EXTRINSIC CALIBRATION FAILED ({exc}) -> using nominal mounting")
        sensing = Sensing(env, soft_cfg, calibration=calibration)
        safety = SafetyGuard(run_cfg)
        last_guard = ""
        estimator.reset((0.0, 0.0, 0.0))     # the robot's own frame: origin = start pose
        # Always learn the markers at run time: no tag / dock poses from the config
        tag_map = TagMap(min_obs=run_cfg.tags.survey_min_obs)
        map_view = MapView(tag_map, dock_id)
        explorer = Explorer(cfg, run_cfg, tag_map, dock_id, scan, dt)
        aim = AimController(run_cfg)
        estimator.set_tag_map(map_view)       # empty map: markers are ignored until learned
        phase = Phase.SURVEY
        if visualizer is not None:
            visualizer.set_world(world_cfg, obstacles)   # evaluation drawing only
            visualizer.start()

        with _viewer(sim, viewer) as view:
            step = 0
            last_rot_t = -1.0
            while env.time < cfg.sim.max_time:
                t0 = time.perf_counter()
                if view is not None and not view.is_running():
                    reason = "viewer closed"
                    break

                # 1) sensing and state estimation (no ground truth)
                frame = sensing.read()
                estimator.predict(frame.odometry.ds, frame.odometry.dtheta)
                if abs(frame.odometry.dtheta) / dt > 0.05:
                    last_rot_t = frame.time
                if frame.time - last_rot_t >= 0.3:
                    for marker in frame.markers:     # dock + extra tags, matched by id in the EKF
                        estimator.update(marker)
                estimator.update_lidar(frame.lidar)
                x_est = np.array(estimator.x, dtype=float)
                x_eval = est_to_world(x_est, start)      # world frame, for logging / display only
                sigma_xy = float(math.sqrt(max(float(np.trace(estimator.P[:2, :2])), 0.0)))
                if sensing.n_camera != last_n_cam:   # a new camera frame was processed
                    last_n_cam = sensing.n_camera
                    last_tags = tuple(sorted(m.marker_id for m in frame.markers))

                # 2) decide the velocity command from the ESTIMATED pose only
                v = w = 0.0
                ncmd = dcmd = None

                # 2-pre) SURVEY: spin at the known start pose and learn the tag poses (no ground truth).
                # dock_first: scan a full turn (or until the dock is seen) and use the learned dock as the goal.
                if phase is Phase.SURVEY:
                    if survey_t0 is None:
                        survey_t0 = frame.time
                        note(frame.time, "SURVEY: scanning 180 deg at the start pose to learn the tag map")
                    first = run_cfg.tags.dock_first
                    w_scan, observe = scan.update(frame.time, frame.odometry.dtheta)
                    if observe:                      # learn only while standing still
                        for marker in frame.markers:
                            is_dock = int(marker.marker_id) == dock_id
                            limit = run_cfg.tags.dock_max_range if is_dock else run_cfg.tags.survey_max_range
                            if sigma_xy > 0.12:
                                limit = min(limit, 2.0)
                            tag_map.add(x_est, marker, max_range=limit)
                    dock_seen = first and tag_map.count(dock_id) >= run_cfg.tags.survey_min_obs
                    angle_limit = run_cfg.tags.survey_angle          # 180 deg maximum
                    if dock_seen or scan.spun >= angle_limit or frame.time - survey_t0 > run_cfg.tags.survey_max_time:
                        learned = tag_map.table()
                        estimator.set_tag_map(map_view)
                        recovery = TagRecovery(run_cfg, tracker=nav.tracker, tag_map=tag_map)
                        scan.reset()
                        if dock_id in learned:
                            dock_lm, dock_yaw = learned[dock_id]
                        elif not first:
                            note(frame.time, "SURVEY: dock too far to learn from the start pose "
                                            "-> navigate and scan for it on the way")
                        if visualizer is not None:
                            visualizer.set_tag_map(table_to_world(learned, start))
                        if first and dock_id not in learned:
                            note(frame.time, "SURVEY: dock not seen -> EXPLORE (move, stop, scan)")
                            phase, explore_t0 = Phase.EXPLORE, frame.time
                            explorer.state = "PICK"
                        elif not learned:
                            note(frame.time, "SURVEY FAILED: no tag could be learned from the start pose")
                            phase, reason = Phase.FAILED, "survey failed"
                        else:
                            note(frame.time, f"SURVEY done: learned ids {sorted(learned)}")
                            if first:      # goal comes from the scan, not from the config waypoints
                                aim_mode = True
                                aim.restart_look()
                            phase = Phase.NAVIGATE
                    else:
                        v, w = 0.0, w_scan

                # 2-pre-a) EXPLORE (dock_first only): drive towards the nearest learned tag that has not been
                # visited yet (otherwise towards the freest direction), stop and scan, until the dock has been
                # seen enough times to learn its pose
                if phase is Phase.EXPLORE:
                    v, w = explorer.update(frame, x_est, sigma_xy)
                    if tag_map.count(dock_id) >= run_cfg.tags.survey_min_obs:
                        learned = tag_map.table()
                        estimator.set_tag_map(map_view)
                        recovery = TagRecovery(run_cfg, tracker=nav.tracker, tag_map=tag_map)
                        dock_lm, dock_yaw = learned[dock_id]
                        scan.reset()
                        aim_mode = True
                        aim.restart_look()
                        if visualizer is not None:
                            visualizer.set_tag_map(table_to_world(learned, start))
                        note(frame.time, f"EXPLORE done: dock learned, ids {sorted(learned)}")
                        phase = Phase.NAVIGATE
                    elif frame.time - explore_t0 > 120.0:
                        note(frame.time, "EXPLORE FAILED: dock not seen within 120 s")
                        phase, reason = Phase.FAILED, "explore failed"

                # 2-pre-c) re-learn the dock from CLOSE observations (accurate), then let the EKF use it
                if phase in (Phase.NAVIGATE, Phase.DOCK) and frame.marker is not None and frame.time - last_rot_t >= 0.3:
                    res = relearner.update(x_est, frame.marker, tag_map)
                    if res is not None:
                        dock_lm, dock_yaw, first_time = res
                        dock_refined = True
                        map_view.show_dock = True
                        estimator.set_tag_map(map_view)
                        dock.set_landmark(dock_lm, dock_yaw)
                        if first_time:
                            note(frame.time, "dock re-learned from a close view")

                # 2-pre-b) learn the tags / dock that were too far during SURVEY.
                # Only close observations while the filter is confident.
                if tag_map is not None and phase in (Phase.NAVIGATE, Phase.DOCK) and not estimator.is_lost() and frame.time - last_rot_t >= 0.3:
                    grew = False
                    for marker in frame.markers:
                        mid = int(marker.marker_id)
                        if tag_map.count(mid) < run_cfg.tags.survey_min_obs:
                            tag_map.add(x_est, marker, max_range=2.0)
                            grew = grew or tag_map.count(mid) >= run_cfg.tags.survey_min_obs
                    if grew:
                        learned = tag_map.table()
                        estimator.set_tag_map(map_view)
                        recovery = TagRecovery(run_cfg, tracker=nav.tracker, tag_map=tag_map)
                        if dock_id in learned and dock_lm is None:
                            dock_lm, dock_yaw = learned[dock_id]
                            dock.set_landmark(dock_lm, dock_yaw)
                        if visualizer is not None:
                            visualizer.set_tag_map(table_to_world(learned, start))
                        note(frame.time, f"map updated while navigating: learned ids {sorted(learned)}")

                # 2a) position uncertainty too large while navigating: start the tag recovery
                if (use_recovery and phase is Phase.NAVIGATE and recovery.available and n_tag_recoveries < MAX_RECOVERIES and estimator.is_lost()):
                    recover_from_aim = aim_mode
                    aim_mode = False
                    recovery.start(x_est)
                    n_tag_recoveries += 1
                    recovery_after_dock = False
                    last_target = None
                    phase = Phase.RECOVER
                    note(frame.time, f"LOST (sigma {sigma_xy:.3f} m > {cfg.tags.lost_std} m) "
                                    f"-> tag recovery #{n_tag_recoveries} started (spinning to look for a tag)")

                # 2b) recovery: spin / drive to the tags until the EKF is localised again
                if phase is Phase.RECOVER:
                    v, w = recovery.update(x_est, frame.lidar, not estimator.is_lost(),
                                        frame.odometry.dtheta)
                    if recovery.phase is RecoveryPhase.GOTO and recovery.target_id != last_target:
                        last_target = recovery.target_id
                        sx, sy = recovery.standoff_point(last_target)
                        note(frame.time, f"recovery: no tag seen, driving to tag {last_target} "
                                        f"stand-off ({sx:+.2f}, {sy:+.2f})")
                    if recovery.phase is RecoveryPhase.MAP_GOTO and last_target != -1:
                        last_target = -1
                        ax_, ay_ = recovery.approach_point()
                        note(frame.time, f"recovery: no tag found -> following the learned map to the "
                                        f"dock approach point ({ax_:+.2f}, {ay_:+.2f})")
                    if recovery.phase is RecoveryPhase.DONE:
                        if recovery.arrived_by_map:
                            note(frame.time, "map-follow finished (not relocalized) -> docking")
                            nav_time = frame.time if nav_time is None else nav_time
                            phase = Phase.DOCK
                            dock.begin(x_est, dock_lm, dock_yaw)
                        else:
                            note(frame.time, f"RELOCALIZED (sigma {sigma_xy:.3f} m, tag updates "
                                            f"{int(getattr(estimator, 'n_tag_updates', 0))}) -> resume navigation")
                            if recover_from_aim:
                                aim_mode = True
                                aim.restart_look()
                                recover_from_aim = False
                                phase = Phase.NAVIGATE
                            else:
                                approach = None if dock_lm is None else dock_approach_point(run_cfg, dock_lm, dock_yaw)
                                nav = resume_navigator(nav, run_cfg, recovery_after_dock, approach)
                                phase = Phase.NAVIGATE
                        v = w = 0.0
                    elif recovery.phase is RecoveryPhase.FAILED:
                        note(frame.time, "recovery FAILED: nothing left to try")
                        phase, reason = Phase.FAILED, "recovery failed"
                        v = w = 0.0

                if (phase is Phase.NAVIGATE and aim_mode and dock_lm is not None
                        and frame.marker is not None and frame.marker.range < 2.2 and frame.marker.range > 0.9
                        and abs(frame.marker.bearing) < 0.3
                        and abs(wrap_angle(frame.marker.bearing - frame.marker.view_angle)) < 0.35
                        and frame.time - last_rot_t >= 0.3 and frame.time - dock_fail_t > 5.0):
                    aim_mode = False
                    phase, nav_time = Phase.DOCK, frame.time
                    dock.begin(x_est, dock_lm, dock_yaw)
                    note(frame.time, f"dock seen at {frame.marker.range:.2f} m -> docking")

                # 2c-pre) AIM: look at the dock while standing still, turn towards it, drive a short step, repeat.
                # Dock not seen -> fallback chain: dock approach point, then the tags (nearest to the dock first),
                # each with a limited (180 deg) scan; nothing left -> blind docking / FAILED.
                if phase is Phase.NAVIGATE and aim_mode:
                    v, w, nav, ev = aim.update(frame, x_est, nav, scan, dock_lm, dock_yaw,
                                            tag_map, dock_id, dt, note)
                    if ev == "DOCK":
                        aim_mode = False
                        phase, nav_time = Phase.DOCK, frame.time
                        dock.begin(x_est, dock_lm, dock_yaw)
                    elif ev == "FAILED":
                        phase, reason = Phase.FAILED, "dock not found"
                    v, w = (v, w) if phase is Phase.NAVIGATE else (0.0, 0.0)

                # 2c) navigation
                if phase is Phase.NAVIGATE and not aim_mode:
                    ncmd = nav.update(x_est, frame.lidar)

                # 2d) docking (a failed docking widens P and starts the tag recovery)
                if phase is Phase.DOCK:
                    dock_t0 = frame.time if dock_t0 is None else dock_t0
                    if frame.time - dock_t0 > 20.0 and dock_retries < 3:
                        dock_t0, dock_fail_t = None, frame.time
                        dock_retries += 1
                        aim_mode = True
                        aim.restart_look()
                        phase = Phase.NAVIGATE
                        note(frame.time, "docking too long -> back to AIM")
                if phase is Phase.DOCK:                      # still docking (not switched to AIM above)
                    dcmd = dock.update(frame.marker, frame.odometry, x_est if (dock_use_estimate and dock_lm is not None and dock_refined) else None)
                    if dcmd.finished:
                        if dcmd.docked:
                            phase, reason = Phase.DONE, "docked"
                            note(frame.time, "DOCKED")
                        elif use_recovery and recovery.available and n_tag_recoveries < MAX_RECOVERIES:
                            estimator.inflate()              # accept the next tag detections
                            recovery.start(x_est)
                            n_tag_recoveries += 1
                            recovery_after_dock = True
                            last_target = None
                            phase = Phase.RECOVER
                            note(frame.time, f"docking failed -> P inflated -> tag recovery "
                                            f"#{n_tag_recoveries} started")
                        elif dock_retries < 3:
                            dock_retries += 1
                            dock_fail_t = frame.time
                            aim_mode = True
                            aim.restart_look()
                            phase = Phase.NAVIGATE
                            note(frame.time, f"docking failed -> back to AIM (retry {dock_retries}/3)")
                        else:
                            phase, reason = Phase.FAILED, "docking failed"
                            note(frame.time, "docking failed (no recovery left)")
                    else:
                        v, w = dcmd.v, dcmd.w
                else:
                    dock_t0 = None                           # reset the timer whenever we are NOT docking
                age = frame.lidar_age + (5.0 if lidar_fault_at is not None and frame.time >= lidar_fault_at else 0.0)
                v, w, g_reason = safety.filter(v, w, frame.lidar, dt, lidar_age=age, sigma_xy=sigma_xy)
                if g_reason != last_guard:
                    last_guard = g_reason
                    if g_reason:
                        note(frame.time, f"SAFETY: {g_reason}")
                if safety.fault:                      # latched E-STOP: end the run, never report success
                    phase, reason = Phase.FAILED, "safety fault"

                # 3) logging and evaluation (the ONLY place ground truth is read)
                hit = sim.collided
                collided_any = collided_any or hit
                mode = mode_label(phase, ncmd, dcmd,
                                recovery.phase if phase is Phase.RECOVER else None)
                if aim_mode and phase is Phase.NAVIGATE:
                    mode = "AIM-" + aim.state
                n_valid = int(frame.lidar.valid.sum())
                n_updates = int(getattr(estimator, "n_updates", 0))
                n_tag_upd = int(getattr(estimator, "n_tag_updates", 0))
                log.record(frame.time, sim.ground_truth(), x_eval, estimator.P,
                        phase.value, v, w, n_valid, n_updates, mode, n_tag_upd)
                if visualizer is not None:
                    visualizer.record(frame.time, x_eval, frame.lidar, mode, frame.marker,
                                    markers=frame.markers, frame=frame)
                    if phase is Phase.DONE:
                        visualizer.capture_target(label="Docked at charging station")
                if verbose and step % report_every == 0:
                    print(table_row(frame.time, x_est, n_valid, frame.lidar.valid.size,
                                    n_updates, v, w, mode, sigma_xy, n_tag_upd, last_tags), flush=True)

                if phase in (Phase.DONE, Phase.FAILED):
                    break
                if hit and stop_on_collision:
                    phase, reason = Phase.FAILED, "collision"
                    note(frame.time, "COLLISION")
                    break

                # 4) act
                if view is None:
                    env.step(v, w)
                else:
                    with view.lock():
                        env.step(v, w)
                    view.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))   # real-time pacing
                step += 1

        if visualizer is not None:
            viz_outputs = visualizer.finish(
                success=phase is Phase.DONE, reason=reason, sim_time=sim.time,
                extra={"seed": seed,
                    "calibration_source": "nominal_mujoco" if calibration is None
                                            else calibration.intrinsics.source,
                    "validation": {"simulation": phase is Phase.DONE,
                                    "hardware_ready": False,      # set True only after the checklist
                                    "hardware_verified": False},
                    "events": events})
        cam_stats = (sensing.n_camera, sensing.n_detect)
        n_detect_any = sensing.n_detect_any
        truth = sim.ground_truth()           # final pose, evaluation only
        sim_time = sim.time

    if verbose:
        print(table_footer())
        if viz_outputs:
            print("Route visualization saved to:", viz_outputs["visualization"])

    diag = {k: int(getattr(estimator, k, 0)) for k in
            ("n_updates", "n_rejected", "n_recovered", "n_lidar_updates", "n_lidar_rejected",
            "n_tag_updates")}
    diag["n_tag_recoveries"] = n_tag_recoveries
    diag["n_detect_any"] = int(n_detect_any)
    diag["nis_per_dof"] = (estimator.mean_nis_per_dof()
                        if hasattr(estimator, "mean_nis_per_dof") else float("nan"))
    if tag_map is not None:                  # EVALUATION ONLY: compare the learned map with the config poses
        truth_tags = world_cfg.tag_table()
        diag["tag_map_errors"] = {i: float(np.hypot(*(p - np.array(truth_tags[i][:2], dtype=float))))
                                    for i, (p, _) in table_to_world(tag_map.table(), start).items() if i in truth_tags}

    return RunResult(phase is Phase.DONE, reason, sim_time, nav_time, collided_any, phase,
                    est_to_world(estimator.x, start), truth, log,
                    camera_stats=cam_stats, diag=diag, obstacles=obstacles, events=events,
                    world_cfg=world_cfg)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track A: navigation + visual docking in MuJoCo")
    parser.add_argument("--seed", type=int, default=None, help="noise seed (default: config)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="run headless without the live MuJoCo window")
    parser.add_argument("--plot", action="store_true", help="save results/run.png")
    parser.add_argument("--compare", action="store_true",
                        help="also run a dead-reckoning baseline (no marker updates) for comparison")
    parser.add_argument("--dock-raw", action="store_true",
                        help="dock on the raw marker measurement instead of the EKF estimate")
    parser.add_argument("--trials", type=int, default=0,
                        help="run N headless trials (seeds seed..seed+N-1) and print statistics")
    parser.add_argument("--marker-cov", action="store_true",
                        help="EKF uses the per-measurement marker covariance from sensing.py")
    parser.add_argument("--sigma-px", type=float, default=0.5,
                        help="assumed ArUco corner noise [px] used to build the marker covariance")
    parser.add_argument("--no-viz", action="store_true",
                        help="do not save the route picture / snapshots")
    parser.add_argument("--no-live-camera", action="store_true",
                        help="do not open the live camera window")
    parser.add_argument("--no-survey", action="store_true",
                        help="skip the start-pose survey and use the tag poses written in the config")
    parser.add_argument("--ideal", action="store_true",
                        help="ideal simulation (no hidden errors, delays or sensor degradation)")
    parser.add_argument("--lidar-map", action="store_true",
                        help="EKF uses LiDAR against the fixed room walls (needs the world frame; off by default)")
    parser.add_argument("--lidar-fault", type=float, default=None,
                        help="simulate a LiDAR outage starting at this time [s]")
    parser.add_argument("--suite-trials", type=int, default=10,
                        help="trials per configuration in the default suite")
    parser.add_argument("--suite", action="store_true",
                        help="run the comparison suite (many seeds, headless) instead of one live run")
    parser.add_argument("--no-recovery", action="store_true", help="disable the tag recovery")
    parser.add_argument("--no-calib-odometry", action="store_true", help="skip the odometry calibration")
    parser.add_argument("--no-plot", action="store_true", help="do not save results/run.png")
    parser.add_argument("--no-calib-camera", action="store_true", help="skip camera calibration")
    parser.add_argument("--camera-delay", type=float, default=0.2,
                        help="camera latency [s] compensated in the EKF (try 0.2)")
    return parser

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.calib_camera = not args.no_calib_camera
    # Final system by default: recovery + odometry calibration + one merged obstacle layout
    args.recovery = not args.no_recovery
    args.calib_odometry = not args.no_calib_odometry
    args.plot = not args.no_plot
    cfg = replace(CONFIG, realism=RealismConfig(enabled=False)) if args.ideal else CONFIG

    if args.suite:
        return run_suite(run_pipeline, args.suite_trials, args.seed or 0, cfg)

    # A random layout needs a random seed; it is printed so the run can be reproduced
    seed = args.seed

    if args.trials > 0:
        return run_trials(run_pipeline, args.trials, seed or 0, cfg,
                        use_marker_cov=args.marker_cov, marker_sigma_px=args.sigma_px,
                        use_lidar_map=args.lidar_map, use_recovery=args.recovery,
                        calib_odometry=args.calib_odometry)
    use_viewer = not args.no_viewer
    if use_viewer:
        _ensure_mjpython()
    cal_seed = seed if seed is not None else cfg.sim.seed
    calibration = None
    if args.calib_camera:
        try:
            calibration = chessboard_demo(cfg, cal_seed, show=use_viewer)
        except ValueError as exc:
            print(f"Camera calibration failed ({exc}) -> using nominal intrinsics")
    viz = None if args.no_viz else RouteVisualizer(
        cfg, output_dir=RESULTS_DIR / "visualization" / f"seed_{seed if seed is not None else cfg.sim.seed}",
        live_view=use_viewer and not args.no_live_camera)
    result = run_pipeline(cfg, seed=seed, viewer=use_viewer, verbose=True,
                        dock_use_estimate=True,
                        marker_sigma_px=args.sigma_px, use_marker_cov=args.marker_cov,
                        visualizer=viz,
                        use_lidar_map=args.lidar_map, use_recovery=args.recovery,
                        lidar_fault_at=args.lidar_fault,
                        calib_odometry=args.calib_odometry, calib_camera=args.calib_camera,
                        camera_delay=args.camera_delay, calibration=calibration)
    ecfg = result.world_cfg or cfg          # the real world, for evaluation only

    dr_rmse = None
    if args.compare:
        # Same seed -> same obstacle layout as the main run
        baseline = run_pipeline(cfg, seed=seed,
                                estimator=EKFEstimator(cfg, use_marker=False, use_lidar=False))
        dr_rmse = compute_report(baseline, ecfg).get("position_rmse")

    print()
    print(result.summary(ecfg))
    print()
    print_report(result, ecfg, dr_rmse)
    if args.plot:
        print("Plot saved to:", plot_run(result, ecfg))
    return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())