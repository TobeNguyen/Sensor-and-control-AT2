# Including libraries
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import time
import numpy as np
from controller import LQRTracker, limit_command, safe_command
from config import CONFIG, RESULTS_DIR
from estimator import EKFEstimator
from robot_model import (analyse_model, default_landmark, jacobian_F, jacobian_G, innovation,
                        measurement_jacobian, measurement_jacobian_pose, measurement_model, measurement_model_pose, motion_model, wrap_angle, pose_error)
from visualization import compute_report

OUT = RESULTS_DIR / "evaluation"
BASE_KW = dict(dock_use_estimate=True, use_recovery=True, calib_odometry=True,
            calib_camera=True, camera_delay=0.2)

ROW_FIELDS = ["experiment", "label", "level", "seed", "success", "reason", "time", "collided",
            "pos_rmse", "heading_rmse", "mean_nees", "mean_nis", "det_pct", "rejected",
            "recoveries", "dock_range", "dock_lateral", "dock_heading", "obs_clear",
            "wall_clear", "mean_v", "mean_abs_w", "sat_frac", "avoid_frac", "path_ratio", "effort"]
METRICS = ["time", "pos_rmse", "heading_rmse", "mean_nees", "mean_nis", "dock_lateral",
        "dock_heading", "obs_clear", "mean_v", "mean_abs_w", "sat_frac", "avoid_frac",
        "path_ratio", "effort"]

ROW_FIELDS += ["xte_rmse", "xte_max", "hdg_err_rmse"]
ROW_FIELDS += ["last_event"]
METRICS += ["xte_rmse", "xte_max", "hdg_err_rmse"]
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")
    
def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

class UKFEstimator(EKFEstimator):
    # Same filter as the EKF (same predict, gating, bookkeeping); the marker correction uses the unscented transform
    def _ut(self, h, z, R):
        n, lam = 3, 1.0
        L = np.linalg.cholesky((n + lam) * self.P)
        X = [self.x] + [self.x + L[:, i] for i in range(n)] + [self.x - L[:, i] for i in range(n)]
        wm = np.full(2 * n + 1, 0.5 / (n + lam)); wm[0] = lam / (n + lam)
        wc = wm.copy(); wc[0] += 2.0                       # beta = 2, alpha = 1
        Z = [h(s) for s in X]
        dZ = np.array([innovation(zi, Z[0]) for zi in Z])  # wrapped residuals
        z_pred = Z[0] + wm @ dZ
        dz = dZ - wm @ dZ
        dX = np.array([s - self.x for s in X])
        dX[:, 2] = [wrap_angle(a) for a in dX[:, 2]]
        S = dz.T @ (wc[:, None] * dz) + R
        Pxz = dX.T @ (wc[:, None] * dz)
        return innovation(z, z_pred), S, Pxz

    def update(self, marker) -> bool:
        if not self.use_marker or marker is None or not self._is_valid(marker):
            return False
        marker = self._to_now(marker)
        mid = int(getattr(marker, "marker_id", self.cfg.camera.marker_id))
        lm, tag_yaw = self.landmarks[mid]
        view = getattr(marker, "view_angle", None)
        if self.use_view and view is not None and math.isfinite(view):
            z = np.array([marker.range, marker.bearing, view], dtype=float)
            R = np.zeros((3, 3)); R[:2, :2] = self.R; R[2, 2] = self.R_view
            h = lambda s: measurement_model_pose(s, lm, tag_yaw)
        else:
            z = np.array([marker.range, marker.bearing], dtype=float)
            R = self.R
            h = lambda s: measurement_model(s, lm)
        try:
            y, S, Pxz = self._ut(h, z, R)
            nis = float(y @ np.linalg.solve(S, y))
            self.last_innovation, self.last_nis = y, nis
            gate = self._gate_for(len(z))
            if gate is not None and nis > gate:
                self.n_rejected += 1
                self._reject_run += 1
                if self._reject_run < self.max_rejects:
                    return False
                self.P = self.P * (nis / gate)             # lock-out recovery, same rule as the EKF
                y, S, Pxz = self._ut(h, z, R)
                nis = float(y @ np.linalg.solve(S, y))
                self.n_recovered += 1
        except (np.linalg.LinAlgError, ValueError):
            return False
        self._reject_run = 0
        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        P = self.P - K @ S @ K.T
        self.P = 0.5 * (P + P.T)
        self.nis_history.append(nis); self.nis_dof.append(len(z))
        self.n_updates += 1
        if mid != int(self.cfg.camera.marker_id):
            self.n_tag_updates += 1
        return True

# Helpers
# Same EKF by default with tunable noise
def make_ekf(cfg, r_scale=1.0, gate=9.21, odom_scale=1.0, use_marker=True, use_lidar=False,
            camera_delay=0.2, kind="ekf"):
    ekf_cfg = replace(cfg, env=replace(cfg.env, obstacles=()))
    cls = UKFEstimator if kind == "ukf" else EKFEstimator
    return cls(ekf_cfg, gate=gate, odom_cov_scale=odom_scale, camera_delay=camera_delay,
                        R=np.diag([0.15 ** 2, 0.05 ** 2]) * r_scale, R_view=(0.15 ** 2) * r_scale,
                        use_marker=use_marker, use_lidar=use_lidar)

@contextmanager
def lqr_scaled(q_scale: float, r_scale: float):
    import navigation
    from controller import lqr_gain
    orig = navigation.LQRTracker

    class Scaled(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.Q, self.R = self.Q * q_scale, self.R * r_scale
            self.K = lqr_gain(self.Ad, self.Bd, self.Q, self.R)

    navigation.LQRTracker = Scaled
    try:
        yield
    finally:
        navigation.LQRTracker = orig

ERR_LOG: list = []   # [e_along, e_cross, e_heading] per compute() call (estimated pose)

def _log_error(x_est, ref):
    x = np.asarray(x_est, dtype=float)
    if x.shape == (3,) and np.all(np.isfinite(x)) and np.all(np.isfinite(ref.pose)):
        e = np.array(pose_error(x, ref.pose), dtype=float)
        if abs(e[0]) > 1e-9 or abs(e[1]) > 1e-9:
            ERR_LOG.append(e)
        return e
    return None

class LoggedLQR(LQRTracker):
    # Same LQR as the pipeline, only records the tracking error
    def compute(self, x_est, ref, front_range=None):
        _log_error(x_est, ref)
        return super().compute(x_est, ref, front_range)

class StanleyTracker(LQRTracker):
    # Baseline: proportional law on the same tracking error
    kx, ky, kth = 1.0, 1.0, 1.0
    def compute(self, x_est, ref, front_range=None):
        e = _log_error(x_est, ref)
        if e is None:
            return 0.0, 0.0
        v, w = limit_command(ref.v + self.kx * e[0],
                             ref.w + ref.v * self.ky * e[1] + self.kth * e[2], self.cfg)
        if front_range is not None:
            v, w = safe_command(v, w, front_range, self.cfg)
        return v, w

@contextmanager
def tracker_swapped(cls):
    import navigation
    orig = navigation.LQRTracker
    navigation.LQRTracker = cls
    try:
        yield
    finally:
        navigation.LQRTracker = orig

@contextmanager
def cruise_speed(v, redesign):
    import main, navigation
    o_nav, o_main, o_trk = navigation.Navigator, main.Navigator, navigation.LQRTracker

    class Fast(o_nav):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.v_cruise = v

    class Redesigned(o_trk):
        def __init__(self, cfg=CONFIG, **k):
            super().__init__(cfg, v_design=v)

    navigation.Navigator = main.Navigator = Fast
    if redesign:
        navigation.LQRTracker = Redesigned
    try:
        yield
    finally:
        navigation.Navigator, main.Navigator, navigation.LQRTracker = o_nav, o_main, o_trk

def track_errors() -> dict:
    E = np.array(ERR_LOG, dtype=float)
    if E.ndim != 2 or len(E) == 0:
        return {}
    return dict(xte_rmse=float(np.sqrt(np.mean(E[:, 1] ** 2))),
                xte_max=float(np.max(np.abs(E[:, 1]))),
                hdg_err_rmse=float(np.sqrt(np.mean(E[:, 2] ** 2))))

def run_once(exp, label, level, cfg, seed, kw=None, ekf=None, lqr=None, tracker=None, speed=None) -> dict:
    import main
    kw = {**BASE_KW, **(kw or {})}
    row = {k: float("nan") for k in ROW_FIELDS}
    row.update(experiment=exp, label=label, level=str(level), seed=seed, success=0, reason="exception")
    try:
        ERR_LOG.clear()
        est = make_ekf(cfg, **ekf) if ekf is not None else None
        with tracker_swapped(tracker or LoggedLQR), (lqr_scaled(*lqr) if lqr else nullcontext()), \
            (cruise_speed(*speed) if speed else nullcontext()):
            res = main.run_pipeline(cfg, seed=seed, estimator=est, **kw)
        wcfg = res.world_cfg or cfg
        rep = compute_report(res, wcfg)
        m = res.metrics(wcfg)
        a = res.log.arrays()
        dt = cfg.sim.control_dt
        n_cam, n_det = res.camera_stats
        truth = a["truth"][:, :2]
        path = float(np.sum(np.hypot(*np.diff(truth, axis=0).T))) if len(truth) > 1 else float("nan")
        straight = float(np.hypot(*(truth[-1] - truth[0]))) if len(truth) > 1 else float("nan")
        row.update(
            success=int(res.success), reason=res.reason, time=res.time, collided=int(res.collided),
            pos_rmse=rep.get("position_rmse", float("nan")), heading_rmse=rep.get("heading_rmse", float("nan")),
            mean_nees=rep.get("mean_nees", float("nan")), mean_nis=res.diag.get("nis_per_dof", float("nan")),
            det_pct=100.0 * n_det / max(1, n_cam), rejected=res.diag.get("n_rejected", 0),
            recoveries=res.diag.get("n_tag_recoveries", 0), dock_range=m["dock_range"],
            dock_lateral=m["dock_lateral"], dock_heading=m["dock_heading"],
            obs_clear=rep.get("min_obstacle_clearance", float("nan")),
            wall_clear=rep.get("min_wall_clearance", float("nan")),
            mean_v=rep.get("mean_v", float("nan")), mean_abs_w=rep.get("mean_abs_w", float("nan")),
            sat_frac=rep.get("saturated_fraction", float("nan")),
            avoid_frac=rep.get("avoid_fraction", float("nan")),
            path_ratio=path / max(straight, 1e-6), effort=float(np.sum(a["v"] ** 2 + a["w"] ** 2) * dt))
        row.update(track_errors())
        row["last_event"] = res.events[-1] if res.events else ""
    except Exception as exc:                                  # a crashed run counts as a failure
        import traceback
        row["reason"] = f"exception: {type(exc).__name__}"
        print("   ", traceback.format_exc().strip().splitlines()[-3:], flush=True)
    print(f"  [{exp}] {label}={level} seed {seed}: {'OK  ' if row['success'] else 'FAIL'} {row['reason']}", flush=True)
    return row

def save_runs(exp: str, rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{exp}_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)

def seeds_of(args):
    return range(args.seed0, args.seed0 + args.seeds)

# A.Model
def fd_jacobian(f, x, wrap_rows=(), eps=1e-6):
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        diff = np.asarray(f(x + d)) - np.asarray(f(x - d))
        for r in wrap_rows:
            diff[r] = wrap_angle(diff[r])
        cols.append(diff / (2 * eps))
    return np.array(cols).T

def gramian_min_eig(x0, v, w, landmark, cfg, steps=100, with_view=True):
    # Smallest eigenvalue of the empirical observability Gramian along a trajectory (0 = unobservable direction)
    dt, cam = cfg.sim.control_dt, cfg.camera
    sig = [cam.range_noise_std, cam.bearing_noise_std] + ([cam.view_noise_std] if with_view else [])
    Rinv = np.diag(1.0 / np.square(sig))
    x, Phi, W = np.array(x0, dtype=float), np.eye(3), np.zeros((3, 3))
    for _ in range(steps):
        H = measurement_jacobian_pose(x, landmark) if with_view else measurement_jacobian(x, landmark)
        W += (H @ Phi).T @ Rinv @ (H @ Phi)
        Phi = jacobian_F(x, v * dt, w * dt) @ Phi
        x = motion_model(x, v * dt, w * dt)
    return float(np.linalg.eigvalsh(W)[0])

def exp_model(args) -> None:
    cfg, dt = CONFIG, CONFIG.sim.control_dt
    x = np.array([0.3, 0.2, 0.3])
    points = [("stopped", 0.0, 0.0), ("straight", 0.2, 0.0), ("turn in place", 0.0, 0.5), ("arc", 0.15, 0.4)]
    lm_near = x[:2] + np.array([0.5, 0.2])
    lm_far = x[:2] + np.array([3.0, 1.0])
    rows = []
    for name, v, w in points:
        for lm_name, lm in (("near", lm_near), ("far", lm_far)):
            r = analyse_model(x, v, w, landmark=lm, cfg=cfg)
            rows.append({"operating_point": name, "v": v, "w": w, "landmark": lm_name, **r})

    # Jacobian check against finite differences
    ds, dth = 0.05, 0.04
    errs = {
        "F": float(np.max(np.abs(jacobian_F(x, ds, dth) - fd_jacobian(lambda s: motion_model(s, ds, dth), x, (2,))))),
        "G": float(np.max(np.abs(jacobian_G(x, ds, dth) - fd_jacobian(lambda u: motion_model(x, u[0], u[1]), [ds, dth], (2,))))),
        "H": float(np.max(np.abs(measurement_jacobian(x, lm_near) - fd_jacobian(lambda s: measurement_model(s, lm_near), x, (1,))))),
    }

    # LQR closed-loop poles at the design point
    from controller import LQRTracker
    tr = LQRTracker(cfg)
    out = {"operating_points": rows, "jacobian_max_abs_error": errs,
        "lqr_gain_K": tr.K, "lqr_pole_magnitudes": np.abs(tr.closed_loop_poles()),
        "lqr_stable": bool(np.all(np.abs(tr.closed_loop_poles()) < 1.0))}
    out["gramian_min_eig"] = {f"{n} | {'with' if wv else 'without'} view":
                            gramian_min_eig(x, v, w, lm_near, cfg, with_view=wv)
                            for n, v, w in points for wv in (False, True)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "modelling.json").write_text(json.dumps(out, indent=2, default=_json), encoding="utf-8")
    with (OUT / "modelling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("A. modelling saved. Jacobian errors:", {k: f"{v:.2e}" for k, v in errs.items()},
        "| LQR stable:", out["lqr_stable"])
    
# B.Sensing
def exp_sensing(args, n_frames: int = 300, warmup: int = 15) -> None:
    from calibration import apply_extrinsics, calibrate_camera_in_sim, calibrate_extrinsics_in_sim
    from environment import MuJoCoEnvironment
    from sensing import Sensing
    from simulation import Simulation
    with Simulation(CONFIG, seed=args.seed0) as sim:
        world = sim.world_cfg
    cal = calibrate_camera_in_sim(world, CONFIG, seed=args.seed0)
    fwd, lat, yaw, rep = calibrate_extrinsics_in_sim(CONFIG, args.seed0, calibration=cal)
    print(f"  [sensing] extrinsics: fwd {fwd:.4f} m (hidden true {rep['hidden_forward']:.4f}), "
        f"lat {lat:+.4f}, yaw {math.degrees(yaw):+.2f} deg")
    setups = [("before calibration", CONFIG, None),
            ("after calibration", apply_extrinsics(CONFIG, fwd, lat, yaw), cal)]
    rows = []
    with Simulation(CONFIG, seed=args.seed0) as sim:
        dx, dy, dyaw = sim.world_cfg.env.dock_pose
        env = MuJoCoEnvironment(sim)
        for tag, cfg_s, cal_s in setups:
            for true_r in (0.6, 0.9, 1.2, 1.6, 2.0, 2.5):
                sim.reset((dx + true_r * math.cos(dyaw), dy + true_r * math.sin(dyaw), wrap_angle(dyaw - math.pi)))
                sensing = Sensing(env, cfg_s, calibration=cal_s)
                last, n_cam = sensing.n_camera, 0
                rng_, brg_, view_ = [], [], []
                for k in range(warmup + n_frames):
                    frame = sensing.read()
                    new, last = sensing.n_camera != last, sensing.n_camera
                    if k >= warmup and new:                       # count camera frames, not control steps
                        n_cam += 1
                        if frame.marker is not None:
                            rng_.append(frame.marker.range)
                            brg_.append(frame.marker.bearing)
                            view_.append(frame.marker.view_angle)
                    env.step(0.0, 0.0)
                n = len(rng_)
                rows.append({"calibration": tag, "true_range_m": true_r,
                             "detected_pct": 100.0 * n / max(1, n_cam),
                            "range_bias_m": float(np.mean(rng_) - true_r) if n else float("nan"),
                            "range_std_m": float(np.std(rng_)) if n else float("nan"),
                            "bearing_std_rad": float(np.std(brg_)) if n else float("nan"),
                            "view_std_rad": float(np.std(view_)) if n else float("nan")})
                print(f"  [sensing] {tag}, range {true_r:.1f} m: {n}/{n_cam} camera frames detected", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sensing_marker_noise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(11, 4))
    a1, a2 = fig.subplots(1, 2)
    for tag, style in (("before calibration", "o--"), ("after calibration", "s-")):
        g = [r for r in rows if r["calibration"] == tag]
        x = [r["true_range_m"] for r in g]
        a1.plot(x, [abs(r["range_bias_m"]) for r in g], style, label=f"|bias| {tag}")
        a2.plot(x, [r["bearing_std_rad"] for r in g], style, label=f"bearing std {tag}")
    a1.set_xlabel("true range [m]"); a1.set_ylabel("|range bias| [m]"); a1.legend(fontsize=7)
    a2.set_xlabel("true range [m]"); a2.set_ylabel("bearing std [rad]"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "sensing_marker_noise.png", dpi=140)
    print("B. sensing saved: bias before vs after calibration, std vs cfg.R_camera().")
    exp_lidar(args)

def exp_lidar(args, n=200):
    from simulation import Simulation
    hx, rows = CONFIG.env.room_half_size[0], []
    with Simulation(CONFIG, seed=args.seed0) as sim:
        for d in (0.5, 1.0, 1.5, 2.0, 3.0):
            sim.reset((hx - d, 0.0, 0.0))              # facing the east wall, beam 0 = straight ahead
            r = np.array([sim.read_lidar()[0] for _ in range(n)])
            r = r[r < CONFIG.lidar.range_max - 1e-6]   # drop dropped beams
            rows.append({"true_range_m": d, "range_bias_m": float(r.mean() - d), "range_std_m": float(r.std())})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "lidar_noise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# C.Estimation
def exp_estimation(args) -> None:
    rows = []
    variants = [("pipeline default", None),
                ("dead reckoning", dict(use_marker=False)),
                ("camera only", dict()),
                ("camera + LiDAR", dict(use_lidar=True)),
                ("EKF tuned R", dict(r_scale=0.03)),
                ("UKF", dict(kind="ukf")),
                ("UKF tuned R", dict(kind="ukf", r_scale=0.03))]
    sweeps = [("R x", [0.01, 0.03, 0.1, 0.25, 0.5, 2, 4], "r_scale"),
            ("gate", [6.0, 13.8, None], "gate"),
            ("odom cov x", [0.5, 2, 5], "odom_scale")]
    for seed in seeds_of(args):
        for name, ekf in variants:
            rows.append(run_once("estimation", name, "-", CONFIG, seed, ekf=ekf))
        for label, levels, key in sweeps:
            for lv in levels:
                rows.append(run_once("estimation", f"sweep {label}", lv, CONFIG, seed, ekf={key: lv}))
    save_runs("estimation", rows)

# D.Control
def exp_control(args) -> None:
    rows = []
    gains = [("Q x0.1", (0.1, 1.0)), ("Q x1 (default)", (1.0, 1.0)), ("Q x10", (10.0, 1.0)),
            ("R x10", (1.0, 10.0)), ("R x0.1", (1.0, 0.1))]
    for seed in seeds_of(args):
        for name, lqr in gains:
            rows.append(run_once("control", name, "-", CONFIG, seed, lqr=lqr))
        rows.append(run_once("control", "Stanley (baseline)", "-", CONFIG, seed, tracker=StanleyTracker))
        for v in (0.10, 0.22):
            rows.append(run_once("control", "LQR design 0.15", v, CONFIG, seed, speed=(v, False)))
            rows.append(run_once("control", "LQR redesigned", v, CONFIG, seed, speed=(v, True)))
    save_runs("control", rows)

# E.Robustness
def _rz(field_name):
    return lambda c, k: replace(c, realism=replace(c.realism, **{field_name: k}))

ROBUST = {   # name: (levels, cfg modifier(cfg, level), extra run kwargs(level))
    "encoder noise x": ([0.5, 1, 2, 4], lambda c, k: replace(c, robot=replace(c.robot, encoder_noise_std=c.robot.encoder_noise_std * k)), None),
    "LiDAR noise x": ([0.5, 1, 2, 4], lambda c, k: replace(c, lidar=replace(c.lidar, noise_std=c.lidar.noise_std * k)), None),
    "LiDAR dropout": ([0.0, 0.05, 0.15, 0.3], _rz("lidar_dropout"), None),
    "camera frame drop": ([0.0, 0.1, 0.3, 0.5], _rz("frame_drop"), None),
    "camera delay s (not compensated)": ([0.0, 0.1, 0.2, 0.4], _rz("camera_delay"), lambda k: dict(camera_delay=0.0)),
    "camera delay s (compensated)": ([0.0, 0.1, 0.2, 0.4], _rz("camera_delay"), lambda k: dict(camera_delay=k)),
    "dock shift x": ([0.0, 1, 2], lambda c, k: replace(c, realism=replace(
        c.realism, dock_shift_xy=tuple(v * k for v in c.realism.dock_shift_xy))), None),
    "wheel slip x": ([0.0, 1, 2, 3], lambda c, k: replace(c, realism=replace(
        c.realism, slip_v=c.realism.slip_v * k, slip_w=c.realism.slip_w * k)), None),
    "combined noise x": ([1, 2, 3], lambda c, k: replace(
        c, robot=replace(c.robot, encoder_noise_std=c.robot.encoder_noise_std * k),
        lidar=replace(c.lidar, noise_std=c.lidar.noise_std * k),
        realism=replace(c.realism, frame_drop=min(0.05 * k, 0.5), lidar_dropout=min(0.03 * k, 0.3))), None),
}

def exp_robust(args) -> None:
    rows = []
    for seed in seeds_of(args):
        for name, (levels, mod, kwf) in ROBUST.items():
            for lv in levels:
                rows.append(run_once("robust", name, lv, mod(CONFIG, lv), seed,
                                    kw=kwf(lv) if kwf else None))
    save_runs("robust", rows)

def exp_ablation(args) -> None:
    rows = []
    variants = [("full system", dict(calib_camera=True)),
                ("no recovery", dict(calib_camera=True, use_recovery=False)),
                ("no odometry calibration", dict(calib_camera=True, calib_odometry=False)),
                ("no camera calibration", dict(calib_camera=False)),
                ("no extrinsics calibration", dict(calib_extrinsics=False)),
                ("no delay compensation", dict(camera_delay=0.0)),
                ("LiDAR map update on", dict(calib_camera=True, use_lidar_map=True))]
    for seed in seeds_of(args):
        for name, kw in variants:
            rows.append(run_once("ablation", name, "-", CONFIG, seed, kw=kw))
    save_runs("ablation", rows)

# Random obstacle layouts and start headings (only cfg changes, the simulator is untouched)
def random_layout(seed, n_obs=3, r=0.2):
    rng = np.random.default_rng(seed + 12345)
    hx, hy = CONFIG.env.room_half_size
    dx, dy, dyaw = CONFIG.env.dock_pose
    keep = [(dx + math.cos(dyaw), dy + math.sin(dyaw), 0.9), (dx, dy, 0.6), (0.0, 0.0, 0.9)]   # dock approach, dock, start
    keep += [(tx + 0.9 * math.cos(ty_), ty + 0.9 * math.sin(ty_), 0.7)                          # tag stand-off points
            for tx, ty, ty_ in ((a, b, c) for a, b, c in CONFIG.tags.fixed_tags)]
    obs = []
    for _ in range(20000):
        if len(obs) == n_obs:
            break
        x, y = rng.uniform(-hx + 0.6, hx - 0.6), rng.uniform(-hy + 0.6, hy - 0.6)
        if any(math.hypot(x - kx, y - ky) < kr + r for kx, ky, kr in keep):
            continue
        if any(math.hypot(x - ox, y - oy) < 2 * r + 0.6 for ox, oy, _ in obs):
            continue
        obs.append((x, y, r))
    return tuple(obs)

def exp_layouts(args) -> None:
    rows = []
    for seed in seeds_of(args):
        heading = float(np.random.default_rng(seed + 999).uniform(-math.pi, math.pi))
        cfg = replace(CONFIG, env=replace(CONFIG.env, obstacles=random_layout(seed),
                                        start_pose=(0.0, 0.0, heading)))
        rows.append(run_once("layouts", "random layout + start heading", "-", cfg, seed))
    save_runs("layouts", rows)

# Summary
def load_all_runs() -> list:
    rows = []
    for p in sorted(OUT.glob("*_runs.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows

def summarize(rows: list) -> list:
    groups = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["label"], r["level"]), []).append(r)
    out = []
    for (e, l, lv), g in groups.items():
        d = {"experiment": e, "label": l, "level": lv, "n": len(g),
            "success_rate": float(np.mean([_num(x["success"]) for x in g]))}
        for m in METRICS:
            v = np.array([_num(x.get(m)) for x in g], dtype=float)
            v = v[np.isfinite(v)]
            d[m] = float(v.mean()) if len(v) else float("nan")
            d[m + "_std"] = float(v.std()) if len(v) else float("nan")
        d["collisions"] = int(sum(_num(x["collided"]) == 1 for x in g))
        d["failures"] = json.dumps(dict(Counter(x["reason"] for x in g if _num(x["success"]) != 1)))
        n_, p, z = len(g), d["success_rate"], 1.96
        den = 1 + z * z / n_
        half = z * math.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_))
        d["success_lo"] = (p + z * z / (2 * n_) - half) / den
        d["success_hi"] = (p + z * z / (2 * n_) + half) / den
        out.append(d)
    return out

def write_summary(summary: list) -> None:
    if not summary:
        return
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    lines = ["| experiment | setting | level | n | success ±95% | time [s] | RMSE [m] | XTE rms [m] | XTE max [m] | hdg rms [rad] | NEES (~3) | NIS/dof (~1) | dock lat [m] | failures |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in summary:
        lines.append(f"| {d['experiment']} | {d['label']} | {d['level']} | {d['n']} | "
                    f"{100 * d['success_rate']:.0f}% [{100 * d['success_lo']:.0f}-{100 * d['success_hi']:.0f}] | "
                    f"{d['time']:.1f} | {d['pos_rmse']:.3f} | {d['xte_rmse']:.3f} | {d['xte_max']:.3f} | "
                    f"{d['hdg_err_rmse']:.3f} | {d['mean_nees']:.2f} | {d['mean_nis']:.2f} | "
                    f"{d['dock_lateral']:.3f} | {d['failures']} |")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

def make_figures(summary: list) -> None:
    from matplotlib.figure import Figure
    by = lambda e: [d for d in summary if d["experiment"] == e]

    rob = by("robust")                      # E: success rate vs level
    if rob:
        names = list(dict.fromkeys(d["label"] for d in rob))
        cols = 3
        fig = Figure(figsize=(4.2 * cols, 3.2 * math.ceil(len(names) / cols)))
        axes = fig.subplots(math.ceil(len(names) / cols), cols, squeeze=False).ravel()
        for ax, n in zip(axes, names):
            g = sorted((d for d in rob if d["label"] == n), key=lambda d: _num(d["level"]))
            ax.plot([_num(d["level"]) for d in g], [100 * d["success_rate"] for d in g], "o-")
            ax.set_ylim(-5, 105); ax.set_title(n, fontsize=9); ax.set_ylabel("success [%]"); ax.grid(alpha=0.3)
        for ax in axes[len(names):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(OUT / "robustness_success.png", dpi=140)

    est = by("estimation")                  # C: accuracy vs consistency
    if est:
        fig = Figure(figsize=(7, 5)); ax = fig.subplots()
        for d in est:
            if math.isfinite(d["pos_rmse"]) and math.isfinite(d["mean_nees"]):
                name = d["label"] + ("" if d["level"] == "-" else f" {d['level']}")
                ax.scatter(d["pos_rmse"], d["mean_nees"]); ax.annotate(name, (d["pos_rmse"], d["mean_nees"]), fontsize=6)
        ax.axhline(3.0, color="gray", ls=":"); ax.set_yscale("log")
        ax.set_xlabel("position RMSE [m]"); ax.set_ylabel("mean NEES (consistent ~ 3)")
        ax.set_title("Estimator accuracy vs consistency"); fig.tight_layout()
        fig.savefig(OUT / "estimation_tradeoff.png", dpi=140)

    for exp, fname in (("control", "control_gains.png"), ("ablation", "ablation.png")):
        g = by(exp)                         # D and ablation: bars
        if not g:
            continue
        keys = [("success_rate", "success rate"), ("dock_lateral", "dock lateral [m]"),
                ("effort", "control effort"), ("time", "time [s]"), ("xte_rmse", "cross-track RMS [m]")]
        fig = Figure(figsize=(14, 4)); axes = fig.subplots(1, len(keys))
        for ax, (k, title) in zip(axes, keys):
            ax.bar(range(len(g)), [d[k] for d in g])
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([d["label"] + ("" if d["level"] == "-" else f" {d['level']}") for d in g], rotation=60, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=140)

def finalize() -> None:
    rows = load_all_runs()
    if not rows:
        print("No run files found in", OUT)
        return
    summary = summarize(rows)
    write_summary(summary)
    make_figures(summary)
    print("Summary + figures written to", OUT)

# Main
# New seed on every run
def fresh_seed() -> int:
    return int(time.time()) % 1_000_000
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluation of the Track A pipeline")
    p.add_argument("--all", action="store_true", help="run everything (default when no flag is given)")
    p.add_argument("--model", action="store_true")
    p.add_argument("--sensing", action="store_true")
    p.add_argument("--estimation", action="store_true")
    p.add_argument("--control", action="store_true")
    p.add_argument("--robust", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--seeds", type=int, default=1, help="seeds per setting (default 1)")
    p.add_argument("--seed0", type=int, default=None, help="first seed (default: new seed every run)")
    a = p.parse_args()
    if a.seed0 is None:
        a.seed0 = fresh_seed()
    print(f"Evaluation seeds: {a.seed0} .. {a.seed0 + a.seeds - 1}  (reproduce with --seed0 {a.seed0} --seeds {a.seeds})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    chosen = any([a.model, a.sensing, a.estimation, a.control, a.robust, a.ablation, a.plot_only])
    if not chosen:
        a.all = True                      # plain `python evaluation.py` = evaluate everything

    if not a.plot_only:
        if a.all or a.model:
            exp_model(a)
        if a.all or a.sensing:
            exp_sensing(a)
        if a.all or a.estimation:
            exp_estimation(a)
        if a.all or a.control:
            exp_control(a)
        if a.all or a.robust:
            exp_robust(a)
        if a.all or a.ablation:
            exp_ablation(a)
    finalize()

if __name__ == "__main__":
    main()
