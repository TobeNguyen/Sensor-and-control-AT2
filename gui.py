# Including libraries
from __future__ import annotations

import argparse
import hmac
import logging
import math
import queue
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional, Sequence

import numpy as np

from config import CONFIG, RESULTS_DIR, Config

LOG = logging.getLogger("track_a.gui")

HOST = "127.0.0.1"                  # never bind anything else
DEFAULT_PORT = 8765                 # 5000 is taken by AirPlay on macOS
MAX_BODY_BYTES = 16 * 1024
MAX_PENDING_COMMANDS = 16
MAX_OBSTACLES = 10
MAX_WAYPOINTS = 20
MAX_SEED = 2**31 - 1
COORD_LIMIT = 100.0                 # [m] sanity limit for any coordinate
RADIUS_LIMIT = 10.0                 # [m]
ANGLE_LIMIT_DEG = 3600.0
APPROACH_EXTRA = 0.30               # [m] approach point = target_distance + this from marker
CLEARANCE_MARGIN = 0.05             # [m] extra clearance around obstacles
MARKER_CLEARANCE = 0.10             # [m] min gap between an obstacle and the marker
RADIUS_RANGE = (0.15, 0.30)         # [m] obstacle radius range used by Randomize
MAX_RANDOM_ATTEMPTS = 2000
COMMANDS = ("apply", "start", "pause", "reset")

# Pure logic
def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

@dataclass(frozen=True)
class StatusMessage:
    state: str = "IDLE"            # IDLE, NAVIGATING, DOCKING, SUCCESS, FAILED, ERROR
    phase: str = ""
    sim_time: float = 0.0          # [s]
    est_pose: tuple = (math.nan, math.nan, math.nan)   # EKF estimate
    gt_pose: tuple = (math.nan, math.nan, math.nan)    # ground truth (eval only)
    goal_distance: float = math.nan   # [m]
    min_lidar: float = math.nan       # [m]
    collisions: int = 0
    message: str = ""

    @property
    def position_error(self) -> float:
        return math.hypot(self.est_pose[0] - self.gt_pose[0],
                        self.est_pose[1] - self.gt_pose[1])

    @property
    def heading_error(self) -> float:
        d = self.est_pose[2] - self.gt_pose[2]
        return _wrap(d) if math.isfinite(d) else math.nan
    
def _clean(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None

def status_to_dict(msg: StatusMessage, running: bool = False, trial_id: int = 0) -> dict:
    he = msg.heading_error
    return {
        "state": msg.state, "phase": msg.phase, "message": msg.message,
        "sim_time": _clean(msg.sim_time),
        "est_pose": [_clean(v) for v in msg.est_pose],
        "gt_pose": [_clean(v) for v in msg.gt_pose],
        "goal_distance": _clean(msg.goal_distance),
        "min_lidar": _clean(msg.min_lidar),
        "position_error": _clean(msg.position_error),
        "heading_error_deg": math.degrees(he) if math.isfinite(he) else None,
        "collisions": int(msg.collisions),
        "running": bool(running), "trial_id": int(trial_id),
    }

def dock_approach_point(dock_pose: Sequence[float], cfg: Config) -> tuple:
    dx, dy, yaw = dock_pose
    d = cfg.docking.target_distance + APPROACH_EXTRA
    return (dx + d * math.cos(yaw), dy + d * math.sin(yaw))

def build_waypoints(intermediate: Sequence[Sequence[float]], dock_pose: Sequence[float],
                    cfg: Config) -> tuple:
    pts = tuple((float(p[0]), float(p[1])) for p in intermediate)
    return pts + (dock_approach_point(dock_pose, cfg),)

def make_config(base: Config, start: Sequence[float], dock: Sequence[float],
                obstacles: Sequence[Sequence[float]],
                intermediate: Sequence[Sequence[float]]) -> Config:
    env = replace(
        base.env,
        start_pose=tuple(float(v) for v in start),
        dock_pose=tuple(float(v) for v in dock),
        obstacles=tuple(tuple(float(v) for v in o) for o in obstacles),
        waypoints=build_waypoints(intermediate, dock, base),
    )
    return replace(base, env=env)

def validate_config(cfg: Config) -> list[str]:
    env, r_body = cfg.env, cfg.robot.body_radius
    hx, hy = env.room_half_size
    errors: list[str] = []

    numbers = [*env.start_pose, *env.dock_pose,
               *(v for o in env.obstacles for v in o),
               *(v for w in env.waypoints for v in w)]
    if not all(math.isfinite(v) for v in numbers):
        return ["All values must be finite numbers."]

    def inside(x: float, y: float, margin: float) -> bool:
        return abs(x) <= hx - margin and abs(y) <= hy - margin

    sx, sy, _ = env.start_pose
    dx, dy, _ = env.dock_pose
    ax, ay = dock_approach_point(env.dock_pose, cfg)

    if not inside(sx, sy, r_body):
        errors.append("Start pose is outside the room (or too close to a wall).")
    if not inside(dx, dy, 0.0):
        errors.append("Dock pose is outside the room.")
    if not inside(ax, ay, r_body):
        errors.append("Dock approach point is outside the room; check the dock heading.")
    if len(env.obstacles) > MAX_OBSTACLES:
        errors.append(f"At most {MAX_OBSTACLES} obstacles are supported.")

    for i, (ox, oy, r) in enumerate(env.obstacles):
        tag = f"Obstacle {i + 1}"
        if r <= 0.0:
            errors.append(f"{tag}: radius must be positive.")
            continue
        if abs(ox) + r > hx or abs(oy) + r > hy:
            errors.append(f"{tag}: outside the room.")
        if math.hypot(ox - sx, oy - sy) < r + r_body + CLEARANCE_MARGIN:
            errors.append(f"{tag}: overlaps the robot start pose.")
        if math.hypot(ox - dx, oy - dy) < r + MARKER_CLEARANCE:
            errors.append(f"{tag}: overlaps the charging station marker.")
        if math.hypot(ox - ax, oy - ay) < r + r_body + CLEARANCE_MARGIN:
            errors.append(f"{tag}: blocks the docking approach point.")
        for j in range(i):
            px, py, pr = env.obstacles[j]
            if math.hypot(ox - px, oy - py) < r + pr:
                errors.append(f"{tag}: overlaps obstacle {j + 1}.")
        for k, (wx, wy) in enumerate(env.waypoints):
            if math.hypot(ox - wx, oy - wy) < r + r_body + CLEARANCE_MARGIN:
                errors.append(f"{tag}: too close to waypoint {k + 1}.")

    for k, (wx, wy) in enumerate(env.waypoints):
        if not inside(wx, wy, r_body):
            errors.append(f"Waypoint {k + 1} is outside the room.")
    return errors

# Random circular obstacles
def randomize_obstacles(base: Config, count: int, seed: int,
                        radius_range: tuple = RADIUS_RANGE) -> tuple:
    empty = replace(base, env=replace(base.env, obstacles=()))
    base_errors = validate_config(empty)
    if base_errors:
        raise ValueError("Cannot randomize: " + "; ".join(base_errors))

    rng = np.random.default_rng(seed)
    hx, hy = base.env.room_half_size
    obstacles: list[tuple] = []
    for _ in range(MAX_RANDOM_ATTEMPTS):
        if len(obstacles) >= count:
            break
        r = float(rng.uniform(*radius_range))
        x = float(rng.uniform(-hx + r, hx - r))
        y = float(rng.uniform(-hy + r, hy - r))
        cand = tuple(obstacles) + ((round(x, 2), round(y, 2), round(r, 2)),)
        trial = replace(base, env=replace(base.env, obstacles=cand))
        if not validate_config(trial):
            obstacles.append(cand[-1])
    if len(obstacles) < count:
        raise RuntimeError(f"Could only place {len(obstacles)} of {count} obstacles.")
    return tuple(obstacles)

class CollisionCounter:
    def __init__(self) -> None:
        self.count = 0
        self._prev = False

    def update(self, collided: bool) -> int:
        if collided and not self._prev:
            self.count += 1
        self._prev = bool(collided)
        return self.count

    def reset(self) -> None:
        self.count = 0
        self._prev = False

# ----------------------------------------------------------------------------
# Strict request parsing: JSON <-> Config
# ----------------------------------------------------------------------------
def _num(value, name: str, limit: float = COORD_LIMIT) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number.")
    v = float(value)
    if not math.isfinite(v) or abs(v) > limit:
        raise ValueError(f"'{name}' is out of range.")
    return v

def _obj(data: dict, key: str) -> dict:
    v = data.get(key)
    if not isinstance(v, dict):
        raise ValueError(f"'{key}' must be an object.")
    return v

def _list(data: dict, key: str, max_len: int) -> list:
    v = data.get(key)
    if not isinstance(v, list):
        raise ValueError(f"'{key}' must be a list.")
    if len(v) > max_len:
        raise ValueError(f"'{key}' has more than {max_len} entries.")
    return v

def _pose(data: dict, key: str) -> tuple:
    p = _obj(data, key)
    return (_num(p.get("x"), f"{key}.x"), _num(p.get("y"), f"{key}.y"),
            math.radians(_num(p.get("deg"), f"{key}.deg", ANGLE_LIMIT_DEG)))


def parse_payload(data, base: Config = CONFIG) -> tuple[Config, int]:
    if not isinstance(data, dict):
        raise ValueError("Body must be a JSON object.")
    start, dock = _pose(data, "start"), _pose(data, "dock")

    obstacles = []
    for i, o in enumerate(_list(data, "obstacles", MAX_OBSTACLES)):
        if not isinstance(o, dict):
            raise ValueError(f"obstacles[{i}] must be an object.")
        obstacles.append((_num(o.get("x"), f"obstacles[{i}].x"),
                        _num(o.get("y"), f"obstacles[{i}].y"),
                        _num(o.get("r"), f"obstacles[{i}].r", RADIUS_LIMIT)))
    waypoints = []
    for i, w in enumerate(_list(data, "waypoints", MAX_WAYPOINTS)):
        if not isinstance(w, dict):
            raise ValueError(f"waypoints[{i}] must be an object.")
        waypoints.append((_num(w.get("x"), f"waypoints[{i}].x"),
                        _num(w.get("y"), f"waypoints[{i}].y")))

    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"'seed' must be an integer between 0 and {MAX_SEED}.")
    return make_config(base, start, dock, obstacles, waypoints), seed

def config_to_payload(cfg: Config, seed: int) -> dict:
    env = cfg.env

    def pose(p):
        return {"x": p[0], "y": p[1], "deg": round(math.degrees(p[2]), 6)}

    return {"start": pose(env.start_pose), "dock": pose(env.dock_pose),
            "obstacles": [{"x": o[0], "y": o[1], "r": o[2]} for o in env.obstacles],
            "waypoints": [{"x": w[0], "y": w[1]} for w in env.waypoints[:-1]],
            "seed": int(seed)}

# ----------------------------------------------------------------------------
# Shared state between the web thread and the simulation (main) thread
# ----------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, errors: Optional[list] = None):
        super().__init__(message)
        self.message, self.status, self.errors = message, status, errors or []

class CommandRejected(ApiError):
    pass

class SharedState:
    def __init__(self, cfg: Config, seed: int) -> None:
        self._lock = threading.Lock()
        self._commands: "queue.Queue[str]" = queue.Queue(maxsize=MAX_PENDING_COMMANDS)
        self._cfg, self._seed = cfg, int(seed)
        self._status = StatusMessage()
        self._running = False
        self._trial_id = 0
        self._stop = threading.Event()

    # ---- web side ----
    def submit(self, name: str, cfg: Optional[Config] = None, seed: Optional[int] = None) -> None:
        if name not in COMMANDS:
            raise ValueError(f"Unknown command '{name}'.")
        with self._lock:
            if name == "apply" and self._running:
                raise CommandRejected(
                    "Pause the simulation before applying a new configuration.", 409)
            try:
                self._commands.put_nowait(name)
            except queue.Full:
                raise CommandRejected("Simulation is busy, try again.", 503) from None
            if name == "apply":
                self._cfg, self._seed = cfg, int(seed)

    def snapshot(self) -> dict:
        with self._lock:
            return status_to_dict(self._status, self._running, self._trial_id)

    # ---- simulation side ----
    def applied(self) -> tuple[Config, int]:
        with self._lock:
            return self._cfg, self._seed

    def next_command(self) -> Optional[str]:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def publish(self, msg: StatusMessage) -> None:
        with self._lock:
            self._status = msg

    def set_running(self, running: bool) -> None:
        with self._lock:
            self._running = bool(running)

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def new_trial(self) -> None:
        with self._lock:
            self._trial_id += 1

    def set_error(self, text: str) -> None:
        with self._lock:
            self._status = StatusMessage(state="ERROR", message=text[:300])
            self._running = False

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def wait_for_rebuild(self) -> bool:
        while not self.should_stop():
            if self.next_command() in ("apply", "reset"):
                return True
            time.sleep(0.05)
        return False
    
# ----------------------------------------------------------------------------
# Simulation loop (main thread)
# ----------------------------------------------------------------------------
def _loop(state: SharedState, trial, handle, dt: float) -> str:
    running = False
    state.set_running(False)
    state.publish(trial.status())
    while not state.should_stop():
        t0 = time.perf_counter()
        if handle is not None and not handle.is_running():
            return "viewer_closed"
        while True:                                  # one command at a time, order kept
            cmd = state.next_command()
            if cmd is None:
                break
            if cmd in ("apply", "reset"):
                return "rebuild"
            if cmd == "start" and not trial.finished:
                running = True
            elif cmd == "pause":
                running = False
        state.set_running(running)

        if running and not trial.finished:
            if handle is None:
                msg = trial.step()
            else:
                with handle.lock():
                    msg = trial.step()
            state.publish(msg)
            if trial.finished:
                running = False
                state.set_running(False)
        if handle is not None:
            handle.sync()
        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
    return "stop"

def _run_trial(state: SharedState, trial, use_viewer: bool, dt: float) -> str:
    if use_viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(trial.model, trial.data) as handle:
            handle.cam.lookat[:] = [0.0, 0.0, 0.0]
            handle.cam.distance = 6.5
            handle.cam.azimuth = 90.0
            handle.cam.elevation = -75.0
            outcome = _loop(state, trial, handle, dt)
        if outcome != "viewer_closed":
            return outcome
        LOG.info("Viewer closed; continuing headless (Apply/Reset reopens it).")
    return _loop(state, trial, None, dt)

def serve_simulation(state: SharedState, trial_factory: Callable, use_viewer: bool = True,
                    dt: float = 0.02) -> None:
    while not state.should_stop():
        cfg, seed = state.applied()
        trial = None
        try:
            trial = trial_factory(cfg, seed)
            state.new_trial()
            outcome = _run_trial(state, trial, use_viewer, dt)
        except Exception as exc:                     # keep the server alive
            LOG.exception("Trial failed")
            state.set_error(f"{type(exc).__name__}: {exc}")
            outcome = "error"
        finally:
            if trial is not None:
                try:
                    trial.close()
                except Exception:
                    LOG.exception("Closing the trial failed")
        if outcome == "stop":
            break
        if outcome == "error" and not state.wait_for_rebuild():
            break

# ----------------------------------------------------------------------------
# The real trial: same logic as main.run_pipeline, one control step at a time
# ----------------------------------------------------------------------------
def _pipeline():
    import main as pipeline        # lazy: keeps this module importable without MuJoCo
    return pipeline

class PipelineTrial:
    def __init__(self, cfg: Config, seed: Optional[int] = None,
                stop_on_collision: bool = True, verbose: bool = True) -> None:
        m = _pipeline()
        self._m, self.cfg = m, cfg
        self.stop_on_collision, self.verbose = stop_on_collision, verbose
        self.sim = m.Simulation(cfg, seed=seed)
        try:
            self.model, self.data = self.sim.model, self.sim.data   # shown by the viewer
            start = np.array(cfg.env.start_pose, dtype=float)
            self.sim.reset(tuple(start))
            self.sensing = m.Sensing(self.sim, cfg)
            self.estimator = m.EKFEstimator(cfg, gate=9.21)
            self.estimator.reset(start)          # known start pose (a prior, not ground truth)
            self.nav, self.dock, self.log = m.Navigator(cfg), m.DockingController(cfg), m.TrialLog()
        except Exception:
            self.sim.close()
            raise
        self.counter = CollisionCounter()
        self.phase, self.reason = m.Phase.NAVIGATE, "timeout"
        self.nav_time: Optional[float] = None
        self.collided_any = False
        self.finished = False
        self._goal = self._min_lidar = math.nan
        self._mode = ""
        self._step = 0
        self._report_every = max(1, int(round(1.0 / cfg.sim.control_dt)))
        if verbose:
            print(m.theory_block(cfg, self.estimator))
            print(m.table_header())

    def _status(self) -> StatusMessage:
        Phase = self._m.Phase
        state = {Phase.NAVIGATE: "NAVIGATING", Phase.DOCK: "DOCKING",
                Phase.DONE: "SUCCESS", Phase.FAILED: "FAILED"}[self.phase]
        msg = self.reason if self.phase is Phase.FAILED else ""
        return StatusMessage(state, self._mode, self.sim.time,
                            tuple(float(v) for v in self.estimator.x),
                            tuple(float(v) for v in self.sim.ground_truth()),  # eval only
                            self._goal, self._min_lidar, self.counter.count, msg)

    def status(self) -> StatusMessage:
        return self._status()

    def step(self) -> StatusMessage:
        if self.finished:
            return self._status()
        m, cfg, sim, est = self._m, self.cfg, self.sim, self.estimator
        Phase = m.Phase

        # 1) sensing and estimation (no ground truth)
        frame = self.sensing.read()
        est.predict(frame.odometry.ds, frame.odometry.dtheta)
        est.update(frame.marker)
        est.update_lidar(frame.lidar)
        x_est = np.array(est.x, dtype=float)

        # 2) command from the ESTIMATED pose only
        v = w = 0.0
        ncmd = dcmd = None
        if self.phase is Phase.NAVIGATE:
            ncmd = self.nav.update(x_est, frame.lidar)
            if ncmd.finished:
                self.phase, self.nav_time = Phase.DOCK, frame.time
                self.dock.begin(x_est)
            else:
                v, w = ncmd.v, ncmd.w
        if self.phase is Phase.DOCK:
            dcmd = self.dock.update(frame.marker, frame.odometry, x_est)
            if dcmd.finished:
                self.phase = Phase.DONE if dcmd.docked else Phase.FAILED
                self.reason = "docked" if dcmd.docked else "docking failed"
            else:
                v, w = dcmd.v, dcmd.w

        # 3) logging and status (ground truth is read here for evaluation only)
        hit = sim.collided
        self.collided_any = self.collided_any or hit
        self.counter.update(hit)
        self._mode = m.mode_label(self.phase, ncmd, dcmd)
        n_valid = int(frame.lidar.valid.sum())
        n_updates = int(getattr(est, "n_updates", 0))
        self.log.record(frame.time, sim.ground_truth(), x_est, est.P, self.phase.value,
                        v, w, n_valid, n_updates, self._mode)
        if self.verbose and self._step % self._report_every == 0:
            print(m.table_row(frame.time, x_est, n_valid, frame.lidar.valid.size,
                            n_updates, v, w, self._mode), flush=True)
        self._step += 1
        self._goal = (ncmd.goal_distance if ncmd is not None
                    else m.dock_errors(x_est, cfg)["dock_range"])        # estimate, not truth
        ranges = getattr(frame.lidar, "ranges", None)   # check this field name in sensing.py
        if ranges is not None and n_valid:
            self._min_lidar = float(np.min(np.asarray(ranges)[frame.lidar.valid]))

        # 4) end conditions, otherwise act
        if self.phase not in (Phase.DONE, Phase.FAILED):
            if hit and self.stop_on_collision:
                self.phase, self.reason = Phase.FAILED, "collision"
            elif sim.time_exceeded:
                self.phase, self.reason = Phase.FAILED, "timeout"
        if self.phase in (Phase.DONE, Phase.FAILED):
            self.finished = True
            self._print_report()
        else:
            sim.step(v, w)
        return self._status()

    def result(self):
        m = self._m
        return m.RunResult(self.phase is m.Phase.DONE, self.reason, self.sim.time,
                        self.nav_time, self.collided_any, self.phase,
                        np.array(self.estimator.x, dtype=float),
                        self.sim.ground_truth(), self.log)

    def _print_report(self) -> None:
        m = self._m
        res = self.result()
        if self.verbose:
            print(m.table_footer())
        print("\n" + res.summary(self.cfg) + "\n")
        m.print_report(res, self.cfg)

    def close(self) -> None:
        self.sim.close()

# ----------------------------------------------------------------------------
# Web app
# ----------------------------------------------------------------------------
def create_app(state: SharedState, token: str, port: int = DEFAULT_PORT,
            base_cfg: Config = CONFIG):
    from flask import Flask, jsonify, make_response, request
    from werkzeug.exceptions import HTTPException

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    hosts = {f"{HOST}:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in hosts}

    def json_error(message: str, status: int, errors: Optional[list] = None):
        resp = jsonify(error=message, errors=errors or [])
        resp.status_code = status
        return resp

    @app.errorhandler(ApiError)
    def _api_error(exc):
        return json_error(exc.message, exc.status, exc.errors)

    @app.errorhandler(Exception)
    def _any_error(exc):
        if isinstance(exc, HTTPException):
            return json_error(exc.description or exc.name, exc.code or 500)
        LOG.exception("Unhandled error in a request")
        return json_error("Internal server error.", 500)

    @app.before_request
    def _guard():
        if request.host.lower() not in hosts:                 # DNS-rebinding protection
            return json_error("Forbidden host.", 403)
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin is not None and origin not in origins:  # cross-site request
                return json_error("Forbidden origin.", 403)
            if not request.is_json:
                return json_error("Content-Type must be application/json.", 415)
            supplied = request.headers.get("X-Token", "")
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return json_error("Missing or invalid token.", 403)

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("Content-Security-Policy",
                                "default-src 'none'; frame-ancestors 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError("Body must be a JSON object.", 400)
        return data

    def checked_config(data: dict) -> tuple[Config, int]:
        try:
            cfg, seed = parse_payload(data, base_cfg)
        except ValueError as exc:
            raise ApiError(str(exc), 400) from None
        errors = validate_config(cfg)
        if errors:
            raise ApiError("Configuration rejected.", 422, errors)
        return cfg, seed

    @app.get("/")
    def index():
        nonce = secrets.token_urlsafe(16)
        resp = make_response(PAGE.replace("__NONCE__", nonce).replace("__TOKEN__", token))
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Content-Security-Policy"] = (
            f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        return resp

    @app.get("/api/health")
    def health():
        return jsonify(ok=True)

    @app.get("/api/status")
    def status():
        return jsonify(state.snapshot())

    @app.get("/api/state")
    def full_state():
        cfg, seed = state.applied()
        return jsonify(config=config_to_payload(cfg, seed),
                    defaults=config_to_payload(base_cfg, base_cfg.sim.seed),
                    status=state.snapshot(),
                    limits={"max_obstacles": MAX_OBSTACLES, "max_waypoints": MAX_WAYPOINTS})

    @app.post("/api/apply")
    def apply_config():
        cfg, seed = checked_config(body())
        state.submit("apply", cfg, seed)
        return jsonify(ok=True)

    def simple(name: str):
        body()
        state.submit(name)
        return jsonify(ok=True)

    @app.post("/api/start")
    def start():
        return simple("start")

    @app.post("/api/pause")
    def pause():
        return simple("pause")

    @app.post("/api/reset")
    def reset():
        return simple("reset")

    @app.post("/api/randomize")
    def randomize():
        data = dict(body())
        count = data.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_OBSTACLES:
            raise ApiError(f"'count' must be an integer between 1 and {MAX_OBSTACLES}.", 400)
        data["obstacles"] = []
        try:
            cfg, seed = parse_payload(data, base_cfg)
            layout = randomize_obstacles(cfg, count, seed)
        except ValueError as exc:
            raise ApiError(str(exc), 400) from None
        except RuntimeError as exc:
            raise ApiError(str(exc), 422) from None
        return jsonify(obstacles=[{"x": x, "y": y, "r": r} for x, y, r in layout])

    return app

def start_server(app, port: int = DEFAULT_PORT):
    from werkzeug.serving import make_server
    server = make_server(HOST, port, app, threaded=True)

    def run() -> None:
        while True:
            try:
                server.serve_forever()
                return                               # normal shutdown
            except Exception:
                LOG.exception("Web server loop crashed; restarting it")
                time.sleep(1.0)

    threading.Thread(target=run, name="web-gui", daemon=True).start()
    return server

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def _setup_logging() -> None:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    LOG.addHandler(console)
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(RESULTS_DIR / "gui.log", maxBytes=1_000_000, backupCount=3)
        handler.setFormatter(fmt)
        LOG.addHandler(handler)
    except OSError:
        pass

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track A: web control panel + MuJoCo simulation")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--seed", type=int, default=None, help="initial noise seed (default: config)")
    p.add_argument("--no-viewer", action="store_true", help="web panel only, no MuJoCo window")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    use_viewer = not args.no_viewer
    if use_viewer:
        _pipeline()._ensure_mjpython()               # macOS: re-run under mjpython (before any socket)
    _setup_logging()

    cfg = CONFIG if args.seed is None else replace(CONFIG, sim=replace(CONFIG.sim, seed=args.seed))
    state = SharedState(cfg, cfg.sim.seed)
    token = secrets.token_urlsafe(32)               # new secret on every run
    try:
        app = create_app(state, token, args.port, cfg)
    except ImportError:
        print("Flask is not installed. Run: pip install flask")
        return 2
    try:
        server = start_server(app, args.port)
    except OSError as exc:
        print(f"Cannot start the web server on port {args.port}: {exc}")
        return 2

    url = f"http://{HOST}:{args.port}/"
    print(f"Control panel: {url}   (Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        serve_simulation(state, lambda c, s: PipelineTrial(c, s), use_viewer, cfg.sim.control_dt)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        state.request_stop()
        server.shutdown()
        server.server_close()
    return 0

# ----------------------------------------------------------------------------
# Single-file page (HTML + CSS + JS). No inline event handlers, no innerHTML.
# ----------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="token" content="__TOKEN__">
<title>Track A - Control Panel</title>
<style nonce="__NONCE__">
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 16px; max-width: 1100px; }
h1 { font-size: 1.2rem; }
small { font-weight: normal; opacity: .7; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
fieldset { border: 1px solid #8886; border-radius: 8px; padding: 8px 12px; margin: 0 0 12px; }
legend { font-weight: 600; }
label { display: inline-block; margin: 2px 8px 2px 0; font-size: .85rem; }
input.num { width: 6.5em; }
table { border-collapse: collapse; }
td, th { padding: 2px 6px; text-align: left; }
button { margin: 2px; padding: 4px 10px; }
#msg { white-space: pre-wrap; min-height: 1.4em; }
#msg.err { color: #c0392b; }
#msg.ok { color: #1e8449; }
#dirty.pending { color: #b9770e; font-weight: 600; }
.badge { padding: 1px 8px; border-radius: 10px; background: #8883; }
.badge.success { background: #1e844966; }
.badge.failed, .badge.error { background: #c0392b66; }
.badge.navigating, .badge.docking { background: #2e86c166; }
.note { font-size: .8rem; opacity: .75; }
</style>
</head>
<body>
<h1>Track A control panel <small id="conn">connecting...</small></h1>
<div class="grid">
<div>
  <fieldset><legend>Robot start pose</legend>
    <label>X (m) <input class="num cfg edit" id="start_x" type="number" step="any"></label>
    <label>Y (m) <input class="num cfg edit" id="start_y" type="number" step="any"></label>
    <label>Heading (deg) <input class="num cfg edit" id="start_deg" type="number" step="any"></label>
  </fieldset>
  <fieldset><legend>Charging station / goal pose</legend>
    <label>X (m) <input class="num cfg edit" id="dock_x" type="number" step="any"></label>
    <label>Y (m) <input class="num cfg edit" id="dock_y" type="number" step="any"></label>
    <label>Heading (deg) <input class="num cfg edit" id="dock_deg" type="number" step="any"></label>
    <div class="note">Heading is the marker normal. The docking approach point is added automatically.</div>
  </fieldset>
  <fieldset><legend>Obstacles (circles)</legend>
    <table><thead><tr><th>#</th><th>X (m)</th><th>Y (m)</th><th>R (m)</th><th></th></tr></thead>
      <tbody id="ob_body"></tbody></table>
    <button class="edit" id="ob_add" type="button">Add obstacle</button>
    <div>
      <label>Count <input class="num edit" id="count" type="number" step="1" min="1" value="3"></label>
      <label>Seed <input class="num cfg edit" id="seed" type="number" step="1" min="0"></label>
      <button class="edit" id="new_seed" type="button">New seed</button>
      <button class="edit" id="randomize" type="button">Randomize</button>
    </div>
  </fieldset>
  <fieldset><legend>Intermediate waypoints</legend>
    <table><thead><tr><th>#</th><th>X (m)</th><th>Y (m)</th><th></th></tr></thead>
      <tbody id="wp_body"></tbody></table>
    <button class="edit" id="wp_add" type="button">Add waypoint</button>
  </fieldset>
  <fieldset><legend>Controls</legend>
    <button class="edit" id="apply" type="button">Apply</button>
    <button id="toggle" type="button">Start</button>
    <button id="reset" type="button">Reset</button>
    <button class="edit" id="defaults" type="button">Defaults</button>
    <div id="dirty">Applied</div>
    <div id="msg"></div>
  </fieldset>
</div>
<div>
  <fieldset><legend>Live status</legend>
    <table>
      <tr><td>State</td><td><span id="s_state" class="badge">--</span> <span id="s_msg"></span></td></tr>
      <tr><td>Phase</td><td id="s_phase">--</td></tr>
      <tr><td>Time</td><td id="s_time">--</td></tr>
      <tr><td>Estimated pose (EKF)</td><td id="s_est">--</td></tr>
      <tr><td>Ground truth (eval only)</td><td id="s_gt">--</td></tr>
      <tr><td>Position error</td><td id="s_perr">--</td></tr>
      <tr><td>Heading error</td><td id="s_herr">--</td></tr>
      <tr><td>Goal distance</td><td id="s_goal">--</td></tr>
      <tr><td>Min LiDAR range</td><td id="s_lidar">--</td></tr>
      <tr><td>Collisions</td><td id="s_coll">--</td></tr>
    </table>
    <div class="note">Detailed logs (EKF, NIS, RMSE) are printed in the terminal.
      Editing is locked while the robot is running.</div>
  </fieldset>
</div>
</div>
<script nonce="__NONCE__">
(() => {
'use strict';
const TOKEN = document.querySelector('meta[name="token"]').content;
const $ = id => document.getElementById(id);
const S = {obstacles: [], waypoints: [], dirty: false, running: false, loaded: false};

async function api(path, body) {
  const opts = body === undefined ? {} : {method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Token': TOKEN}, body: JSON.stringify(body)};
  const resp = await fetch(path, opts);
  let data = {};
  try { data = await resp.json(); } catch (e) { data = {error: 'Unreadable server response.'}; }
  if (!resp.ok) {
    const err = new Error(data.error || ('HTTP ' + resp.status));
    err.details = data.errors || [];
    throw err;
  }
  return data;
}

function say(text, details, ok) {
  const el = $('msg');
  el.textContent = [text].concat(details || []).filter(Boolean).join('\n- ');
  el.className = ok ? 'ok' : 'err';
}
function setDirty(v) {
  S.dirty = v;
  $('dirty').textContent = v ? 'Unapplied changes - press Apply' : 'Applied';
  $('dirty').className = v ? 'pending' : '';
}
function applyLock() {
  document.querySelectorAll('.edit').forEach(el => { el.disabled = S.running; });
}

function renderList(bodyId, list, keys) {
  const tb = $(bodyId);
  tb.replaceChildren();
  list.forEach((row, i) => {
    const tr = document.createElement('tr');
    const idx = document.createElement('td');
    idx.textContent = String(i + 1);
    tr.appendChild(idx);
    keys.forEach(k => {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = 'any'; inp.className = 'num edit';
      inp.value = row[k];
      inp.addEventListener('input', () => { row[k] = inp.value; setDirty(true); });
      td.appendChild(inp);
      tr.appendChild(td);
    });
    const td = document.createElement('td');
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'edit'; b.textContent = 'Remove';
    b.addEventListener('click', () => { list.splice(i, 1); renderAll(); setDirty(true); });
    td.appendChild(b);
    tr.appendChild(td);
    tb.appendChild(tr);
  });
}
function renderAll() {
  renderList('ob_body', S.obstacles, ['x', 'y', 'r']);
  renderList('wp_body', S.waypoints, ['x', 'y']);
  applyLock();
}

function fill(cfg) {
  ['start', 'dock'].forEach(k => {
    $(k + '_x').value = cfg[k].x; $(k + '_y').value = cfg[k].y; $(k + '_deg').value = cfg[k].deg;
  });
  $('seed').value = cfg.seed;
  S.obstacles = cfg.obstacles.map(o => ({x: o.x, y: o.y, r: o.r}));
  S.waypoints = cfg.waypoints.map(w => ({x: w.x, y: w.y}));
  renderAll();
  setDirty(false);
}

const num = v => parseFloat(v);
function collect() {
  const pose = k => ({x: num($(k + '_x').value), y: num($(k + '_y').value), deg: num($(k + '_deg').value)});
  return {
    start: pose('start'), dock: pose('dock'),
    obstacles: S.obstacles.map(o => ({x: num(o.x), y: num(o.y), r: num(o.r)})),
    waypoints: S.waypoints.map(w => ({x: num(w.x), y: num(w.y)})),
    seed: parseInt($('seed').value, 10),
  };
}
function hasBad(v) {
  if (typeof v === 'number') return !Number.isFinite(v);
  if (v && typeof v === 'object') return Object.values(v).some(hasBad);
  return false;
}

const f = (v, d, unit) => (v === null || v === undefined) ? '--' : v.toFixed(d) + unit;
const pose = p => (!p || p.some(v => v === null)) ? '--' :
  'x=' + p[0].toFixed(3) + ' y=' + p[1].toFixed(3) + ' th=' + (p[2] * 180 / Math.PI).toFixed(1) + ' deg';

function showStatus(s) {
  S.running = s.running;
  const badge = $('s_state');
  badge.textContent = s.state;
  badge.className = 'badge ' + s.state.toLowerCase();
  $('s_msg').textContent = s.message || '';
  $('s_phase').textContent = s.phase || '--';
  $('s_time').textContent = f(s.sim_time, 2, ' s');
  $('s_est').textContent = pose(s.est_pose);
  $('s_gt').textContent = pose(s.gt_pose);
  $('s_perr').textContent = f(s.position_error, 3, ' m');
  $('s_herr').textContent = f(s.heading_error_deg, 1, ' deg');
  $('s_goal').textContent = f(s.goal_distance, 3, ' m');
  $('s_lidar').textContent = f(s.min_lidar, 3, ' m');
  $('s_coll').textContent = String(s.collisions);
  $('toggle').textContent = s.running ? 'Pause' : 'Start';
  applyLock();
}

async function guard(fn) {
  try { say('', [], true); await fn(); }
  catch (e) { say(e.message, e.details, false); }
}

$('apply').addEventListener('click', () => guard(async () => {
    const cfg = collect();
    if (hasBad(cfg)) throw new Error('Fill in every field with a valid number.');
    await api('/api/apply', cfg);
    setDirty(false);
    say('Configuration applied.', [], true);
}));
$('toggle').addEventListener('click', () => guard(async () => {
    if (S.running) { await api('/api/pause', {}); return; }
    if (S.dirty) throw new Error('Press Apply first.');
    await api('/api/start', {});
}));
$('reset').addEventListener('click', () => guard(async () => {
    await api('/api/reset', {});
    say(S.dirty ? 'Reset uses the last APPLIED configuration.' : 'Trial reset.', [], true);
}));
$('defaults').addEventListener('click', () => guard(async () => {
    fill((await api('/api/state')).defaults);
}));
$('randomize').addEventListener('click', () => guard(async () => {
    const body = collect();
    body.obstacles = [];
    body.count = parseInt($('count').value, 10);
    const r = await api('/api/randomize', body);
    S.obstacles = r.obstacles.map(o => ({x: o.x, y: o.y, r: o.r}));
    renderAll();
    setDirty(true);
}));
$('new_seed').addEventListener('click', () => {
  $('seed').value = String(Math.floor(Math.random() * 1000000));
    setDirty(true);
});
$('ob_add').addEventListener('click', () => { S.obstacles.push({x: 0, y: 0, r: 0.2}); renderAll(); setDirty(true); });
$('wp_add').addEventListener('click', () => { S.waypoints.push({x: 0, y: 0}); renderAll(); setDirty(true); });
document.querySelectorAll('.cfg').forEach(el => el.addEventListener('input', () => setDirty(true)));

async function poll() {
    try {
    if (!S.loaded) {
        const d = await api('/api/state');
        fill(d.config);
        S.loaded = true;
    }
    showStatus(await api('/api/status'));
    $('conn').textContent = 'connected';
    } catch (e) {
    $('conn').textContent = 'server unreachable, retrying...';
    }
    setTimeout(poll, 200);
}
poll();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())