# Including libraries
import json
import math
import re
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace

import gui
from config import CONFIG
from gui import (MAX_OBSTACLES, MAX_PENDING_COMMANDS, ApiError, CollisionCounter,
                CommandRejected, SharedState, StatusMessage, config_to_payload, create_app,
                make_config, parse_payload, randomize_obstacles, serve_simulation,
                status_to_dict, validate_config)

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False

START, DOCK = CONFIG.env.start_pose, CONFIG.env.dock_pose
INTERMEDIATE = CONFIG.env.waypoints[:-1]


def cfg_with(**env_changes):
    return replace(CONFIG, env=replace(CONFIG.env, **env_changes))


def good(**over):
    payload = config_to_payload(CONFIG, 42)
    payload.update(over)
    return payload


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


# ----------------------------------------------------------------------------
class TestParsePayload(unittest.TestCase):
    def test_round_trip_of_default_config(self):
        cfg, seed = parse_payload(good(), CONFIG)
        self.assertEqual(seed, 42)
        self.assertEqual(validate_config(cfg), [])
        self.assertEqual(cfg.env.obstacles, CONFIG.env.obstacles)
        for a, b in zip(cfg.env.start_pose, START):
            self.assertAlmostEqual(a, b)
        for a, b in zip(cfg.env.dock_pose, DOCK):
            self.assertAlmostEqual(a, b)
        for a, b in zip(cfg.env.waypoints[-1], CONFIG.env.waypoints[-1]):
            self.assertAlmostEqual(a, b)

    def test_degrees_are_converted_to_radians(self):
        p = good(start={"x": 0.0, "y": 0.0, "deg": 90})
        cfg, _ = parse_payload(p, CONFIG)
        self.assertAlmostEqual(cfg.env.start_pose[2], math.pi / 2)

    def test_base_config_is_not_modified(self):
        old = CONFIG.env.obstacles
        parse_payload(good(obstacles=[]), CONFIG)
        self.assertEqual(CONFIG.env.obstacles, old)

    def test_rejects_malformed_input(self):
        pose = {"x": 0, "y": 0, "deg": 0}
        bad_cases = {
            "none": None, "list": [], "string": "x",
            "start missing deg": good(start={"x": 0, "y": 0}),
            "start string": good(start={"x": "1", "y": 0, "deg": 0}),
            "start bool": good(start={"x": True, "y": 0, "deg": 0}),
            "start nan": good(start={"x": math.nan, "y": 0, "deg": 0}),
            "start inf": good(start={"x": math.inf, "y": 0, "deg": 0}),
            "start huge": good(start={"x": 1e6, "y": 0, "deg": 0}),
            "angle huge": good(dock={"x": 0, "y": 0, "deg": 1e5}),
            "dock not object": good(dock=[1, 2, 3]),
            "obstacles not list": good(obstacles="abc"),
            "obstacle not object": good(obstacles=[1]),
            "too many obstacles": good(obstacles=[{"x": 0, "y": 0, "r": 0.2}] * (MAX_OBSTACLES + 1)),
            "too many waypoints": good(waypoints=[{"x": 0, "y": 0}] * 21),
            "huge radius": good(obstacles=[{"x": 0, "y": 0, "r": 99}]),
            "seed float": good(seed=1.5), "seed negative": good(seed=-1),
            "seed bool": good(seed=True), "seed string": good(seed="7"),
            "seed none": good(seed=None), "seed too big": good(seed=2 ** 31),
        }
        for name, payload in bad_cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_payload(payload, CONFIG)
        for key in ("start", "dock", "obstacles", "waypoints", "seed"):
            p = good()
            del p[key]
            with self.subTest("missing " + key), self.assertRaises(ValueError):
                parse_payload(p, CONFIG)
        self.assertIsNotNone(pose)


class TestValidate(unittest.TestCase):
    def test_default_is_valid(self):
        self.assertEqual(validate_config(CONFIG), [])

    def test_invalid_cases(self):
        cases = {
            "outside room": cfg_with(obstacles=((1.95, 0.0, 0.2),)),
            "on start": cfg_with(obstacles=((-1.5, -1.3, 0.2),)),
            "blocks dock": cfg_with(obstacles=((1.6, 1.0, 0.2),)),
            "on waypoint": cfg_with(obstacles=((0.6, -1.2, 0.2),)),
            "overlap": cfg_with(obstacles=((0.0, 0.0, 0.2), (0.1, 0.0, 0.2))),
            "zero radius": cfg_with(obstacles=((0.0, 0.0, 0.0),)),
            "start outside": cfg_with(start_pose=(5.0, 0.0, 0.0)),
            "dock faces wall": cfg_with(dock_pose=(1.9, 1.0, 0.0)),
            "nan": cfg_with(start_pose=(math.nan, 0.0, 0.0)),
        }
        for name, cfg in cases.items():
            self.assertTrue(validate_config(cfg), name)

    def test_dock_approach_point_follows_heading(self):
        x, y = gui.dock_approach_point((0.0, 0.0, math.pi / 2), CONFIG)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, CONFIG.docking.target_distance + 0.3)


class TestRandomize(unittest.TestCase):
    def setUp(self):
        self.base = make_config(CONFIG, START, DOCK, [], INTERMEDIATE)

    def test_reproducible_and_different_seeds(self):
        self.assertEqual(randomize_obstacles(self.base, 4, 7), randomize_obstacles(self.base, 4, 7))
        self.assertNotEqual(randomize_obstacles(self.base, 4, 1), randomize_obstacles(self.base, 4, 2))

    def test_always_valid(self):
        for seed in range(20):
            obs = randomize_obstacles(self.base, 4, seed)
            cfg = make_config(CONFIG, START, DOCK, obs, INTERMEDIATE)
            self.assertEqual(validate_config(cfg), [], f"seed {seed}")

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            randomize_obstacles(self.base, 500, 0)
        with self.assertRaises(ValueError):
            randomize_obstacles(make_config(CONFIG, (9, 9, 0), DOCK, [], INTERMEDIATE), 2, 0)


class TestCollisionCounter(unittest.TestCase):
    def test_counts_only_rising_edges(self):
        c = CollisionCounter()
        for flag in (False, True, True, True, False, True, False):
            c.update(flag)
        self.assertEqual(c.count, 2)

    def test_reset(self):
        c = CollisionCounter()
        c.update(True)
        c.reset()
        self.assertEqual((c.count, c.update(True)), (0, 1))


class TestStatus(unittest.TestCase):
    def test_errors(self):
        m = StatusMessage(est_pose=(1.0, 1.0, 0.1), gt_pose=(1.0, 0.0, 0.0))
        self.assertAlmostEqual(m.position_error, 1.0)
        self.assertAlmostEqual(m.heading_error, 0.1)

    def test_heading_error_wraps(self):
        m = StatusMessage(est_pose=(0, 0, math.pi - 0.1), gt_pose=(0, 0, -math.pi + 0.1))
        self.assertAlmostEqual(m.heading_error, -0.2, places=6)

    def test_dict_is_strict_json_with_none_for_unknowns(self):
        d = status_to_dict(StatusMessage(), running=True, trial_id=3)
        text = json.dumps(d, allow_nan=False)           # raises if NaN/Infinity slipped in
        d = json.loads(text)
        self.assertEqual(d["est_pose"], [None, None, None])
        self.assertIsNone(d["position_error"])
        self.assertIsNone(d["heading_error_deg"])
        self.assertTrue(d["running"])
        self.assertEqual(d["trial_id"], 3)

    def test_dict_values(self):
        m = StatusMessage("NAVIGATING", "track", 1.5, (1, 1, 0.1), (1, 0, 0), 2.0, 0.5, 2)
        d = status_to_dict(m)
        self.assertAlmostEqual(d["position_error"], 1.0)
        self.assertAlmostEqual(d["heading_error_deg"], math.degrees(0.1))
        self.assertEqual(d["collisions"], 2)


# ----------------------------------------------------------------------------
class TestSharedState(unittest.TestCase):
    def setUp(self):
        self.state = SharedState(CONFIG, 42)

    def test_apply_stores_config_and_queues_command(self):
        cfg = cfg_with(obstacles=())
        self.state.submit("apply", cfg, 7)
        self.assertEqual(self.state.applied(), (cfg, 7))
        self.assertEqual(self.state.next_command(), "apply")
        self.assertIsNone(self.state.next_command())

    def test_apply_rejected_while_running_and_config_unchanged(self):
        self.state.set_running(True)
        with self.assertRaises(CommandRejected) as ctx:
            self.state.submit("apply", cfg_with(obstacles=()), 7)
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(self.state.applied(), (CONFIG, 42))
        self.assertIsNone(self.state.next_command())

    def test_full_queue_is_rejected_without_changing_config(self):
        for _ in range(MAX_PENDING_COMMANDS):
            self.state.submit("start")
        with self.assertRaises(CommandRejected) as ctx:
            self.state.submit("apply", cfg_with(obstacles=()), 9)
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(self.state.applied(), (CONFIG, 42))

    def test_unknown_command(self):
        with self.assertRaises(ValueError):
            self.state.submit("explode")

    def test_commands_keep_order(self):
        for name in ("start", "pause", "reset"):
            self.state.submit(name)
        self.assertEqual([self.state.next_command() for _ in range(3)], ["start", "pause", "reset"])

    def test_publish_and_snapshot(self):
        self.state.publish(StatusMessage(state="NAVIGATING", collisions=1))
        self.state.set_running(True)
        self.state.new_trial()
        snap = self.state.snapshot()
        self.assertEqual((snap["state"], snap["collisions"], snap["running"], snap["trial_id"]),
                        ("NAVIGATING", 1, True, 1))

    def test_set_error_stops_running(self):
        self.state.set_running(True)
        self.state.set_error("boom")
        snap = self.state.snapshot()
        self.assertEqual((snap["state"], snap["message"], snap["running"]), ("ERROR", "boom", False))

    def test_wait_for_rebuild(self):
        self.state.submit("start")
        self.state.submit("apply", CONFIG, 1)
        self.assertTrue(self.state.wait_for_rebuild())      # ignores 'start', returns on 'apply'
        self.state.request_stop()
        self.assertFalse(self.state.wait_for_rebuild())


# ----------------------------------------------------------------------------
class FakeTrial:
    def __init__(self, steps=3, on_finish=None, fail_at=None):
        self.n, self.steps, self.on_finish, self.fail_at = 0, steps, on_finish, fail_at
        self.finished = self.closed = False
        self.model = self.data = None

    def status(self):
        return StatusMessage(state="IDLE", sim_time=self.n * 0.02)

    def step(self):
        self.n += 1
        if self.fail_at == self.n:
            raise RuntimeError("boom")
        if self.n >= self.steps:
            self.finished = True
            if self.on_finish:
                self.on_finish()
        return StatusMessage(state="SUCCESS" if self.finished else "NAVIGATING",
                             sim_time=self.n * 0.02)

    def close(self):
        self.closed = True


class FakeHandle:
    def __init__(self, alive=True):
        self.alive, self.syncs, self.locked = alive, 0, 0

    def is_running(self):
        return self.alive

    @contextmanager
    def lock(self):
        self.locked += 1
        yield

    def sync(self):
        self.syncs += 1


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.state = SharedState(CONFIG, 1)

    def test_idle_trial_does_not_step(self):
        trial = FakeTrial(on_finish=self.state.request_stop)
        self.state.request_stop()
        self.assertEqual(gui._loop(self.state, trial, None, 0.0), "stop")
        self.assertEqual(trial.n, 0)
        self.assertEqual(self.state.snapshot()["state"], "IDLE")

    def test_start_runs_until_finished(self):
        trial = FakeTrial(3, on_finish=self.state.request_stop)
        self.state.submit("start")
        self.assertEqual(gui._loop(self.state, trial, None, 0.0), "stop")
        self.assertEqual(trial.n, 3)
        snap = self.state.snapshot()
        self.assertEqual((snap["state"], snap["running"]), ("SUCCESS", False))

    def test_start_then_pause_never_steps(self):
        trial = FakeTrial()
        self.state.submit("start")
        self.state.submit("pause")
        self.state.request_stop()
        # one iteration is skipped because stop is set; run with a thread-free trick:
        self.state._stop.clear()
        t = threading.Thread(target=gui._loop, args=(self.state, trial, None, 0.001))
        t.start()
        time.sleep(0.05)
        self.state.request_stop()
        t.join(2)
        self.assertEqual(trial.n, 0)

    def test_apply_and_reset_request_rebuild(self):
        for cmd in ("apply", "reset"):
            self.state.submit(cmd, CONFIG, 3) if cmd == "apply" else self.state.submit(cmd)
            trial = FakeTrial()
            self.assertEqual(gui._loop(self.state, trial, None, 0.0), "rebuild")
            self.assertEqual(trial.n, 0)
        self.assertEqual(self.state.applied()[1], 3)

    def test_commands_after_rebuild_are_kept(self):
        self.state.submit("reset")
        self.state.submit("start")
        self.assertEqual(gui._loop(self.state, FakeTrial(), None, 0.0), "rebuild")
        self.assertEqual(self.state.next_command(), "start")

    def test_viewer_closed(self):
        handle = FakeHandle(alive=False)
        self.assertEqual(gui._loop(self.state, FakeTrial(), handle, 0.0), "viewer_closed")

    def test_steps_run_under_viewer_lock(self):
        handle = FakeHandle()
        trial = FakeTrial(2, on_finish=self.state.request_stop)
        self.state.submit("start")
        gui._loop(self.state, trial, handle, 0.0)
        self.assertEqual(handle.locked, 2)
        self.assertGreaterEqual(handle.syncs, 1)

    def test_step_exception_propagates(self):
        self.state.submit("start")
        with self.assertRaises(RuntimeError):
            gui._loop(self.state, FakeTrial(fail_at=1), None, 0.0)


class TestServeSimulation(unittest.TestCase):
    def test_recovers_from_factory_error(self):
        state, calls = SharedState(CONFIG, 1), []

        def factory(cfg, seed):
            calls.append(seed)
            if len(calls) == 1:
                raise RuntimeError("no model")
            return FakeTrial(steps=100)

        t = threading.Thread(target=serve_simulation, args=(state, factory, False, 0.001))
        t.start()
        self.assertTrue(wait_for(lambda: state.snapshot()["state"] == "ERROR"))
        state.submit("apply", CONFIG, 9)
        self.assertTrue(wait_for(lambda: len(calls) == 2))
        self.assertTrue(wait_for(lambda: state.snapshot()["state"] == "IDLE"))
        self.assertEqual(calls[1], 9)
        self.assertEqual(state.snapshot()["trial_id"], 1)
        state.request_stop()
        t.join(3)
        self.assertFalse(t.is_alive())

    def test_step_error_goes_to_error_state_and_closes_trial(self):
        state, trials = SharedState(CONFIG, 1), []

        def factory(cfg, seed):
            trials.append(FakeTrial(fail_at=2))
            return trials[-1]

        t = threading.Thread(target=serve_simulation, args=(state, factory, False, 0.001))
        t.start()
        state.submit("start")
        self.assertTrue(wait_for(lambda: state.snapshot()["state"] == "ERROR"))
        self.assertIn("boom", state.snapshot()["message"])
        self.assertTrue(wait_for(lambda: trials[0].closed))
        state.request_stop()
        t.join(3)
        self.assertFalse(t.is_alive())

    def test_reset_builds_new_trial_with_same_config(self):
        state, trials = SharedState(CONFIG, 5), []

        def factory(cfg, seed):
            trials.append((cfg, seed))
            return FakeTrial(steps=100)

        t = threading.Thread(target=serve_simulation, args=(state, factory, False, 0.001))
        t.start()
        self.assertTrue(wait_for(lambda: len(trials) == 1))
        state.submit("reset")
        self.assertTrue(wait_for(lambda: len(trials) == 2))
        self.assertEqual(trials[0], trials[1])
        state.request_stop()
        t.join(3)


# ----------------------------------------------------------------------------
@unittest.skipUnless(HAVE_FLASK, "flask is not installed")
class TestWebApp(unittest.TestCase):
    PORT = 8765
    TOKEN = "t" * 32
    BASE = "http://127.0.0.1:8765"

    def setUp(self):
        self.state = SharedState(CONFIG, 42)
        self.app = create_app(self.state, self.TOKEN, self.PORT, CONFIG)
        self.client = self.app.test_client()

    def get(self, path, **kw):
        return self.client.get(path, base_url=kw.pop("base_url", self.BASE), **kw)

    def post(self, path, payload=None, token=None, **kw):
        headers = {"X-Token": self.TOKEN if token is None else token, **kw.pop("headers", {})}
        return self.client.post(path, json={} if payload is None else payload,
                                headers=headers, base_url=self.BASE, **kw)

    # ---- page and headers ----
    def test_health(self):
        r = self.get("/api/health")
        self.assertEqual((r.status_code, r.get_json()), (200, {"ok": True}))

    def test_index_has_nonce_csp_and_token(self):
        r = self.get("/")
        html = r.get_data(as_text=True)
        csp = r.headers["Content-Security-Policy"]
        nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
        self.assertIn(f'nonce="{nonce}"', html)
        self.assertIn(self.TOKEN, html)
        self.assertNotIn("__TOKEN__", html)
        self.assertNotIn("__NONCE__", html)
        self.assertIn("default-src 'none'", csp)
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(r.headers["Cache-Control"], "no-store")
        self.assertEqual(r.headers["X-Frame-Options"], "DENY")

    def test_nonce_changes_between_requests(self):
        a = self.get("/").headers["Content-Security-Policy"]
        b = self.get("/").headers["Content-Security-Policy"]
        self.assertNotEqual(a, b)

    def test_page_does_not_use_innerhtml_or_inline_handlers(self):
        self.assertNotIn("innerHTML", gui.PAGE)
        self.assertNotIn("onclick=", gui.PAGE)

    # ---- request guards ----
    def test_wrong_host_is_forbidden(self):
        self.assertEqual(self.client.get("/api/health").status_code, 403)          # host 'localhost'
        self.assertEqual(self.get("/api/health", base_url="http://evil.example:8765").status_code, 403)

    def test_localhost_with_right_port_is_allowed(self):
        self.assertEqual(self.get("/api/health", base_url="http://localhost:8765").status_code, 200)

    def test_post_needs_valid_token(self):
        self.assertEqual(self.post("/api/start", token="").status_code, 403)
        self.assertEqual(self.post("/api/start", token="wrong").status_code, 403)
        r = self.client.post("/api/start", json={}, base_url=self.BASE)             # header missing
        self.assertEqual(r.status_code, 403)
        self.assertIsNone(self.state.next_command())

    def test_post_needs_json_content_type(self):
        r = self.client.post("/api/start", data="x", content_type="text/plain",
                            headers={"X-Token": self.TOKEN}, base_url=self.BASE)
        self.assertEqual(r.status_code, 415)

    def test_foreign_origin_is_forbidden(self):
        r = self.post("/api/start", headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = self.post("/api/start", headers={"Origin": self.BASE})
        self.assertEqual(r.status_code, 200)

    # ---- commands ----
    def test_apply_valid(self):
        r = self.post("/api/apply", good(seed=7))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.state.next_command(), "apply")
        self.assertEqual(self.state.applied()[1], 7)

    def test_apply_malformed_is_400(self):
        r = self.post("/api/apply", good(seed="x"))
        self.assertEqual(r.status_code, 400)
        self.assertIn("seed", r.get_json()["error"])
        self.assertIsNone(self.state.next_command())

    def test_apply_invalid_config_is_422_and_not_queued(self):
        r = self.post("/api/apply", good(obstacles=[{"x": -1.5, "y": -1.3, "r": 0.2}]))
        self.assertEqual(r.status_code, 422)
        self.assertTrue(r.get_json()["errors"])
        self.assertIsNone(self.state.next_command())
        self.assertEqual(self.state.applied(), (CONFIG, 42))

    def test_apply_while_running_is_409(self):
        self.state.set_running(True)
        self.assertEqual(self.post("/api/apply", good()).status_code, 409)

    def test_start_pause_reset_are_queued(self):
        for name in ("start", "pause", "reset"):
            self.assertEqual(self.post(f"/api/{name}").status_code, 200)
        self.assertEqual([self.state.next_command() for _ in range(3)], ["start", "pause", "reset"])

    def test_busy_queue_is_503(self):
        for _ in range(MAX_PENDING_COMMANDS):
            self.assertEqual(self.post("/api/start").status_code, 200)
        self.assertEqual(self.post("/api/start").status_code, 503)

    # ---- randomize ----
    def test_randomize_valid_and_reproducible(self):
        p = good(seed=5, count=3)
        a, b = self.post("/api/randomize", p), self.post("/api/randomize", p)
        self.assertEqual(a.status_code, 200)
        self.assertEqual(a.get_json(), b.get_json())
        obstacles = a.get_json()["obstacles"]
        self.assertEqual(len(obstacles), 3)
        cfg, _ = parse_payload(good(seed=5, obstacles=obstacles), CONFIG)
        self.assertEqual(validate_config(cfg), [])
        self.assertIsNone(self.state.next_command())             # randomize never applies

    def test_randomize_bad_count(self):
        for count in (0, 99, "3", None, True, 2.5):
            with self.subTest(count=count):
                self.assertEqual(self.post("/api/randomize", good(count=count)).status_code, 400)

    def test_randomize_with_invalid_start_is_400(self):
        p = good(count=2, start={"x": 50, "y": 50, "deg": 0})
        self.assertIn(self.post("/api/randomize", p).status_code, (400, 422))

    # ---- state and errors ----
    def test_state_shape(self):
        d = self.get("/api/state").get_json()
        self.assertEqual(set(d), {"config", "defaults", "status", "limits"})
        self.assertEqual(d["config"]["seed"], 42)
        self.assertEqual(d["limits"]["max_obstacles"], MAX_OBSTACLES)

    def test_status_is_strict_json(self):
        r = self.get("/api/status")
        self.assertNotIn("NaN", r.get_data(as_text=True))
        self.assertEqual(r.get_json()["est_pose"], [None, None, None])

    def test_errors_are_json(self):
        r = self.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.get_json())
        r = self.get("/api/apply")
        self.assertEqual(r.status_code, 405)
        self.assertIn("error", r.get_json())

    def test_oversized_body_is_413(self):
        r = self.post("/api/apply", {"pad": "x" * 40000})
        self.assertEqual(r.status_code, 413)


if __name__ == "__main__":
    unittest.main()