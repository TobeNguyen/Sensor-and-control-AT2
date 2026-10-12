# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

from config import CONFIG
from estimator import EKFEstimator
from main import (
    Phase, RunResult, TrialLog, build_parser, dock_errors, plot_run, run_pipeline,
)
from robot_model import wrap_angle

DOCK_X, DOCK_Y, DOCK_YAW = CONFIG.env.dock_pose
HEAD_ON = wrap_angle(DOCK_YAW - math.pi)
D = CONFIG.docking

# ------------------------------------------------------- evaluation helpers (no sim)
def test_dock_errors_perfect_pose():
    e = dock_errors((DOCK_X - D.target_distance, DOCK_Y, HEAD_ON))
    assert e["dock_range"] == pytest.approx(D.target_distance)
    assert e["dock_lateral"] == pytest.approx(0.0, abs=1e-9)
    assert e["dock_heading"] == pytest.approx(0.0, abs=1e-9)


def test_dock_errors_lateral_and_heading():
    e = dock_errors((DOCK_X - 0.4, DOCK_Y + 0.05, HEAD_ON + 0.1))
    assert e["dock_lateral"] == pytest.approx(0.05)
    assert e["dock_heading"] == pytest.approx(0.1)


def test_parser_options():
    args = build_parser().parse_args(["--seed", "3", "--plot"])
    assert args.seed == 3 and args.plot and not args.no_viewer
    assert build_parser().parse_args(["--no-viewer"]).no_viewer


def make_fake_result():
    log = TrialLog()
    for k in range(50):
        pose = np.array([k * 0.01, 0.0, 0.0])
        log.record(k * 0.02, pose, pose + 0.01, np.diag([1e-4] * 3), "navigate", 0.2, 0.0)
    pose = np.array([DOCK_X - 0.42, DOCK_Y, HEAD_ON])
    return RunResult(True, "docked", 1.0, 0.5, False, Phase.DONE, pose + 0.01, pose, log)


def test_run_result_metrics_and_summary():
    result = make_fake_result()
    m = result.metrics()
    assert m["position_error"] == pytest.approx(math.hypot(0.01, 0.01))
    assert m["dock_range"] == pytest.approx(0.42)
    assert "SUCCESS" in result.summary() and "docked" in result.summary()


def test_plot_run_writes_a_png(tmp_path):
    out = plot_run(make_fake_result(), CONFIG, tmp_path / "run.png")
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------------------ real MuJoCo pipeline
@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    pytest.importorskip("mujoco")
    from simulation import Simulation
    path = tmp_path_factory.mktemp("scene")
    try:
        with Simulation(scene_dir=path) as sim:
            sim.read_camera()
    except Exception as exc:  # no OpenGL context available
        pytest.skip(f"MuJoCo offscreen rendering not available: {exc}")
    return path


@pytest.fixture(scope="module")
def result(scene_dir):
    return run_pipeline(seed=0, scene_dir=scene_dir)


def test_pipeline_navigates_and_docks(result):
    assert result.success, f"pipeline failed: {result.reason}"
    assert result.reason == "docked"
    assert result.phase is Phase.DONE
    assert result.time < 90.0


def test_pipeline_has_no_collision(result):
    assert not result.collided


def test_pipeline_runs_navigation_before_docking(result):
    phases = result.log.phase
    first_dock = phases.index("dock")
    assert phases[0] == "navigate" and phases[-1] == "done"
    assert all(p == "navigate" for p in phases[:first_dock])
    assert result.nav_time is not None and 5.0 < result.nav_time < result.time


def test_pipeline_docking_accuracy(result):
    m = result.metrics()
    assert D.target_distance - D.distance_tolerance <= m["dock_range"] <= (
        D.target_distance + D.distance_tolerance + 0.04)
    assert m["dock_lateral"] < 0.06
    assert m["dock_heading"] < 0.2


def test_pipeline_estimate_stays_close_to_the_truth(result):
    m = result.metrics()
    assert m["position_error"] < 0.1
    assert m["heading_error"] < 0.1


def test_pipeline_log_is_consistent(result):
    a = result.log.arrays()
    n = len(a["t"])
    assert n > 100
    assert a["truth"].shape == (n, 3) and a["est"].shape == (n, 3) and a["sigma"].shape == (n, 3)
    assert np.all(np.diff(a["t"]) > 0)


def test_control_does_not_depend_on_ground_truth(scene_dir, result, monkeypatch):
    """Replace the simulator's ground truth by garbage: the run must not change."""
    import simulation
    monkeypatch.setattr(simulation.Simulation, "ground_truth",
                        lambda self: np.array([99.0, -99.0, 2.0]))
    poisoned = run_pipeline(seed=0, scene_dir=scene_dir)
    assert poisoned.success == result.success
    assert poisoned.time == pytest.approx(result.time)
    np.testing.assert_allclose(poisoned.log.arrays()["est"], result.log.arrays()["est"])


def test_pipeline_reports_a_timeout(scene_dir):
    short = replace(CONFIG, sim=replace(CONFIG.sim, max_time=3.0))
    res = run_pipeline(short, seed=0, scene_dir=scene_dir)
    assert not res.success and res.reason == "timeout"
    assert res.time == pytest.approx(3.0, abs=0.05)


def test_pipeline_reports_a_collision(scene_dir):
    ox, oy, _ = CONFIG.env.obstacles[0]
    blocked = replace(CONFIG, env=replace(CONFIG.env, start_pose=(ox, oy, 0.0)))
    res = run_pipeline(blocked, seed=0, scene_dir=scene_dir)
    assert not res.success and res.reason == "collision" and res.collided

def test_pipeline_applies_marker_updates(scene_dir):
    ekf = EKFEstimator(CONFIG)
    run_pipeline(seed=0, scene_dir=scene_dir, estimator=ekf)
    assert ekf.n_updates > 0          # the EKF correction step really ran


def test_ekf_is_not_worse_than_dead_reckoning(scene_dir, result):
    dr = run_pipeline(seed=0, scene_dir=scene_dir,
                    estimator=EKFEstimator(CONFIG, use_marker=False))
    assert result.metrics()["position_error"] <= dr.metrics()["position_error"] + 0.02