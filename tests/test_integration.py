# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

from config import CONFIG
from estimator import EKFEstimator
from main import (
    Phase, RunResult, TrialLog, build_parser, dock_errors, plot_run, print_report,
    run_pipeline, run_trials, theory_block,
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


def test_parser_new_options():
    args = build_parser().parse_args(["--trials", "5", "--marker-cov", "--sigma-px", "1.0"])
    assert args.trials == 5 and args.marker_cov and args.sigma_px == 1.0
    default = build_parser().parse_args([])
    assert default.trials == 0 and not default.marker_cov and default.sigma_px == 0.5


def test_run_result_new_fields_have_defaults():
    r = make_fake_result()
    assert r.camera_stats == (0, 0) and r.diag == {}


def test_print_report_shows_sensing_and_ekf_diagnostics(capsys):
    r = make_fake_result()
    r.camera_stats = (10, 8)
    r.diag = {"n_updates": 8, "n_rejected": 1, "n_recovered": 0,
            "n_lidar_updates": 40, "n_lidar_rejected": 2, "nis_per_dof": 1.1}
    print_report(r, CONFIG)
    out = capsys.readouterr().out
    assert "marker accepted 8/10" in out and "(80 %)" in out
    assert "mean NIS/dof 1.10" in out and "LiDAR updates 40" in out


def test_theory_block_mentions_marker_noise_model():
    text = theory_block(CONFIG, EKFEstimator(CONFIG))
    assert "Marker noise model" in text and "fixed (config)" in text
    assert "per-measurement" in theory_block(CONFIG, EKFEstimator(CONFIG, use_marker_cov=True))


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

def test_pipeline_records_camera_and_ekf_diagnostics(result):
    n_cam, n_det = result.camera_stats
    assert n_cam > 0 and 0 < n_det <= n_cam
    assert result.diag["n_updates"] > 0
    assert result.diag["n_lidar_updates"] > 0


def test_pipeline_filter_is_roughly_consistent(result):
    # Loose sanity band for mean NIS/dof (ideal ~1). Tight chi-square bounds go in the report.
    assert 0.1 < result.diag["nis_per_dof"] < 10.0


def test_pipeline_with_per_measurement_covariance(scene_dir):
    res = run_pipeline(seed=0, scene_dir=scene_dir, use_marker_cov=True)
    assert res.success and not res.collided
    assert res.metrics()["position_error"] < 0.15


def test_run_trials_prints_statistics(scene_dir, capsys):
    code = run_trials(2, seed0=0, scene_dir=scene_dir)
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "Success rate" in out and "Mean NEES" in out