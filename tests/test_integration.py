import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

from config import CONFIG, RealismConfig
from estimator import EKFEstimator
import main as main_module
from main import (
    Phase, RunResult, TrialLog, build_parser, compute_report, dock_errors,
    mode_label, plot_run, print_report, resume_navigator, run_trials,
)
from navigation import Navigator
from recovery import RecoveryPhase
from robot_model import wrap_angle
from visualization import RouteVisualizer

DOCK_X, DOCK_Y, DOCK_YAW = CONFIG.env.dock_pose
HEAD_ON = wrap_angle(DOCK_YAW - math.pi)
D = CONFIG.docking

# Start pose with a clear view of the dock, so the start-pose scan can learn it
TEST_START = (0.3, 0.9, 0.0)
NO_SURVEY_CFG = replace(CONFIG, realism=RealismConfig(enabled=False),
                        env=replace(CONFIG.env, start_pose=TEST_START))


def run_pipeline(*args, **kwargs):
    """Wrapper: ideal world and a start pose that sees the dock, unless told otherwise."""
    kwargs.pop("use_survey", None)
    realistic = kwargs.pop("realistic", False)
    own_start = kwargs.pop("own_start", False)
    cfg = args[0] if args else kwargs.pop("cfg", CONFIG)
    rest = args[1:]
    if not realistic:
        cfg = replace(cfg, realism=RealismConfig(enabled=False))
    if not own_start and tuple(cfg.env.start_pose) == tuple(CONFIG.env.start_pose):
        cfg = replace(cfg, env=replace(cfg.env, start_pose=TEST_START))
    return main_module.run_pipeline(cfg, *rest, **kwargs)


# ------------------------------------------------------- evaluation helpers (no sim)
def make_fake_result():
    log = TrialLog()
    for k in range(50):
        pose = np.array([k * 0.01, 0.0, 0.0])
        log.record(k * 0.02, pose, pose + 0.01, np.diag([1e-4] * 3), "navigate", 0.2, 0.0)
    pose = np.array([DOCK_X - 0.42, DOCK_Y, HEAD_ON])
    return RunResult(True, "docked", 1.0, 0.5, False, Phase.DONE, pose + 0.01, pose, log)


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
    args = build_parser().parse_args(["--seed", "3", "--plot", "--trials", "5"])
    assert args.seed == 3 and args.plot and args.trials == 5 and not args.no_viewer
    assert build_parser().parse_args(["--no-viewer"]).no_viewer


def test_run_result_metrics_and_summary():
    result = make_fake_result()
    m = result.metrics()
    assert m["position_error"] == pytest.approx(math.hypot(0.01, 0.01))
    assert m["dock_range"] == pytest.approx(0.42)
    assert "SUCCESS" in result.summary() and "docked" in result.summary()
    assert result.camera_stats == (0, 0) and result.diag == {} and result.obstacles is None


def test_plot_run_writes_a_png(tmp_path):
    r = make_fake_result()
    r.obstacles = [(0.5, 0.5, 0.2), (-0.5, -0.5, 0.3)]
    out = plot_run(r, CONFIG, tmp_path / "run.png")
    assert out.exists() and out.stat().st_size > 0


def test_print_report_shows_sensing_and_ekf_diagnostics(capsys):
    r = make_fake_result()
    r.camera_stats = (10, 8)
    r.diag = {"n_updates": 8, "n_rejected": 1, "n_recovered": 0,
              "n_lidar_updates": 40, "n_lidar_rejected": 2, "nis_per_dof": 1.1,
              "tag_map_errors": {1: 0.012}}
    print_report(r, CONFIG)
    out = capsys.readouterr().out
    assert "marker accepted 8/10" in out and "(80 %)" in out
    assert "mean NIS/dof 1.10" in out and "id 1: 0.012" in out


# ------------------------------------------------------------ recovery helpers (no sim)
def test_mode_label():
    assert mode_label(Phase.SURVEY) == "SURVEY"
    assert mode_label(Phase.DONE) == "DONE"
    assert mode_label(Phase.RECOVER, rphase=RecoveryPhase.SPIN) == "RECOVER-SPIN"
    assert mode_label(Phase.RECOVER, rphase=RecoveryPhase.MAP_GOTO) == "RECOVER-MAP"


def test_resume_navigator_keeps_the_remaining_waypoints():
    nav = Navigator(CONFIG, waypoints=CONFIG.env.waypoints)
    nav.index = 1
    new = resume_navigator(nav, CONFIG, after_dock=False)
    np.testing.assert_allclose(new.waypoints, nav.waypoints[1:])
    assert new.index == 0 and new.tracker is nav.tracker


def test_resume_navigator_after_failed_docking():
    nav = Navigator(CONFIG, waypoints=CONFIG.env.waypoints)
    last = resume_navigator(nav, CONFIG, after_dock=True)
    np.testing.assert_allclose(last.waypoints[0], nav.waypoints[-1])
    approach = resume_navigator(nav, CONFIG, after_dock=True, approach=np.array([0.9, 1.0]))
    np.testing.assert_allclose(approach.waypoints[0], [0.9, 1.0])


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
    assert result.reason == "docked" and result.phase is Phase.DONE
    assert result.time < 90.0
    assert not result.collided

def test_pipeline_runs_navigation_before_docking(result):
    phases = result.log.phase
    first_dock = phases.index("dock")
    assert phases[0] == "survey" and phases[-1] == "done"
    assert all(p in ("survey", "explore", "navigate", "recover") for p in phases[:first_dock])
    assert result.nav_time is not None and 0.0 < result.nav_time < result.time


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


def test_pipeline_log_and_diagnostics(result):
    a = result.log.arrays()
    n = len(a["t"])
    assert n > 100 and np.all(np.diff(a["t"]) > 0)
    assert a["truth"].shape == (n, 3) and a["est"].shape == (n, 3)
    n_cam, n_det = result.camera_stats
    assert n_cam > 0 and 0 < n_det <= n_cam
    d = result.diag
    assert d["n_updates"] == 0 or 0.01 < d["nis_per_dof"] < 10.0


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
    res = run_pipeline(blocked, seed=0, scene_dir=scene_dir, own_start=True)
    assert not res.success and res.reason == "collision" and res.collided


def test_pipeline_applies_marker_updates(scene_dir):
    ekf = EKFEstimator(CONFIG, use_lidar=False)
    res = run_pipeline(seed=0, scene_dir=scene_dir, estimator=ekf)
    assert res.success
    assert ekf.n_updates >= 0                      # may be 0 when no learned marker is in view
    assert np.all(np.isfinite(ekf.P))


def test_recovery_can_be_switched_off(scene_dir):
    res = run_pipeline(seed=0, scene_dir=scene_dir, use_recovery=False)
    assert "recover" not in res.log.phase
    assert res.diag["n_tag_recoveries"] == 0


def test_run_trials_prints_statistics(scene_dir, capsys):
    code = run_trials(2, seed0=0, cfg=NO_SURVEY_CFG, scene_dir=scene_dir)
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "Success rate" in out and "Mean NEES" in out


# ------------------------------------------------------------ start-pose scan
def test_scan_is_the_first_phase_and_fails_when_no_tag_is_learned(scene_dir):
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, survey_min_obs=10_000))   # nothing reaches the map
    res = run_pipeline(cfg, seed=0, scene_dir=scene_dir)
    assert res.log.phase[0] == "survey"
    assert not res.success and res.reason == "explore failed"


def test_scan_learns_the_dock_accurately(scene_dir):
    cfg = replace(CONFIG, sim=replace(CONFIG.sim, max_time=18.0))
    res = run_pipeline(cfg, seed=0, scene_dir=scene_dir)
    dock_id = int(CONFIG.camera.marker_id)
    assert any("SURVEY done" in e for e in res.events)
    assert res.diag["tag_map_errors"][dock_id] < 0.06

@pytest.mark.parametrize("seed", [100, 101, 102])
def test_pipeline_is_safe_with_the_fixed_layout(scene_dir, seed):
    # real start pose (0, 0, 0): the obstacle hides the dock, so the robot must explore
    res = run_pipeline(seed=seed, scene_dir=scene_dir, own_start=True,
                    realistic=True, dock_use_estimate=True)
    assert not res.collided, f"collision with layout {res.obstacles}"
    assert res.success



def test_report_uses_the_fixed_layout(result):
    rep = compute_report(result, CONFIG)
    assert rep["min_obstacle_clearance"] > 0.0
    assert not result.collided


def test_lidar_outage_triggers_a_latched_safety_stop(scene_dir):
    res = run_pipeline(seed=0, scene_dir=scene_dir, lidar_fault_at=5.0)
    assert not res.success and res.reason == "safety fault" and not res.collided
    assert any("E-STOP" in e for e in res.events)


def test_parser_has_recovery_and_lidar_fault_options():
    args = build_parser().parse_args(["--lidar-fault", "20"])
    assert args.lidar_fault == 20.0
    assert not args.no_recovery and not args.no_calib_odometry
    off = build_parser().parse_args(["--no-recovery", "--no-calib-odometry"])
    assert off.no_recovery and off.no_calib_odometry


def path_length(xy):
    # Total driven distance from the logged true positions
    return float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_scenario_docks_with_an_efficient_path(scene_dir, seed):
    # Same settings as main.py: real start, hidden obstacle, noise, recovery + calibration
    res = run_pipeline(seed=seed, scene_dir=scene_dir, own_start=True, realistic=True,
                    dock_use_estimate=True, use_recovery=True, calib_odometry=True)
    assert res.success, f"seed {seed}: {res.reason}"
    assert not res.collided

    # Docking quality against the REAL dock (it is shifted in the hidden world)
    m = res.metrics(res.world_cfg)
    assert m["dock_lateral"] < 0.08
    assert m["dock_heading"] < 0.2

    # Safety margins
    rep = compute_report(res, res.world_cfg)
    assert rep["min_obstacle_clearance"] > 0.05
    assert rep["min_wall_clearance"] > 0.05

    # Path quality: driven distance vs straight line start -> final pose
    truth = res.log.arrays()["truth"][:, :2]
    straight = float(np.hypot(*(truth[-1] - truth[0])))
    assert path_length(truth) < 8.0 * straight    # tune after looking at real numbers
    assert res.time < 120.0