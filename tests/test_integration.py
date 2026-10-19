# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

from config import CONFIG, RealismConfig
from estimator import EKFEstimator
import main as main_module
from main import (
    MAX_RECOVERIES, Phase, RunResult, TrialLog, build_parser, compute_report, dock_errors,
    mode_label, plot_run, print_report, resume_navigator, run_trials, theory_block,
)
from navigation import Navigator
from recovery import RecoveryPhase
from robot_model import wrap_angle

DOCK_X, DOCK_Y, DOCK_YAW = CONFIG.env.dock_pose
HEAD_ON = wrap_angle(DOCK_YAW - math.pi)
D = CONFIG.docking
# The start-pose SURVEY is a separate feature with its own tests at the end of this file.
# The older tests check the rest of the pipeline with the tag poses from the config,
# so they run with the survey off (explicit use_survey=True still wins).
TEST_START = (0.3, 0.9, 0.0)   # clear view of the dock, so SURVEY can learn it
NO_SURVEY_CFG = replace(CONFIG, realism=RealismConfig(enabled=False),
                        env=replace(CONFIG.env, start_pose=TEST_START))

def run_pipeline(*args, **kwargs):
    kwargs.pop("use_survey", None)           # the survey always runs now
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


# ------------------------------------------------------- random obstacles (no sim)
def test_parser_fixed_obstacles_flag():
    assert not build_parser().parse_args([]).fixed_obstacles
    assert build_parser().parse_args(["--fixed-obstacles"]).fixed_obstacles


def test_run_result_obstacles_default_is_none():
    assert make_fake_result().obstacles is None


def test_report_clearance_uses_the_obstacles_of_the_run():
    r = make_fake_result()
    r.obstacles = [(0.25, 0.0, 0.2)]      # sits right on the fake trajectory (y = 0)
    clearance = compute_report(r, CONFIG)["min_obstacle_clearance"]
    assert clearance == pytest.approx(-(0.2 + CONFIG.robot.body_radius), abs=1e-3)


def test_plot_run_draws_custom_obstacles(tmp_path):
    r = make_fake_result()
    r.obstacles = [(0.5, 0.5, 0.2), (-0.5, -0.5, 0.3)]
    out = plot_run(r, CONFIG, tmp_path / "random.png")
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
    assert phases[0] == "survey" and phases[-1] == "done"
    assert "navigate" in phases[:first_dock]
    assert all(p in ("survey", "explore", "navigate", "recover") for p in phases[:first_dock])
    assert result.nav_time is not None and 1.0 < result.nav_time < result.time


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
    ekf = EKFEstimator(CONFIG, use_lidar=False)
    run_pipeline(seed=0, scene_dir=scene_dir, estimator=ekf)
    assert ekf.n_updates > 0          # the EKF correction step really ra

def test_pipeline_records_camera_and_ekf_diagnostics(result):
    n_cam, n_det = result.camera_stats
    assert n_cam > 0 and 0 < n_det <= n_cam
    assert result.diag["n_updates"] > 0


def test_pipeline_filter_is_roughly_consistent(result):
    # Loose sanity band for mean NIS/dof (ideal ~1). Tight chi-square bounds go in the report.
    assert 0.1 < result.diag["nis_per_dof"] < 10.0


def test_pipeline_with_per_measurement_covariance(scene_dir):
    res = run_pipeline(seed=0, scene_dir=scene_dir, use_marker_cov=True)
    assert res.success and not res.collided
    assert res.metrics()["position_error"] < 0.15


def test_run_trials_prints_statistics(scene_dir, capsys):
    code = run_trials(2, seed0=0, cfg=NO_SURVEY_CFG, scene_dir=scene_dir)
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "Success rate" in out and "Mean NEES" in out

# ------------------------------------------------------------ random obstacles (real pipeline)
RANDOM_SEEDS = [0, 1, 2]


@pytest.fixture(scope="module")
def random_result(scene_dir):
    return run_pipeline(seed=1, scene_dir=scene_dir, randomize_obstacles=True)


def test_default_pipeline_uses_the_fixed_layout(result):
    assert result.obstacles == list(CONFIG.env.obstacles)


def test_random_pipeline_uses_a_different_layout(random_result):
    assert len(random_result.obstacles) == len(CONFIG.env.obstacles)
    assert random_result.obstacles != list(CONFIG.env.obstacles)


def test_random_pipeline_is_reproducible_with_same_seed(scene_dir, random_result):
    again = run_pipeline(seed=1, scene_dir=scene_dir, randomize_obstacles=True)
    assert again.obstacles == random_result.obstacles
    assert again.time == pytest.approx(random_result.time)


def test_random_layout_changes_with_seed(scene_dir, random_result):
    other = run_pipeline(seed=2, scene_dir=scene_dir, randomize_obstacles=True)
    assert other.obstacles != random_result.obstacles


def test_random_pipeline_report_uses_the_random_layout(random_result):
    # Clearance must be computed against the obstacles of the scene, not the config ones
    rep = compute_report(random_result, CONFIG)
    assert rep["min_obstacle_clearance"] > 0.0
    assert not random_result.collided


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_pipeline_navigates_and_docks_with_random_obstacles(scene_dir, seed):
    res = run_pipeline(seed=seed, scene_dir=scene_dir, randomize_obstacles=True)
    assert not res.collided, f"collision with layout {res.obstacles}"
    assert res.success, f"failed ({res.reason}) with layout {res.obstacles}"


def test_run_trials_with_random_obstacles(scene_dir, capsys):
    code = run_trials(2, seed0=0, cfg=NO_SURVEY_CFG, scene_dir=scene_dir, randomize_obstacles=True)
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "Success rate" in out

def test_random_pipeline_filter_uses_the_random_map(random_result):
    # With a wrong LiDAR map the NIS/dof was 18-28; a correct map keeps it near 1
    assert 0.1 < random_result.diag["nis_per_dof"] < 10.0

# ------------------------------------------------------------ tag recovery (no sim)
def test_mode_label_shows_the_recovery_phase():
    assert mode_label(Phase.RECOVER, rphase=RecoveryPhase.SPIN) == "RECOVER-SPIN"
    assert mode_label(Phase.RECOVER, rphase=RecoveryPhase.GOTO) == "RECOVER-GOTO"
    assert mode_label(Phase.RECOVER) == "RECOVER"
    assert mode_label(Phase.DONE) == "DONE"


def test_resume_navigator_keeps_the_remaining_waypoints():
    nav = Navigator(CONFIG, waypoints=CONFIG.env.waypoints)
    nav.index = 2
    new = resume_navigator(nav, CONFIG, after_dock=False)
    np.testing.assert_allclose(new.waypoints, nav.waypoints[2:])
    assert new.index == 0 and new.tracker is nav.tracker


def test_resume_navigator_after_failed_docking_returns_to_the_last_waypoint():
    nav = Navigator(CONFIG, waypoints=CONFIG.env.waypoints)
    nav.index = 1
    new = resume_navigator(nav, CONFIG, after_dock=True)
    assert len(new.waypoints) == 1
    np.testing.assert_allclose(new.waypoints[0], nav.waypoints[-1])


def test_resume_navigator_falls_back_to_the_last_waypoint_when_none_remain():
    nav = Navigator(CONFIG, waypoints=CONFIG.env.waypoints)
    nav.index = len(nav.waypoints)
    new = resume_navigator(nav, CONFIG, after_dock=False)
    assert len(new.waypoints) == 1
    np.testing.assert_allclose(new.waypoints[0], nav.waypoints[-1])


def test_resume_navigator_after_failed_docking_uses_the_learned_approach_point():
    nav = Navigator(CONFIG, waypoints=CONFIG.env.waypoints)
    new = resume_navigator(nav, CONFIG, after_dock=True, approach=np.array([0.9, 1.0]))
    assert len(new.waypoints) == 1
    np.testing.assert_allclose(new.waypoints[0], [0.9, 1.0])


# ------------------------------------------------------------ tag recovery (real pipeline)
def lost_start_config():
    """Huge initial covariance and a start heading that faces an empty wall: filter starts lost."""
    sx, sy, _ = CONFIG.env.start_pose
    return replace(CONFIG,
                sim=replace(CONFIG.sim, max_time=60.0),
                env=replace(CONFIG.env, start_pose=(sx, sy, math.pi)),   # facing the west wall
                ekf=replace(CONFIG.ekf, init_pose_std=(0.5, 0.5, 0.3)))


def run_lost(scene_dir):
    cfg = lost_start_config()
    # LiDAR off, otherwise the map update shrinks P before the recovery can start
    return run_pipeline(cfg, seed=0, scene_dir=scene_dir,
                        estimator=EKFEstimator(cfg, use_lidar=False))


@pytest.fixture(scope="module")
def lost_result(scene_dir):
    return run_lost(scene_dir)


def test_recovery_can_be_switched_off(scene_dir):
    res = run_pipeline(seed=0, scene_dir=scene_dir, use_recovery=False)
    assert "recover" not in res.log.phase
    assert res.diag["n_tag_recoveries"] == 0


def test_pipeline_without_extra_tags_never_recovers(scene_dir):
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, n_tags=0))
    res = run_pipeline(cfg, seed=0, scene_dir=scene_dir)
    assert res.success and not res.collided
    assert "recover" not in res.log.phase
    assert res.diag["n_tag_recoveries"] == 0 and res.diag["n_tag_updates"] == 0


def test_recovery_does_not_depend_on_ground_truth(scene_dir, lost_result, monkeypatch):
    import simulation
    monkeypatch.setattr(simulation.Simulation, "ground_truth",
                        lambda self: np.array([99.0, -99.0, 2.0]))
    poisoned = run_lost(scene_dir)
    assert poisoned.log.phase == lost_result.log.phase
    np.testing.assert_allclose(poisoned.log.arrays()["est"], lost_result.log.arrays()["est"])

# ------------------------------------------------------------ start-pose SURVEY
def test_mode_label_shows_the_survey_and_map_follow_phases():
    assert mode_label(Phase.SURVEY) == "SURVEY"
    assert mode_label(Phase.RECOVER, rphase=RecoveryPhase.MAP_GOTO) == "RECOVER-MAP"


def test_print_report_shows_the_learned_tag_map_error(capsys):
    r = make_fake_result()
    r.diag = {"n_updates": 8, "n_rejected": 0, "n_recovered": 0, "n_lidar_updates": 40,
            "n_lidar_rejected": 0, "nis_per_dof": 1.0, "tag_map_errors": {1: 0.012, 2: 0.034}}
    print_report(r, CONFIG)
    out = capsys.readouterr().out
    assert "learned tag map error" in out and "id 1: 0.012" in out and "id 2: 0.034" in out


def test_older_tests_run_without_the_survey(result):
    assert result.log.phase[0] == "survey"


def test_survey_is_the_first_phase_and_fails_when_no_tag_is_learned(scene_dir):
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, survey_min_obs=10_000))   # nothing reaches the map
    res = run_pipeline(cfg, seed=0, scene_dir=scene_dir, use_survey=True)
    assert res.log.phase[0] == "survey"
    assert not res.success and res.reason == "explore failed"
    assert any("EXPLORE FAILED" in e for e in res.events)


def test_survey_learns_the_nearest_tag_accurately(scene_dir):
    cfg = replace(CONFIG, sim=replace(CONFIG.sim, max_time=20.0))
    res = run_pipeline(cfg, seed=0, scene_dir=scene_dir, use_survey=True, own_start=True)
    errs = res.diag["tag_map_errors"]
    assert errs and min(errs.values()) < 0.05          # tag 1 is 0.7 m from the start pose


def test_survey_learns_the_dock_when_it_is_in_range(scene_dir):
    # Start pose with a clear view of the dock (1.6 m away, no obstacle in between)
    cfg = replace(CONFIG, sim=replace(CONFIG.sim, max_time=18.0),
                env=replace(CONFIG.env, start_pose=(0.3, 0.9, 0.0)))
    res = run_pipeline(cfg, seed=0, scene_dir=scene_dir, use_survey=True)
    dock_id = int(CONFIG.camera.marker_id)
    assert any("SURVEY done" in e for e in res.events)
    assert dock_id in res.diag["tag_map_errors"]
    assert res.diag["tag_map_errors"][dock_id] < 0.06


def test_survey_does_not_depend_on_ground_truth(scene_dir, monkeypatch):
    cfg = replace(CONFIG, sim=replace(CONFIG.sim, max_time=17.0))
    base = run_pipeline(cfg, seed=0, scene_dir=scene_dir, use_survey=True)
    import simulation
    monkeypatch.setattr(simulation.Simulation, "ground_truth",
                        lambda self: np.array([99.0, -99.0, 2.0]))
    poisoned = run_pipeline(cfg, seed=0, scene_dir=scene_dir, use_survey=True)
    assert poisoned.diag["tag_map_errors"] == base.diag["tag_map_errors"]
    np.testing.assert_allclose(poisoned.log.arrays()["est"], base.log.arrays()["est"])