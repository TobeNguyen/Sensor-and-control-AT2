# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG
from recovery import MAX_TAGS, RecoveryPhase, TagRecovery
from robot_model import TagMap, measurement_model_pose
from sensing import MarkerMeasurement, make_lidar_scan
TWO_PI = 2 * math.pi
DOCK_ID = CONFIG.camera.marker_id
OMEGA = CONFIG.tags.spin_omega
HALF_TURN = math.pi
class ConfigTagMap:
    # Test helper: pretends the config poses were "learned" at run time
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def table(self):
        return {int(i): (np.array(p[:2], dtype=float), float(p[2]))
                for i, p in self.cfg.tag_table().items()}

def free_scan():
    angles = np.linspace(0.0, TWO_PI, CONFIG.lidar.num_beams, endpoint=False)
    return make_lidar_scan(np.full(len(angles), CONFIG.lidar.range_max), angles)


def make_recovery(cfg=CONFIG):
    return TagRecovery(cfg, tag_map=ConfigTagMap(cfg))


def make_tag_map(poses):
    """TagMap learned from exact measurements taken at the origin: poses = {id: (x, y, yaw)}."""
    tm = TagMap()
    origin = np.zeros(3)
    for i, (x, y, yaw) in poses.items():
        r, b, view = measurement_model_pose(origin, np.array([x, y]), yaw)
        tm.add(origin, MarkerMeasurement(r, b, view, i, 0.0))
    return tm


# ------------------------------------------------------------------ setup
def test_recovery_knows_only_the_extra_tags():
    rec = make_recovery()
    assert DOCK_ID not in rec.tags
    assert sorted(rec.tags) == sorted(i for i in CONFIG.tag_table() if i != DOCK_ID)


def test_initial_state_is_idle_and_outputs_zero():
    rec = make_recovery()
    assert rec.phase is RecoveryPhase.IDLE
    assert not rec.active and not rec.finished
    assert rec.update(np.zeros(3), free_scan(), False, 0.1) == (0.0, 0.0)
    assert rec.phase is RecoveryPhase.IDLE


def test_start_begins_with_a_spin():
    rec = make_recovery()
    rec.start()
    assert rec.phase is RecoveryPhase.SPIN
    assert rec.active and not rec.finished


def test_start_without_extra_tags_but_with_the_dock_still_spins():
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, n_tags=0))
    rec = make_recovery(cfg)
    assert rec.tags == {} and rec.dock is not None and rec.available
    rec.start()
    assert rec.phase is RecoveryPhase.SPIN


def test_start_with_an_empty_map_has_nothing_to_recover_to():
    for rec in (TagRecovery(CONFIG, tag_map=TagMap()), TagRecovery(CONFIG)):
        assert rec.tags == {} and rec.dock is None and not rec.available
        rec.start()
        assert rec.phase is RecoveryPhase.FAILED


def test_standoff_point_is_in_front_of_the_tag():
    rec = make_recovery()
    for tid, (pos, yaw) in rec.tags.items():
        p = rec.standoff_point(tid)
        d = p - pos
        assert np.hypot(*d) == pytest.approx(CONFIG.tags.standoff)
        np.testing.assert_allclose(d / np.hypot(*d), [math.cos(yaw), math.sin(yaw)], atol=1e-9)


# ------------------------------------------------------------------ SPIN
def test_spin_rotates_in_place_at_the_configured_speed():
    rec = make_recovery()
    rec.start()
    assert rec.update(np.zeros(3), free_scan(), False, 0.01) == (0.0, OMEGA)
    assert rec.phase is RecoveryPhase.SPIN


def test_spin_lasts_half_a_turn_before_driving_to_a_tag():
    rec, scan, step = make_recovery(), free_scan(), 0.05
    rec.start()
    turned = 0.0
    for _ in range(500):
        rec.update(np.zeros(3), scan, False, step)
        turned += step
        if rec.phase is not RecoveryPhase.SPIN:
            break
    assert rec.phase is RecoveryPhase.GOTO
    assert turned >= HALF_TURN - 1e-9 and turned - step < HALF_TURN
    assert rec.target_id in rec.tags


def test_spin_counts_rotation_in_both_directions():
    rec = make_recovery()
    rec.start()
    rec.update(np.zeros(3), free_scan(), False, -TWO_PI)
    assert rec.phase is RecoveryPhase.GOTO


def test_localized_during_spin_finishes_recovery():
    rec = make_recovery()
    rec.start()
    assert rec.update(np.zeros(3), free_scan(), True, 0.01) == (0.0, 0.0)
    assert rec.phase is RecoveryPhase.DONE and rec.finished


# ------------------------------------------------------------------ GOTO
@pytest.mark.parametrize("tag_id", sorted(i for i in CONFIG.tag_table() if i != DOCK_ID))
def test_goto_picks_the_tag_with_the_nearest_standoff_point(tag_id):
    rec = make_recovery()
    rec.start()
    standoff = rec.standoff_point(tag_id)
    x = np.array([standoff[0], standoff[1] + 0.3, 0.0])
    rec.update(x, free_scan(), False, TWO_PI)
    assert rec.phase is RecoveryPhase.GOTO
    assert rec.target_id == tag_id and tag_id in rec._visited


def test_goto_drives_towards_the_standoff_point():
    rec = make_recovery()
    rec.start()
    v, w = rec.update(np.array([0.0, 0.0, 0.0]), free_scan(), False, TWO_PI)
    assert rec.phase is RecoveryPhase.GOTO
    assert abs(v) > 0.0 or abs(w) > 0.0
    assert 0.0 <= v <= CONFIG.robot.max_linear_vel + 1e-9


def test_arriving_at_the_standoff_point_starts_a_new_spin():
    rec = make_recovery()
    rec.start()
    rec.update(np.zeros(3), free_scan(), False, TWO_PI)
    x = np.append(rec.standoff_point(rec.target_id), 0.0)
    assert rec.update(x, free_scan(), False, 0.0) == (0.0, OMEGA)
    assert rec.phase is RecoveryPhase.SPIN


def test_localized_during_goto_finishes_recovery():
    rec = make_recovery()
    rec.start()
    rec.update(np.zeros(3), free_scan(), False, TWO_PI)
    assert rec.phase is RecoveryPhase.GOTO
    assert rec.update(np.zeros(3), free_scan(), True, 0.0) == (0.0, 0.0)
    assert rec.phase is RecoveryPhase.DONE


def test_only_max_tags_are_visited_then_the_map_is_followed():
    rec, scan = make_recovery(), free_scan()
    rec.start()
    x = np.zeros(3)
    targets = []
    for _ in range(min(MAX_TAGS, len(rec.tags))):
        rec.update(x, scan, False, TWO_PI)                 # half turn without a tag -> next tag
        assert rec.phase is RecoveryPhase.GOTO
        targets.append(rec.target_id)
        x = np.append(rec.standoff_point(rec.target_id), 0.0)
        rec.update(x, scan, False, 0.0)                    # arrived -> look around again
        assert rec.phase is RecoveryPhase.SPIN
    assert len(set(targets)) == len(targets)               # no tag twice
    assert len(targets) == min(MAX_TAGS, len(rec.tags))
    v, w = rec.update(x, scan, False, TWO_PI)              # tag limit reached: follow the learned map
    assert rec.phase is RecoveryPhase.MAP_GOTO and rec.target_id == DOCK_ID
    assert (v, w) != (0.0, 0.0)
    assert rec.active and not rec.finished


# ------------------------------------------------------------------ end states
def test_finished_recovery_outputs_zero():
    rec = make_recovery()
    rec.start()
    rec.update(np.zeros(3), free_scan(), True, 0.0)
    assert rec.phase is RecoveryPhase.DONE
    assert rec.update(np.zeros(3), free_scan(), False, 0.1) == (0.0, 0.0)
    assert rec.phase is RecoveryPhase.DONE


def test_reset_restores_the_initial_state():
    rec = make_recovery()
    rec.start()
    rec.update(np.zeros(3), free_scan(), False, TWO_PI)
    rec.reset()
    assert rec.phase is RecoveryPhase.IDLE
    assert rec.target_id is None and rec._visited == set() and rec._spun == 0.0
    assert not rec.active and not rec.finished


def test_start_clears_the_previous_run():
    rec = make_recovery()
    rec.start()
    rec.update(np.zeros(3), free_scan(), False, TWO_PI)
    rec.start()
    assert rec.phase is RecoveryPhase.SPIN and rec._visited == set()

# ------------------------------------------------------------------ learned map + map follow
def test_recovery_uses_the_learned_map_instead_of_the_config():
    tm = make_tag_map({DOCK_ID: (1.0, 1.0, math.pi), 1: (-1.0, 1.9, -math.pi / 2)})
    rec = TagRecovery(CONFIG, tag_map=tm)
    assert sorted(rec.tags) == [1]
    np.testing.assert_allclose(rec.tags[1][0], [-1.0, 1.9], atol=1e-9)
    np.testing.assert_allclose(rec.dock[0], [1.0, 1.0], atol=1e-9)


def test_approach_point_is_in_front_of_the_dock():
    rec = make_recovery()
    pos, yaw = rec.dock
    p = rec.approach_point()
    assert np.hypot(*(p - pos)) == pytest.approx(CONFIG.tags.approach_dist)
    assert np.dot(p - pos, [math.cos(yaw), math.sin(yaw)]) > 0.0


def test_map_follow_ends_at_the_approach_point_without_relocalising():
    rec, scan = make_recovery(), free_scan()
    rec.start()
    rec._visited = set(rec.tags)                           # pretend every tag was tried
    rec.update(np.zeros(3), scan, False, TWO_PI)           # full turn -> map follow
    assert rec.phase is RecoveryPhase.MAP_GOTO
    x = np.append(rec.approach_point(), 0.0)
    assert rec.update(x, scan, False, 0.0) == (0.0, 0.0)
    assert rec.phase is RecoveryPhase.DONE and rec.arrived_by_map and rec.finished


def test_localised_during_map_follow_is_a_normal_relocalisation():
    rec, scan = make_recovery(), free_scan()
    rec.start()
    rec._visited = set(rec.tags)
    rec.update(np.zeros(3), scan, False, TWO_PI)
    assert rec.phase is RecoveryPhase.MAP_GOTO
    assert rec.update(np.zeros(3), scan, True, 0.0) == (0.0, 0.0)
    assert rec.phase is RecoveryPhase.DONE and not rec.arrived_by_map


def test_map_without_extra_tags_spins_once_then_follows_the_map():
    tm = make_tag_map({DOCK_ID: (1.0, 1.0, math.pi)})
    rec, scan = TagRecovery(CONFIG, tag_map=tm), free_scan()
    rec.start()
    assert rec.phase is RecoveryPhase.SPIN
    rec.update(np.zeros(3), scan, False, TWO_PI)
    assert rec.phase is RecoveryPhase.MAP_GOTO


def test_reset_clears_the_map_follow_flag():
    rec, scan = make_recovery(), free_scan()
    rec.start()
    rec._visited = set(rec.tags)
    rec.update(np.zeros(3), scan, False, TWO_PI)
    rec.update(np.append(rec.approach_point(), 0.0), scan, False, 0.0)
    assert rec.arrived_by_map
    rec.reset()
    assert not rec.arrived_by_map and rec.phase is RecoveryPhase.IDLE