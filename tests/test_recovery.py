# Including libraries
import math
from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG
from recovery import RecoveryPhase, TagRecovery
from sensing import make_lidar_scan

DOCK_ID = CONFIG.camera.marker_id
TWO_PI = 2.0 * math.pi
OMEGA = CONFIG.tags.spin_omega


def free_scan():
    angles = np.linspace(0.0, TWO_PI, CONFIG.lidar.num_beams, endpoint=False)
    return make_lidar_scan(np.full(len(angles), CONFIG.lidar.range_max), angles)


def make_recovery(cfg=CONFIG):
    return TagRecovery(cfg)


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


def test_start_without_tags_fails_immediately():
    cfg = replace(CONFIG, tags=replace(CONFIG.tags, n_tags=0))
    rec = make_recovery(cfg)
    assert rec.tags == {}
    rec.start()
    assert rec.phase is RecoveryPhase.FAILED and rec.finished


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


def test_spin_lasts_one_full_turn_before_driving_to_a_tag():
    rec, scan, step = make_recovery(), free_scan(), 0.05
    rec.start()
    turned = 0.0
    for _ in range(500):
        rec.update(np.zeros(3), scan, False, step)
        turned += step
        if rec.phase is not RecoveryPhase.SPIN:
            break
    assert rec.phase is RecoveryPhase.GOTO
    assert turned >= TWO_PI - 1e-9 and turned - step < TWO_PI
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


def test_each_tag_is_visited_once_then_recovery_fails():
    rec, scan = make_recovery(), free_scan()
    rec.start()
    x = np.zeros(3)
    targets = []
    for _ in range(len(rec.tags)):
        rec.update(x, scan, False, TWO_PI)                 # full turn without a tag -> next tag
        assert rec.phase is RecoveryPhase.GOTO
        targets.append(rec.target_id)
        x = np.append(rec.standoff_point(rec.target_id), 0.0)
        rec.update(x, scan, False, 0.0)                    # arrived -> look around again
        assert rec.phase is RecoveryPhase.SPIN
    assert sorted(targets) == sorted(rec.tags)             # no tag twice
    assert rec.update(x, scan, False, TWO_PI) == (0.0, 0.0)  # nothing left to try
    assert rec.phase is RecoveryPhase.FAILED and rec.finished


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