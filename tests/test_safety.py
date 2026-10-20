import math

import numpy as np
import pytest

from config import CONFIG
from safety import SafetyGuard
from sensing import make_lidar_scan

N = CONFIG.lidar.num_beams
ANGLES = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
DT = CONFIG.sim.control_dt
FRONT, BACK = 0, N // 2


def scan(hits=None):
    """Free scan (2 m everywhere); hits = {beam_index: range}."""
    r = np.full(N, 2.0)
    for i, d in (hits or {}).items():
        r[i] = d
    return make_lidar_scan(r, ANGLES)


def test_free_command_passes_after_the_rate_limiter_settles():
    g = SafetyGuard(CONFIG)
    for _ in range(50):
        v, w, why = g.filter(0.1, 0.0, scan(), DT)
    assert v == pytest.approx(0.1) and why == ""


def test_first_step_respects_the_acceleration_limit():
    v, _, _ = SafetyGuard(CONFIG).filter(0.2, 0.0, scan(), DT)
    assert 0.0 < v <= g_max_step()


def g_max_step():
    return SafetyGuard(CONFIG).sc.max_lin_acc * DT + 1e-9


def test_braking_is_never_delayed():
    g = SafetyGuard(CONFIG)
    for _ in range(50):
        g.filter(0.1, 0.0, scan(), DT)
    assert g.filter(0.0, 0.0, scan(), DT)[:2] == (0.0, 0.0)


def test_non_finite_command_is_stopped_without_latching():
    g = SafetyGuard(CONFIG)
    v, w, why = g.filter(float("nan"), 0.0, scan(), DT)
    assert (v, w) == (0.0, 0.0) and why == "invalid command" and g.fault == ""


def test_stale_lidar_stops_but_does_not_latch():
    g = SafetyGuard(CONFIG)
    v, w, why = g.filter(0.1, 0.3, scan(), DT, lidar_age=0.6)
    assert (v, w) == (0.0, 0.0) and "stale" in why and g.fault == ""


def test_too_few_valid_beams_is_treated_as_invalid_lidar():
    empty = make_lidar_scan(np.full(N, CONFIG.lidar.range_max), ANGLES)
    assert SafetyGuard(CONFIG).filter(0.1, 0.0, empty, DT)[:2] == (0.0, 0.0)


def test_lidar_loss_latches_an_estop_until_reset():
    g = SafetyGuard(CONFIG)
    assert g.filter(0.1, 0.0, scan(), DT, lidar_age=3.0)[:2] == (0.0, 0.0)
    assert g.fault
    v, w, why = g.filter(0.1, 0.0, scan(), DT, lidar_age=0.0)
    assert (v, w) == (0.0, 0.0) and why == g.fault
    g.reset()
    assert g.fault == ""


def test_forward_obstacle_blocks_forward_translation():
    v, _, why = SafetyGuard(CONFIG).filter(0.1, 0.0, scan({FRONT: 0.2}), DT)
    assert v == 0.0 and "translation blocked" in why


def test_obstacle_behind_blocks_reversing_but_not_driving_forward():
    g = SafetyGuard(CONFIG)
    assert g.filter(-0.1, 0.0, scan({BACK: 0.2}), DT)[0] == 0.0
    v, _, why = SafetyGuard(CONFIG).filter(0.1, 0.0, scan({BACK: 0.2}), DT)
    assert v > 0.0 and "translation blocked" not in why


def test_forward_obstacle_does_not_block_reversing():
    v, _, why = SafetyGuard(CONFIG).filter(-0.1, 0.0, scan({FRONT: 0.2}), DT)
    assert v < 0.0 and "translation blocked" not in why


def test_rotation_checks_the_whole_footprint_ring():
    _, w, why = SafetyGuard(CONFIG).filter(0.0, 0.5, scan({BACK: 0.2}), DT)
    assert w == 0.0 and "rotation blocked" in why
    _, w, _ = SafetyGuard(CONFIG).filter(0.0, 0.5, scan(), DT)
    assert w > 0.0


def test_high_position_uncertainty_stops_translation_but_allows_rotation():
    v, w, why = SafetyGuard(CONFIG).filter(0.1, 0.3, scan(), DT, sigma_xy=2.0)
    assert v == 0.0 and w > 0.0 and "uncertainty" in why