"""Unit and integration tests for Subphase 15.7: Authoritative 3D Terrain & Downhill Braking Safety Governor.

Verifies:
1. Analytical downhill stopping distance formula:
   d_stop(theta) = d_stop_0 * a_max / max(0.30, a_max - g * sin(theta_downhill)).
2. Earlier braking on downhill grades: an obstacle at 1.15m ahead that is ALLOWED on flat ground
   triggers immediate STOP_FORWARD_HAZARD when descending a 10 deg downhill pitch.
3. 3D IMU rollover tilt protection:
   - Cautious speed clamp (<= 0.15 m/s) when roll or pitch is in [18 deg, 26 deg).
   - Authoritative zero-velocity emergency stop when roll or pitch >= 26 deg.
4. Forward corridor 2.5D terrain hazard halt:
   - Immediate halt before a >= 28 deg slope drop-off (`costmap.slope`) or >= 0.12 m step/ditch (`costmap.height_var`).
5. `AuthoritativeSafetyGovernor` integration in `production/safety/safety_governor.py`.
"""

from __future__ import annotations

import math
import pytest

from ai.planning.safety_governor import (
    LatchedHazard,
    SafetyGovernor,
    SafetyGovernorConfig,
)
from ai.spatial_fusion.world_map import WorldTerrainMap
from production.runtime.adaptation_manager import RuntimeMetrics
from production.safety.safety_governor import AuthoritativeSafetyGovernor


def test_downhill_stopping_distance_mathematical_formula():
    """Verify exact downhill stopping distance scaling against gravitational component g*sin(theta)."""
    gov = SafetyGovernor(SafetyGovernorConfig(stop_distance_m=0.40, max_linear_deceleration_mps2=1.50))

    # Flat ground (0 deg): d_stop = 0.40m
    assert abs(gov.compute_downhill_stopping_distance(0.0) - 0.40) < 1e-6

    # 5 deg downhill: g*sin(5 deg) = 0.854999 -> scale = 1.50 / (1.50 - 0.854999) = 2.32558 -> d_stop = 0.9302m
    theta_5 = math.radians(5.0)
    expected_5 = 0.40 * (1.50 / max(0.30, 1.50 - 9.81 * math.sin(theta_5)))
    assert abs(gov.compute_downhill_stopping_distance(theta_5) - expected_5) < 1e-6
    assert gov.compute_downhill_stopping_distance(theta_5) > 0.90

    # 10 deg downhill: g*sin(10 deg) = 1.7035 > 1.20 -> clamped denominator 0.30 -> scale = 5.0 -> d_stop = 2.00m
    theta_10 = math.radians(10.0)
    assert abs(gov.compute_downhill_stopping_distance(theta_10) - 2.00) < 1e-6


def test_earlier_braking_on_downhill_grade_vs_flat_ground():
    """Verify an obstacle at effective distance 1.10m is ALLOW on flat ground, CAUTIOUS on 4 deg downhill,
    and STOP on 9 deg downhill.
    """
    gov = SafetyGovernor()
    # Hazard at x=1.50m with bbox 0.80m -> radius 0.40m -> effective_dist_x = 1.10m
    hazard = LatchedHazard(hazard_id="boulder_1", x=1.50, y=0.0, bbox_size=(0.80, 0.80))

    # 1. Flat ground (0 deg pitch): 1.10m > cautious_distance_m (0.95m) -> ALLOW full speed
    v_flat, _, dec_flat, diag_flat = gov.evaluate(
        requested_v=0.60,
        requested_w=0.0,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[hazard],
        robot_roll_pitch=(0.0, 0.0),
    )
    assert dec_flat == "ALLOW"
    assert v_flat == 0.60
    assert diag_flat["effective_stop_distance_m"] == 0.40

    # 2. 3 deg downhill pitch: eff_cautious = 0.95 * (1.50 / (1.50 - 9.81*sin(3 deg))) = 1.44m >= 1.10m -> CAUTIOUS_SLOWDOWN
    v_caut, _, dec_caut, _ = gov.evaluate(
        requested_v=0.60,
        requested_w=0.0,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[hazard],
        robot_roll_pitch=(0.0, math.radians(3.0)),
    )
    assert "CAUTIOUS_SLOWDOWN" in dec_caut
    assert v_caut <= 0.15

    # 3. 8 deg downhill pitch: eff_stop = 0.40 * (1.50 / 0.30) = 2.00m >= 1.10m -> STOP_FORWARD_HAZARD
    v_stop, w_stop, dec_stop, diag_stop = gov.evaluate(
        requested_v=0.60,
        requested_w=0.20,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[hazard],
        robot_roll_pitch=(0.0, math.radians(8.0)),
    )
    assert "STOP_FORWARD_HAZARD" in dec_stop
    assert v_stop == 0.0
    assert w_stop == 0.0
    assert diag_stop["effective_stop_distance_m"] >= 1.10


def test_imu_tilt_18deg_cautious_and_26deg_rollover_halt():
    """Verify 18 deg IMU tilt clamps speed to <= 0.15 m/s and 26 deg IMU tilt triggers emergency stop."""
    gov = SafetyGovernor()

    # 20 deg roll -> cautious slowdown
    v_caut, _, dec_caut, _ = gov.evaluate(
        requested_v=0.55,
        requested_w=0.10,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[],
        robot_roll_pitch=(math.radians(20.0), 0.0),
    )
    assert "CAUTIOUS_TILT_SLOWDOWN" in dec_caut
    assert v_caut == pytest.approx(0.15)

    # 26.5 deg roll -> authoritative rollover stop
    v_stop, w_stop, dec_stop, _ = gov.evaluate(
        requested_v=0.55,
        requested_w=0.30,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[],
        robot_roll_pitch=(-math.radians(26.5), 0.0),
    )
    assert "STOP_TILT_ROLLOVER_HAZARD" in dec_stop
    assert v_stop == 0.0
    assert w_stop == 0.0


def test_forward_corridor_halts_before_28deg_dropoff_and_15cm_ditch():
    """Verify SafetyGovernor inspects costmap.slope and costmap.height_var in forward corridor and halts."""
    gov = SafetyGovernor()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 5.0), y_bounds_m=(-2.0, 2.0))
    wmap.unknown.fill(False)

    # Case A: 30 deg slope drop-off at x = 0.30m ahead (even with obstacle=False)
    r_drop, c_drop = wmap.world_to_grid(0.30, 0.0)
    wmap.slope[r_drop, c_drop] = 30.0

    v_s, w_s, dec_s, diag_s = gov.evaluate(
        requested_v=0.45,
        requested_w=0.0,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[],
        costmap=wmap,
    )
    assert "STOP_FORWARD_HAZARD" in dec_s
    assert v_s == 0.0 and w_s == 0.0
    assert diag_s["blocking_hazard_id"] == "terrain_slope_dropoff"

    # Case B: 15 cm step / ditch hazard at x = 0.30m ahead
    wmap.slope.fill(0.0)
    wmap.height_var[r_drop, c_drop] = 0.15

    v_h, w_h, dec_h, diag_h = gov.evaluate(
        requested_v=0.45,
        requested_w=0.0,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[],
        costmap=wmap,
    )
    assert "STOP_FORWARD_HAZARD" in dec_h
    assert v_h == 0.0 and w_h == 0.0
    assert diag_h["blocking_hazard_id"] == "terrain_step_ditch_hazard"


def test_authoritative_production_safety_governor_integration():
    """Verify AuthoritativeSafetyGovernor propagates 3D tilt and terrain costmap checks."""
    auth_gov = AuthoritativeSafetyGovernor()
    metrics = RuntimeMetrics(
        timestamp_s=1.0,
        perception_latency_ms=15.0,
        planning_latency_ms=2.0,
        total_loop_latency_ms=18.0,
    )
    v, w, decision, diag = auth_gov.evaluate_safe_command(
        requested_v=0.50,
        requested_w=0.10,
        robot_pose=(0.0, 0.0, 0.0),
        hazards=[],
        metrics=metrics,
        robot_roll_pitch=(math.radians(27.0), 0.0),
    )
    assert v == 0.0 and w == 0.0
    assert "STOP_TILT_ROLLOVER_HAZARD" in decision
    assert diag["max_tilt_deg"] == pytest.approx(27.0)
