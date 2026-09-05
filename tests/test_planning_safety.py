"""Unit tests for SafetyController, staleness guards, watchdog, and emergency stop."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.safety import SafetyController
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_empty_world_map(resolution: float = 0.10, bounds: tuple[float, float] = (-5.0, 5.0)) -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def test_safety_lost_state_emergency_stop():
    """Section 60: Localization state LOST triggers immediate zero velocity emergency stop."""
    safety = SafetyController()
    wmap = create_empty_world_map()
    st = RobotState(x=0.0, y=0.0, yaw=0.0, timestamp=10.0)

    # State 2 = LOST
    is_safe, status, reason, is_deg = safety.evaluate_safety_guards(
        current_time_s=10.0, current_state=st, terrain_map=wmap, tracking_state=2
    )
    assert not is_safe
    assert status == PlannerStatus.EMERGENCY_STOP

    # Command arbitration must return 0.0
    v, w, arb_status, diag = safety.arbitrate_command(
        requested_v=0.8, requested_w=0.2, dt_s=0.1, current_time_s=10.0,
        planner_status=status, is_emergency=True
    )
    assert v == 0.0
    assert w == 0.0
    assert diag["emergency_override"] is True


def test_safety_degraded_velocity_cap():
    """Section 61: DEGRADED tracking state enforces conservative speed v <= 0.35 m/s."""
    safety = SafetyController()
    wmap = create_empty_world_map()
    st = RobotState(x=0.0, y=0.0, yaw=0.0, timestamp=10.0)

    # State 1 = DEGRADED
    is_safe, status, reason, is_deg = safety.evaluate_safety_guards(
        current_time_s=10.0, current_state=st, terrain_map=wmap, tracking_state=1
    )
    assert is_safe
    assert is_deg is True

    # Request high speed of 1.0 m/s in degraded mode
    v, w, arb_status, diag = safety.arbitrate_command(
        requested_v=1.0, requested_w=0.0, dt_s=0.5, current_time_s=10.0,
        planner_status=PlannerStatus.FOLLOWING, is_degraded=True
    )
    assert v <= 0.35 + 1e-4
    assert diag["clamped"] is True


def test_safety_map_staleness():
    """Section 62: Map timestamp older than max_map_age triggers MAP_STALE stop."""
    safety = SafetyController()
    wmap = create_empty_world_map()
    wmap.created_timestamp = 5.0  # Old map
    st = RobotState(x=0.0, y=0.0, yaw=0.0, timestamp=10.0)

    # Current time = 10.0s, map age = 5.0s > max_map_age (2.0s)
    is_safe, status, reason, is_deg = safety.evaluate_safety_guards(
        current_time_s=10.0, current_state=st, terrain_map=wmap,
        map_timestamp_s=5.0, loc_timestamp_s=10.0
    )
    assert not is_safe
    assert status == PlannerStatus.MAP_STALE


def test_safety_loc_staleness():
    """Localization timestamp older than max_loc_age triggers LOCALIZATION_INVALID stop."""
    safety = SafetyController()
    wmap = create_empty_world_map()
    st = RobotState(x=0.0, y=0.0, yaw=0.0, timestamp=8.0)

    # Current time = 10.0s, loc age = 2.0s > max_loc_age (1.0s)
    is_safe, status, reason, is_deg = safety.evaluate_safety_guards(
        current_time_s=10.0, current_state=st, terrain_map=wmap,
        map_timestamp_s=10.0, loc_timestamp_s=8.0
    )
    assert not is_safe
    assert status == PlannerStatus.LOCALIZATION_INVALID


def test_safety_current_footprint_in_obstacle():
    """Section 36: Robot already inside an obstacle triggers RECOVERY_REQUIRED."""
    safety = SafetyController()
    wmap = create_empty_world_map()
    # Place obstacle directly at robot location
    r, c = wmap.world_to_grid(0.0, 0.0)
    wmap.obstacle[r, c] = True

    st = RobotState(x=0.0, y=0.0, yaw=0.0, timestamp=10.0)
    is_safe, status, reason, is_deg = safety.evaluate_safety_guards(
        current_time_s=10.0, current_state=st, terrain_map=wmap
    )
    assert not is_safe
    assert status == PlannerStatus.RECOVERY_REQUIRED


def test_safety_command_watchdog():
    """Section 40: Watchdog emits zero velocity if command interval exceeds command_timeout."""
    safety = SafetyController()

    # Cycle 1 at t=1.0s
    v1, w1, s1, d1 = safety.arbitrate_command(
        requested_v=0.5, requested_w=0.0, dt_s=0.1, current_time_s=1.0,
        planner_status=PlannerStatus.FOLLOWING
    )
    assert v1 > 0.0

    # Cycle 2 delayed until t=1.5s (> timeout 0.25s)
    v2, w2, s2, d2 = safety.arbitrate_command(
        requested_v=0.5, requested_w=0.0, dt_s=0.1, current_time_s=1.5,
        planner_status=PlannerStatus.FOLLOWING
    )
    assert v2 == 0.0
    assert d2["watchdog_override"] is True


def test_safety_nan_inf_rejection():
    """Section 86: NaN or Inf command values are rejected and zero velocity is commanded."""
    safety = SafetyController()

    v, w, s, d = safety.arbitrate_command(
        requested_v=float("nan"), requested_w=0.0, dt_s=0.1, current_time_s=1.0,
        planner_status=PlannerStatus.FOLLOWING
    )
    assert v == 0.0
    assert w == 0.0
    assert s == PlannerStatus.EMERGENCY_STOP
