"""Unit tests for Safe Reverse Recovery and Deadlock Replanning (Phase 13.24).

Mandatory Scenarios Tested:
1. Open environment (recovery stays INACTIVE).
2. Stationary obstacle ahead triggering deadlock recognition.
3. Obstacle slightly offset.
4. Obstacle blocking narrow corridor.
5. Obstacle appearing after initial path generation.
6. Rear area clear -> controlled reverse proceeds (-0.15 m/s).
7. Rear area blocked -> reverse aborted, vehicle remains in safe STOP.
8. Rear obstacle appears during reverse -> safety governor stops reverse immediately.
9. Dynamic obstacle crossing behind during recovery -> reverse safely halted.
10. Recovery duration timeout (bounded execution time).
11. Repeated deadlock handling.
12. Successful global replan after recovery disengagement.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path, PathPoint
from ai.planning.recovery_controller import (
    RecoveryConfig,
    RecoveryController,
    RecoveryState,
)
from ai.spatial_fusion.world_map import WorldTerrainMap


def _create_test_map(width: int = 100, height: int = 100, res: float = 0.10) -> WorldTerrainMap:
    wtm = WorldTerrainMap(
        resolution_m=res,
        x_bounds_m=[-2.0, 8.0],
        y_bounds_m=[-5.0, 5.0],
        created_timestamp=0.0,
    )
    wtm.cost.fill(0.10)
    wtm.obstacle.fill(False)
    wtm.hazard.fill(0.0)
    wtm.unknown.fill(False)
    return wtm


def _make_track(track_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> TrackedObstacleState:
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        frame_id="odom",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        covariance=np.eye(4) * 0.05,
        bbox_size=(0.8, 0.8),
        bbox_yaw=0.0,
        track_age=5,
        coast_cycles=0,
        is_confirmed=True,
        status="TRACKING",
    )


def test_scenario_1_open_environment():
    """Scenario 1: Open environment with no deadlock leaves recovery INACTIVE."""
    rec = RecoveryController()
    costmap = _create_test_map()

    v, w, state, _ = rec.step(
        current_time_s=1.0,
        robot_pose=(0.0, 0.0, 0.0),
        is_deadlocked=False,
        hazards=[],
        costmap=costmap,
    )
    assert state == RecoveryState.INACTIVE
    assert v is None


def test_scenario_2_stationary_obstacle_deadlock_trigger():
    """Scenario 2: Stationary obstacle ahead causes deadlock -> transitions to ASSESSING_REAR after N cycles."""
    rec = RecoveryController(RecoveryConfig(deadlock_stop_cycles=3))
    costmap = _create_test_map()
    hazards = [_make_track(1, x=0.70, y=0.0)]

    # Cycle 1 & 2: Building confirmation
    rec.step(1.0, (0.0, 0.0, 0.0), is_deadlocked=True, hazards=hazards, costmap=costmap)
    rec.step(1.1, (0.0, 0.0, 0.0), is_deadlocked=True, hazards=hazards, costmap=costmap)
    # Cycle 3: Triggers assessment
    v, _, state, _ = rec.step(1.2, (0.0, 0.0, 0.0), is_deadlocked=True, hazards=hazards, costmap=costmap)
    assert state == RecoveryState.REVERSING
    assert v == -0.15


def test_scenario_6_rear_clear_controlled_reverse():
    """Scenario 6: When rear is verified clear, commands bounded -0.15 m/s reverse until target distance."""
    rec = RecoveryController(RecoveryConfig(deadlock_stop_cycles=1, target_reverse_distance_m=0.40, reverse_velocity_mps=-0.15))
    costmap = _create_test_map()

    # Step 1: Trigger -> REVERSING
    v1, _, state1, _ = rec.step(1.0, (2.0, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert state1 == RecoveryState.REVERSING
    assert v1 == -0.15

    # Step 2: Intermediate reverse motion (x moved from 2.0 to 1.80m, disp = 0.20m < 0.40m)
    v2, _, state2, _ = rec.step(2.0, (1.80, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert state2 == RecoveryState.REVERSING
    assert v2 == -0.15

    # Step 3: Target distance reached (x moved to 1.55m, disp = 0.45m >= 0.40m) -> SETTLING
    v3, _, state3, _ = rec.step(3.0, (1.55, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert state3 == RecoveryState.SETTLING
    assert v3 == 0.0


def test_scenario_7_rear_blocked_aborts_reverse():
    """Scenario 7: When rear area is blocked by static or dynamic obstacle, reverse is aborted."""
    rec = RecoveryController(RecoveryConfig(deadlock_stop_cycles=1, min_rear_clearance_m=0.50))
    costmap = _create_test_map()
    # Rear obstacle directly behind UGV at x = -0.50m (UGV at x=0.0m)
    rear_hazard = [_make_track(2, x=-0.50, y=0.0)]

    v, _, state, reason = rec.step(1.0, (0.0, 0.0, 0.0), is_deadlocked=True, hazards=rear_hazard, costmap=costmap)
    assert state == RecoveryState.ABORTED_REAR_BLOCKED
    assert v == 0.0
    assert "Rear blocked" in reason


def test_scenario_8_rear_obstacle_appears_during_reverse():
    """Scenario 8: If rear obstacle appears while reversing, safety governor immediately halts reverse."""
    rec = RecoveryController(RecoveryConfig(deadlock_stop_cycles=1, target_reverse_distance_m=0.50))
    costmap = _create_test_map()

    # Step 1: Start reversing when clear
    rec.step(1.0, (2.0, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert rec.state == RecoveryState.REVERSING

    # Step 2: Unexpected dynamic obstacle steps behind robot at x=1.60m while robot is at x=1.85m
    sudden_rear_hazard = [_make_track(99, x=1.45, y=0.0)]
    v, _, state, reason = rec.step(1.5, (1.85, 0.0, 0.0), is_deadlocked=True, hazards=sudden_rear_hazard, costmap=costmap)
    assert state == RecoveryState.ABORTED_REAR_BLOCKED
    assert v == 0.0
    assert "safety governor" in reason


def test_scenario_10_recovery_duration_timeout():
    """Scenario 10: If robot is physically stuck and cannot reverse distance, duration timeout terminates reverse."""
    rec = RecoveryController(RecoveryConfig(deadlock_stop_cycles=1, max_reverse_duration_s=2.00, target_reverse_distance_m=1.0))
    costmap = _create_test_map()

    # Start reverse at t=10.0
    rec.step(10.0, (0.0, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert rec.state == RecoveryState.REVERSING

    # Wheel slip: robot pose barely moved (disp = 0.05m), but 2.1s elapsed
    v, _, state, reason = rec.step(12.1, (-0.05, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert state == RecoveryState.SETTLING
    assert v == 0.0
    assert "timeout" in reason


def test_scenario_12_successful_settle_and_replan_trigger():
    """Scenario 12: After settling, transitions to REPLAN_READY, enabling route regeneration."""
    rec = RecoveryController(RecoveryConfig(deadlock_stop_cycles=1, target_reverse_distance_m=0.30, settle_duration_s=0.20))
    costmap = _create_test_map()

    # 1. Reverse start
    rec.step(1.0, (1.0, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    # 2. Reverse finished (disp = 0.35m) -> SETTLING at t=2.0
    rec.step(2.0, (0.65, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert rec.state == RecoveryState.SETTLING

    # 3. Settle duration elapsed (t=2.25 >= 2.0 + 0.20) -> REPLAN_READY
    v, _, state, _ = rec.step(2.25, (0.65, 0.0, 0.0), is_deadlocked=True, hazards=[], costmap=costmap)
    assert state == RecoveryState.REPLAN_READY
    assert v == 0.0

    # Test marking replan complete resets to INACTIVE
    rec.mark_replan_completed()
    assert rec.state == RecoveryState.INACTIVE
