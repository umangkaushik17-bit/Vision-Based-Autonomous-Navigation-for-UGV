"""Integration tests for Phase 14.1 Mission Executive recovery orchestration and multi-waypoint navigation."""

from dataclasses import dataclass
import math
import numpy as np
import pytest

from ai.control.command_buffer import SafeCommandBuffer, SafeCommandBufferConfig
from ai.planning.base import PlannerStatus
from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.path import Path, PathPoint
from ai.planning.recovery_controller import RecoveryConfig, RecoveryState
from ai.planning.safety_governor import LatchedHazard, SafetyGovernor
from ai.spatial_fusion.world_map import WorldTerrainMap

from ugv_mission import (
    MissionAbortReason,
    MissionExecutive,
    MissionExecutiveConfig,
    MissionState,
    Waypoint,
)


def _create_test_map() -> WorldTerrainMap:
    """Create lightweight test terrain costmap."""
    wtm = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=[-5.0, 15.0],
        y_bounds_m=[-5.0, 5.0],
        created_timestamp=0.0,
    )
    wtm.cost.fill(0.10)
    wtm.obstacle.fill(False)
    wtm.hazard.fill(0.0)
    wtm.unknown.fill(False)
    return wtm


def _make_hazard(track_id: int, x: float, y: float) -> TrackedObstacleState:
    """Construct dynamic obstacle hazard."""
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        frame_id="odom",
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        covariance=np.eye(4) * 0.05,
        bbox_size=(0.8, 0.8),
        bbox_yaw=0.0,
        track_age=5,
        coast_cycles=0,
        is_confirmed=True,
        status="TRACKING",
    )


class MockGlobalPlanner:
    """Mock global planner providing deterministic straight-line paths for integration tests."""

    def __init__(self, succeed: bool = True):
        self.succeed = succeed
        self.replan_calls = 0
        self.last_goal = None

    def plan(self, costmap, start, goal):
        self.replan_calls += 1
        self.last_goal = (goal.x, goal.y)
        if not self.succeed:
            return None, PlannerStatus.NO_PATH, {}

        pts = [
            PathPoint(x=start.x, y=start.y, yaw=0.0, cost=0.1),
            PathPoint(x=goal.x, y=goal.y, yaw=0.0, cost=0.1),
        ]
        return Path(points=pts), PlannerStatus.SUCCESS, {}


# ==============================================================================
# CATEGORY D: Recovery Integration & Same-Waypoint Invariant
# ==============================================================================

def test_recovery_at_w3_resumes_same_waypoint():
    """Verify: Deadlock at W3 enters RECOVERY, replans toward W3 (NEVER skips to W4)."""
    planner = MockGlobalPlanner(succeed=True)
    cmd_buf = SafeCommandBuffer(SafeCommandBufferConfig())
    exec_engine = MissionExecutive(
        config=MissionExecutiveConfig(mission_name="recovery_same_wp", max_recovery_attempts=3),
        global_planner=planner,
        command_buffer=cmd_buf,
    )
    exec_engine.recovery_orchestrator.controller.config.deadlock_stop_cycles = 1
    exec_engine.recovery_orchestrator.controller.config.target_reverse_distance_m = 0.30

    wps = [
        Waypoint(id="W1", x=2.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W2", x=4.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W3", x=6.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W4", x=8.0, y=0.0, xy_tolerance_m=0.35),
    ]
    costmap = _create_test_map()
    exec_engine.load_mission(wps)
    exec_engine.start_mission((0.0, 0.0, 0.0), costmap=costmap)

    # Fast-forward to W3
    exec_engine.step(1.0, (2.0, 0.0, 0.0), costmap=costmap)  # W1 reached -> W2
    exec_engine.step(2.0, (4.0, 0.0, 0.0), costmap=costmap)  # W2 reached -> W3
    assert exec_engine.active_waypoint.id == "W3"
    assert exec_engine.active_index == 2

    # UGV halts before obstacle at (5.0, 0.0), dist to W3 is 1.0m > 0.35m
    # Deadlock occurs (DWA status = LOCAL_NO_SAFE_TRAJECTORY)
    state, cmd_v, _, _ = exec_engine.step(
        current_time_s=3.0,
        robot_pose=(5.0, 0.0, 0.0),
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        costmap=costmap,
    )
    assert state == MissionState.RECOVERY
    assert exec_engine.active_waypoint.id == "W3"  # Invariant: Still W3!
    assert cmd_v == -0.15  # Reverse commanded

    # Step reverse: robot reverses from 5.0 to 4.65m (disp = 0.35m >= 0.30m target)
    state, _, _, _ = exec_engine.step(
        current_time_s=4.0,
        robot_pose=(4.65, 0.0, 0.0),
        costmap=costmap,
    )
    # Settle past duration (0.25s) -> transitions through SETTLING -> REPLANNING -> NAVIGATING
    state, _, _, reason = exec_engine.step(
        current_time_s=4.5,
        robot_pose=(4.65, 0.0, 0.0),
        costmap=costmap,
    )
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    # CRITICAL INVARIANT VERIFICATION:
    assert exec_engine.active_waypoint.id == "W3", "Active waypoint must remain W3 after recovery!"
    assert exec_engine.active_index == 2
    assert planner.last_goal == (6.0, 0.0), "Replanner must replan toward W3!"


# ==============================================================================
# CATEGORY E: Recovery Failure & Bounded Retry Exhaustion
# ==============================================================================

def test_recovery_failure_retry_limit_aborts():
    """Verify: Repeated recovery failures exceeding retry limit triggers MISSION_ABORTED."""
    planner = MockGlobalPlanner(succeed=False)
    exec_engine = MissionExecutive(
        config=MissionExecutiveConfig(mission_name="retry_exhaustion", max_recovery_attempts=2),
        global_planner=planner,
    )
    exec_engine.recovery_orchestrator.controller.config.deadlock_stop_cycles = 1

    wps = [
        Waypoint(id="W1", x=2.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W2", x=4.0, y=0.0, xy_tolerance_m=0.35),
    ]
    costmap = _create_test_map()
    exec_engine.load_mission(wps)
    exec_engine.start_mission((0.0, 0.0, 0.0), costmap=costmap)

    # Robot at (1.0, 0.0). Rear bumper is at 1.0 - 0.325 = 0.675m.
    # Place rear hazard directly behind rear bumper at x=0.40m (< 0.50m rear clearance).
    rear_hazard = _make_hazard(track_id=99, x=0.40, y=0.0)

    # Attempt 1: Deadlock with rear blocked
    state, cmd_v, cmd_w, _ = exec_engine.step(
        current_time_s=1.0,
        robot_pose=(1.0, 0.0, 0.0),
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        hazards=[rear_hazard],
        costmap=costmap,
    )
    # Attempt 1 fails: holds in RECOVERY awaiting retry
    assert state == MissionState.RECOVERY
    assert cmd_v == 0.0
    assert exec_engine.recovery_orchestrator.attempt_count == 1

    # Attempt 2: Deadlock persists with rear still blocked
    state, cmd_v, cmd_w, reason = exec_engine.step(
        current_time_s=2.0,
        robot_pose=(1.0, 0.0, 0.0),
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        hazards=[rear_hazard],
        costmap=costmap,
    )
    # Max attempts (2) reached -> Aborts safely!
    assert state == MissionState.MISSION_ABORTED
    assert exec_engine.state == MissionState.MISSION_ABORTED
    assert exec_engine.abort_reason == MissionAbortReason.RECOVERY_PERMANENTLY_BLOCKED
    assert cmd_v == 0.0 and cmd_w == 0.0
    assert exec_engine.active_waypoint.id == "W1"  # Never advanced to W2


# ==============================================================================
# CATEGORY F: Recovery then Waypoint Completion
# ==============================================================================

def test_recovery_then_waypoint_completion():
    """Verify: Deadlock at W2 -> recovery -> replan -> robot resumes and reaches W2 -> W3 becomes active."""
    planner = MockGlobalPlanner(succeed=True)
    exec_engine = MissionExecutive(
        config=MissionExecutiveConfig(mission_name="rec_then_complete", max_recovery_attempts=3),
        global_planner=planner,
    )
    exec_engine.recovery_orchestrator.controller.config.deadlock_stop_cycles = 1
    exec_engine.recovery_orchestrator.controller.config.target_reverse_distance_m = 0.20

    wps = [
        Waypoint(id="W1", x=2.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W2", x=4.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W3", x=6.0, y=0.0, xy_tolerance_m=0.35),
    ]
    costmap = _create_test_map()
    exec_engine.load_mission(wps)
    exec_engine.start_mission((0.0, 0.0, 0.0), costmap=costmap)

    # Reach W1
    exec_engine.step(1.0, (2.0, 0.0, 0.0), costmap=costmap)
    assert exec_engine.active_waypoint.id == "W2"

    # Deadlock before W2 at x=3.2m
    state, cmd_v, _, _ = exec_engine.step(
        current_time_s=2.0,
        robot_pose=(3.2, 0.0, 0.0),
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        costmap=costmap,
    )
    assert state == MissionState.RECOVERY

    # Complete reverse (reverses to 2.95m, disp = 0.25m >= 0.20m)
    exec_engine.step(2.5, (2.95, 0.0, 0.0), costmap=costmap)
    # Settle and replan
    state, _, _, _ = exec_engine.step(3.0, (2.95, 0.0, 0.0), costmap=costmap)
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.active_waypoint.id == "W2"

    # Robot drives forward around obstacle and arrives at W2 (x=4.0m)
    state, _, _, _ = exec_engine.step(4.0, (4.0, 0.0, 0.0), costmap=costmap)
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    # Verifies that W2 was NOT skipped: W3 is now active
    assert exec_engine.active_waypoint.id == "W3"
    assert "W2" in exec_engine.waypoint_manager.reached_ids


# ==============================================================================
# CATEGORY H: Safety Invariants Enforcement
# ==============================================================================

def test_safety_rear_clearance_enforcement():
    """Verify: Safety Governor strictly halts reverse motion if rear hazard appears during recovery."""
    planner = MockGlobalPlanner(succeed=True)
    exec_engine = MissionExecutive(global_planner=planner)
    exec_engine.recovery_orchestrator.controller.config.deadlock_stop_cycles = 1

    wps = [Waypoint(id="W1", x=5.0, y=0.0, xy_tolerance_m=0.35)]
    costmap = _create_test_map()
    exec_engine.load_mission(wps)
    exec_engine.start_mission((0.0, 0.0, 0.0), costmap=costmap)

    # 1. Initiate recovery when rear is clear
    state, cmd_v, _, _ = exec_engine.step(
        current_time_s=1.0,
        robot_pose=(2.0, 0.0, 0.0),
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        hazards=[],
        costmap=costmap,
    )
    assert state == MissionState.RECOVERY
    assert cmd_v == -0.15

    # 2. While reversing, a new dynamic hazard appears directly behind bumper at x=1.5m
    sudden_rear_hazard = _make_hazard(track_id=42, x=1.50, y=0.0)
    state, cmd_v, _, _ = exec_engine.step(
        current_time_s=1.5,
        robot_pose=(1.90, 0.0, 0.0),
        hazards=[sudden_rear_hazard],
        costmap=costmap,
    )
    # Reverse velocity is halted immediately by the safety governor
    assert cmd_v == 0.0


# ==============================================================================
# SECTION 23: MANDATORY 5-WAYPOINT DETERMINISTIC INTEGRATION SCENARIO
# ==============================================================================

def test_complete_5_waypoint_mission_with_recovery():
    """Mandatory Section 23 scenario:
    
    Mission: W1 -> W2 -> W3 -> W4 -> W5
    
    Execution Trace:
    1. Mission initialization
    2. W1 navigation -> W1 arrival
    3. W2 activation -> W2 arrival
    4. W3 activation -> deadlock before W3
    5. Recovery at W3 -> safe reverse -> replan toward W3
    6. W3 arrival
    7. W4 activation -> W4 arrival
    8. W5 activation -> W5 arrival (Final)
    9. MISSION_COMPLETE!
    """
    planner = MockGlobalPlanner(succeed=True)
    cmd_buf = SafeCommandBuffer(SafeCommandBufferConfig())
    exec_engine = MissionExecutive(
        config=MissionExecutiveConfig(mission_name="section_23_benchmark", max_recovery_attempts=3),
        global_planner=planner,
        command_buffer=cmd_buf,
    )
    exec_engine.recovery_orchestrator.controller.config.deadlock_stop_cycles = 1
    exec_engine.recovery_orchestrator.controller.config.target_reverse_distance_m = 0.25

    wps = [
        Waypoint(id="W1", x=2.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W2", x=4.0, y=0.5, xy_tolerance_m=0.35),
        Waypoint(id="W3", x=6.0, y=0.0, xy_tolerance_m=0.35),
        Waypoint(id="W4", x=8.0, y=-0.5, xy_tolerance_m=0.35),
        Waypoint(id="W5", x=10.0, y=0.0, xy_tolerance_m=0.35),
    ]
    costmap = _create_test_map()

    # 1. Mission Initialization
    ok, _ = exec_engine.load_mission(wps)
    assert ok is True
    assert exec_engine.state == MissionState.INITIALIZING
    ok_start, _ = exec_engine.start_mission((0.0, 0.0, 0.0), costmap=costmap)
    assert ok_start is True
    assert exec_engine.state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.active_waypoint.id == "W1"

    # 2. W1 Navigation -> W1 Arrival
    exec_engine.step(1.0, (1.0, 0.0, 0.0), costmap=costmap)
    assert exec_engine.active_waypoint.id == "W1"
    exec_engine.step(2.0, (2.0, 0.0, 0.0), costmap=costmap)  # W1 reached
    assert "W1" in exec_engine.waypoint_manager.reached_ids

    # 3. W2 Activation -> W2 Arrival
    assert exec_engine.active_waypoint.id == "W2"
    exec_engine.step(3.0, (3.0, 0.25, 0.0), costmap=costmap)
    exec_engine.step(4.0, (4.0, 0.50, 0.0), costmap=costmap)  # W2 reached
    assert "W2" in exec_engine.waypoint_manager.reached_ids

    # 4. W3 Activation
    assert exec_engine.active_waypoint.id == "W3"
    exec_engine.step(5.0, (5.0, 0.25, 0.0), costmap=costmap)

    # 5. Deadlock before W3 at x=5.4m
    state, cmd_v, _, _ = exec_engine.step(
        current_time_s=6.0,
        robot_pose=(5.40, 0.10, 0.0),
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        costmap=costmap,
    )
    assert state == MissionState.RECOVERY
    assert exec_engine.active_waypoint.id == "W3"
    assert cmd_v == -0.15

    # Reverse disengagement (reverses to 5.10m, disp = 0.30m >= 0.25m)
    exec_engine.step(7.0, (5.10, 0.10, 0.0), costmap=costmap)
    # Settle & Replan
    state, _, _, _ = exec_engine.step(7.5, (5.10, 0.10, 0.0), costmap=costmap)
    assert state == MissionState.NAVIGATING_TO_WAYPOINT
    assert exec_engine.active_waypoint.id == "W3"

    # 6. W3 Arrival
    exec_engine.step(8.0, (6.0, 0.0, 0.0), costmap=costmap)
    assert "W3" in exec_engine.waypoint_manager.reached_ids

    # 7. W4 Activation -> W4 Arrival
    assert exec_engine.active_waypoint.id == "W4"
    exec_engine.step(9.0, (7.0, -0.25, 0.0), costmap=costmap)
    exec_engine.step(10.0, (8.0, -0.50, 0.0), costmap=costmap)
    assert "W4" in exec_engine.waypoint_manager.reached_ids

    # 8. W5 Activation -> W5 Arrival (Final)
    assert exec_engine.active_waypoint.id == "W5"
    assert exec_engine.waypoint_manager.is_final_waypoint is True
    exec_engine.step(11.0, (9.0, -0.25, 0.0), costmap=costmap)
    final_state, final_v, final_w, _ = exec_engine.step(12.0, (10.0, 0.0, 0.0), costmap=costmap)

    # 9. MISSION_COMPLETE!
    assert final_state == MissionState.MISSION_COMPLETE
    assert exec_engine.state == MissionState.MISSION_COMPLETE
    assert final_v == 0.0 and final_w == 0.0
    assert exec_engine.waypoint_manager.is_complete is True
    assert exec_engine.waypoint_manager.reached_ids == ["W1", "W2", "W3", "W4", "W5"]
    assert exec_engine.recovery_orchestrator.total_recovery_count == 1

    status = exec_engine.get_status((10.0, 0.0, 0.0))
    assert status.is_complete is True
    assert status.is_aborted is False
    assert status.state == MissionState.MISSION_COMPLETE
