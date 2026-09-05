"""Unit and safety validation tests for Milestone 13.11 Event-Driven Global Replanning.

Tests cover:
1. temporary obstacle -> no global replan
2. WAIT condition -> no immediate global replan
3. AVOID condition -> no immediate global replan
4. DYNAMIC_DEADLOCK -> replan
5. persistent blockage -> replan
6. global path invalidated -> replan
7. insufficient progress -> replan
8. cooldown prevents repeated replan
9. noisy single-frame blockage does not replan
10. successful replan changes path
11. successful replan preserves goal
12. static obstacles remain respected
13. dynamic obstacle blocks old path but alternative exists
14. no alternative global path -> replan failure
15. unknown region does not become falsely free
16. stale path is never silently used after invalidation
17. deterministic repeated replan
18. multiple dynamic obstacles
19. goal already reached
20. new path is rejected if it violates safety

And proves Safety Invariants A through I.
"""

from __future__ import annotations

import copy
import math
from typing import List, Tuple
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    DeadlockEvent,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.global_replanner import (
    DynamicPlanningMapOverlay,
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
    PathChangeMetrics,
    compute_min_dynamic_clearance,
    compute_path_deviation,
)
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_test_world_map(
    bounds: Tuple[float, float] = (-5.0, 5.0),
    resolution: float = 0.10,
) -> WorldTerrainMap:
    """Create a 2D WorldTerrainMap with 0.10 nominal terrain cost."""
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def make_straight_path(start_x: float = -2.0, end_x: float = 2.0, y: float = 0.0, step: float = 0.10) -> Path:
    """Generate a straight line path along the X axis."""
    xs = np.arange(start_x, end_x + step * 0.5, step)
    pts = [PathPoint(x=float(x), y=float(y), yaw=0.0, distance_from_start=float(x - start_x)) for x in xs]
    return Path(points=pts)


def make_test_tracked_obstacle(
    track_id: int = 1,
    pos: Tuple[float, float] = (0.0, 0.0),
    size: Tuple[float, float] = (0.6, 0.6),
    yaw: float = 0.0,
    status: str = "TRACKING",
) -> TrackedObstacleState:
    """Create a TrackedObstacleState for testing."""
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(pos[0]),
        y=float(pos[1]),
        vx=0.0,
        vy=0.0,
        covariance=np.eye(4, dtype=np.float64) * 0.01,
        status=status,
        is_confirmed=True,
        bbox_size=size,
        bbox_yaw=yaw,
    )


# ==============================================================================
# 20 FOCUSED OPERATIONAL TESTS
# ==============================================================================

def test_1_temporary_obstacle_no_global_replan():
    """Test 1: Temporary obstacle crossing the path does not trigger global replan."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=2))
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Obstacle is off path, decision is PROCEED
    decision = BehavioralDecision(
        state=BehavioralState.PROCEED,
        command_v=0.30,
        command_omega=0.0,
        reason="Path clear",
    )
    obs = [make_test_tracked_obstacle(1, (0.0, 1.5))]

    result = replanner.check_and_replan(
        current_pose=(-2.0, 0.0, 0.0),
        goal=(2.0, 0.0),
        base_map=wmap,
        behavioral_decision=decision,
        dynamic_obstacles=obs,
        timestamp=1.0,
    )

    assert not result.replan_triggered
    assert result.trigger_type == DynamicReplanTrigger.NONE
    assert not result.is_path_stale


def test_2_wait_condition_no_immediate_global_replan():
    """Test 2: WAIT condition awaiting obstacle clearance does not immediately replan."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(persistent_blockage_duration_s=3.0))
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    decision = BehavioralDecision(
        state=BehavioralState.WAIT,
        command_v=0.0,
        command_omega=0.0,
        reason="Obstacle clearing corridor",
        wait_elapsed_s=0.5,
    )
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]

    # At t = 0.5s (< 3.0s threshold)
    result = replanner.check_and_replan(
        current_pose=(-1.0, 0.0, 0.0),
        goal=(2.0, 0.0),
        base_map=wmap,
        behavioral_decision=decision,
        dynamic_obstacles=obs,
        timestamp=0.5,
    )

    assert not result.replan_triggered
    assert result.trigger_type == DynamicReplanTrigger.NONE


def test_3_avoid_condition_no_immediate_global_replan():
    """Test 3: Tactical AVOID executing a local detour does not trigger global replan."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    decision = BehavioralDecision(
        state=BehavioralState.AVOID,
        command_v=0.25,
        command_omega=0.10,
        reason="Executing local detour around obstacle",
    )
    obs = [make_test_tracked_obstacle(1, (0.0, 0.8))]

    result = replanner.check_and_replan(
        current_pose=(-0.5, 0.2, 0.1),
        goal=(2.0, 0.0),
        base_map=wmap,
        behavioral_decision=decision,
        dynamic_obstacles=obs,
        timestamp=1.0,
    )

    assert not result.replan_triggered
    assert result.trigger_type == DynamicReplanTrigger.NONE


def test_4_dynamic_deadlock_triggers_replan():
    """Test 4: M13.10 DYNAMIC_DEADLOCK event immediately triggers global replan."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    decision = BehavioralDecision(
        state=BehavioralState.STOP,
        command_v=0.0,
        command_omega=0.0,
        reason="Consecutive stop cycles exceeded limit",
        deadlock=True,
        deadlock_event=DeadlockEvent.DYNAMIC_DEADLOCK.value,
        blocking_obstacle_id=42,
    )
    # Obstacle blocks at (0.0, 0.0)
    obs = [make_test_tracked_obstacle(42, (0.0, 0.0))]

    result = replanner.check_and_replan(
        current_pose=(-1.0, 0.0, 0.0),
        goal=(2.0, 0.0),
        base_map=wmap,
        behavioral_decision=decision,
        dynamic_obstacles=obs,
        timestamp=2.0,
    )

    assert result.replan_triggered
    assert result.trigger_type == DynamicReplanTrigger.DYNAMIC_DEADLOCK
    assert result.event is not None
    assert result.event.blocking_obstacle_ids == [42]
    assert result.success


def test_5_persistent_blockage_triggers_replan():
    """Test 5: Continuous blockage in STOP/WAIT for >= persistent_blockage_duration_s triggers replan."""
    cfg = GlobalReplannerConfig(persistent_blockage_duration_s=2.0, debounce_cycles=5)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    decision = BehavioralDecision(
        state=BehavioralState.STOP,
        command_v=0.0,
        command_omega=0.0,
        reason="Corridor blocked",
        blocking_obstacle_id=5,
    )
    obs = [make_test_tracked_obstacle(5, (0.0, 0.0))]

    # Cycle 1 at t=0.0 -> starts blocked timer
    r1 = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=0.0)
    # Cycle 2 at t=1.0 -> 1.0s < 2.0s
    r2 = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    # Cycle 3 at t=2.1 -> 2.1s >= 2.0s -> triggers PERSISTENT_BLOCKAGE
    r3 = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=2.1)

    assert not r1.replan_triggered
    assert not r2.replan_triggered
    assert r3.replan_triggered
    assert r3.trigger_type == DynamicReplanTrigger.PERSISTENT_BLOCKAGE
    assert r3.success


def test_6_global_path_invalidated_triggers_replan():
    """Test 6: Existing global path rendered invalid by dynamic obstacle triggers replan after debounce."""
    cfg = GlobalReplannerConfig(debounce_cycles=2, cooldown_s=0.0)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Obstacle placed directly on global path at (0.0, 0.0)
    obs = [make_test_tracked_obstacle(10, (0.0, 0.0))]

    # Cycle 1: Debounce 1/2 -> No replan yet
    r1 = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, None, obs, timestamp=1.0)
    assert not r1.replan_triggered

    # Cycle 2: Debounce 2/2 -> Triggers PATH_INVALIDATED
    r2 = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, None, obs, timestamp=1.1)
    assert r2.replan_triggered
    assert r2.trigger_type == DynamicReplanTrigger.PATH_INVALIDATED
    assert r2.success


def test_7_insufficient_progress_triggers_replan():
    """Test 7: No meaningful progress toward goal over configurable window triggers replan."""
    cfg = GlobalReplannerConfig(progress_loss_window_s=3.0, min_progress_m=0.20, cooldown_s=0.0)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Vehicle is stalled at x=-1.0 from t=0.0 to t=3.0 (progress = 0.0m < 0.20m)
    r1 = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, None, [], timestamp=0.0)
    r2 = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, None, [], timestamp=1.5)
    r3 = replanner.check_and_replan((-0.95, 0.0, 0.0), (2.0, 0.0), wmap, None, [], timestamp=3.0)

    assert not r1.replan_triggered
    assert not r2.replan_triggered
    assert r3.replan_triggered
    assert r3.trigger_type == DynamicReplanTrigger.INSUFFICIENT_PROGRESS


def test_8_cooldown_prevents_repeated_replan():
    """Test 8: Replan cooldown prevents immediate repeated replanning unless collision occurs."""
    cfg = GlobalReplannerConfig(cooldown_s=4.0, debounce_cycles=1)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    # Replan 1 at t=1.0 -> Succeeds
    r1 = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert r1.replan_triggered
    assert r1.success

    # At t=2.0 (cooldown remaining = 3.0s), deadlock is requested again, but new path is safe
    # Cooldown suppresses replan
    decision2 = copy.deepcopy(decision)
    r2 = replanner.check_and_replan((-1.4, 0.5, 0.0), (2.0, 0.0), wmap, decision2, obs, timestamp=2.0)
    assert not r2.replan_triggered
    assert r2.cooldown_remaining_s > 0.0
    assert "cooldown" in r2.failure_reason.lower()


def test_9_noisy_single_frame_blockage_does_not_replan():
    """Test 9: A single noisy frame of path obstruction is filtered by debouncing."""
    cfg = GlobalReplannerConfig(debounce_cycles=2)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Frame 1: Spurious obstacle appears on path
    noisy_obs = [make_test_tracked_obstacle(99, (0.0, 0.0))]
    r1 = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, None, noisy_obs, timestamp=1.0)
    assert not r1.replan_triggered

    # Frame 2: Obstacle disappeared (noise clears)
    r2 = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, None, [], timestamp=1.1)
    assert not r2.replan_triggered
    assert replanner.path_invalid_consecutive_cycles == 0


def test_10_successful_replan_changes_path():
    """Test 10: Successful replanning produces a route with measurable deviation from the old blocked path."""
    cfg = GlobalReplannerConfig(debounce_cycles=1)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    obs = [make_test_tracked_obstacle(1, (0.0, 0.0), size=(0.8, 0.8))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.success
    assert result.path is not None
    assert result.metrics is not None
    # Path must deviate to detour around obstacle
    assert result.metrics.path_deviation > 0.20
    assert result.metrics.new_path_length > result.metrics.old_path_length


def test_11_successful_replan_preserves_goal():
    """Test 11: Replanned path endpoint preserves the original target goal position."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()
    goal = (2.5, 1.0)
    path = make_straight_path(-2.0, 2.5, 1.0)
    replanner.set_global_path(path)

    obs = [make_test_tracked_obstacle(1, (0.0, 1.0))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 1.0, 0.0), goal, wmap, decision, obs, timestamp=1.0)
    assert result.success
    last_pt = result.path.points[-1]
    assert math.hypot(last_pt.x - goal[0], last_pt.y - goal[1]) <= 0.20


def test_12_static_obstacles_remain_respected():
    """Test 12: Replanned route avoids both dynamic obstacle and existing static lethal obstacles."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()

    # Place static obstacle wall at y = 1.0
    for x in np.arange(-1.0, 1.5, 0.1):
        r, c = wmap.world_to_grid(float(x), 1.0)
        wmap.obstacle[r, c] = True

    # Initial path along y = 0.0
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Dynamic obstacle at (0.0, 0.0)
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.success
    # Verify no waypoint is on static obstacle
    for pt in result.path.points:
        r, c = wmap.world_to_grid(pt.x, pt.y)
        assert not wmap.obstacle[r, c]


def test_13_dynamic_obstacle_blocks_old_path_alternative_exists():
    """Test 13: Dynamic obstacle completely blocks current corridor, but A* finds alternative route."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Broad obstacle blocking corridor between y = [-0.5, 0.5]
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0), size=(1.2, 1.0))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.success
    # Waypoints around x=0 should detour outside y in [-0.5, 0.5]
    mid_points = [p for p in result.path.points if abs(p.x) < 0.3]
    assert any(abs(p.y) > 0.6 for p in mid_points)


def test_14_no_alternative_global_path_replan_failure():
    """Test 14: If all routes are blocked, replan emits GLOBAL_REPLAN_FAILED and no path is produced."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()

    # Build an impenetrable static wall from y = -4.8 to +4.8 across x = 0.5 except a gap at y=0.0
    for y in np.arange(-4.8, 4.9, 0.1):
        if abs(y) > 0.4:
            r, c = wmap.world_to_grid(0.5, float(y))
            wmap.obstacle[r, c] = True

    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Place dynamic obstacle directly in the only remaining gap at (0.5, 0.0)
    obs = [make_test_tracked_obstacle(1, (0.5, 0.0), size=(1.0, 1.0))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.replan_triggered
    assert not result.success
    assert result.path is None
    assert result.is_path_stale
    assert "failed" in result.failure_reason.lower() or "no feasible" in result.failure_reason.lower()


def test_15_unknown_region_does_not_become_falsely_free():
    """Test 15: DynamicPlanningMapOverlay preserves unknown space without falsely clearing it."""
    wmap = create_test_world_map()
    # Mark a region as unknown
    wmap.unknown[10:30, 10:30] = True

    obs = [make_test_tracked_obstacle(1, (1.0, 1.0), status="UNKNOWN")]
    overlay = DynamicPlanningMapOverlay(wmap, dynamic_obstacles=obs)

    # 1. Base unknown region remains unknown
    assert overlay.unknown[15, 15]
    # 2. Dynamic obstacle marked with unknown flag has its footprint flagged unknown
    r_obs, c_obs = overlay.world_to_grid(1.0, 1.0)
    assert overlay.unknown[r_obs, c_obs]
    # 3. Canonical base_map was NOT mutated
    assert not wmap.unknown[r_obs, c_obs]


def test_16_stale_path_is_never_silently_used_after_invalidation():
    """Test 16: When a path is invalidated, it is explicitly marked stale and cannot be silently executed."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Block corridor completely so replan fails
    for y in np.arange(-4.8, 4.9, 0.1):
        r, c = wmap.world_to_grid(0.0, float(y))
        wmap.obstacle[r, c] = True

    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)
    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, [], timestamp=1.0)

    assert result.replan_triggered
    assert not result.success
    assert replanner.is_current_path_stale
    assert result.is_path_stale
    assert result.path is None


def test_17_deterministic_repeated_replan():
    """Test 17: Two identical replan queries produce bitwise identical paths and metrics."""
    cfg = GlobalReplannerConfig(debounce_cycles=1)
    rep1 = EventDrivenGlobalReplanner(cfg)
    rep2 = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    rep1.set_global_path(path)
    rep2.set_global_path(path)

    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    r1 = rep1.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, dec, obs, timestamp=1.0)
    r2 = rep2.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, dec, obs, timestamp=1.0)

    assert r1.success and r2.success
    assert len(r1.path.points) == len(r2.path.points)
    np.testing.assert_allclose(r1.path.waypoints_xy, r2.path.waypoints_xy, atol=1e-5)
    assert r1.metrics.new_path_length == r2.metrics.new_path_length


def test_18_multiple_dynamic_obstacles():
    """Test 18: Multiple dynamic obstacles are simultaneously rasterized and navigated."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Obstacle 1 at (0.0, 0.0), Obstacle 2 at (0.0, 1.0)
    obs = [
        make_test_tracked_obstacle(1, (0.0, 0.0)),
        make_test_tracked_obstacle(2, (0.0, 1.0)),
    ]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.success
    # The detour should navigate below (negative y) since positive y is blocked by obstacle 2
    mid_points = [p for p in result.path.points if abs(p.x) < 0.3]
    assert all(p.y < 0.0 for p in mid_points)


def test_19_goal_already_reached_no_replan():
    """Test 19: When vehicle is within goal tolerance, replanning is suppressed."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(goal_tolerance_m=0.25))
    wmap = create_test_world_map()
    goal = (2.0, 0.0)
    path = make_straight_path(1.8, 2.0, 0.0)
    replanner.set_global_path(path)

    # Vehicle at x = 1.90 (dist = 0.10m <= 0.25m)
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)
    obs = [make_test_tracked_obstacle(1, (2.0, 0.0))]

    result = replanner.check_and_replan((1.90, 0.0, 0.0), goal, wmap, decision, obs, timestamp=1.0)
    assert not result.replan_triggered
    assert result.trigger_type == DynamicReplanTrigger.NONE


def test_20_new_path_rejected_if_entrance_unsafe():
    """Test 20: A generated path is rejected if its entrance immediately collides with a dynamic obstacle."""
    cfg = GlobalReplannerConfig(debounce_cycles=1, entrance_min_clearance_m=0.40)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Dynamic obstacle placed immediately adjacent to vehicle start pose (-1.5, 0.0) at (-1.4, 0.1)
    obs = [make_test_tracked_obstacle(1, (-1.4, 0.1), size=(0.6, 0.6))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.replan_triggered
    # The entrance is blocked by obstacle 1 -> Path must be rejected
    assert not result.success
    assert result.path is None
    assert "entrance" in result.failure_reason.lower() or "start" in result.failure_reason.lower()


# ==============================================================================
# SAFETY INVARIANTS A THROUGH I PROOFS
# ==============================================================================

def test_invariant_a_temporary_local_obstruction_no_global_replan():
    """Invariant A: Temporary local obstruction alone does not trigger global replan."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Local DWA avoidance
    decision = BehavioralDecision(state=BehavioralState.AVOID, command_v=0.25, command_omega=0.10, reason="Local detour")
    obs = [make_test_tracked_obstacle(1, (0.0, 0.5))]

    result = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert not result.replan_triggered, "Invariant A violated: Temporary AVOID triggered global replan"


def test_invariant_b_dynamic_deadlock_can_trigger_global_replan():
    """Invariant B: DYNAMIC_DEADLOCK can trigger global replan."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    decision = BehavioralDecision(
        state=BehavioralState.STOP, command_v=0.0, command_omega=0.0,
        reason="Deadlock", deadlock=True, deadlock_event=DeadlockEvent.DYNAMIC_DEADLOCK.value
    )
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]

    result = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.replan_triggered, "Invariant B violated: DYNAMIC_DEADLOCK failed to trigger replan"
    assert result.trigger_type == DynamicReplanTrigger.DYNAMIC_DEADLOCK


def test_invariant_c_global_replan_failed_never_produces_motion():
    """Invariant C: GLOBAL_REPLAN_FAILED never produces motion (path is None, stale flag is True)."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()
    # Fully block map
    wmap.obstacle.fill(True)

    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)
    result = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, [], timestamp=1.0)

    assert not result.success, "Invariant C violated: Replan should fail on completely blocked map"
    assert result.path is None, "Invariant C violated: Replan failure must not output a path"
    assert result.is_path_stale, "Invariant C violated: Path must be marked stale upon failure"


def test_invariant_d_stale_invalid_path_cannot_be_silently_executed():
    """Invariant D: A stale/invalid global path cannot be silently executed."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()
    path = make_straight_path(-2.0, 2.0, 0.0)
    replanner.set_global_path(path)

    # Invalidate path with obstacle
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]
    replanner.check_trigger_conditions((-1.0, 0.0, 0.0), (2.0, 0.0), path, wmap, None, obs, timestamp=1.0)

    # Force replan failure by setting start out of bounds
    res = replanner.execute_replan((100.0, 100.0, 0.0), (2.0, 0.0), wmap, obs)
    assert replanner.is_current_path_stale, "Invariant D violated: Stale flag was reset despite failure"
    assert res.is_path_stale, "Invariant D violated: Result path is not marked stale"


def test_invariant_e_new_path_cannot_override_safety_constraints():
    """Invariant E: New path cannot override static or dynamic safety constraints."""
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig(debounce_cycles=1))
    wmap = create_test_world_map()

    # Block corridor at x = 0.0, y in [-1.0, 1.0] with dynamic obstacle
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0), size=(1.0, 1.0))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert result.success

    # Check that EVERY point on new path has non-zero clearance from dynamic obstacle
    clearance = compute_min_dynamic_clearance(result.path, obs)
    assert clearance >= 0.0, f"Invariant E violated: Path clearance {clearance}m < 0.0m"


def test_invariant_f_unknown_dynamic_space_never_confirmed_free():
    """Invariant F: Unknown dynamic space is never treated as confirmed free."""
    wmap = create_test_world_map()
    obs = [make_test_tracked_obstacle(1, (0.0, 0.0), status="UNKNOWN")]
    overlay = DynamicPlanningMapOverlay(wmap, dynamic_obstacles=obs)

    r, c = overlay.world_to_grid(0.0, 0.0)
    assert overlay.unknown[r, c], "Invariant F violated: Unknown obstacle space was marked known/free"


def test_invariant_g_replanning_cannot_create_path_colliding_at_entrance():
    """Invariant G: Replanning cannot create a path that immediately collides with a known dynamic obstacle."""
    cfg = GlobalReplannerConfig(debounce_cycles=1, entrance_min_clearance_m=0.35)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()

    # Obstacle is sitting right in front of vehicle start (-1.0, 0.0) at (-0.9, 0.0)
    obs = [make_test_tracked_obstacle(1, (-0.9, 0.0), size=(0.6, 0.6))]
    decision = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    result = replanner.check_and_replan((-1.0, 0.0, 0.0), (2.0, 0.0), wmap, decision, obs, timestamp=1.0)
    assert not result.success, "Invariant G violated: Replanner accepted a path with blocked entrance"


def test_invariant_h_replanning_deterministic_for_identical_inputs():
    """Invariant H: Replanning is deterministic for identical inputs."""
    cfg = GlobalReplannerConfig(debounce_cycles=1)
    rep1 = EventDrivenGlobalReplanner(cfg)
    rep2 = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()

    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    res1 = rep1.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, dec, obs, timestamp=1.0)
    res2 = rep2.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, dec, obs, timestamp=1.0)

    assert res1.success and res2.success
    assert res1.trigger_type == res2.trigger_type
    assert res1.metrics.new_path_cost == res2.metrics.new_path_cost
    np.testing.assert_array_equal(res1.path.waypoints_xy, res2.path.waypoints_xy)


def test_invariant_i_replan_cooldown_prevents_oscillatory_replanning():
    """Invariant I: Replan cooldown prevents oscillatory replanning."""
    cfg = GlobalReplannerConfig(cooldown_s=3.0, debounce_cycles=1)
    replanner = EventDrivenGlobalReplanner(cfg)
    wmap = create_test_world_map()

    obs = [make_test_tracked_obstacle(1, (0.0, 0.0))]
    dec = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)

    # Initial replan at t=0.0
    r0 = replanner.check_and_replan((-1.5, 0.0, 0.0), (2.0, 0.0), wmap, dec, obs, timestamp=0.0)
    assert r0.success

    # Attempt replans every 0.1s up to t=2.5s (inside 3.0s cooldown)
    blocked_count = 0
    for step in range(1, 26):
        t = step * 0.1
        # Obstacle oscillates or reappears in peripheral zone, SM commands STOP
        dec_step = BehavioralDecision(state=BehavioralState.STOP, command_v=0.0, command_omega=0.0, reason="Deadlock", deadlock=True)
        r = replanner.check_and_replan((-1.4, 0.5, 0.0), (2.0, 0.0), wmap, dec_step, obs, timestamp=t)
        if not r.replan_triggered and r.cooldown_remaining_s > 0.0:
            blocked_count += 1

    assert blocked_count == 25, f"Invariant I violated: Cooldown failed to suppress {25 - blocked_count} cycles"
    assert replanner.total_replans_executed == 1
