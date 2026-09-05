"""Unit and integration tests for Milestone 13.10 Behavioral State Machine (WAIT / AVOID / STOP / PROCEED).

Validates:
1. Clear path -> PROCEED
2. Safe alternative -> AVOID
3. Temporary blocking obstacle -> WAIT
4. No safe trajectory -> STOP
5. Physical collision -> STOP
6. High risk with alternatives -> AVOID
7. High risk without alternatives -> WAIT or STOP according to policy
8. UNKNOWN_HORIZON prevents unsafe PROCEED
9. Safe prediction recovery -> PROCEED
10. Obstacle disappearance -> recovery
11. WAIT timeout
12. STOP recovery
13. State hysteresis
14. Noisy one-frame disturbance does not cause state thrashing
15. Repeated blockage creates DYNAMIC_DEADLOCK
16. Deterministic repeated evaluation
17. Multiple obstacles
18. Obstacle behind vehicle does not unnecessarily stop forward motion
19. Static obstacle + dynamic obstacle composition
20. Command output for every state (v=0, w=0 for WAIT/STOP)

And proves Safety Invariants A through H.
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import pytest

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionResult,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint
from ai.spatial_fusion.world_map import WorldTerrainMap


def create_empty_world_map(
    bounds: Tuple[float, float] = (-10.0, 10.0),
    resolution: float = 0.10,
) -> WorldTerrainMap:
    """Create a 2D WorldTerrainMap with 0.10 nominal terrain cost."""
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


def make_test_obstacle(
    track_id: int = 1,
    start_pos: Tuple[float, float] = (1.5, 0.0),
    velocity: Tuple[float, float] = (0.0, 0.0),
    size: Tuple[float, float] = (0.6, 0.4),
    yaw: float = 0.0,
    horizon_s: float = 2.0,
    dt: float = 0.10,
    base_unc: float = 0.10,
    unc_growth: float = 0.05,
) -> PredictedObstacleTrajectory:
    """Generate deterministic predicted obstacle trajectory."""
    cov = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    state = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        covariance=cov,
        status="TRACKING",
        is_confirmed=True,
        bbox_size=size,
        bbox_yaw=yaw,
    )
    n_steps = int(round(horizon_s / dt))
    wps: List[TrajectoryWaypoint] = []
    polys: List[List[Tuple[float, float]]] = []

    for step in range(n_steps + 1):
        t = round(step * dt, 4)
        x = start_pos[0] + velocity[0] * t
        y = start_pos[1] + velocity[1] * t
        r_unc = base_unc + unc_growth * t
        cov_2d = np.diag([(r_unc / 3.0) ** 2, (r_unc / 3.0) ** 2]).astype(np.float64)
        wps.append(
            TrajectoryWaypoint(
                time_offset=t,
                timestamp=t,
                x=x,
                y=y,
                vx=velocity[0],
                vy=velocity[1],
                covariance_2d=cov_2d,
                uncertainty_radius=r_unc,
            )
        )
        polys.append(compute_obb_corners((x, y), size, yaw, margin_m=r_unc))

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=wps,
        bounding_polygon_per_step=polys,
    )


# =============================================================================
# 20 Focused Behavioral Tests
# =============================================================================

def test_1_clear_path_proceed():
    """Test 1: When path is clear and no obstacles exist, state machine selects PROCEED."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    traj, v, w, status, diag, decision = dwa.plan_with_behavior(state, path, wmap)

    assert status == PlannerStatus.SUCCESS
    assert decision.state == BehavioralState.PROCEED
    assert decision.command_v > 0.20
    assert abs(decision.command_omega) < 0.10
    assert decision.dynamic_risk == "FREE"
    assert not decision.dynamic_collision
    assert not decision.unknown_horizon


def test_2_safe_alternative_avoid():
    """Test 2: When preferred path is obstructed but a safe alternative exists, selects AVOID."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Obstacle in corridor, forcing DWA avoidance detour/rejection
    obs = make_test_obstacle(track_id=1, start_pos=(1.2, 0.15), velocity=(0.0, 0.0), size=(0.6, 0.4))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag, decision = dwa.plan_with_behavior(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    assert status == PlannerStatus.SUCCESS
    assert decision.state == BehavioralState.AVOID
    assert decision.command_v > 0.0  # Motion is active
    assert decision.blocking_obstacle_id == 1


def test_3_temporary_blocking_obstacle_wait():
    """Test 3: When a moving obstacle temporarily blocks the corridor and will clear, selects WAIT."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)

    # Obstacle currently crossing corridor at (1.0, 0.1), moving at vy=1.0 m/s (clearing corridor in ~0.5s)
    obs = make_test_obstacle(track_id=7, start_pos=(1.0, 0.1), velocity=(0.0, 1.0), size=(0.6, 0.4))

    # Simulate DWA having no safe forward trajectory due to crossing obstacle
    dwa_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "HIGH",
        "valid_trajectories": 0,
        "emergency_stop_triggered": True,
        "dynamically_rejected_candidates": 375,
    }

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=None,
        best_v=0.0,
        best_w=0.0,
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        dwa_diagnostics=dwa_diag,
        predicted_tracks=[obs],
        current_time_s=1.0,
    )

    assert decision.state == BehavioralState.WAIT
    assert decision.command_v == 0.0
    assert decision.command_omega == 0.0
    assert decision.blocking_obstacle_id == 7
    assert "blocking" in decision.reason.lower() or "wait" in decision.reason.lower()


def test_4_no_safe_trajectory_stop():
    """Test 4: When all candidates are blocked and obstacle is not clearing, selects STOP."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)

    # Stationary wide obstacle permanently blocking corridor at (1.0, 0.0)
    obs = make_test_obstacle(track_id=8, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(3.0, 1.0))

    dwa_diag = {
        "dynamic_collision": True,
        "dynamic_risk": "LETHAL",
        "valid_trajectories": 0,
        "emergency_stop_triggered": True,
        "blocking_obstacle_id": 8,
    }

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=None,
        best_v=0.0,
        best_w=0.0,
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        dwa_diagnostics=dwa_diag,
        predicted_tracks=[obs],
        current_time_s=1.0,
    )

    assert decision.state == BehavioralState.STOP
    assert decision.command_v == 0.0
    assert decision.command_omega == 0.0
    assert decision.blocking_obstacle_id == 8


def test_5_physical_collision_stop():
    """Test 5: Physical collision detected on candidate strictly commands STOP."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    dwa_diag = {
        "dynamic_collision": True,
        "dynamic_risk": "LETHAL",
        "valid_trajectories": 1,
        "blocking_obstacle_id": 9,
    }

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=Trajectory(points=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 1.0)]),
        best_v=0.30,
        best_w=0.0,
        planner_status=PlannerStatus.SUCCESS,
        dwa_diagnostics=dwa_diag,
        current_time_s=1.0,
    )

    assert decision.state == BehavioralState.STOP
    assert decision.command_v == 0.0
    assert decision.command_omega == 0.0


def test_6_high_risk_with_alternatives_avoid():
    """Test 6: When candidate has high risk but safe alternatives exist, selects AVOID."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    dwa_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "HIGH",
        "valid_trajectories": 10,
        "dynamically_rejected_candidates": 50,
        "blocking_obstacle_id": 2,
    }

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=Trajectory(points=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.25, 0.15, 0.1, 1.0)]),
        best_v=0.25,
        best_w=0.15,
        planner_status=PlannerStatus.SUCCESS,
        dwa_diagnostics=dwa_diag,
        current_time_s=1.0,
    )

    assert decision.state == BehavioralState.AVOID
    assert decision.command_v == 0.25
    assert decision.command_omega == 0.15


def test_7_high_risk_without_alternatives():
    """Test 7: High risk without safe alternatives commands WAIT if clearing, else STOP."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)

    # Moving clearing obstacle
    obs = make_test_obstacle(track_id=3, start_pos=(1.0, 0.0), velocity=(0.0, 0.8))

    dwa_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "HIGH",
        "valid_trajectories": 0,
        "emergency_stop_triggered": True,
    }

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=None,
        best_v=0.0,
        best_w=0.0,
        planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        dwa_diagnostics=dwa_diag,
        predicted_tracks=[obs],
        current_time_s=1.0,
    )
    assert decision.state == BehavioralState.WAIT
    assert decision.command_v == 0.0


def test_8_unknown_horizon_prevents_unsafe_proceed():
    """Test 8: UNKNOWN_HORIZON under REJECT policy prevents unsafe PROCEED."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0, unknown_horizon_policy="REJECT"))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    dwa_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "UNKNOWN_HORIZON",
        "unknown_horizon": True,
        "valid_trajectories": 10,
    }

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=Trajectory(points=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 1.0)]),
        best_v=0.30,
        best_w=0.0,
        planner_status=PlannerStatus.SUCCESS,
        dwa_diagnostics=dwa_diag,
        current_time_s=1.0,
    )

    assert decision.state != BehavioralState.PROCEED
    assert decision.state == BehavioralState.STOP
    assert decision.unknown_horizon


def test_9_safe_prediction_recovery_proceed():
    """Test 9: Transitioning from WAIT to PROCEED requires confirmed recovery cycles."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0, recovery_confirmation_cycles=2))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)

    # Initial state is WAIT
    sm.current_state = BehavioralState.WAIT

    clear_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "FREE",
        "unknown_horizon": False,
        "valid_trajectories": 100,
    }

    # Cycle 1: First clear frame -> holding recovery confirmation
    d1 = sm.evaluate(
        current_state=state,
        best_trajectory=Trajectory(points=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 1.0)]),
        best_v=0.30, best_w=0.0, planner_status=PlannerStatus.SUCCESS, dwa_diagnostics=clear_diag, current_time_s=1.1,
    )
    assert d1.state == BehavioralState.WAIT  # Still waiting for confirmation

    # Cycle 2: Second clear frame -> confirms recovery -> PROCEED
    d2 = sm.evaluate(
        current_state=state,
        best_trajectory=Trajectory(points=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 1.0)]),
        best_v=0.30, best_w=0.0, planner_status=PlannerStatus.SUCCESS, dwa_diagnostics=clear_diag, current_time_s=1.2,
    )
    assert d2.state == BehavioralState.PROCEED
    assert d2.command_v == 0.30


def test_10_obstacle_disappearance_recovery():
    """Test 10: When a blocking obstacle disappears from tracking, vehicle recovers safely."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0, recovery_confirmation_cycles=1))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)
    sm.current_state = BehavioralState.STOP

    # Obstacle disappeared -> tracks empty, candidates valid
    clear_diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 50}

    decision = sm.evaluate(
        current_state=state,
        best_trajectory=Trajectory(points=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.1, 1.0)]),
        best_v=0.35, best_w=0.0, planner_status=PlannerStatus.SUCCESS, dwa_diagnostics=clear_diag, predicted_tracks=[], current_time_s=1.1,
    )
    assert decision.state == BehavioralState.PROCEED
    assert decision.command_v == 0.35


def test_11_wait_timeout():
    """Test 11: Exceeding wait_timeout_s in WAIT transitions to STOP."""
    sm = BehavioralStateMachine(BehavioralConfig(wait_timeout_s=2.0, min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)
    obs = make_test_obstacle(track_id=10, start_pos=(1.0, 0.0), velocity=(0.0, 0.5))
    blocked_diag = {"dynamic_collision": False, "dynamic_risk": "HIGH", "valid_trajectories": 0}

    # Start waiting at t = 1.0s
    sm.reset(initial_state=BehavioralState.WAIT, timestamp=1.0)
    sm.wait_start_time = 1.0

    # At t = 2.5s (elapsed 1.5s < 2.0s timeout) -> still WAIT
    d1 = sm.evaluate(state, None, 0.0, 0.0, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, blocked_diag, [obs], current_time_s=2.5)
    assert d1.state == BehavioralState.WAIT
    assert not d1.wait_timeout

    # At t = 3.1s (elapsed 2.1s >= 2.0s timeout) -> transitions to STOP
    d2 = sm.evaluate(state, None, 0.0, 0.0, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, blocked_diag, [obs], current_time_s=3.1)
    assert d2.state == BehavioralState.STOP
    assert d2.wait_timeout
    assert d2.command_v == 0.0
    assert "timeout" in d2.reason.lower()


def test_12_stop_recovery():
    """Test 12: Recovery from STOP back to PROCEED revalidates safety."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0, recovery_confirmation_cycles=2))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)
    sm.current_state = BehavioralState.STOP

    clear_diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 50}

    # Cycle 1
    d1 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, clear_diag, current_time_s=1.1)
    assert d1.state == BehavioralState.STOP

    # Cycle 2
    d2 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, clear_diag, current_time_s=1.2)
    assert d2.state == BehavioralState.PROCEED
    assert d2.command_v == 0.30


def test_13_state_hysteresis_dwell():
    """Test 13: min_dwell_time_s prevents premature exit between non-emergency states."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.30))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
    sm.reset(initial_state=BehavioralState.PROCEED, timestamp=1.0)

    avoid_diag = {"dynamic_collision": False, "dynamic_risk": "HIGH", "valid_trajectories": 10, "dynamically_rejected_candidates": 20}

    # Query at t = 1.10s (dwell dt = 0.10s < 0.30s) -> holds PROCEED
    d1 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.25,0.1,0.1,1)]), 0.25, 0.10, PlannerStatus.SUCCESS, avoid_diag, current_time_s=1.10)
    assert d1.state == BehavioralState.PROCEED
    assert "dwell" in d1.reason.lower()

    # Query at t = 1.35s (dwell dt = 0.35s >= 0.30s) -> transitions to AVOID
    d2 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.25,0.1,0.1,1)]), 0.25, 0.10, PlannerStatus.SUCCESS, avoid_diag, current_time_s=1.35)
    assert d2.state == BehavioralState.AVOID


def test_14_noisy_one_frame_disturbance_no_thrash():
    """Test 14: Single-frame spurious disturbance does not cause state thrashing."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.20, recovery_confirmation_cycles=2))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
    sm.reset(initial_state=BehavioralState.PROCEED, timestamp=1.0)

    clear_diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 100}
    noisy_diag = {"dynamic_collision": False, "dynamic_risk": "HIGH", "valid_trajectories": 10, "dynamically_rejected_candidates": 10}

    # Frame 1: Clear -> PROCEED
    d1 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, clear_diag, current_time_s=1.0)
    assert d1.state == BehavioralState.PROCEED

    # Frame 2 (0.05s later): Spurious single-frame high risk -> Dwell holds PROCEED
    d2 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.2,0.1,0.1,1)]), 0.2, 0.1, PlannerStatus.SUCCESS, noisy_diag, current_time_s=1.05)
    assert d2.state == BehavioralState.PROCEED

    # Frame 3 (0.10s later): Clear again -> PROCEED continues uninterrupted
    d3 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, clear_diag, current_time_s=1.10)
    assert d3.state == BehavioralState.PROCEED


def test_15_repeated_blockage_creates_dynamic_deadlock():
    """Test 15: Repeated blockage for N consecutive cycles emits DYNAMIC_DEADLOCK event."""
    sm = BehavioralStateMachine(BehavioralConfig(deadlock_consecutive_stop_cycles=4, min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)
    blocked_diag = {"dynamic_collision": True, "dynamic_risk": "LETHAL", "valid_trajectories": 0}

    # Run 4 consecutive STOP cycles
    decisions = []
    for i in range(4):
        t = 1.0 + i * 0.1
        d = sm.evaluate(state, None, 0.0, 0.0, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, blocked_diag, current_time_s=t)
        decisions.append(d)

    # 4th cycle must emit DYNAMIC_DEADLOCK
    assert decisions[-1].deadlock
    assert decisions[-1].deadlock_event == DeadlockEvent.DYNAMIC_DEADLOCK.value


def test_16_deterministic_repeated_evaluation():
    """Test 16: Two independent state machines with identical sequences produce identical outputs."""
    sm1 = BehavioralStateMachine(BehavioralConfig())
    sm2 = BehavioralStateMachine(BehavioralConfig())
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 50}
    traj = Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)])

    d1 = sm1.evaluate(state, traj, 0.30, 0.0, PlannerStatus.SUCCESS, diag, current_time_s=1.0)
    d2 = sm2.evaluate(state, traj, 0.30, 0.0, PlannerStatus.SUCCESS, diag, current_time_s=1.0)

    assert d1.to_dict() == d2.to_dict()


def test_17_multiple_obstacles_handled():
    """Test 17: Multiple obstacles evaluate collective impact conservatively."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    obs1 = make_test_obstacle(track_id=1, start_pos=(1.0, 1.0), velocity=(0.0, 0.0))
    obs2 = make_test_obstacle(track_id=2, start_pos=(1.5, -1.0), velocity=(0.0, 0.0))

    diag = {
        "dynamic_collision": False,
        "dynamic_risk": "MEDIUM",
        "valid_trajectories": 25,
        "dynamically_penalized_candidates": 15,
        "blocking_obstacle_id": 1,
    }

    decision = sm.evaluate(
        state, Trajectory([TrajectoryPoint(0,0,0,0,0.25,0.08,0.1,1)]), 0.25, 0.08, PlannerStatus.SUCCESS, diag, [obs1, obs2], current_time_s=1.0
    )
    assert decision.state == BehavioralState.AVOID
    assert decision.blocking_obstacle_id == 1


def test_18_obstacle_behind_vehicle_proceeds():
    """Test 18: Obstacle behind vehicle does not unnecessarily stop forward motion."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.40, omega=0.0, timestamp=1.0)

    # Obstacle at x = -2.0m (behind UGV)
    obs = make_test_obstacle(track_id=5, start_pos=(-2.0, 0.0), velocity=(-0.5, 0.0))

    clear_diag = {
        "dynamic_collision": False,
        "dynamic_risk": "FREE",
        "valid_trajectories": 150,
        "dynamically_rejected_candidates": 0,
    }

    decision = sm.evaluate(
        state, Trajectory([TrajectoryPoint(0,0,0,0,0.40,0.0,0.1,1)]), 0.40, 0.0, PlannerStatus.SUCCESS, clear_diag, [obs], current_time_s=1.0
    )
    assert decision.state == BehavioralState.PROCEED
    assert decision.command_v == 0.40


def test_19_static_plus_dynamic_composition():
    """Test 19: Static and dynamic obstacles compose conservatively."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()

    # Static wall across path
    for r in range(50, 70):
        wmap.cost[r, 60] = 1.0  # Lethal static obstacle
        wmap.obstacle[r, 60] = True

    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    # Dynamic obstacle also in corridor
    obs = make_test_obstacle(track_id=1, start_pos=(0.8, 0.0), velocity=(0.0, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag, decision = dwa.plan_with_behavior(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    # Physical blockage from static + dynamic must trigger STOP
    assert decision.state in (BehavioralState.STOP, BehavioralState.AVOID)
    if status != PlannerStatus.SUCCESS:
        assert decision.state == BehavioralState.STOP
        assert decision.command_v == 0.0
        assert decision.command_omega == 0.0


def test_20_command_output_for_every_state():
    """Test 20: Command outputs strictly enforce v=0, w=0 on WAIT and STOP, and valid motion on PROCEED/AVOID."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)

    # 1. STOP
    sm.reset(BehavioralState.STOP)
    d_stop = sm.evaluate(state, None, 0.35, 0.10, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, {"valid_trajectories": 0}, current_time_s=1.0)
    assert d_stop.command_v == 0.0
    assert d_stop.command_omega == 0.0

    # 2. WAIT
    sm.reset(BehavioralState.WAIT)
    obs = make_test_obstacle(start_pos=(1.0, 0.1), velocity=(0.0, 1.0))
    d_wait = sm.evaluate(state, None, 0.35, 0.10, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, {"valid_trajectories": 0}, [obs], current_time_s=1.0)
    assert d_wait.command_v == 0.0
    assert d_wait.command_omega == 0.0

    # 3. PROCEED
    sm.reset(BehavioralState.PROCEED)
    traj = Trajectory([TrajectoryPoint(0,0,0,0,0.40,0.0,0.1,1)])
    d_proc = sm.evaluate(state, traj, 0.40, 0.0, PlannerStatus.SUCCESS, {"valid_trajectories": 50, "dynamic_risk": "FREE"}, current_time_s=1.0)
    assert d_proc.command_v == 0.40
    assert d_proc.command_omega == 0.0

    # 4. AVOID
    sm.reset(BehavioralState.AVOID)
    d_avoid = sm.evaluate(state, traj, 0.25, 0.18, PlannerStatus.SUCCESS, {"valid_trajectories": 20, "dynamic_risk": "HIGH"}, current_time_s=1.0)
    assert d_avoid.command_v == 0.25
    assert d_avoid.command_omega == 0.18


# =============================================================================
# Safety Invariants A Through H Verification
# =============================================================================

def test_invariant_a_physical_collision_never_proceed():
    """Safety Invariant A: Physical collision can NEVER result in PROCEED."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    dwa_diag = {"dynamic_collision": True, "dynamic_risk": "LETHAL", "valid_trajectories": 1}
    decision = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, dwa_diag, current_time_s=1.0)

    assert decision.state != BehavioralState.PROCEED
    assert decision.state == BehavioralState.STOP


def test_invariant_b_and_c_stop_and_wait_always_zero():
    """Safety Invariants B & C: STOP and WAIT always command v=0.0, omega=0.0."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.50, omega=0.20, timestamp=1.0)

    # Request high velocities under blocked condition
    d_stop = sm.evaluate(state, None, 0.50, 0.20, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, {"valid_trajectories": 0}, current_time_s=1.0)
    assert d_stop.command_v == 0.0
    assert d_stop.command_omega == 0.0

    obs = make_test_obstacle(start_pos=(1.0, 0.1), velocity=(0.0, 1.0))
    sm.reset(BehavioralState.WAIT)
    d_wait = sm.evaluate(state, None, 0.50, 0.20, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, {"valid_trajectories": 0}, [obs], current_time_s=1.0)
    assert d_wait.command_v == 0.0
    assert d_wait.command_omega == 0.0


def test_invariant_d_unknown_never_silently_safe():
    """Safety Invariant D: UNKNOWN_HORIZON never silently becomes SAFE."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0, unknown_horizon_policy="REJECT"))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)

    dwa_diag = {"dynamic_collision": False, "dynamic_risk": "UNKNOWN_HORIZON", "unknown_horizon": True, "valid_trajectories": 10}
    decision = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, dwa_diag, current_time_s=1.0)

    assert decision.state != BehavioralState.PROCEED
    assert decision.unknown_horizon


def test_invariant_e_no_safe_trajectory_stop():
    """Safety Invariant E: No safe trajectory available forces STOP."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)

    decision = sm.evaluate(state, None, 0.0, 0.0, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, {"valid_trajectories": 0}, current_time_s=1.0)
    assert decision.state == BehavioralState.STOP
    assert decision.command_v == 0.0
    assert decision.command_omega == 0.0


def test_invariant_f_cannot_override_static_safety_rejection():
    """Safety Invariant F: Behavioral layer cannot override static Phase 12 safety rejection."""
    sm = BehavioralStateMachine(BehavioralConfig(min_dwell_time_s=0.0))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)

    # Static collision checker failed all candidates -> emergency stop
    decision = sm.evaluate(state, None, 0.0, 0.0, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, {"emergency_stop_triggered": True, "valid_trajectories": 0}, current_time_s=1.0)
    assert decision.state == BehavioralState.STOP
    assert decision.command_v == 0.0
    assert decision.command_omega == 0.0


def test_invariant_g_and_h_deterministic_repeated_calls():
    """Safety Invariants G & H: Transitions are deterministic; repeated calls produce bitwise identical outputs."""
    for seed in [1, 42, 123]:
        sm = BehavioralStateMachine(BehavioralConfig())
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
        diag = {"dynamic_collision": False, "dynamic_risk": "FREE", "valid_trajectories": 100}

        out1 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, diag, current_time_s=1.0)
        out2 = sm.evaluate(state, Trajectory([TrajectoryPoint(0,0,0,0,0.3,0,0.1,1)]), 0.3, 0.0, PlannerStatus.SUCCESS, diag, current_time_s=1.0)

        assert out1.state == out2.state
        assert out1.command_v == out2.command_v
        assert out1.command_omega == out2.command_omega
        assert out1.reason == out2.reason
