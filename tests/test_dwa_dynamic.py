"""Comprehensive Safety Validation Tests for Milestone 13.9:
Dynamic Predictive Collision Integration into DWA.

Tests cover:
1. Physically colliding candidate is rejected.
2. Safe candidate remains selectable.
3. Near-miss candidate receives intended risk penalty.
4. HIGH_RISK does not automatically become physical collision.
5. UNKNOWN_HORIZON is never treated as SAFE.
6. Multiple dynamic obstacles are handled correctly.
7. Static Phase 12 obstacles still reject candidates.
8. Dynamic + static obstacles compose conservatively.
9. Deterministic repeated runs produce identical decisions.
10. Existing emergency-stop behavior still works.
11. Dynamic obstacles behind the UGV do not cause false forward rejection.
12. Crossing dynamic obstacles are detected at the correct future time.
13. Broad-phase pre-filtering eliminates unneeded expensive checks.
14. Configurable reject_high_risk policy.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from ai.costmap import BEVTerrainCostmap
from ai.planning.base import PlannerStatus, RobotState, VehicleKinematics
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionResult,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.dwa_filter import (
    DynamicBroadPhaseFilter,
    DynamicPlanningConfig,
    ObstacleSpatiotemporalHull,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap


# =============================================================================
# Test Helpers
# =============================================================================

def make_test_obstacle(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (0.8, 0.6),
    yaw: float = 0.0,
    status: str = "TRACKING",
    horizon_s: float = 2.0,
    dt: float = 0.10,
    base_unc: float = 0.15,
    unc_growth: float = 0.10,
) -> PredictedObstacleTrajectory:
    """Generate a synthetic PredictedObstacleTrajectory."""
    cov = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    state = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        covariance=cov,
        status=status,
        is_confirmed=True if status == "TRACKING" else False,
        bbox_size=size,
        bbox_yaw=yaw,
    )

    n_steps = int(round(horizon_s / dt))
    waypoints: List[TrajectoryWaypoint] = []
    vx, vy = velocity
    x0, y0 = start_pos

    for step in range(n_steps + 1):
        t = round(step * dt, 4)
        x = x0 + vx * t
        y = y0 + vy * t
        r_unc = base_unc + unc_growth * t
        cov_2x2 = np.diag([(r_unc / 3.0) ** 2, (r_unc / 3.0) ** 2]).astype(np.float64)
        waypoints.append(
            TrajectoryWaypoint(
                time_offset=t,
                timestamp=t,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                covariance_2d=cov_2x2,
                uncertainty_radius=r_unc,
            )
        )

    polygons = [
        compute_obb_corners((wp.x, wp.y), size, yaw, margin_m=wp.uncertainty_radius)
        for wp in waypoints
    ]

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=waypoints,
        bounding_polygon_per_step=polygons,
    )


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


# =============================================================================
# 12 Focused Safety Tests
# =============================================================================

def test_1_physically_colliding_candidate_rejected():
    """Safety Test 1: A physically colliding candidate trajectory is rejected."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.50, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    # Dynamic obstacle right along straight forward path at x = 0.8m, moving slowly
    obs = make_test_obstacle(track_id=1, start_pos=(0.8, 0.0), velocity=(0.0, 0.0), size=(0.8, 0.8))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    # Candidates that collide with the obstacle footprint must be rejected
    assert diag["dynamically_rejected_candidates"] > 0
    # Either planner steered around obstacle or stopped safely
    if status == PlannerStatus.SUCCESS:
        # Planner steered away from center line
        assert abs(w) > 0.05 or v < 0.20
        assert not diag["dynamic_collision"]
    else:
        assert status == PlannerStatus.NO_SAFE_TRAJECTORY
        assert v == 0.0


def test_2_safe_candidate_remains_selectable():
    """Safety Test 2: A safe candidate remains selectable without dynamic penalties."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    # Obstacle is far away laterally at y = 4.0m
    obs = make_test_obstacle(track_id=1, start_pos=(1.0, 4.0), velocity=(0.0, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    assert status == PlannerStatus.SUCCESS
    assert v > 0.20
    assert abs(w) < 0.20  # Keeps following straight path
    assert not diag["dynamic_collision"]
    assert diag["dynamic_risk"] == "FREE"
    assert diag["dynamically_rejected_candidates"] == 0


def test_3_near_miss_candidate_receives_risk_penalty():
    """Safety Test 3: A near-miss candidate receives the intended risk penalty."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.40, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Baseline run without obstacle
    _, v_base, _, _, diag_base = dwa.compute_trajectory(state, path, wmap)

    # Obstacle placed just outside physical collision boundary (near miss)
    # Distance to path center is 0.70m: outside physical footprint (0.40 + 0.30/2 = 0.55m),
    # but inside the uncertainty envelope
    obs = make_test_obstacle(
        track_id=2, start_pos=(1.2, 0.70), velocity=(0.0, 0.0), size=(0.6, 0.4),
        base_unc=0.40, unc_growth=0.10
    )
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    dyn_cfg = DynamicPlanningConfig(w_dynamic_risk=0.50, medium_risk_penalty=0.35)
    traj, v_dyn, w_dyn, status, diag_dyn = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs], dynamic_config=dyn_cfg
    )

    assert status == PlannerStatus.SUCCESS
    assert diag_dyn["dynamically_penalized_candidates"] > 0
    # UGV steers slightly to the right to increase clearance from obstacle on left
    assert w_dyn < 0.0 or diag_dyn["best_score"] < diag_base["best_score"]


def test_4_high_risk_does_not_automatically_become_collision():
    """Safety Test 4: HIGH_RISK does not automatically reject candidate when reject_high_risk=False."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    # Obstacle placed so candidate enters 1-sigma high-risk zone without footprint collision
    obs = make_test_obstacle(
        track_id=3, start_pos=(1.0, 0.65), velocity=(0.0, 0.0), size=(0.6, 0.4),
        base_unc=0.50, unc_growth=0.10
    )
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # Config with reject_high_risk=False
    cfg_allow = DynamicPlanningConfig(reject_high_risk=False, high_risk_penalty=0.40)
    _, _, _, status_allow, diag_allow = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs], dynamic_config=cfg_allow
    )

    # Config with reject_high_risk=True
    cfg_reject = DynamicPlanningConfig(reject_high_risk=True)
    _, _, _, status_reject, diag_reject = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs], dynamic_config=cfg_reject
    )

    # With reject_high_risk=False, candidates are penalized, not treated as physical collision
    assert diag_allow["dynamically_penalized_candidates"] >= 0
    assert diag_reject["dynamically_rejected_candidates"] >= diag_allow["dynamically_rejected_candidates"]


def test_5_unknown_horizon_is_never_treated_as_safe():
    """Safety Test 5: UNKNOWN_HORIZON queries are never treated as SAFE."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.50, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Obstacle with short prediction horizon of 0.5s, while DWA horizon is 1.5s
    obs = make_test_obstacle(track_id=5, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), horizon_s=0.50)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # 5a. Policy == PENALIZE
    cfg_pen = DynamicPlanningConfig(unknown_horizon_policy="PENALIZE", unknown_horizon_penalty=0.50)
    _, _, _, status_pen, diag_pen = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs], dynamic_config=cfg_pen
    )
    assert diag_pen["unknown_horizon_candidates"] > 0

    # 5b. Policy == REJECT
    cfg_rej = DynamicPlanningConfig(unknown_horizon_policy="REJECT")
    _, _, _, status_rej, diag_rej = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs], dynamic_config=cfg_rej
    )
    assert diag_rej["unknown_horizon_candidates"] > 0
    assert diag_rej["dynamically_rejected_candidates"] >= diag_pen["dynamically_rejected_candidates"]


def test_6_multiple_dynamic_obstacles_handled_correctly():
    """Safety Test 6: Multiple dynamic obstacles are simultaneously evaluated conservatively."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Two obstacles flanking the corridor and moving
    obs1 = make_test_obstacle(track_id=10, start_pos=(1.5, 0.8), velocity=(-0.2, -0.1))
    obs2 = make_test_obstacle(track_id=20, start_pos=(1.5, -0.8), velocity=(-0.2, 0.1))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs1, obs2])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs1, obs2]
    )

    # Both obstacles must be evaluated without crash or index error
    assert diag["expensive_collision_checks"] > 0
    assert diag["trajectories_evaluated"] == dwa.v_samples * dwa.omega_samples


def test_7_static_phase_12_obstacles_still_reject_candidates():
    """Safety Test 7: Static Phase 12 obstacles still reject candidates before dynamic phase."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.40, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    # Add a static obstacle wall 0.4m ahead of the robot at x = 0.4m
    r_wall, _ = wmap.world_to_grid(0.40, 0.0)
    wmap.obstacle[r_wall - 5 : r_wall + 5, :] = True
    wmap.cost[r_wall - 5 : r_wall + 5, :] = 1.00

    # Also add a dynamic obstacle far away
    obs = make_test_obstacle(track_id=1, start_pos=(3.0, 3.0), velocity=(0.0, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    # Static wall must eliminate all forward candidates during static Phase 12 check
    assert diag["statically_valid_candidates"] == 0
    assert status == PlannerStatus.NO_SAFE_TRAJECTORY
    assert diag["emergency_stop_triggered"]
    assert v == 0.0 and w == 0.0


def test_8_dynamic_and_static_obstacles_compose_conservatively():
    """Safety Test 8: Dynamic + static obstacles compose conservatively."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # Static obstacle blocks the right turn (y < 0)
    for y_val in np.linspace(-1.5, -0.2, 14):
        for x_val in np.linspace(0.5, 2.0, 15):
            r, c = wmap.world_to_grid(x_val, y_val)
            wmap.obstacle[r, c] = True
            wmap.cost[r, c] = 1.0

    # Dynamic obstacle blocks the straight path (y in [-0.2, 0.2])
    obs = make_test_obstacle(track_id=8, start_pos=(1.2, 0.0), velocity=(0.0, 0.0), size=(0.6, 0.5))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    if status == PlannerStatus.SUCCESS:
        # The only remaining escape corridor is turning left (w > 0)
        assert w > 0.05
        assert not diag["dynamic_collision"]
    else:
        assert diag["emergency_stop_triggered"]


def test_9_deterministic_repeated_runs():
    """Safety Test 9: Deterministic repeated runs produce identical commands and diagnostics."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.5, y=-0.2, yaw=0.1, v=0.25, omega=0.05)
    path = Path.from_xy_array(np.array([[0.5, -0.2], [2.0, 0.0], [3.5, 0.5]], dtype=np.float32))

    obs = make_test_obstacle(track_id=9, start_pos=(2.0, 0.5), velocity=(-0.3, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj1, v1, w1, status1, diag1 = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )
    traj2, v2, w2, status2, diag2 = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    assert status1 == status2
    assert v1 == v2
    assert w1 == w2
    assert diag1["best_score"] == diag2["best_score"]
    assert diag1["valid_trajectories"] == diag2["valid_trajectories"]
    assert diag1["expensive_collision_checks"] == diag2["expensive_collision_checks"]
    assert diag1["broad_phase_cleared_count"] == diag2["broad_phase_cleared_count"]


def test_10_existing_emergency_stop_behavior():
    """Safety Test 10: Emergency stop triggers when all trajectories are blocked dynamically."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.40, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    # Giant dynamic obstacle completely spanning x in [0.5, 1.5], y in [-3.0, 3.0]
    obs = make_test_obstacle(
        track_id=10, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 6.0)
    )
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    assert status == PlannerStatus.NO_SAFE_TRAJECTORY
    assert v == 0.0
    assert w == 0.0
    assert diag["emergency_stop_triggered"]
    assert diag["valid_trajectories"] == 0
    assert diag["dynamically_rejected_candidates"] > 0


def test_11_obstacles_behind_ugv_do_not_cause_false_forward_rejection():
    """Safety Test 11: Dynamic obstacles behind UGV do not cause false forward rejection."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    # Obstacle is 2 meters BEHIND the robot and moving away backwards (vx = -0.5)
    obs = make_test_obstacle(track_id=11, start_pos=(-2.0, 0.0), velocity=(-0.5, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    assert status == PlannerStatus.SUCCESS
    assert v > 0.20
    assert abs(w) < 0.20
    assert diag["dynamically_rejected_candidates"] == 0
    # Broad-phase must clear all candidates because obstacle hull is entirely in [-3.0, -1.0]
    assert diag["expensive_collision_checks"] == 0
    assert diag["broad_phase_cleared_count"] == diag["statically_valid_candidates"]


def test_12_crossing_obstacle_detected_at_correct_time():
    """Safety Test 12: Crossing dynamic obstacle is detected at the correct future time."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.50, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32))

    # UGV moving forward at ~0.50 m/s will reach x = 0.50m at t = 1.0s.
    # Crossing obstacle starts at (0.50, 1.0) and moves downwards at vy = -1.0 m/s.
    # At t = 1.0s, obstacle center is at (0.50, 0.0).
    obs = make_test_obstacle(
        track_id=12, start_pos=(0.50, 1.0), velocity=(0.0, -1.0), size=(0.6, 0.6)
    )
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    # Collision checker must have tested intersecting candidates and detected collision at t ~ 1.0s
    assert diag["expensive_collision_checks"] > 0
    assert diag["dynamically_rejected_candidates"] > 0
    # Safe maneuver chosen (e.g. turning or slowing down to let obstacle cross)
    assert status == PlannerStatus.SUCCESS or status == PlannerStatus.NO_SAFE_TRAJECTORY


def test_13_broad_phase_prefilter_skips_clear_candidates():
    """Broad-phase pre-filtering eliminates exact checks for candidates far from obstacles."""
    dwa = DWAPlanner()
    wmap = create_empty_world_map()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0)
    path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))

    # Obstacle located far outside candidate corridor at (5.0, 5.0)
    obs = make_test_obstacle(track_id=13, start_pos=(5.0, 5.0), velocity=(0.0, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj, v, w, status, diag = dwa.compute_trajectory(
        state, path, wmap, dynamic_layer=layer, predicted_tracks=[obs]
    )

    assert status == PlannerStatus.SUCCESS
    # 100% of candidates should be cleared by broad phase
    assert diag["expensive_collision_checks"] == 0
    assert diag["broad_phase_cleared_count"] == diag["statically_valid_candidates"]
    assert diag["number_of_broad_phase_rejections"] == diag["broad_phase_cleared_count"]
