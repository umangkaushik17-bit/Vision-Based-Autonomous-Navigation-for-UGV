"""Comprehensive unit and safety test suite for Milestone 13.8 Predictive Collision Checking.

Validates:
- 20 Required Functional Test Scenarios:
  1. Static obstacle directly blocking UGV
  2. Moving obstacle crossing UGV path
  3. Moving obstacle missing the UGV
  4. Obstacle crossing behind UGV
  5. Turning obstacle
  6. Multiple obstacles
  7. Physical footprint overlap
  8. Near miss with positive clearance
  9. Uncertainty-region intersection without physical collision
  10. High-risk region intersection
  11. Medium/low risk intersection
  12. Query between occupancy bins
  13. Trajectory extending past 2.0 s
  14. LOST trajectory capped at 1.0 s
  15. EXPIRED obstacle
  16. Empty dynamic layer
  17. Invalid trajectory input
  18. Deterministic repeated result
  19. Footprint rotation during motion
  20. Boundary/grid clipping
- 5 Explicit Safety Proofs:
  A. Predicted physical footprint intersection is always reported as collision.
  B. Large 3-sigma envelope alone does NOT automatically become a physical collision.
  C. Out-of-horizon prediction is NOT reported as FREE.
  D. LOST/expired trajectories do not fabricate future certainty.
  E. Multiple obstacles cannot overwrite a more critical collision/risk result.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from ai.planning.base import VehicleKinematics
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
    PredictiveCollisionResult,
    TrajectorySample,
    compute_polygon_clearance,
    point_to_segment_distance,
    sample_ugv_footprint_points,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
    polygon_intersection_sat,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ai.planning.trajectory import Trajectory, TrajectoryPoint


# =============================================================================
# Test Helpers and Fixtures
# =============================================================================

def make_test_obstacle(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (1.5, 0.8),
    yaw: float = 0.0,
    status: str = "TRACKING",
    horizon_s: float = 2.0,
    dt: float = 0.25,
    base_unc: float = 0.20,
    unc_growth: float = 0.15,
) -> PredictedObstacleTrajectory:
    """Generate a synthetic PredictedObstacleTrajectory for testing."""
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
        t = step * dt
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


def make_straight_ugv_trajectory(
    start_x: float = 0.0,
    start_y: float = 0.0,
    speed: float = 1.0,
    yaw: float = 0.0,
    duration_s: float = 2.0,
    dt: float = 0.1,
) -> List[TrajectorySample]:
    """Generate linear forward UGV trajectory samples."""
    samples: List[TrajectorySample] = []
    n_steps = int(round(duration_s / dt))
    for step in range(n_steps + 1):
        t = step * dt
        x = start_x + speed * math.cos(yaw) * t
        y = start_y + speed * math.sin(yaw) * t
        samples.append(TrajectorySample(time_offset=t, x=x, y=y, yaw=yaw, v=speed))
    return samples


# =============================================================================
# 20 Focused Test Scenarios
# =============================================================================

def test_scenario_01_static_obstacle_blocking() -> None:
    """Scenario 1: Static obstacle directly blocking UGV along forward trajectory."""
    # UGV starts at (0, 0) moving +X at 1.0 m/s; static obstacle at (1.5, 0.0)
    obs = make_test_obstacle(track_id=1, start_pos=(1.5, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(start_x=0.0, start_y=0.0, speed=1.0, yaw=0.0, duration_s=2.0)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.obstacle_id == 1
    assert result.collision_time is not None
    assert 0.6 <= result.collision_time <= 1.4
    assert result.minimum_clearance == 0.0
    assert result.maximum_cost == 1.0


def test_scenario_02_moving_obstacle_crossing_path() -> None:
    """Scenario 2: Moving obstacle crossing UGV path at t ≈ 1.0s."""
    # UGV moving +X at (x=t, y=0). Obstacle starts at (1.0, -2.0) moving +Y at 2.0 m/s.
    # At t=1.0s, UGV is at (1.0, 0.0) and obstacle is at (1.0, 0.0) -> COLLISION!
    obs = make_test_obstacle(track_id=2, start_pos=(1.0, -2.0), velocity=(0.0, 2.0), size=(1.0, 0.8))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0, dt=0.1)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.obstacle_id == 2
    assert 0.65 <= result.collision_time <= 1.25
    assert result.minimum_clearance == 0.0


def test_scenario_03_moving_obstacle_missing_ugv() -> None:
    """Scenario 3: Moving obstacle passing parallel with safe lateral offset."""
    # UGV moving along y=0. Obstacle moving along y=3.0 (lateral offset 3.0m).
    obs = make_test_obstacle(track_id=3, start_pos=(0.0, 3.0), velocity=(1.0, 0.0), size=(1.0, 0.8), base_unc=0.2, unc_growth=0.1)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False
    assert result.status == CollisionStatus.SAFE
    assert result.minimum_clearance > 1.5
    assert result.maximum_cost == 0.0


def test_scenario_04_obstacle_crossing_behind_ugv() -> None:
    """Scenario 4: Obstacle crosses the path after UGV has already cleared the point."""
    # UGV starts at (0, 0) and reaches (2.0, 0) at t=2.0s.
    # Obstacle starts at y=-4.0m and crosses y=0 at t=2.0s when UGV is already at x=2.0m.
    obs = make_test_obstacle(track_id=4, start_pos=(0.5, -4.0), velocity=(0.0, 2.0), size=(0.8, 0.8))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0, dt=0.1)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False
    # At t=0.5s, UGV is at x=0.5m, obstacle is at (0.5, -2.0m) (distance 2m, safe)
    # At t=1.5s, obstacle is at (0.5, 0.0m), UGV is at (1.5, 0.0m) (distance 1m, safe)
    assert result.status in (CollisionStatus.SAFE, CollisionStatus.PREDICTED_LOW_RISK)


def test_scenario_05_turning_obstacle() -> None:
    """Scenario 5: Obstacle with curved waypoint trajectory intersecting UGV path."""
    cov = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    state = TrackedObstacleState(
        track_id=5, timestamp=0.0, x=2.0, y=2.0, vx=-1.0, vy=-1.0,
        covariance=cov, status="TRACKING", is_confirmed=True, bbox_size=(1.0, 0.8), bbox_yaw=0.0,
    )
    # Waypoints curving toward (1.0, 0.0) at t=1.0s
    waypoints = [
        TrajectoryWaypoint(0.0, 0.0, 2.0, 2.0, -1.0, -1.0, np.eye(2)*0.01, 0.2),
        TrajectoryWaypoint(0.5, 0.5, 1.5, 1.0, -1.0, -2.0, np.eye(2)*0.01, 0.3),
        TrajectoryWaypoint(1.0, 1.0, 1.0, 0.0, -1.0, -2.0, np.eye(2)*0.01, 0.4),
        TrajectoryWaypoint(1.5, 1.5, 0.5, -1.0, -1.0, -2.0, np.eye(2)*0.01, 0.5),
        TrajectoryWaypoint(2.0, 2.0, 0.0, -2.0, -1.0, -2.0, np.eye(2)*0.01, 0.6),
    ]
    polygons = [
        compute_obb_corners((wp.x, wp.y), (1.0, 0.8), 0.0, margin_m=wp.uncertainty_radius)
        for wp in waypoints
    ]
    obs = PredictedObstacleTrajectory(
        track_id=5,
        current_state=state,
        horizon_s=2.0,
        dt=0.5,
        waypoints=waypoints,
        bounding_polygon_per_step=polygons,
    )
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0, dt=0.1)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.obstacle_id == 5


def test_scenario_06_multiple_obstacles() -> None:
    """Scenario 6: Multiple obstacles (4 tracks); checker detects the critical colliding one."""
    obs1 = make_test_obstacle(track_id=10, start_pos=(5.0, 5.0), velocity=(0.0, 0.0))    # Far away
    obs2 = make_test_obstacle(track_id=20, start_pos=(0.0, -4.0), velocity=(1.0, 0.0))   # Parallel offset
    obs3 = make_test_obstacle(track_id=30, start_pos=(1.2, 0.0), velocity=(0.0, 0.0))    # Direct collision at t~1.0s
    obs4 = make_test_obstacle(track_id=40, start_pos=(-5.0, 0.0), velocity=(-1.0, 0.0))  # Moving away behind

    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs1, obs2, obs3, obs4])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0, dt=0.1)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs1, obs2, obs3, obs4])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.obstacle_id == 30  # Critical colliding obstacle ID preserved!


def test_scenario_07_physical_footprint_overlap() -> None:
    """Scenario 7: Exact physical footprint overlap with rotated bounding boxes."""
    # UGV heading 45 deg, Obstacle heading -45 deg, intersecting
    obs = make_test_obstacle(track_id=7, start_pos=(1.0, 1.0), velocity=(0.0, 0.0), size=(1.2, 0.8), yaw=-math.pi/4)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(start_x=0.0, start_y=0.0, speed=1.0, yaw=math.pi/4, duration_s=1.5, dt=0.1)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.minimum_clearance == 0.0


def test_scenario_08_near_miss_positive_clearance() -> None:
    """Scenario 8: Near miss with positive geometric clearance outside footprint."""
    # Obstacle at lateral offset y = 0.85m.
    # UGV half-width = 0.45/2 + 0.05 = 0.275m.
    # Obstacle half-width = 0.6/2 = 0.30m.
    # Center separation = 0.85m -> Expected clearance = 0.85 - (0.275 + 0.30) = 0.275m > 0!
    obs = make_test_obstacle(track_id=8, start_pos=(1.0, 0.85), velocity=(0.0, 0.0), size=(1.0, 0.6), base_unc=0.15, unc_growth=0.05)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0, dt=0.1)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False  # Physical collision must be FALSE!
    assert result.minimum_clearance > 0.10  # Positive metric clearance!
    assert result.minimum_clearance < 0.50


def test_scenario_09_uncertainty_region_intersection() -> None:
    """Scenario 9: UGV passes through 2-3 sigma low-risk uncertainty zone without physical contact."""
    # Obstacle physical footprint ends at y=0.5m. Uncertainty extends to y=1.2m (r_unc=0.7m, 3-sigma).
    # UGV path at y = 1.0m (penetrates outer low-risk zone [2-3 sigma]).
    obs = make_test_obstacle(track_id=9, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0), base_unc=0.9, unc_growth=0.0)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = [TrajectorySample(time_offset=t, x=1.0, y=1.50, yaw=0.0, v=0.0) for t in [0.0, 0.5, 1.0]]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False  # Physical collision MUST NOT occur!
    assert result.status == CollisionStatus.PREDICTED_LOW_RISK
    assert result.minimum_clearance > 0.30


def test_scenario_10_high_risk_region_intersection() -> None:
    """Scenario 10: UGV penetrates immediate 0-1 sigma high-risk zone without physical contact."""
    # Obstacle at (1.0, 0.0), half-width 0.5m. UGV at y = 0.85m.
    # Geometric clearance ~ 0.85 - (0.50 + 0.275) = 0.075m.
    # With r_unc = 0.6m (sigma = 0.2m), clearance 0.075m is inside [0, 1*sigma] -> HIGH_RISK!
    obs = make_test_obstacle(track_id=10, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0), base_unc=0.6, unc_growth=0.0)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = [TrajectorySample(time_offset=1.0, x=1.0, y=0.85, yaw=0.0, v=0.0)]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False  # NOT a physical collision!
    assert result.status == CollisionStatus.PREDICTED_HIGH_RISK
    assert result.maximum_cost >= 0.70
    assert result.minimum_clearance > 0.0


def test_scenario_11_medium_low_risk_intersection() -> None:
    """Scenario 11: UGV penetrates 1-2 sigma intermediate zone -> MEDIUM_RISK."""
    # Clearance ~ 0.25m with r_unc = 0.6m (sigma = 0.2m). Clearance is in (0.2, 0.4] -> MEDIUM_RISK!
    obs = make_test_obstacle(track_id=11, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0), base_unc=0.6, unc_growth=0.0)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = [TrajectorySample(time_offset=1.0, x=1.0, y=1.03, yaw=0.0, v=0.0)]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False
    assert result.status == CollisionStatus.PREDICTED_MEDIUM_RISK
    assert 0.40 <= result.maximum_cost < 0.70


def test_scenario_12_query_between_occupancy_bins() -> None:
    """Scenario 12: Query at arbitrary intermediate time (t = 0.375s) between 0.25s and 0.50s bins."""
    obs = make_test_obstacle(track_id=12, start_pos=(0.0, 0.0), velocity=(2.0, 0.0), size=(1.0, 1.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # Sample at t = 0.375s
    sample = TrajectorySample(time_offset=0.375, x=0.75, y=0.0, yaw=0.0, v=2.0)
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory([sample], layer, [obs])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.samples_checked == 1


def test_scenario_13_trajectory_extending_past_2s() -> None:
    """Scenario 13: Trajectory extending past the valid 2.0s prediction horizon."""
    # Clear path, but trajectory extends to t = 2.5s
    layer = DynamicOccupancyLayer()
    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.5, dt=0.5)

    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(max_horizon_s=2.0))
    result = checker.check_trajectory(traj, layer)

    assert result.collision is False
    assert result.unknown_horizon_encountered is True
    assert result.status == CollisionStatus.UNKNOWN_HORIZON  # MUST NOT be SAFE!
    assert "exceeds layer max horizon" in result.reason


def test_scenario_14_lost_trajectory_capped_at_1s() -> None:
    """Scenario 14: LOST obstacle trajectory capped at 1.0s; trajectory queried past 1.0s."""
    # Obstacle is LOST, horizon_s = 1.0s
    obs = make_test_obstacle(track_id=14, start_pos=(0.0, 3.0), velocity=(0.0, 0.0), status="LOST", horizon_s=1.0)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # Trajectory extends up to 1.5s
    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=1.5, dt=0.5)
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(check_track_horizons=True))
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.unknown_horizon_encountered is True
    assert result.status == CollisionStatus.UNKNOWN_HORIZON
    assert "LOST" in result.reason or "exceeds valid horizon" in result.reason


def test_scenario_15_expired_obstacle() -> None:
    """Scenario 15: EXPIRED obstacle rejected cleanly without unhandled exception."""
    obs = make_test_obstacle(track_id=15, start_pos=(2.0, 0.0), velocity=(0.0, 0.0), status="EXPIRED")
    layer = DynamicOccupancyLayer()

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=1.0)
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(raise_on_invalid=False))
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.status == CollisionStatus.INVALID_INPUT
    assert "EXPIRED" in result.reason


def test_scenario_16_empty_dynamic_layer() -> None:
    """Scenario 16: Empty dynamic layer with no active obstacles."""
    layer = DynamicOccupancyLayer()
    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=1.5, dt=0.25)

    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer)

    assert result.collision is False
    assert result.status == CollisionStatus.SAFE
    assert result.maximum_cost == 0.0
    assert result.minimum_clearance >= checker.config.max_clearance_search_m
    assert result.risk_classification == DynamicRiskLevel.FREE


def test_scenario_17_invalid_trajectory_input() -> None:
    """Scenario 17: Invalid trajectory inputs (empty, non-monotonic, non-finite)."""
    layer = DynamicOccupancyLayer()
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(raise_on_invalid=False))

    # Empty trajectory
    res_empty = checker.check_trajectory([], layer)
    assert res_empty.status == CollisionStatus.INVALID_INPUT

    # Non-monotonic timestamps
    bad_traj = [
        TrajectorySample(0.0, 0.0, 0.0),
        TrajectorySample(1.0, 1.0, 0.0),
        TrajectorySample(0.5, 2.0, 0.0),  # Decreasing time!
    ]
    res_non_mono = checker.check_trajectory(bad_traj, layer)
    assert res_non_mono.status == CollisionStatus.INVALID_INPUT

    # Non-finite coordinates
    nan_traj = [TrajectorySample(0.0, float("nan"), 0.0)]
    res_nan = checker.check_trajectory(nan_traj, layer)
    assert res_nan.status == CollisionStatus.INVALID_INPUT


def test_scenario_18_deterministic_repeated_result() -> None:
    """Scenario 18: Deterministic repeated execution yields identical result."""
    obs = make_test_obstacle(track_id=18, start_pos=(1.5, 0.0), velocity=(0.0, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=2.0)
    checker = PredictiveCollisionChecker()

    res1 = checker.check_trajectory(traj, layer, [obs])
    for _ in range(10):
        res_repeat = checker.check_trajectory(traj, layer, [obs])
        assert res_repeat.collision == res1.collision
        assert res_repeat.status == res1.status
        assert res_repeat.collision_time == res1.collision_time
        assert res_repeat.minimum_clearance == res1.minimum_clearance
        assert res_repeat.maximum_cost == res1.maximum_cost


def test_scenario_19_footprint_rotation_during_motion() -> None:
    """Scenario 19: UGV rotating footprint during motion sweeps into obstacle."""
    # Obstacle is at (0.0, 0.45)
    # At yaw=0, UGV half-width=0.275, clearance ~ 0.45 - 0.275 = 0.175m (clear)
    # At yaw=90 deg, UGV half-length=0.375, so reaches y=0.375m, which collides with obstacle!
    obs = make_test_obstacle(track_id=19, start_pos=(0.0, 0.50), velocity=(0.0, 0.0), size=(0.6, 0.6))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # Stationary UGV rotating from 0 to 90 degrees
    traj = [
        TrajectorySample(time_offset=0.0, x=0.0, y=0.0, yaw=0.0),
        TrajectorySample(time_offset=0.5, x=0.0, y=0.0, yaw=math.pi / 4),
        TrajectorySample(time_offset=1.0, x=0.0, y=0.0, yaw=math.pi / 2),
    ]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.collision_time is not None


def test_scenario_20_boundary_grid_clipping() -> None:
    """Scenario 20: Trajectory exceeding spatial map grid bounds is reported as collision."""
    layer = DynamicOccupancyLayer(DynamicOccupancyConfig(x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0)))

    # Trajectory starts inside and exits at x = 12.0m (> 10.0m boundary)
    traj = [
        TrajectorySample(time_offset=0.0, x=8.0, y=0.0, yaw=0.0),
        TrajectorySample(time_offset=0.5, x=9.5, y=0.0, yaw=0.0),
        TrajectorySample(time_offset=1.0, x=12.0, y=0.0, yaw=0.0),  # Out of bounds!
    ]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer)

    assert result.collision is True
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert "bounds" in result.reason.lower()


# =============================================================================
# 5 Mandatory Safety Proofs (A through E)
# =============================================================================

def test_safety_proof_a_physical_footprint_intersection_always_collision() -> None:
    """Safety Proof A: A predicted physical footprint intersection is ALWAYS reported as collision."""
    obs = make_test_obstacle(track_id=101, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # Trajectory intersects obstacle directly
    traj = [TrajectorySample(time_offset=0.5, x=1.0, y=0.0, yaw=0.0)]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is True, "Proof A Violation: Physical overlap MUST set collision=True"
    assert result.status == CollisionStatus.PREDICTED_COLLISION
    assert result.minimum_clearance == 0.0


def test_safety_proof_b_large_uncertainty_envelope_not_physical_collision() -> None:
    """Safety Proof B: A large 3-sigma envelope alone does NOT automatically become a physical collision."""
    # Obstacle at (0.0, 0.0) with very large 3-sigma uncertainty radius = 5.0m
    obs = make_test_obstacle(track_id=102, start_pos=(0.0, 0.0), velocity=(0.0, 0.0), size=(1.0, 1.0), base_unc=5.0, unc_growth=0.0)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # UGV is positioned at (2.5, 0.0).
    # Center separation is 2.5m. Obstacle half-extent is 0.5m. UGV half-extent is 0.375m.
    # Geometric clearance is 2.5 - (0.5 + 0.375) = 1.625m > 0!
    # But it is inside the 5.0m uncertainty envelope!
    traj = [TrajectorySample(time_offset=1.0, x=2.5, y=0.0, yaw=0.0)]
    checker = PredictiveCollisionChecker()
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.collision is False, "Proof B Violation: Uncertainty penetration MUST NOT be flagged as physical collision"
    assert result.status != CollisionStatus.PREDICTED_COLLISION
    assert result.minimum_clearance > 1.0


def test_safety_proof_c_out_of_horizon_never_reported_as_free() -> None:
    """Safety Proof C: Out-of-horizon prediction is NOT reported as FREE or SAFE."""
    layer = DynamicOccupancyLayer(DynamicOccupancyConfig(max_horizon_s=2.0))

    # Clean trajectory extending beyond 2.0s
    traj = [TrajectorySample(time_offset=2.5, x=5.0, y=0.0, yaw=0.0)]
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(max_horizon_s=2.0))
    result = checker.check_trajectory(traj, layer)

    assert result.collision is False
    assert result.status != CollisionStatus.SAFE, "Proof C Violation: Out-of-horizon MUST NOT silently become SAFE"
    assert result.status == CollisionStatus.UNKNOWN_HORIZON
    assert result.unknown_horizon_encountered is True


def test_safety_proof_d_lost_track_does_not_fabricate_future_certainty() -> None:
    """Safety Proof D: LOST/expired trajectories do not fabricate future certainty."""
    # Obstacle is LOST, valid horizon ends at 1.0s
    obs = make_test_obstacle(track_id=104, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), status="LOST", horizon_s=1.0)
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    # UGV trajectory queries at t = 1.5s (> 1.0s)
    traj = [TrajectorySample(time_offset=1.5, x=1.0, y=0.0, yaw=0.0)]
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(check_track_horizons=True))
    result = checker.check_trajectory(traj, layer, [obs])

    assert result.status != CollisionStatus.SAFE, "Proof D Violation: Query beyond LOST track horizon MUST NOT become SAFE"
    assert result.unknown_horizon_encountered is True
    assert result.status == CollisionStatus.UNKNOWN_HORIZON


def test_safety_proof_e_multiple_obstacles_cannot_overwrite_critical_result() -> None:
    """Safety Proof E: Multiple obstacles cannot overwrite a more critical collision/risk result."""
    # Obstacle 1: clear (SAFE)
    obs1 = make_test_obstacle(track_id=1, start_pos=(10.0, 10.0), velocity=(0.0, 0.0))
    # Obstacle 2: crossing path to collide at t ≈ 0.5s
    obs2 = make_test_obstacle(track_id=2, start_pos=(0.5, -1.0), velocity=(0.0, 2.0), size=(0.6, 0.6))
    # Obstacle 3: low risk at t = 1.0s
    obs3 = make_test_obstacle(track_id=3, start_pos=(3.0, 1.2), velocity=(0.0, 0.0), base_unc=1.0)

    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs1, obs2, obs3])

    traj = make_straight_ugv_trajectory(speed=1.0, duration_s=1.5, dt=0.1)
    checker = PredictiveCollisionChecker()

    # Pass in various orders: [obs1, obs2, obs3] and [obs3, obs1, obs2]
    res_a = checker.check_trajectory(traj, layer, [obs1, obs2, obs3])
    res_b = checker.check_trajectory(traj, layer, [obs3, obs1, obs2])

    for res in (res_a, res_b):
        assert res.collision is True, "Proof E Violation: Critical collision must not be overwritten"
        assert res.status == CollisionStatus.PREDICTED_COLLISION
        assert res.obstacle_id == 2
        assert abs(res.collision_time - 0.5) <= 0.25


# =============================================================================
# Additional Interface and Footprint Contract Tests
# =============================================================================

def test_check_point_api() -> None:
    """Validate single point check_point API."""
    obs = make_test_obstacle(track_id=88, start_pos=(1.0, 0.0), velocity=(0.0, 0.0))
    layer = DynamicOccupancyLayer()
    layer.update_predictions([obs])

    checker = PredictiveCollisionChecker()
    res_hit = checker.check_point(x=1.0, y=0.0, yaw=0.0, time_offset=0.5, dynamic_layer=layer, predicted_tracks=[obs])
    assert res_hit.collision is True

    res_miss = checker.check_point(x=5.0, y=5.0, yaw=0.0, time_offset=0.5, dynamic_layer=layer, predicted_tracks=[obs])
    assert res_miss.collision is False
    assert res_miss.status == CollisionStatus.SAFE


def test_vehicle_kinematics_initialization() -> None:
    """Verify initialization from Phase 12 VehicleKinematics."""
    vk = VehicleKinematics(length_m=0.70, width_m=0.50, collision_margin_m=0.08)
    checker = PredictiveCollisionChecker(vehicle=vk)

    assert checker.length == 0.70
    assert checker.width == 0.50
    assert checker.margin == 0.08


def test_polygon_clearance_calculation() -> None:
    """Verify mathematical accuracy of compute_polygon_clearance."""
    # Two axis-aligned 1x1 boxes
    # Box A: center (0, 0), corners [(-0.5, -0.5), (0.5, 0.5)]
    # Box B: center (3, 0), corners [(2.5, -0.5), (3.5, 0.5)]
    # Expected clearance = 2.5 - 0.5 = 2.0m
    box_a = compute_obb_corners((0.0, 0.0), (1.0, 1.0), 0.0)
    box_b = compute_obb_corners((3.0, 0.0), (1.0, 1.0), 0.0)

    clearance = compute_polygon_clearance(box_a, box_b)
    assert abs(clearance - 2.0) < 1e-6

    # Overlapping boxes -> 0.0
    box_overlap = compute_obb_corners((0.5, 0.0), (1.0, 1.0), 0.0)
    assert compute_polygon_clearance(box_a, box_overlap) == 0.0


def test_trajectory_adapter_types() -> None:
    """Verify Trajectory (ai.planning.trajectory) and TrajectoryPoint adapters."""
    layer = DynamicOccupancyLayer()
    pts = [TrajectoryPoint(t=0.1 * i, x=float(i), y=0.0, yaw=0.0, v=1.0) for i in range(5)]
    traj_obj = Trajectory(points=pts)

    checker = PredictiveCollisionChecker()
    res = checker.check_trajectory(traj_obj, layer)
    assert res.samples_checked == 5
    assert res.status == CollisionStatus.SAFE
