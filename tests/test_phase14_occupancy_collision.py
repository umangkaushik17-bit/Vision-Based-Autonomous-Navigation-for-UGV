"""Group 3: Multi-obstacle dynamic occupancy grid and predictive collision checking tests."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.trajectory import Trajectory, TrajectoryPoint
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    PredictedObstacleTrajectory,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
    polygon_intersection_sat,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer


def _make_dummy_pred(track_id: int, start_x: float, start_y: float, vx: float, vy: float) -> PredictedObstacleTrajectory:
    """Helper to construct synthetic PredictedObstacleTrajectory for unit testing."""
    dt = 0.10
    horizon_s = 2.0
    waypoints = []
    polys = []
    t_state = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=start_x,
        y=start_y,
        vx=vx,
        vy=vy,
        covariance=np.eye(4) * 0.05,
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.0,
    )
    for step in range(int(horizon_s / dt)):
        t_off = step * dt
        x = start_x + vx * t_off
        y = start_y + vy * t_off
        wp = TrajectoryWaypoint(
            time_offset=t_off,
            timestamp=t_off,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            covariance_2d=np.eye(2) * 0.05,
            uncertainty_radius=0.20,
        )
        waypoints.append(wp)
        polys.append(compute_obb_corners((x, y), (0.8, 0.6), 0.0))

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=t_state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=waypoints,
        bounding_polygon_per_step=polys,
    )


def test_multi_obstacle_occupancy_aggregation():
    """Assert D: Multiple dynamic obstacle predictions correctly aggregate in DynamicOccupancyLayer."""
    layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))
    p1 = _make_dummy_pred(1, 2.0, 0.0, 0.0, 0.0)
    p2 = _make_dummy_pred(2, 4.0, 2.0, 0.0, 0.0)

    layer.update_predictions([p1, p2], header_timestamp=0.0)

    # Cost at obstacle 1 and 2 must be elevated
    sl = layer.get_slice(0.0)
    r1, c1 = layer.world_to_grid(2.0, 0.0)
    r2, c2 = layer.world_to_grid(4.0, 2.0)
    cost1 = sl.cost[r1, c1]
    cost2 = sl.cost[r2, c2]

    assert cost1 > 0.5, f"Expected high cost at obs 1, got {cost1}"
    assert cost2 > 0.5, f"Expected high cost at obs 2, got {cost2}"


def test_predictive_collision_checker_multiple_obstacles():
    """Assert E: Predictive collision checker detects collision against the closest threatening obstacle."""
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig())
    layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    # Obstacle 1 at x=2.0, obstacle 2 at x=5.0
    p1 = _make_dummy_pred(101, 2.0, 0.0, -0.2, 0.0)
    p2 = _make_dummy_pred(102, 5.0, 0.0, 0.0, 0.0)
    layer.update_predictions([p1, p2], header_timestamp=0.0)

    # Robot trajectory heading straight forward into Obstacle 1
    points = [TrajectoryPoint(t=float(i) * 0.10, x=float(i) * 0.15, y=0.0, v=0.5, omega=0.0) for i in range(20)]
    traj = Trajectory(points=points)

    res = checker.check_trajectory(traj, layer, [p1, p2])
    assert res.status in (CollisionStatus.PREDICTED_COLLISION, CollisionStatus.PREDICTED_HIGH_RISK)
    assert res.collision_time is not None
    assert res.obstacle_id == 101, f"Expected collision with track 101, got {res.obstacle_id}"


def test_sat_ground_truth_oracle_precision():
    """Exact SAT polygon intersection correctly determines overlap."""
    poly_ugv = np.array([[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]])
    # Overlapping obstacle at (0.2, 0.1)
    poly_obs_overlap = np.array([[-0.1, -0.1], [0.5, -0.1], [0.5, 0.3], [-0.1, 0.3]])
    assert polygon_intersection_sat(poly_ugv, poly_obs_overlap) is True

    # Separated obstacle at (2.0, 2.0)
    poly_obs_clear = np.array([[1.8, 1.8], [2.2, 1.8], [2.2, 2.2], [1.8, 2.2]])
    assert polygon_intersection_sat(poly_ugv, poly_obs_clear) is False
