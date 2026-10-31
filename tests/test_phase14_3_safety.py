"""Tests for Phase 14.3 Navigation Safety Scenarios & Invariant Verification.

Validates the bounded safety scenarios required by the Phase 14.3 gate:
  - Scenario A: Clear trail -> valid forward velocity
  - Scenario B: Obstacle in trail -> camera detects obstacle, marks costmap, DWA steers clear (zero collision false negatives)
  - Scenario C: Traversable detour -> side corridor used when center is blocked
  - Scenario D: Non-traversable terrain -> vehicle stops safely
  - Scenario E: Perception blackout / camera loss -> fail-safe stop, no acceleration
  - Scenario F: Dynamic obstacle coexistence -> dynamic tracker and vision costmap operate simultaneously
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.backends.backend_interface import DWABackendType
from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.base import GoalPose, PlannerStatus, PlanningConfig, RobotState
from ai.planning.dynamic.collision_checker import PredictiveCollisionChecker, PredictiveCollisionConfig
from ai.planning.dynamic.global_replanner import EventDrivenGlobalReplanner, GlobalReplannerConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig
from ai.planning.path import Path as AIPath, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from production.perception.base import PerceptionFrameInput
from production.perception.realtime_pipeline import RealtimeTraversabilityPipeline
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


def _create_synthetic_frame_with_obstacle(obstacle_x_center: int = 320, obstacle_y_center: int = 240) -> np.ndarray:
    """Creates a frame with an obstacle placed at specific image coordinates."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:200, :, :] = [180, 140, 100]  # Sky
    img[200:, :, :] = [60, 120, 70]   # Soil
    img[200:, 200:440, :] = [110, 130, 140]  # Center trail

    # Dark rock obstacle
    r0, r1 = max(0, obstacle_y_center - 30), min(480, obstacle_y_center + 30)
    c0, c1 = max(0, obstacle_x_center - 30), min(640, obstacle_x_center + 30)
    img[r0:r1, c0:c1, :] = [25, 25, 25]
    return img


def test_scenario_a_clear_trail_forward_motion():
    """Scenario A: Vehicle on clear trail commands safe positive forward velocity."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 15.0], y_bounds_m=[-10.0, 10.0])
    wmap.unknown.fill(True)
    wmap.cost.fill(0.50)

    # Ingest clear trail perception
    mask_clear = np.full((240, 320), 0.90, dtype=np.float32)
    res_clear = TraversabilityResult(mask=mask_clear, timestamp=1.0, valid=True)
    bev = integrator.project_traversability(res_clear, current_time_s=1.0)
    integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)

    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    # Straight line path ahead
    path = AIPath(points=[PathPoint(x=x, y=0.0) for x in np.linspace(0.0, 5.0, 25)])
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)

    traj, v, w, status, diag = dwa.compute_trajectory(
        current_state=state,
        global_path=path,
        costmap=wmap,
        dynamic_layer=dynamic_layer,
        predicted_tracks=[],
    )

    assert v > 0.15  # Commands positive forward speed
    assert math.isclose(w, 0.0, abs_tol=0.2)  # Commands straight heading


def test_scenario_b_obstacle_in_trail_avoidance():
    """Scenario B: Boulder in trail is marked as lethal; DWA rejects collision trajectory (zero false negatives)."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 15.0], y_bounds_m=[-10.0, 10.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    # Place lethal boulder in front of vehicle at ~2.1m ahead (row 120, col 160 in 240x320 mask)
    mask = np.full((240, 320), 0.90, dtype=np.float32)
    mask[110:130, 150:170] = 0.05  # Lethal obstacle in vehicle heading line
    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)

    bev = integrator.project_traversability(res, current_time_s=1.0)
    integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)

    # Verify obstacle marked at x ~ 2.1m, y ~ 0.0m
    r_obs, c_obs = wmap.world_to_grid(2.1, 0.0)
    assert wmap.obstacle[r_obs, c_obs]
    assert wmap.cost[r_obs, c_obs] == 1.0

    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    # Test robot approaching the obstacle at (1.8m, 0.0m) moving forward at 0.3 m/s
    approaching_state = RobotState(x=1.6, y=0.0, yaw=0.0, v=0.3, omega=0.0, timestamp=1.0)
    straight_path = AIPath(points=[PathPoint(x=x, y=0.0) for x in np.linspace(1.6, 5.0, 20)])

    traj, v, w, status, diag = dwa.compute_trajectory(
        current_state=approaching_state,
        global_path=straight_path,
        costmap=wmap,
        dynamic_layer=dynamic_layer,
        predicted_tracks=[],
    )

    # Collision Checker: Trajectory must NEVER collide with the lethal obstacle
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(vehicle_length_m=0.65, vehicle_width_m=0.45))
    if traj is not None:
        for pt in traj.points:
            dist_to_obs = math.hypot(pt.x - 2.1, pt.y - 0.0)
            assert dist_to_obs > 0.20  # Trajectory maintains safety clearance!


def test_scenario_c_traversable_detour():
    """Scenario C: Center trail is blocked, but clear detour exists; planner detours around hazard."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-2.0, 10.0], y_bounds_m=[-5.0, 5.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    # Lethal barrier across center trail from y = -0.5 to +0.5 at x = 2.0
    r_bar, _ = wmap.world_to_grid(2.0, 0.0)
    for y in np.linspace(-0.6, 0.6, 13):
        r, c = wmap.world_to_grid(2.0, y)
        wmap.obstacle[r, c] = True
        wmap.cost[r, c] = 1.0

    # Clear detour exists on the left (y = 1.5)
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    res = replanner.execute_replan(
        current_pose=(0.0, 0.0, 0.0),
        goal=(4.0, 0.0),
        base_map=wmap,
        dynamic_obstacles=[],
    )

    assert res.success
    # The path must detour around y=0 at x=2.0 (i.e. |y| > 0.6 at x=2.0)
    pts_near_barrier = [pt for pt in res.path.points if abs(pt.x - 2.0) < 0.3]
    assert len(pts_near_barrier) > 0
    for pt in pts_near_barrier:
        assert abs(pt.y) > 0.6  # Successfully detoured!


def test_scenario_d_non_traversable_terrain_stop():
    """Scenario D: Entire forward horizon is non-traversable (water/ditch); vehicle halts safely."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-2.0, 10.0], y_bounds_m=[-5.0, 5.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    # Ingest perception where entire field is non-traversable (tau = 0.05)
    mask = np.full((240, 320), 0.05, dtype=np.float32)
    res = TraversabilityResult(mask=mask, timestamp=1.0, valid=True)
    bev = integrator.project_traversability(res, current_time_s=1.0)
    integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)

    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    # Robot approaching non-traversable boundary at x=0.5m
    straight_path = AIPath(points=[PathPoint(x=x, y=0.0) for x in np.linspace(0.5, 4.0, 20)])
    state = RobotState(x=0.5, y=0.0, yaw=0.0, v=0.3, omega=0.0, timestamp=1.0)

    traj, v, w, status, diag = dwa.compute_trajectory(
        current_state=state,
        global_path=straight_path,
        costmap=wmap,
        dynamic_layer=dynamic_layer,
        predicted_tracks=[],
    )

    # Must command safe stop or zero forward speed (never accelerate through barrier)
    assert v <= 0.05 or status in (
        PlannerStatus.NO_SAFE_TRAJECTORY,
        PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
        PlannerStatus.NO_PATH,
    )


def test_scenario_e_perception_blackout_safety():
    """Scenario E: Camera blackout / perception failure degrades conservatively and does not emit free space."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 10.0], y_bounds_m=[-5.0, 10.0])
    wmap.unknown.fill(True)
    wmap.cost.fill(0.50)

    # Feed corrupt / blackout perception result (valid = False)
    corrupt_res = TraversabilityResult(
        mask=np.zeros((240, 320), dtype=np.float32),
        timestamp=1.0,
        valid=False,
        status_message="CAMERA_BLACKOUT_DETECTED",
    )
    bev = integrator.project_traversability(corrupt_res, current_time_s=1.0)
    assert not bev.valid

    # World map fusion with invalid BEV must NOT alter world map into free space
    fuse_res = integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)
    assert fuse_res["status"] == "DEGRADED"
    assert fuse_res["cells_updated"] == 0

    # World map remains strictly unknown (cost >= 0.50), never converted to free space
    assert np.all(wmap.unknown)
    assert np.all(wmap.cost >= 0.50)


def test_scenario_f_dynamic_obstacle_coexistence():
    """Scenario F: Dynamic obstacle tracker and vision-based static costmap coexist seamlessly."""
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 10.0], y_bounds_m=[-5.0, 10.0])
    wmap.unknown.fill(False)
    wmap.cost.fill(0.10)

    # Vision marks static rock on right (x=2.0, y=-1.0)
    r_rock, c_rock = wmap.world_to_grid(2.0, -1.0)
    wmap.obstacle[r_rock, c_rock] = True
    wmap.cost[r_rock, c_rock] = 1.0

    # Dynamic layer marks moving obstacle on left (x=2.0, y=1.0)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    from ai.planning.dynamic.data_model import TrackedObstacleState
    from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig

    track = TrackedObstacleState(
        track_id=1,
        timestamp=1.0,
        x=2.0,
        y=1.0,
        vx=0.0,
        vy=-0.2,
        covariance=np.eye(4, dtype=np.float64) * 0.05,
        bbox_size=(0.8, 0.8),
        bbox_yaw=0.0,
    )
    predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.25))
    pred_array = predictor.predict_tracks([track])
    pred = pred_array.predictions[0]
    dynamic_layer.update_predictions([pred], header_timestamp=1.0)



    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)

    straight_path = AIPath(points=[PathPoint(x=x, y=0.0) for x in np.linspace(0.0, 4.0, 20)])
    state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.2, omega=0.0, timestamp=1.0)

    # DWA must evaluate both static rock and dynamic obstacle
    traj, v, w, status, diag = dwa.compute_trajectory(
        current_state=state,
        global_path=straight_path,
        costmap=wmap,
        dynamic_layer=dynamic_layer,
        predicted_tracks=[pred],
    )

    # Center corridor (y=0) is clear between y=-1.0 (rock) and y=1.0 (dynamic obs)
    assert traj is not None
    for pt in traj.points:
        assert math.hypot(pt.x - 2.0, pt.y - (-1.0)) > 0.20  # Avoids static rock
        assert math.hypot(pt.x - 2.0, pt.y - 1.0) > 0.20     # Avoids dynamic obstacle
