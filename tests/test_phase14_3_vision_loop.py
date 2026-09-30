"""Tests for Phase 14.3 Closed-Loop Vision-in-the-Loop Navigation.

Validates the complete closed loop:
  RGB Image -> Perception Pipeline -> Ground Projection -> World Costmap -> A* Replanner -> Native C++ DWA -> cmd_vel.
Measures latency breakdown and verifies real-time execution.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.spatial_fusion.world_map import WorldTerrainMap
from ai.traversability.base import TraversabilityResult
from ai.planning.base import GoalPose, PlannerStatus, RobotState, PlanningConfig
from ai.planning.backends.backend_interface import DWABackendType
from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.dynamic.behavioral_state import BehavioralConfig, BehavioralStateMachine
from ai.planning.dynamic.global_replanner import (
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
)
from production.perception.base import PerceptionFrameInput
from production.perception.realtime_pipeline import RealtimeTraversabilityPipeline
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


def _create_synthetic_test_frame(has_obstacle: bool = False) -> np.ndarray:
    """Generate a realistic RGB outdoor test frame (640x480)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Sky (top half)
    img[:200, :, :] = [180, 140, 100]  # Light sky
    # Trail ground (bottom half)
    img[200:, :, :] = [60, 120, 70]   # Soil / grass
    # Center trail
    img[200:, 220:420, :] = [110, 130, 140]  # Compact dirt trail

    if has_obstacle:
        # Lethal boulder in the trail at center ground (rows 240:340, cols 280:360)
        img[260:320, 290:350, :] = [40, 40, 40]  # Dark rock boulder

    return img


def test_closed_loop_vision_to_dwa_execution():
    """Verify live perception feeds costmap, triggers A* replanning, and yields valid DWA commands."""
    # 1. Initialize complete stack (navigation pipeline with overlay generation disabled for speed)
    perception = RealtimeTraversabilityPipeline(generate_overlay=False)
    integrator = VisionCostmapIntegrator.create_simulation_default()

    wmap = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=[-5.0, 15.0],
        y_bounds_m=[-10.0, 10.0],
        unknown_cost_default=0.50,
    )
    # Initialize nominal terrain as unobserved (unknown)
    wmap.unknown.fill(True)
    wmap.cost.fill(0.50)

    # Planners
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    # Initial state and goal
    robot_pose = (0.0, 0.0, 0.0)  # at origin facing +X
    goal = (8.0, 0.0)             # 8 meters ahead
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)

    # 2. Step 1: Ingest clear trail frame
    frame_clear = _create_synthetic_test_frame(has_obstacle=False)

    # Warmup pipeline (one pass to warm up caches, JIT, and kernels with valid frame)
    _ = perception.predict(PerceptionFrameInput(image=frame_clear, timestamp=0.5))


    t0 = time.perf_counter()
    trav_res = perception.predict(
        PerceptionFrameInput(image=frame_clear, timestamp=1.0, frame_id="camera_optical")
    )
    t_perc = (time.perf_counter() - t0) * 1000.0



    t0 = time.perf_counter()
    bev_costmap = integrator.project_traversability(trav_res, current_time_s=1.0)
    fusion_res = integrator.fuse_into_world_map(bev_costmap, robot_pose, wmap)
    t_proj_fuse = (time.perf_counter() - t0) * 1000.0

    camera_to_costmap_ms = t_perc + t_proj_fuse
    assert fusion_res["valid"]
    assert fusion_res["cells_updated"] > 0

    # Step 2: Global Replanner
    t0 = time.perf_counter()
    replan_res = replanner.execute_replan(
        current_pose=robot_pose,
        goal=goal,
        base_map=wmap,
        dynamic_obstacles=[],
    )
    t_plan = (time.perf_counter() - t0) * 1000.0
    assert replan_res.success
    assert len(replan_res.path.points) > 0

    # Step 3: DWA Local Trajectory
    t0 = time.perf_counter()
    traj, v_cmd, w_cmd, status, diag = dwa.compute_trajectory(
        current_state=robot_state,
        global_path=replan_res.path,
        costmap=wmap,
        dynamic_layer=dynamic_layer,
        predicted_tracks=[],
    )
    t_dwa = (time.perf_counter() - t0) * 1000.0

    steady_state_loop_ms = camera_to_costmap_ms + t_dwa
    total_initial_ms = camera_to_costmap_ms + t_plan + t_dwa
    eff_freq = 1000.0 / max(steady_state_loop_ms, 1e-3)

    assert v_cmd > 0.0  # Clear path allows forward motion
    assert steady_state_loop_ms < 120.0  # Steady state vision+costmap+DWA loop runs in <120ms (tolerating OS jitter)

    assert total_initial_ms < 1000.0     # Cold global A* plan + vision + DWA in <1.0s


    print(
        f"\nClosed-Loop Latency Metrics:"
        f"\n  camera_to_costmap_ms:    {camera_to_costmap_ms:.2f} ms"
        f"\n  costmap_to_plan_ms:      {t_plan:.2f} ms"
        f"\n  plan_to_command_ms:      {t_dwa:.2f} ms"
        f"\n  steady_state_loop_ms:    {steady_state_loop_ms:.2f} ms"
        f"\n  effective_frequency:     {eff_freq:.1f} Hz"
    )


def test_closed_loop_sequential_stability():
    """Verify stability of closed loop over 5 consecutive frames."""
    perception = RealtimeTraversabilityPipeline(generate_overlay=False)
    integrator = VisionCostmapIntegrator.create_simulation_default()
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=[-5.0, 15.0], y_bounds_m=[-10.0, 10.0])
    wmap.unknown.fill(True)
    wmap.cost.fill(0.50)

    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    frame = _create_synthetic_test_frame(has_obstacle=False)

    # Warmup
    _ = perception.predict(PerceptionFrameInput(image=frame, timestamp=0.5))

    # Initial global path
    plan = replanner.execute_replan((0.0, 0.0, 0.0), (8.0, 0.0), base_map=wmap, dynamic_obstacles=[])

    latencies = []
    for i in range(5):
        t_start = time.perf_counter()
        t_now = 1.0 + i * 0.05
        res = perception.predict(PerceptionFrameInput(image=frame, timestamp=t_now))
        bev = integrator.project_traversability(res, current_time_s=t_now)
        integrator.fuse_into_world_map(bev, (0.0, 0.0, 0.0), wmap)

        robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.2, omega=0.0, timestamp=t_now)
        traj, v, w, status, diag = dwa.compute_trajectory(
            current_state=robot_state,
            global_path=plan.path,
            costmap=wmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=[],
        )
        loop_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(loop_ms)
        assert v >= 0.0

    mean_loop_ms = np.mean(latencies)
    assert mean_loop_ms < 100.0

