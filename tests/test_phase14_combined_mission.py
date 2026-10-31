"""Phase 14 Final Combined Validation: End-to-End 5-Waypoint Mission with Vision-in-the-Loop & EKF Localization.

Verifies the complete closed-loop navigation architecture:
- Live RGB Camera -> RealtimeTraversabilityPipeline
- Metric Pinhole Ground-Plane IPM Projection -> VisionCostmapIntegrator
- Costmap Fusion -> WorldTerrainMap
- Multi-Sensor Fusion Extended Kalman Filter (EKF) -> FusedLocalizationState
- High-Level Mission Executive & Sequential Waypoint Manager (W1 -> W2 -> W3 -> W4 -> W5 -> MISSION_COMPLETE)
- Event-Driven Global Replanner (A*)
- Native C++ AVX2 Dynamic Window Approach (DWA) Local Planner
- Authoritative Safety Governor & High-Frequency SafeCommandBuffer
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from ai.control.command_buffer import SafeCommandBuffer, SafeCommandBufferConfig
from ai.planning.backends.backend_interface import DWABackendType
from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.base import PlanningConfig, RobotState
from ai.planning.dynamic.global_replanner import EventDrivenGlobalReplanner, GlobalReplannerConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.safety_governor import SafetyGovernor, SafetyGovernorConfig
from ai.spatial_fusion.world_map import WorldTerrainMap
from production.localization import (
    IMUMeasurement,
    MultiSensorEKF,
    StereoVIOMeasurement,
    WheelOdometryMeasurement,
)
from production.perception.realtime_pipeline import PerceptionFrameInput, RealtimeTraversabilityPipeline
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator
from ugv_mission import MissionExecutive, MissionExecutiveConfig, MissionState


def _create_synthetic_trail_frame() -> np.ndarray:
    """Generates an outdoor synthetic trail camera frame matching Rellis3D ground signatures."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Sky / horizon
    frame[:180, :] = [180, 150, 120]
    # Drivable outdoor terrain (soil/vegetation)
    frame[180:, :] = [60, 120, 70]
    return frame


def test_final_combined_5waypoint_mission() -> None:
    """Executes the complete 5-waypoint integrated mission verification required by Section 48."""
    # 1. Initialize Perception & Projection
    perception = RealtimeTraversabilityPipeline(generate_overlay=False)
    integrator = VisionCostmapIntegrator.create_simulation_default()

    # 2. World Costmap
    wmap = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=[-5.0, 15.0],
        y_bounds_m=[-8.0, 8.0],
        unknown_cost_default=0.50,
    )
    wmap.unknown.fill(True)
    wmap.cost.fill(0.50)

    # 3. Localization: Multi-Sensor EKF
    ekf = MultiSensorEKF()
    ekf.reset(initial_pose=(0.0, 0.0, 0.0))

    # 4. Planning Stack
    replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
    dwa_cfg = PlanningConfig.load()
    dwa_cfg.vehicle.min_linear_velocity_mps = 0.0
    dwa_cfg.vehicle.max_linear_velocity_mps = 0.55
    dwa = UnifiedDWAPlanner(config=dwa_cfg, backend=DWABackendType.AUTO)
    dynamic_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))

    # 5. Safety & Command Execution
    safety_governor = SafetyGovernor(SafetyGovernorConfig(stop_distance_m=0.35, cautious_distance_m=0.85))
    cmd_buffer = SafeCommandBuffer(SafeCommandBufferConfig(publish_rate_hz=20.0))

    # 6. Mission Executive
    mission = MissionExecutive(
        config=MissionExecutiveConfig(mission_name="phase14_final_combined_validation"),
        global_planner=replanner,
        safety_governor=safety_governor,
        command_buffer=cmd_buffer,
    )
    ok, err = mission.load_from_yaml("configs/mission_phase14_1_5wp.yaml")
    assert ok, f"Failed to load mission config: {err}"
    assert mission.total_waypoints == 5

    # Mission execution loop
    sim_t = 0.0
    dt = 0.10  # 10 Hz control loop
    max_steps = 500

    waypoints_visited = []
    rgb_frames_processed = 0
    costmap_updates_count = 0

    frame_rgb = _create_synthetic_trail_frame()

    # Initial path planning & Start Mission
    curr_wp = mission.active_waypoint
    assert curr_wp is not None
    init_res = replanner.execute_replan(
        current_pose=(0.0, 0.0, 0.0),
        goal=(curr_wp.x, curr_wp.y),
        base_map=wmap,
        dynamic_obstacles=[],
    )
    assert init_res.success
    active_path = init_res.path
    last_wp_id = curr_wp.id

    # Start mission executive
    start_ok, start_msg = mission.start_mission((0.0, 0.0, 0.0), wmap)
    assert start_ok, f"Failed to start mission: {start_msg}"
    assert mission.state == MissionState.NAVIGATING_TO_WAYPOINT

    print(f"\n--- COMMENCING FINAL COMBINED 5-WAYPOINT VALIDATION MISSION ---")
    start_wall_time = time.perf_counter()
    current_v = 0.0
    current_w = 0.0

    for step in range(max_steps):
        sim_t = step * dt

        # A. Live Perception Step (every 2 steps = 5 Hz)
        if step % 2 == 0:
            trav_result = perception.predict(PerceptionFrameInput(image=frame_rgb, timestamp=sim_t))
            rgb_frames_processed += 1
            assert trav_result.valid

            # B. Metric Projection & Costmap Fusion
            bev_map = integrator.project_traversability(trav_result, current_time_s=sim_t)
            fuse_stat = integrator.fuse_into_world_map(bev_map, ekf.get_state(sim_t).pose_xy_yaw, wmap)
            if fuse_stat["valid"]:
                costmap_updates_count += 1

        # C. Localization Step: Multi-Sensor EKF Update
        ekf.update_wheel_odom(WheelOdometryMeasurement(
            timestamp=sim_t,
            vx=current_v,
            vyaw=current_w,
        ))
        ekf.update_imu(IMUMeasurement(
            timestamp=sim_t,
            angular_velocity=np.array([0.0, 0.0, current_w]),
            linear_acceleration=np.array([0.0, 0.0, 9.80665]),
        ))
        state_fused = ekf.get_state(sim_t)
        ekf.update_stereo_vio(StereoVIOMeasurement(
            timestamp=sim_t,
            x=state_fused.x,
            y=state_fused.y,
            z=0.0,
            roll=0.0, pitch=0.0, yaw=state_fused.yaw,
            vx=current_v * math.cos(state_fused.yaw),
            vy=current_v * math.sin(state_fused.yaw),
            vz=0.0,
            inliers=95, tracking_state="TRACKING_OK", valid=True,
        ))

        robot_state = RobotState(
            x=state_fused.x,
            y=state_fused.y,
            yaw=state_fused.yaw,
            v=current_v,
            omega=current_w,
            timestamp=sim_t,
        )

        # D. Mission Executive Step (checks waypoint arrival and manages state transitions)
        m_state, m_v, m_w, m_reason = mission.step(
            current_time_s=sim_t,
            robot_pose=(robot_state.x, robot_state.y, robot_state.yaw),
            costmap=wmap,
        )

        # Track waypoints visited
        if mission.active_waypoint is not None:
            if not waypoints_visited or waypoints_visited[-1] != mission.active_waypoint.id:
                waypoints_visited.append(mission.active_waypoint.id)

        # Check mission complete
        if mission.state == MissionState.MISSION_COMPLETE:
            print(f"[MISSION SUCCESS] Mission reached COMPLETE state at step {step} (t = {sim_t:.1f}s)!")
            break

        # If active waypoint changed, replan global path to next waypoint
        curr_wp = mission.active_waypoint
        if curr_wp is not None and (active_path is None or last_wp_id != curr_wp.id):
            replan_res = replanner.execute_replan(
                current_pose=(robot_state.x, robot_state.y, robot_state.yaw),
                goal=(curr_wp.x, curr_wp.y),
                base_map=wmap,
                dynamic_obstacles=[],
            )
            if replan_res.success:
                active_path = replan_res.path
                last_wp_id = curr_wp.id

        # E. DWA Local Planner
        # Enforce localization safety limit
        safe_v = ekf.get_safe_speed_limit(nominal_max_speed=0.55)
        dwa.config.vehicle.max_linear_velocity_mps = min(0.55, safe_v)

        cand_traj, cand_v, cand_w, status, diag = dwa.compute_trajectory(
            current_state=robot_state,
            global_path=active_path,
            costmap=wmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=[],
        )

        # F. Safety Governor & Command Buffer
        gov_v, gov_w, gov_dec, _ = safety_governor.evaluate(
            requested_v=cand_v,
            requested_w=cand_w,
            robot_pose=(robot_state.x, robot_state.y, robot_state.yaw),
            hazards=[],
            current_time_s=sim_t,
        )
        cmd_buffer.set_command(linear_v=gov_v, angular_w=gov_w, timestamp_s=sim_t)
        current_v = gov_v
        current_w = gov_w

    total_wall_time = time.perf_counter() - start_wall_time
    final_fused_state = ekf.get_state(sim_t)

    print(f"\n--- FINAL COMBINED VALIDATION MISSION METRICS ---")
    print(f"Total Steps:              {step}")
    print(f"Simulation Duration:      {sim_t:.1f} s")
    print(f"Wall Execution Time:      {total_wall_time:.2f} s")
    print(f"RGB Frames Processed:     {rgb_frames_processed}")
    print(f"Costmap Fusions:          {costmap_updates_count}")
    print(f"Waypoints Visited:        {waypoints_visited}")
    print(f"Final Waypoint ID:        {waypoints_visited[-1] if waypoints_visited else 'None'}")
    print(f"Mission State:            {mission.state.name}")
    print(f"Final Fused Pose:         ({final_fused_state.x:.2f}, {final_fused_state.y:.2f}, yaw={math.degrees(final_fused_state.yaw):.1f} deg)")
    print(f"Distance Traveled:        {final_fused_state.distance_traveled_m:.2f} m")
    print(f"Final EKF Tracking State: {final_fused_state.tracking_state}")
    print(f"Final EKF Confidence:     {final_fused_state.confidence:.2f}")

    # Assertions for Section 48 gate
    assert rgb_frames_processed > 50, "RGB perception frames were not actively consumed!"
    assert costmap_updates_count > 50, "Live perception did not update costmap!"
    assert len(waypoints_visited) == 5, f"Expected all 5 waypoints visited, got: {waypoints_visited}"
    assert mission.state == MissionState.MISSION_COMPLETE, f"Mission did not reach completion! State: {mission.state.name}"
    assert final_fused_state.distance_traveled_m >= 9.5, f"UGV did not traverse full course: {final_fused_state.distance_traveled_m:.2f}m"
    assert final_fused_state.confidence > 0.80, "EKF confidence degraded unexpectedly during clean mission"
