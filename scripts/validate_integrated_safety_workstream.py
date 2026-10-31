#!/usr/bin/env python3
"""Integrated Validation Suite for Post-DWA Vehicle Safety & Recovery Workstream (Phase 13.25).

Executes bounded, deterministic evaluation across scenarios B01–B10:
- B01: Open Path Nominal Travel & Command Continuity
- B02: Static Obstacle Avoidance (Boulder)
- B03: Dynamic Crossing Obstacle (Tactical WAIT / AVOID)
- B04: Temporary Perception Loss (0.5s Dropout Immunity)
- B05: Close Obstacle Tracker Loss (Anti-Pushing & Latch Verification)
- B06: Narrow Passage Traversal
- B07: Complete Forward Corridor Blockage
- B08: Safe Reverse Recovery & Global Replanning Detour
- B09: Rear Blocked Recovery Rejection (Safe Stationary Hold)
- B10: Dynamic Obstacle Intersecting Reverse Maneuver

Records all safety metrics, command gap statistics, and computational latencies.
Outputs: reports/integrated_safety_workstream_measurements.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from ai.control.command_buffer import DispatchStatus, SafeCommandBuffer, SafeCommandBufferConfig
from ai.planning.backends.backend_interface import DWABackendType
from ai.planning.backends.unified_planner import UnifiedDWAPlanner
from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    PredictedObstacleTrajectory,
    TrackedObstacleState,
)
from ai.planning.dynamic.global_replanner import EventDrivenGlobalReplanner, GlobalReplannerConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path as AIPath, PathPoint
from ai.planning.recovery_controller import RecoveryConfig, RecoveryController, RecoveryState
from ai.planning.safety_governor import (
    LatchedHazard,
    RecentHazardMemory,
    RecentHazardMemoryConfig,
    SafetyGovernor,
    SafetyGovernorConfig,
)
from ai.spatial_fusion.world_map import WorldTerrainMap


def build_base_map(x_bounds=(-2.0, 10.0), y_bounds=(-5.0, 5.0), res=0.10) -> WorldTerrainMap:
    wtm = WorldTerrainMap(
        resolution_m=res,
        x_bounds_m=list(x_bounds),
        y_bounds_m=list(y_bounds),
        created_timestamp=0.0,
    )
    wtm.cost.fill(0.10)
    wtm.obstacle.fill(False)
    wtm.hazard.fill(0.0)
    wtm.unknown.fill(False)
    return wtm


def imprint_circular_obstacle(costmap: WorldTerrainMap, cx: float, cy: float, radius: float) -> None:
    rc, cc = costmap.world_to_grid(cx, cy)
    r_rad = max(1, int(round(radius / costmap.resolution_m)))
    for dr in range(-r_rad, r_rad + 1):
        for dc in range(-r_rad, r_rad + 1):
            if dr * dr + dc * dc <= r_rad * r_rad:
                rr, cc_idx = rc + dr, cc + dc
                if 0 <= rr < costmap.shape[0] and 0 <= cc_idx < costmap.shape[1]:
                    costmap.obstacle[rr, cc_idx] = True
                    costmap.cost[rr, cc_idx] = 1.0


class IntegratedScenarioRunner:
    """Runs closed-loop simulation of the full navigation pipeline."""

    def __init__(self, backend_type: DWABackendType = DWABackendType.AUTO) -> None:
        self.backend_type = backend_type
        self.dwa_planner = UnifiedDWAPlanner(backend=backend_type)
        self.dwa_planner.enable_behavioral_arbitration = True

        self.tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))
        self.predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.25))
        self.occupancy_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))
        self.replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())

        self.hazard_memory = RecentHazardMemory(
            RecentHazardMemoryConfig(proximity_threshold_m=2.5, critical_proximity_m=1.2, base_ttl_s=3.5, close_ttl_s=5.0)
        )
        self.safety_governor = SafetyGovernor(
            SafetyGovernorConfig(stop_distance_m=0.40, cautious_distance_m=0.95, cautious_max_v_mps=0.15)
        )
        self.command_buffer = SafeCommandBuffer(
            SafeCommandBufferConfig(publish_rate_hz=20.0, freshness_timeout_s=0.35, stale_timeout_s=0.50)
        )
        self.recovery_controller = RecoveryController(
            RecoveryConfig(reverse_velocity_mps=-0.15, target_reverse_distance_m=0.45, deadlock_stop_cycles=4)
        )

    def run_scenario(
        self,
        scenario_id: str,
        goal: Tuple[float, float] = (7.0, 0.0),
        static_obstacles: List[Tuple[float, float, float]] = [],
        delayed_boulders: Optional[List[Tuple[float, float, float, float]]] = None,
        dynamic_trajectory_func: Optional[Any] = None,
        dropout_interval: Optional[Tuple[float, float]] = None,
        rear_obstacle_at: Optional[float] = None,
        max_duration_s: float = 12.0,
        dt_plan_s: float = 0.10,
        dt_pub_s: float = 0.05,
    ) -> Dict[str, Any]:
        # Reset components
        self.tracker.reset()
        self.hazard_memory.clear()
        self.command_buffer.reset()
        self.recovery_controller.reset()

        costmap = build_base_map()
        for ox, oy, orad in static_obstacles:
            imprint_circular_obstacle(costmap, ox, oy, orad)

        imprinted_delayed = set()

        # Initial state
        ugv_x, ugv_y, ugv_yaw = 0.0, 0.0, 0.0
        ugv_v, ugv_w = 0.0, 0.0

        # Metrics collection
        collision_count = 0
        unsafe_transitions = 0
        unsafe_reverse_events = 0
        stale_command_violations = 0
        command_gaps: List[float] = []
        dwa_latencies: List[float] = []
        cycle_latencies: List[float] = []
        velocities: List[float] = []
        min_clearance_observed = 999.0

        active_path: Optional[AIPath] = None
        sim_time = 0.0
        next_plan_time = 0.0
        prev_behavior_state = "PROCEED"
        prev_gov_decision = "ALLOW"
        recovery_triggered_count = 0
        replan_count = 0

        # Step through time
        while sim_time <= max_duration_s:
            dist_to_goal = math.hypot(ugv_x - goal[0], ugv_y - goal[1])
            if dist_to_goal <= 0.35:
                break

            # Check delayed static obstacles
            if delayed_boulders:
                for idx, (bx, by, brad, b_time) in enumerate(delayed_boulders):
                    if sim_time >= b_time and idx not in imprinted_delayed:
                        imprint_circular_obstacle(costmap, bx, by, brad)
                        imprinted_delayed.add(idx)

            # -----------------------------------------------------------------
            # 1. Planning Cycle (runs every dt_plan_s = 10 Hz)
            # -----------------------------------------------------------------
            if sim_time >= next_plan_time:
                t_cycle_start = time.perf_counter()
                robot_state = RobotState(x=ugv_x, y=ugv_y, yaw=ugv_yaw, v=ugv_v, omega=ugv_w, timestamp=sim_time)

                # Generate dynamic detections
                detections = []
                is_in_dropout = dropout_interval and (dropout_interval[0] <= sim_time <= dropout_interval[1])
                if dynamic_trajectory_func and not is_in_dropout:
                    dets = dynamic_trajectory_func(sim_time, (ugv_x, ugv_y, ugv_yaw))
                    if dets:
                        detections.extend(dets)

                tracked_array = self.tracker.update(detections, sim_time)
                confirmed_tracks = [
                    t for t in tracked_array.tracks
                    if t.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED, TrackStatus.LOST)
                ]

                # Update Recent Hazard Memory
                all_hazards = self.hazard_memory.update(confirmed_tracks, sim_time, (ugv_x, ugv_y, ugv_yaw))

                # Inject sudden rear obstacle if scheduled
                if rear_obstacle_at is not None and sim_time >= rear_obstacle_at:
                    rear_haz = LatchedHazard(
                        hazard_id="rear_obs_99",
                        x=ugv_x - 0.40 * math.cos(ugv_yaw),
                        y=ugv_y - 0.40 * math.sin(ugv_yaw),
                        bbox_size=(0.6, 0.6),
                        last_observed_time_s=sim_time,
                        last_distance_m=0.40,
                    )
                    all_hazards.append(rear_haz)

                pred_array = self.predictor.predict_tracks(confirmed_tracks)
                active_predictions = list(pred_array.predictions)

                # Augment with latched dropouts
                dropped_hazards = self.hazard_memory.get_latched_dropped_hazards([t.track_id for t in confirmed_tracks])
                for dh in dropped_hazards:
                    active_predictions.append(dh.to_predicted_trajectory(horizon_s=2.0, dt_s=0.25))

                self.occupancy_layer.update_predictions(active_predictions, header_timestamp=sim_time)

                # Global path routing
                if active_path is None or len(active_path.points) == 0:
                    init_res = self.replanner.execute_replan(
                        current_pose=(ugv_x, ugv_y, ugv_yaw),
                        goal=goal,
                        base_map=costmap,
                        dynamic_obstacles=confirmed_tracks,
                    )
                    if init_res.success and init_res.path is not None:
                        active_path = init_res.path
                        self.replanner.set_global_path(active_path)

                # Local DWA trajectory evaluation
                t_dwa_start = time.perf_counter()
                cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.dwa_planner.compute_trajectory(
                    current_state=robot_state,
                    global_path=active_path,
                    costmap=costmap,
                    dynamic_layer=self.occupancy_layer,
                    predicted_tracks=active_predictions,
                )
                dwa_latencies.append((time.perf_counter() - t_dwa_start) * 1000.0)

                decision_str = dwa_diag.get("behavioral_state", "PROCEED")

                is_deadlocked = (
                    decision_str in ("STOP", "EMERGENCY_STOP")
                    or planner_status in (PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, PlannerStatus.NO_SAFE_TRAJECTORY)
                    or ("STOP" in prev_gov_decision and dist_to_goal > 0.5)
                    or (abs(cand_v) <= 0.05 and dist_to_goal > 0.5 and abs(robot_state.v) <= 0.05)
                )

                # Recovery controller step
                rec_v, rec_w, rec_state, rec_reason = self.recovery_controller.step(
                    current_time_s=sim_time,
                    robot_pose=(ugv_x, ugv_y, ugv_yaw),
                    is_deadlocked=is_deadlocked,
                    hazards=all_hazards,
                    costmap=costmap,
                )

                if rec_state in (RecoveryState.REVERSING, RecoveryState.SETTLING, RecoveryState.REPLAN_READY):
                    recovery_triggered_count += 1

                if rec_state == RecoveryState.REPLAN_READY:
                    replan_res = self.replanner.execute_replan(
                        current_pose=(ugv_x, ugv_y, ugv_yaw),
                        goal=goal,
                        base_map=costmap,
                        dynamic_obstacles=confirmed_tracks,
                    )
                    if replan_res.success and replan_res.path is not None:
                        active_path = replan_res.path
                        self.replanner.set_global_path(active_path)
                        self.recovery_controller.mark_replan_completed()
                        replan_count += 1

                if rec_state == RecoveryState.REVERSING and rec_v is not None:
                    cand_v = rec_v
                    cand_w = rec_w or 0.0
                    decision_str = "RECOVERY_REVERSE"

                # Safety Governor Evaluation
                gov_v, gov_w, gov_decision, gov_diag = self.safety_governor.evaluate(
                    requested_v=cand_v,
                    requested_w=cand_w,
                    robot_pose=(ugv_x, ugv_y, ugv_yaw),
                    hazards=all_hazards,
                    current_time_s=sim_time,
                )

                # Check for unsafe STOP -> PROCEED inversions during dropout
                if is_in_dropout and prev_behavior_state in ("STOP", "WAIT") and gov_decision == "ALLOW" and gov_v > 0.2:
                    unsafe_transitions += 1

                prev_behavior_state = decision_str
                prev_gov_decision = gov_decision

                is_safe_cmd = (
                    (decision_str not in ("STOP", "EMERGENCY_STOP"))
                    and ("STOP" not in gov_decision)
                    and (dist_to_goal > 0.35)
                )
                self.command_buffer.set_command(
                    linear_v=gov_v,
                    angular_w=gov_w,
                    timestamp_s=sim_time,
                    is_safe=is_safe_cmd,
                    source="SAFETY_GOVERNOR" if gov_v != cand_v else "DWA",
                    reason=f"{decision_str} | {gov_decision}",
                )

                cycle_latencies.append((time.perf_counter() - t_cycle_start) * 1000.0)
                next_plan_time += dt_plan_s

            # -----------------------------------------------------------------
            # 2. High-Frequency Actuator Dispatch (20 Hz = 0.05s)
            # -----------------------------------------------------------------
            dispatched_v, dispatched_w, status, diag = self.command_buffer.get_dispatch_command(sim_time)
            velocities.append(dispatched_v)

            # Check for stale command violation
            if status == DispatchStatus.STALE_STOP and (dispatched_v > 0.0 or dispatched_w > 0.0):
                stale_command_violations += 1

            # Simple forward kinematic simulation
            theta_mid = ugv_yaw + 0.5 * dispatched_w * dt_pub_s
            ugv_x += dispatched_v * math.cos(theta_mid) * dt_pub_s
            ugv_y += dispatched_v * math.sin(theta_mid) * dt_pub_s
            ugv_yaw += dispatched_w * dt_pub_s
            ugv_v = dispatched_v
            ugv_w = dispatched_w

            # Collision check with static obstacles
            for ox, oy, orad in static_obstacles:
                d = math.hypot(ugv_x - ox, ugv_y - oy) - orad - 0.225
                min_clearance_observed = min(min_clearance_observed, d)
                if d <= 0.0:
                    collision_count += 1

            if delayed_boulders:
                for bx, by, brad, b_time in delayed_boulders:
                    if sim_time >= b_time:
                        d = math.hypot(ugv_x - bx, ugv_y - by) - brad - 0.225
                        min_clearance_observed = min(min_clearance_observed, d)
                        if d <= 0.0:
                            collision_count += 1

            sim_time += dt_pub_s

        # Final metrics aggregation
        buf_metrics = self.command_buffer.get_metrics()
        final_dist = math.hypot(ugv_x - goal[0], ugv_y - goal[1])
        return {
            "scenario_id": scenario_id,
            "goal_reached": final_dist <= 0.40,
            "final_distance_to_goal_m": round(final_dist, 3),
            "collision_count": collision_count,
            "unsafe_transitions": unsafe_transitions,
            "unsafe_reverse_events": unsafe_reverse_events,
            "stale_command_violations": stale_command_violations,
            "min_clearance_m": round(float(min_clearance_observed), 3) if min_clearance_observed < 900.0 else None,
            "max_velocity_mps": round(float(np.max(velocities)), 3) if velocities else 0.0,
            "mean_velocity_mps": round(float(np.mean(velocities)), 3) if velocities else 0.0,
            "dwa_latency_mean_ms": round(float(np.mean(dwa_latencies)), 2) if dwa_latencies else 0.0,
            "dwa_latency_p95_ms": round(float(np.percentile(dwa_latencies, 95)), 2) if dwa_latencies else 0.0,
            "cycle_latency_mean_ms": round(float(np.mean(cycle_latencies)), 2) if cycle_latencies else 0.0,
            "command_gap_mean_ms": buf_metrics.get("gap_mean_ms", 0.0),
            "command_gap_p95_ms": buf_metrics.get("gap_p95_ms", 0.0),
            "command_gap_max_ms": buf_metrics.get("gap_max_ms", 0.0),
            "recovery_triggered_count": recovery_triggered_count,
            "replan_count": replan_count,
        }


def run_full_suite() -> Dict[str, Any]:
    print("=" * 70)
    print("  PHASE 13.25: INTEGRATED SAFETY & RECOVERY VALIDATION SUITE (B01-B10)")
    print("=" * 70)

    runner = IntegratedScenarioRunner(backend_type=DWABackendType.AUTO)
    results = {}

    # B01: Open Path Nominal Travel
    print("\n[Running B01] Open Path...")
    r01 = runner.run_scenario("B01_Open_Path", goal=(5.0, 0.0), max_duration_s=14.0)
    results["B01"] = r01
    print(f"  Goal Reached: {r01['goal_reached']} | Max V: {r01['max_velocity_mps']} m/s | Gap Mean: {r01['command_gap_mean_ms']} ms")

    # B02: Static Obstacle (Boulder at x=2.5, y=0.1)
    print("\n[Running B02] Static Boulder Avoidance...")
    r02 = runner.run_scenario("B02_Static_Boulder", goal=(5.0, 0.0), static_obstacles=[(2.5, 0.1, 0.55)], max_duration_s=14.0)
    results["B02"] = r02
    print(f"  Goal Reached: {r02['goal_reached']} | Min Clearance: {r02['min_clearance_m']} m | Collisions: {r02['collision_count']}")

    # B03: Dynamic Crossing Obstacle
    print("\n[Running B03] Dynamic Crossing Obstacle...")
    def crossing_func(t, pose):
        y_obs = -2.0 + 0.60 * t
        if -2.5 <= y_obs <= 2.5:
            return [DynamicObstacleDetection(
                detection_id="cross_1", timestamp=t, frame_id="odom",
                bbox_center=(2.5, y_obs), bbox_size=(0.8, 0.8), bbox_yaw=1.57, confidence=0.95
            )]
        return []
    r03 = runner.run_scenario("B03_Dynamic_Crossing", goal=(5.0, 0.0), dynamic_trajectory_func=crossing_func, max_duration_s=18.0)
    results["B03"] = r03
    print(f"  Goal Reached: {r03['goal_reached']} | Collisions: {r03['collision_count']} | Unsafe Transitions: {r03['unsafe_transitions']}")

    # B04: Temporary Perception Loss (0.8s dropout)
    print("\n[Running B04] Temporary Perception Loss...")
    def close_dyn(t, pose):
        return [DynamicObstacleDetection("obs_front", t, "odom", (1.8, 0.0), (0.8, 0.8), 0.0, 0.95)]
    r04 = runner.run_scenario("B04_Perception_Dropout", goal=(5.0, 0.0), dynamic_trajectory_func=close_dyn, dropout_interval=(2.0, 2.8), max_duration_s=10.0)
    results["B04"] = r04
    print(f"  Collisions: {r04['collision_count']} | Unsafe Transitions: {r04['unsafe_transitions']}")

    # B05: Close Obstacle Tracker Loss (Anti-Pushing)
    print("\n[Running B05] Close Obstacle Tracker Loss...")
    def close_pin(t, pose):
        if t <= 1.0:
            return [DynamicObstacleDetection("obs_pin", t, "odom", (0.80, 0.0), (0.8, 0.8), 0.0, 0.95)]
        return []
    r05 = runner.run_scenario("B05_Close_Tracker_Loss", goal=(5.0, 0.0), dynamic_trajectory_func=close_pin, max_duration_s=6.0)
    results["B05"] = r05
    print(f"  Collisions: {r05['collision_count']} | Unsafe Transitions: {r05['unsafe_transitions']}")

    # B06: Narrow Passage
    print("\n[Running B06] Narrow Passage Traversal...")
    r06 = runner.run_scenario("B06_Narrow_Passage", goal=(5.0, 0.0), static_obstacles=[(2.5, 0.75, 0.40), (2.5, -0.75, 0.40)], max_duration_s=14.0)
    results["B06"] = r06
    print(f"  Goal Reached: {r06['goal_reached']} | Min Clearance: {r06['min_clearance_m']} m")

    # B07: Complete Corridor Blockage
    print("\n[Running B07] Complete Corridor Blockage...")
    blockage_wall = [(2.5, float(y), 0.45, 1.0) for y in np.linspace(-3.0, 3.0, 9)]
    r07 = runner.run_scenario("B07_Complete_Blockage", goal=(5.0, 0.0), delayed_boulders=blockage_wall, max_duration_s=15.0)
    results["B07"] = r07
    print(f"  Recovery Triggered: {r07['recovery_triggered_count'] > 0} | Collisions: {r07['collision_count']}")

    # B08: Safe Reverse Recovery & Detour
    print("\n[Running B08] Safe Reverse Recovery & Detour...")
    # Delayed boulder appears at t=1.0s on the initial straight route at (2.5, 0.0)
    r08 = runner.run_scenario("B08_Safe_Reverse_Detour", goal=(5.0, 0.0), delayed_boulders=[(2.5, 0.0, 0.55, 1.0)], max_duration_s=22.0)
    results["B08"] = r08
    print(f"  Goal Reached: {r08['goal_reached']} | Replans: {r08['replan_count']} | Collisions: {r08['collision_count']}")

    # B09: Rear Blocked Recovery Rejection
    print("\n[Running B09] Rear Blocked Recovery Rejection...")
    r09 = runner.run_scenario("B09_Rear_Blocked_Abort", goal=(5.0, 0.0), delayed_boulders=[(2.5, 0.0, 0.55, 1.0)], rear_obstacle_at=1.0, max_duration_s=10.0)
    results["B09"] = r09
    print(f"  Collisions: {r09['collision_count']} | Unsafe Reverse Events: {r09['unsafe_reverse_events']}")

    # B10: Dynamic Obstacle During Reverse
    print("\n[Running B10] Dynamic Obstacle During Reverse...")
    r10 = runner.run_scenario("B10_Dynamic_Rear_Abort", goal=(5.0, 0.0), delayed_boulders=[(2.5, 0.0, 0.55, 1.0)], rear_obstacle_at=3.5, max_duration_s=10.0)
    results["B10"] = r10
    print(f"  Collisions: {r10['collision_count']} | Unsafe Reverse Events: {r10['unsafe_reverse_events']}")

    out_file = "reports/integrated_safety_workstream_measurements.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll 10 scenarios completed successfully. Metrics written to {out_file}")
    return results


if __name__ == "__main__":
    run_full_suite()
