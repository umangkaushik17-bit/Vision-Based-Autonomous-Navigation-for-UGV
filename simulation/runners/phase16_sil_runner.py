"""Closed-loop Software-in-the-Loop (SIL) runner for Phase 13.16 Full Closed-Loop Benchmark.

Extends Phase14SILRunner to provide:
1. Multi-region traversability / terrain cost injection (for B02 rough terrain and B03 static hazards).
2. Controlled realistic localization degradation with pose noise and drift (for B10).
3. Comprehensive per-step timing profiling (mean, P50, P95, P99, max latency in ms).
4. DWA candidate trajectory diagnostics (valid vs rejected candidate counts).
5. Ground-truth isolation: Navigation stack operates purely on sensor observations and noisy state,
   while ground-truth state is isolated within the post-action evaluator and collision oracle.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import BehavioralState
from ai.planning.dynamic.collision_checker import CollisionStatus
from ai.planning.dynamic.data_model import polygon_intersection_sat
from ai.planning.dynamic.tracker import TrackStatus
from simulation.metrics.evaluator import QuantitativeScenarioMetrics, ScenarioEvaluator
from simulation.runners.phase14_sil_runner import Phase14SILRunner
from simulation.runners.dynamic_sil_runner import ScenarioRunMetrics
from simulation.scenarios.dynamic.data_model import GroundTruthObstacleState
from simulation.scenarios.dynamic.phase16_scenarios import Phase16ScenarioConfig


class Phase16SILRunner(Phase14SILRunner):
    """Phase 13.16 SIL Runner executing the 12 canonical benchmark missions."""

    def __init__(
        self,
        config: Phase16ScenarioConfig,
        enable_replanner: bool = True,
    ) -> None:
        super().__init__(config=config, enable_replanner=enable_replanner)
        self.phase16_config = config
        self.rng = np.random.RandomState(config.random_seed)

        # 1. Inject terrain patches into WorldTerrainMap
        for patch in getattr(config, "terrain_patches", []):
            x_min = float(patch["x_min"])
            x_max = float(patch["x_max"])
            y_min = float(patch["y_min"])
            y_max = float(patch["y_max"])
            cost = float(patch["cost"])

            r_min, c_min = self.world_map.world_to_grid(x_min, y_min)
            r_max, c_max = self.world_map.world_to_grid(x_max, y_max)
            r_start, r_end = min(r_min, r_max), max(r_min, r_max) + 1
            c_start, c_end = min(c_min, c_max), max(c_min, c_max) + 1

            r_start = max(0, min(self.world_map.shape[0], r_start))
            r_end = max(0, min(self.world_map.shape[0], r_end))
            c_start = max(0, min(self.world_map.shape[1], c_start))
            c_end = max(0, min(self.world_map.shape[1], c_end))

            self.world_map.cost[r_start:r_end, c_start:c_end] = cost
            self.world_map.hazard[r_start:r_end, c_start:c_end] = max(0.0, (cost - 0.10) / 0.90)

        # Timing and candidate statistics
        self.step_latencies_ms: List[float] = []
        self.dwa_valid_candidates_history: List[int] = []
        self.dwa_rejected_candidates_history: List[int] = []

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step with localization noise handling and step profiling."""
        t_step_start = time.perf_counter()
        t_now = float(self.motion.state.timestamp)
        st = self.motion.state

        # Apply simulated localization degradation if configured (B10)
        loc_noise_xy = getattr(self.phase16_config, "localization_noise_xy_std", 0.0)
        loc_noise_yaw = getattr(self.phase16_config, "localization_noise_yaw_std", 0.0)
        loc_drift_rate = getattr(self.phase16_config, "localization_drift_rate_m_s", 0.0)

        if loc_noise_xy > 0.0 or loc_noise_yaw > 0.0 or loc_drift_rate > 0.0:
            drift = loc_drift_rate * t_now
            est_x = st.x + self.rng.normal(0.0, loc_noise_xy) + drift
            est_y = st.y + self.rng.normal(0.0, loc_noise_xy)
            est_yaw = st.yaw + self.rng.normal(0.0, loc_noise_yaw)
        else:
            est_x, est_y, est_yaw = st.x, st.y, st.yaw

        # Robot state exposed to planning/control (localized estimate)
        robot_state = RobotState(
            x=est_x, y=est_y, yaw=est_yaw, v=st.linear_x, omega=st.angular_z, timestamp=t_now
        )

        # 1. Goal Arrival Check (evaluated against localized estimate by controller)
        dist_to_goal = math.hypot(robot_state.x - self.goal_pose.x, robot_state.y - self.goal_pose.y)
        if dist_to_goal <= self.goal_pose.xy_tolerance_m:
            dt_step_ms = (time.perf_counter() - t_step_start) * 1000.0
            self.step_latencies_ms.append(dt_step_ms)
            step_record = {
                "step": len(self.telemetry),
                "sim_time": t_now,
                "robot_x": st.x,
                "robot_y": st.y,
                "robot_yaw": st.yaw,
                "est_x": est_x,
                "est_y": est_y,
                "est_yaw": est_yaw,
                "robot_v": 0.0,
                "robot_w": 0.0,
                "cmd_v": 0.0,
                "cmd_w": 0.0,
                "behavior_state": self.last_behavior.state.value if self.last_behavior else "PROCEED",
                "replan_triggered": False,
                "failed_replan": False,
                "emergency_stop": False,
                "deadlock": False,
                "unknown_horizon": False,
                "dynamic_collision": False,
                "tracked_count": 0,
                "predicted_count": 0,
                "collision_status": "CLEAR",
                "dist_to_goal": dist_to_goal,
                "gt_collision": False,
                "min_clearance_m": self.min_clearance_observed if self.min_clearance_observed != float("inf") else 0.0,
                "tracked_details": [],
                "prediction_details": [],
                "blocked_duration_s": self.replanner.blocked_duration if self.enable_replanner else None,
                "active_faults": [f.fault_id for f in self.fault_injector.get_active_faults(t_now)],
                "step_latency_ms": dt_step_ms,
                "status": "GOAL_REACHED",
            }
            self.telemetry.append(step_record)
            return step_record

        # 2. Perception Generation & Fault Interception
        raw_detections = self.observation_generator.generate_observations(
            timestamp_s=t_now,
            robot_pose=(robot_state.x, robot_state.y, robot_state.yaw),
        )
        intercepted_detections = self.fault_injector.intercept_observations(raw_detections, t_now)

        # Invariant K: Reject stale or out-of-order observations
        valid_detections = []
        for d in intercepted_detections:
            if d.timestamp < (t_now - 1e-4) or d.timestamp > (t_now + 1e-4):
                self.stale_observations_rejected += 1
            else:
                valid_detections.append(d)

        # 3. Tracking: MultiFrameTracker update
        tracked_array = self.tracker.update(valid_detections, t_now)
        confirmed_tracks = [
            t for t in tracked_array.tracks if t.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED, TrackStatus.LOST)
        ]

        # 4. Prediction: MotionPredictor update
        predictions_array = self.predictor.predict_tracks(confirmed_tracks)
        active_predictions = list(predictions_array.predictions)

        # 5. Dynamic Occupancy Layer update
        self.occupancy_layer.update_predictions(active_predictions, header_timestamp=t_now)

        # 6. Global Plan generation if needed
        if self.active_global_path is None or len(self.active_global_path.points) == 0:
            init_res = self.replanner.execute_replan(
                current_pose=robot_state.pose,
                goal=(self.goal_pose.x, self.goal_pose.y),
                base_map=self.world_map,
                dynamic_obstacles=confirmed_tracks,
            )
            if init_res.success and init_res.path is not None:
                self.active_global_path = init_res.path
                self.replanner.set_global_path(self.active_global_path)

        # 7. Check Event-Driven Global Replanning Triggers
        replan_triggered = False
        failed_replan = False
        if self.enable_replanner and self.active_global_path is not None:
            replan_call = lambda: self.replanner.check_and_replan(
                current_pose=robot_state.pose,
                goal=(self.goal_pose.x, self.goal_pose.y),
                base_map=self.world_map,
                behavioral_decision=self.last_behavior,
                dynamic_obstacles=confirmed_tracks,
                timestamp=t_now,
            )
            replan_res = self.fault_injector.intercept_replan_execution(replan_call, t_now)
            if replan_res.replan_triggered:
                if replan_res.success and replan_res.path is not None:
                    self.active_global_path = replan_res.path
                    self.global_replan_count += 1
                    self.replan_count += 1
                    replan_triggered = True
                else:
                    self.failed_global_replan_count += 1
                    self.failed_replan_count += 1
                    failed_replan = True

        # 8. Local Planning: DWA Trajectory Evaluation
        t_plan_start = time.perf_counter()
        self.local_planner_cycles += 1
        dwa_state = RobotState(
            x=est_x, y=est_y, yaw=est_yaw, v=st.linear_x, omega=st.angular_z, timestamp=0.0
        )
        cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.dwa_planner.compute_trajectory(
            current_state=dwa_state,
            global_path=self.active_global_path,
            costmap=self.world_map,
            dynamic_layer=self.occupancy_layer,
            predicted_tracks=active_predictions,
        )
        cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.fault_injector.intercept_dwa_output(
            cand_traj, cand_v, cand_w, planner_status, dwa_diag, t_now
        )
        t_plan_end = time.perf_counter()
        self.planner_latencies.append(t_plan_end - t_plan_start)

        # Track candidate statistics safely
        if isinstance(dwa_diag, dict):
            valid_cands = int(dwa_diag.get("valid_trajectories", dwa_diag.get("valid_candidates", 0)))
            total_cands = int(dwa_diag.get("trajectories_evaluated", 375))
        else:
            valid_cands = int(getattr(dwa_diag, "valid_trajectories_count", getattr(dwa_diag, "valid_trajectories", 0)))
            total_cands = int(getattr(dwa_diag, "total_trajectories_evaluated", getattr(dwa_diag, "trajectories_evaluated", 375)))
        rej_cands = max(0, total_cands - valid_cands)
        self.dwa_valid_candidates_history.append(valid_cands)
        self.dwa_rejected_candidates_history.append(rej_cands)

        # 9. Predictive Dynamic Collision Check
        collision_res = self.collision_checker.check_trajectory(
            trajectory=cand_traj,
            dynamic_layer=self.occupancy_layer,
            predicted_tracks=active_predictions,
        )

        # 10. Behavioral State Machine Decision
        decision = self.behavior_machine.evaluate(
            current_state=robot_state,
            best_trajectory=cand_traj,
            best_v=cand_v,
            best_w=cand_w,
            planner_status=planner_status,
            dwa_diagnostics=dwa_diag,
            predicted_tracks=active_predictions,
            global_path=self.active_global_path,
            current_time_s=t_now,
        )
        self.last_behavior = decision

        if decision.state == BehavioralState.WAIT:
            self.wait_count += 1

        # 11. Command Arbitration & Supervisory Safety Clamp
        cmd_v = float(decision.command_v)
        cmd_w = float(decision.command_omega)

        is_emergency_stop = False
        if collision_res.collision_time is not None and collision_res.collision_time < 0.5:
            cmd_v = 0.0
            cmd_w = 0.0
            is_emergency_stop = True
            self.emergency_stop_count += 1
        elif collision_res.minimum_clearance < 0.35:
            cmd_v = 0.0
            cmd_w = 0.0
            is_emergency_stop = True
            self.emergency_stop_count += 1
        elif collision_res.minimum_clearance < 0.60:
            cmd_v = min(cmd_v, 0.20)

        # 12. Kinematics Propagation (Ground-Truth Robot State)
        self.motion.set_command(cmd_v, cmd_w)
        next_state = self.motion.step(self.config.control_dt_s)

        d_step = math.hypot(next_state.x - self.last_pose[0], next_state.y - self.last_pose[1])
        self.travel_distance += d_step
        self.last_pose = (next_state.x, next_state.y)

        # 13. Ground-Truth Safety Audit (Outside navigation loop)
        step_collision = False
        ugv_corners = self.footprint_checker.get_footprint_corners(next_state.x, next_state.y, next_state.yaw)
        step_min_clearance = float("inf")

        for model in self.motion_models:
            gt_obs: Optional[GroundTruthObstacleState] = model.evaluate(t_now)
            if gt_obs is None or not gt_obs.is_active:
                continue

            obs_corners = gt_obs.get_obb_corners()
            has_overlap = polygon_intersection_sat(ugv_corners, obs_corners)
            if has_overlap:
                step_collision = True
                self.collision_events.append({
                    "timestamp": t_now,
                    "obstacle_id": gt_obs.obstacle_id,
                    "ugv_pose": (next_state.x, next_state.y, next_state.yaw),
                    "obs_pose": (gt_obs.x, gt_obs.y, gt_obs.yaw),
                })

            dist_centers = math.hypot(next_state.x - gt_obs.x, next_state.y - gt_obs.y)
            r_ugv = 0.395
            r_obs = math.hypot(gt_obs.length_m / 2.0, gt_obs.width_m / 2.0)
            approx_clearance = max(0.0, dist_centers - (r_ugv + r_obs))
            if approx_clearance < step_min_clearance:
                step_min_clearance = approx_clearance

        # Check static obstacles clearance
        for s_obs in self.config.static_obstacles:
            so_x = float(s_obs["x"])
            so_y = float(s_obs["y"])
            so_rad = float(s_obs.get("radius", 0.35))
            dist_so = math.hypot(next_state.x - so_x, next_state.y - so_y)
            so_clearance = max(0.0, dist_so - (0.395 + so_rad))
            if so_clearance < step_min_clearance:
                step_min_clearance = so_clearance
            if dist_so <= (0.325 + so_rad):  # UGV inscribed radius collision
                step_collision = True
                self.collision_events.append({
                    "timestamp": t_now,
                    "obstacle_id": "static_hazard",
                    "ugv_pose": (next_state.x, next_state.y, next_state.yaw),
                    "obs_pose": (so_x, so_y, 0.0),
                })

        if step_min_clearance < self.min_clearance_observed:
            self.min_clearance_observed = step_min_clearance

        dt_step_ms = (time.perf_counter() - t_step_start) * 1000.0
        self.step_latencies_ms.append(dt_step_ms)

        tracked_summaries = [
            {
                "track_id": trk.track_id,
                "x": trk.x,
                "y": trk.y,
                "vx": trk.vx,
                "vy": trk.vy,
                "yaw": trk.bbox_yaw,
                "status": trk.status if isinstance(trk.status, str) else trk.status.value,
            }
            for trk in confirmed_tracks
        ]
        prediction_summaries = [
            {
                "track_id": pred.track_id,
                "waypoints": [
                    (wp.timestamp, wp.x, wp.y, wp.vx, wp.vy, wp.uncertainty_radius)
                    for wp in pred.waypoints
                ],
            }
            for pred in active_predictions
        ]

        blocked_dur = self.replanner.blocked_duration if self.enable_replanner else None
        active_fault_ids = [f.fault_id for f in self.fault_injector.get_active_faults(t_now)]

        step_record = {
            "step": len(self.telemetry),
            "sim_time": t_now,
            "robot_x": next_state.x,
            "robot_y": next_state.y,
            "robot_yaw": next_state.yaw,
            "est_x": est_x,
            "est_y": est_y,
            "est_yaw": est_yaw,
            "robot_v": next_state.linear_x,
            "robot_w": next_state.angular_z,
            "cmd_v": cmd_v,
            "cmd_w": cmd_w,
            "behavior_state": decision.state.value,
            "replan_triggered": replan_triggered,
            "failed_replan": failed_replan,
            "emergency_stop": is_emergency_stop,
            "deadlock": bool(decision.deadlock),
            "unknown_horizon": bool(decision.unknown_horizon),
            "dynamic_collision": bool(decision.dynamic_collision),
            "tracked_count": len(confirmed_tracks),
            "predicted_count": len(active_predictions),
            "collision_status": collision_res.status.value,
            "dist_to_goal": dist_to_goal,
            "gt_collision": step_collision,
            "min_clearance_m": step_min_clearance,
            "valid_candidates": valid_cands,
            "rejected_candidates": rej_cands,
            "tracked_details": tracked_summaries,
            "prediction_details": prediction_summaries,
            "blocked_duration_s": blocked_dur,
            "active_faults": active_fault_ids,
            "step_latency_ms": dt_step_ms,
        }
        self.telemetry.append(step_record)
        return step_record
