"""Closed-loop Software-in-the-Loop (SIL) runner for Phase 13.14 validation."""

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
from simulation.metrics.evaluator import ScenarioEvaluator
from simulation.runners.dynamic_sil_runner import DynamicSILRunner, ScenarioRunMetrics
from simulation.scenarios.dynamic.data_model import GroundTruthObstacleState
from simulation.scenarios.dynamic.phase14.data_model import (
    Phase14ScenarioConfig,
)
from simulation.scenarios.dynamic.phase14.failure_models import (
    Phase14FaultInjector,
)


class Phase14SILRunner(DynamicSILRunner):
    """Phase 13.14 SIL Runner executing multi-obstacle stress and fault injection suites."""

    def __init__(
        self,
        config: Phase14ScenarioConfig,
        enable_replanner: bool = True,
    ) -> None:
        super().__init__(config=config, enable_replanner=enable_replanner)
        self.phase14_config = config
        self.fault_injector = Phase14FaultInjector(
            fault_specs=getattr(config, "fault_specs", []),
            random_seed=config.random_seed,
        )

        # Performance and validation metrics
        self.tracking_errors_m: List[float] = []
        self.velocity_errors_m_s: List[float] = []
        self.prediction_errors_m: List[float] = []
        self.dwa_rejected_counts: List[int] = []
        self.dwa_valid_counts: List[int] = []

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step with non-invasive fault injection interceptors."""
        t_now = float(self.motion.state.timestamp)
        st = self.motion.state
        robot_state = RobotState(
            x=st.x, y=st.y, yaw=st.yaw, v=st.linear_x, omega=st.angular_z, timestamp=t_now
        )

        # 1. Goal Arrival Check
        dist_to_goal = math.hypot(robot_state.x - self.goal_pose.x, robot_state.y - self.goal_pose.y)
        if dist_to_goal <= self.goal_pose.xy_tolerance_m:
            step_record = {
                "step": len(self.telemetry),
                "sim_time": t_now,
                "robot_x": st.x,
                "robot_y": st.y,
                "robot_yaw": st.yaw,
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
                "status": "GOAL_REACHED",
            }
            self.telemetry.append(step_record)
            return step_record

        # 2. Perception Generation & Fault Interception
        raw_detections = self.observation_generator.generate_observations(
            timestamp_s=t_now,
            robot_pose=(robot_state.x, robot_state.y, robot_state.yaw),
        )
        # Intercept perception faults (dropout, stale timestamp, noise)
        intercepted_detections = self.fault_injector.intercept_observations(raw_detections, t_now)

        # Invariant K: Reject stale or out-of-order observations
        valid_detections = []
        for d in intercepted_detections:
            if d.timestamp < (t_now - 1e-4) or d.timestamp > (t_now + 1e-4):
                self.stale_observations_rejected += 1
            else:
                valid_detections.append(d)

        # 3. Tracking Update & Fault Interception
        tracked_array = self.tracker.update(valid_detections, t_now)
        confirmed_tracks = [
            t for t in tracked_array.tracks if t.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED, TrackStatus.LOST)
        ]
        # Intercept tracking faults (track loss, reset)
        confirmed_tracks = self.fault_injector.intercept_tracks(confirmed_tracks, t_now)

        # 4. Prediction Update & Fault Interception
        predictions_array = self.predictor.predict_tracks(confirmed_tracks)
        active_predictions = list(predictions_array.predictions)
        # Intercept prediction faults (dropout, horizon truncate, degrade)
        active_predictions = self.fault_injector.intercept_predictions(active_predictions, t_now)

        # 5. Dynamic Occupancy Layer Update
        self.occupancy_layer.update_predictions(active_predictions, header_timestamp=t_now)

        # 6. Global Plan Initialization
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

        # 7. Event-Driven Global Replanning with Interception
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

        # 8. Local Planning: DWA Trajectory Evaluation with Interception
        t_plan_start = time.perf_counter()
        self.local_planner_cycles += 1
        dwa_state = RobotState(
            x=st.x, y=st.y, yaw=st.yaw, v=st.linear_x, omega=st.angular_z, timestamp=0.0
        )
        cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.dwa_planner.compute_trajectory(
            current_state=dwa_state,
            global_path=self.active_global_path,
            costmap=self.world_map,
            dynamic_layer=self.occupancy_layer,
            predicted_tracks=active_predictions,
        )

        # Intercept DWA output faults (zero valid candidates, malformed output)
        cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.fault_injector.intercept_dwa_output(
            cand_traj, cand_v, cand_w, planner_status, dwa_diag, t_now
        )
        valid_count = int(dwa_diag.get("valid_trajectories", dwa_diag.get("valid_candidates", 0)))
        eval_count = int(dwa_diag.get("trajectories_evaluated", 375))
        rejected_count = max(0, eval_count - valid_count)
        self.dwa_valid_counts.append(valid_count)
        self.dwa_rejected_counts.append(rejected_count)

        # 9. Dynamic Predictive Collision Check
        if cand_traj is not None and len(cand_traj.points) > 0 and not math.isnan(cand_v):
            collision_res = self.collision_checker.check_trajectory(
                trajectory=cand_traj,
                dynamic_layer=self.occupancy_layer,
                predicted_tracks=active_predictions,
            )
        else:
            from ai.planning.dynamic.collision_checker import PredictiveCollisionResult
            from ai.planning.dynamic.occupancy_layer import DynamicRiskLevel
            collision_res = PredictiveCollisionResult(
                collision=True,
                collision_time=0.0,
                collision_position=(st.x, st.y),
                obstacle_id=None,
                minimum_clearance=0.0,
                maximum_cost=1.0,
                risk_classification=DynamicRiskLevel.LETHAL,
                unknown_horizon_encountered=False,
                samples_checked=0,
                reason="Planner failed or empty trajectory",
                status=CollisionStatus.PREDICTED_COLLISION,
            )

        # 10. Behavioral Decision
        decision = self.behavior_machine.evaluate(
            current_state=robot_state,
            best_trajectory=cand_traj,
            best_v=cand_v if not math.isnan(cand_v) else 0.0,
            best_w=cand_w if not math.isnan(cand_w) else 0.0,
            planner_status=planner_status,
            dwa_diagnostics=dwa_diag,
            predicted_tracks=active_predictions,
            global_path=self.active_global_path,
            current_time_s=t_now,
        )
        self.last_behavior = decision
        self.planner_latencies.append(time.perf_counter() - t_plan_start)

        if decision.state == BehavioralState.WAIT:
            self.wait_count += 1

        # 11. Command Arbitration & Phase 12 Supervisory Safety Clamp
        cmd_v = float(decision.command_v)
        cmd_w = float(decision.command_omega)

        # Mandatory Safety Invariant: Never propagate NaN / Inf / unhandled planner failure
        if math.isnan(cmd_v) or math.isnan(cmd_w) or math.isinf(cmd_v) or math.isinf(cmd_w):
            cmd_v = 0.0
            cmd_w = 0.0

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

        # If planner failed or produced no trajectory, command must be 0
        if cand_traj is None or planner_status != PlannerStatus.SUCCESS:
            cmd_v = 0.0
            cmd_w = 0.0

        # 12. Kinematics Propagation
        self.motion.set_command(cmd_v, cmd_w)
        next_state = self.motion.step(self.config.control_dt_s)

        d_step = math.hypot(next_state.x - self.last_pose[0], next_state.y - self.last_pose[1])
        self.travel_distance += d_step
        self.last_pose = (next_state.x, next_state.y)

        # 13. Ground-Truth Collision Check & Perception Audit (Outside autonomy loop)
        step_collision = False
        ugv_corners = self.footprint_checker.get_footprint_corners(next_state.x, next_state.y, next_state.yaw)
        step_min_clearance = float("inf")

        active_gt_obstacles: List[GroundTruthObstacleState] = []
        for model in self.motion_models:
            gt_obs = model.evaluate(t_now)
            if gt_obs is not None and gt_obs.is_active:
                active_gt_obstacles.append(gt_obs)
                obs_corners = gt_obs.get_obb_corners()
                if polygon_intersection_sat(ugv_corners, obs_corners):
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

        if step_min_clearance < self.min_clearance_observed:
            self.min_clearance_observed = step_min_clearance

        # Audit Tracking and Velocity Errors against Ground Truth (Evaluation only)
        for trk in confirmed_tracks:
            # Match to closest GT obstacle
            if active_gt_obstacles:
                best_gt = min(active_gt_obstacles, key=lambda g: math.hypot(trk.x - g.x, trk.y - g.y))
                pos_err = math.hypot(trk.x - best_gt.x, trk.y - best_gt.y)
                vel_err = math.hypot(trk.vx - best_gt.vx, trk.vy - best_gt.vy)
                self.tracking_errors_m.append(pos_err)
                self.velocity_errors_m_s.append(vel_err)

        tracked_summaries = [
            {
                "track_id": trk.track_id,
                "x": round(trk.x, 3),
                "y": round(trk.y, 3),
                "vx": round(trk.vx, 3),
                "vy": round(trk.vy, 3),
                "yaw": round(trk.bbox_yaw, 3),
                "status": trk.status if isinstance(trk.status, str) else trk.status.value,
            }
            for trk in confirmed_tracks
        ]

        active_fault_ids = [f.fault_id for f in self.fault_injector.get_active_faults(t_now)]
        step_record = {
            "step": len(self.telemetry),
            "sim_time": round(t_now, 2),
            "robot_x": round(next_state.x, 4),
            "robot_y": round(next_state.y, 4),
            "robot_yaw": round(next_state.yaw, 4),
            "robot_v": round(next_state.linear_x, 4),
            "robot_w": round(next_state.angular_z, 4),
            "cmd_v": round(cmd_v, 4),
            "cmd_w": round(cmd_w, 4),
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
            "dist_to_goal": round(dist_to_goal, 4),
            "gt_collision": step_collision,
            "min_clearance_m": round(step_min_clearance, 3) if step_min_clearance != float("inf") else 0.0,
            "tracked_details": tracked_summaries,
            "blocked_duration_s": round(self.replanner.blocked_duration, 3) if self.enable_replanner and self.replanner.blocked_duration else None,
            "active_faults": active_fault_ids,
        }
        self.telemetry.append(step_record)
        return step_record

    def run(self) -> ScenarioRunMetrics:
        """Run complete scenario and compile extended Phase 14 quantitative metrics."""
        base_metrics = super().run()

        # Compute additional Phase 14 metrics
        trk_rmse = float(np.sqrt(np.mean(np.square(self.tracking_errors_m)))) if self.tracking_errors_m else 0.0
        vel_rmse = float(np.sqrt(np.mean(np.square(self.velocity_errors_m_s)))) if self.velocity_errors_m_s else 0.0
        total_fault_dur = self.fault_injector.calculate_total_fault_duration()

        # Augment telemetry with fault events
        base_metrics.detailed_metrics = base_metrics.detailed_metrics

        # Inject extra dictionary fields for reporting
        extra = {
            "tracking_rmse": round(trk_rmse, 4),
            "velocity_rmse": round(vel_rmse, 4),
            "fault_events": len(self.fault_injector.fault_event_log),
            "fault_duration_s": round(total_fault_dur, 2),
            "fault_log": self.fault_injector.fault_event_log,
            "dwa_avg_valid_candidates": round(float(np.mean(self.dwa_valid_counts)), 1) if self.dwa_valid_counts else 0.0,
            "dwa_avg_rejected_candidates": round(float(np.mean(self.dwa_rejected_counts)), 1) if self.dwa_rejected_counts else 0.0,
        }
        # Attach to object for access
        for k, v in extra.items():
            setattr(base_metrics, k, v)

        return base_metrics
