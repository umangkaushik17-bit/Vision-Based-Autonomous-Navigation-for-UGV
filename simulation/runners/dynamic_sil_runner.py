"""Deterministic closed-loop Software-in-the-Loop (SIL) simulation runner for Phase 13."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, PlannerStatus, RobotState, VehicleKinematics
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    polygon_intersection_sat,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyLayer,
)
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap
from simulation.motion_controller import RobotState2D, UGVMotionController
from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    GroundTruthObstacleState,
)
from simulation.scenarios.dynamic.motion_models import (
    BaseObstacleMotionModel,
    create_motion_model,
)
from simulation.scenarios.dynamic.observation_model import DynamicObservationGenerator


from simulation.metrics.evaluator import QuantitativeScenarioMetrics, ScenarioEvaluator


@dataclass
class ScenarioRunMetrics:
    """Consolidated quantitative outcome metrics for a completed SIL scenario run."""

    scenario_id: str
    scenario_name: str
    mission_outcome: str  # GOAL_REACHED, INCOMPLETE
    safety_outcome: str   # SAFE, COLLISION
    termination_reason: str  # GOAL_REACHED, HORIZON_BOUNDED_TIMEOUT, COLLISION
    mission_success: bool
    configured_duration_s: float
    actual_simulation_time_s: float
    steps_executed: int
    exit_reason: str
    goal_position_error_m: float
    goal_tolerance_m: float
    collision_count: int
    collision_detected: bool
    minimum_clearance_m: float
    local_planner_cycles: int
    global_replan_count: int
    failed_global_replan_count: int
    proceed_count: int = 0
    avoid_count: int = 0
    wait_count: int = 0
    stop_count: int = 0
    deadlock_count: int = 0
    blocked_duration_s: Optional[float] = None
    wall_clock_duration_s: float = 0.0
    real_time_factor: float = 0.0
    travel_distance_m: float = 0.0
    average_speed_m_s: float = 0.0
    max_speed_m_s: float = 0.0
    planner_latency_mean_s: float = 0.0
    planner_latency_max_s: float = 0.0
    emergency_stop_count: int = 0
    stale_observations_rejected: int = 0
    # Legacy fields retained for backward compatibility
    replan_count: int = 0
    failed_replan_count: int = 0
    min_clearance_m: float = 0.0
    total_time_s: float = 0.0
    goal_distance_error_m: float = 0.0
    detailed_metrics: Optional[QuantitativeScenarioMetrics] = None
    telemetry: List[Dict[str, Any]] = field(default_factory=list)
    collision_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "configured_duration_s": round(self.configured_duration_s, 2),
            "actual_simulation_time_s": round(self.actual_simulation_time_s, 2),
            "steps_executed": self.steps_executed,
            "exit_reason": self.exit_reason,
            "mission_outcome": self.mission_outcome,
            "safety_outcome": self.safety_outcome,
            "termination_reason": self.termination_reason,
            "mission_success": self.mission_success,
            "goal_position_error_m": round(self.goal_position_error_m, 4),
            "goal_tolerance_m": round(self.goal_tolerance_m, 3),
            "collision_count": self.collision_count,
            "collision_detected": self.collision_detected,
            "minimum_clearance_m": round(self.minimum_clearance_m, 3),
            "local_planner_cycles": self.local_planner_cycles,
            "global_replan_count": self.global_replan_count,
            "failed_global_replan_count": self.failed_global_replan_count,
            "proceed_count": self.proceed_count,
            "avoid_count": self.avoid_count,
            "wait_count": self.wait_count,
            "stop_count": self.stop_count,
            "deadlock_count": self.deadlock_count,
            "blocked_duration_s": round(self.blocked_duration_s, 3) if self.blocked_duration_s is not None else None,
            "wall_clock_duration_s": round(self.wall_clock_duration_s, 2),
            "real_time_factor": round(self.real_time_factor, 3),
            "travel_distance_m": round(self.travel_distance_m, 2),
            "average_speed_m_s": round(self.average_speed_m_s, 3),
            "max_speed_m_s": round(self.max_speed_m_s, 3),
            "planner_latency_mean_s": round(self.planner_latency_mean_s, 4),
            "planner_latency_max_s": round(self.planner_latency_max_s, 4),
            "emergency_stop_count": self.emergency_stop_count,
            "stale_observations_rejected": self.stale_observations_rejected,
            # Legacy compatibility fields
            "total_time_s": round(self.total_time_s, 2),
            "min_clearance_m": round(self.min_clearance_m, 3),
            "goal_distance_error_m": round(self.goal_distance_error_m, 3),
            "replan_count": self.replan_count,
            "failed_replan_count": self.failed_replan_count,
            "collision_details": self.collision_details,
            "detailed_metrics": self.detailed_metrics.to_dict() if self.detailed_metrics is not None else None,
        }


class DynamicSILRunner:
    """Full closed-loop dynamic SIL simulation runner executing Phase 13 navigation stack."""

    def __init__(
        self,
        config: DynamicScenarioConfig,
        enable_replanner: bool = True,
    ) -> None:
        self.config = config
        self.enable_replanner = enable_replanner

        # 1. Motion Controller & Kinematics
        self.motion = UGVMotionController(
            track_width_m=0.48,
            wheel_radius_m=0.125,
            max_linear_vel_m_s=0.60,
            max_angular_vel_rad_s=0.80,
        )
        sx, sy, syaw = config.start_pose
        self.motion.state = RobotState2D(x=sx, y=sy, yaw=syaw, linear_x=0.0, angular_z=0.0, timestamp=0.0)

        # 2. World Terrain Map
        x_b, y_b = config.world_bounds_m
        self.world_map = WorldTerrainMap(
            resolution_m=0.10,
            x_bounds_m=x_b,
            y_bounds_m=y_b,
            created_timestamp=0.0,
        )
        self.world_map.cost.fill(0.10)
        self.world_map.obstacle.fill(False)
        self.world_map.hazard.fill(0.0)
        self.world_map.unknown.fill(False)

        # Inject static obstacles if present
        for obs in config.static_obstacles:
            ox = float(obs.get("x", 0.0))
            oy = float(obs.get("y", 0.0))
            rad = float(obs.get("radius", 0.35))
            r_c, c_c = self.world_map.world_to_grid(ox, oy)
            r_rad = max(1, int(round(rad / self.world_map.resolution_m)))
            for dr in range(-r_rad, r_rad + 1):
                for dc in range(-r_rad, r_rad + 1):
                    if dr * dr + dc * dc <= r_rad * r_rad:
                        rr, cc = r_c + dr, c_c + dc
                        if 0 <= rr < self.world_map.shape[0] and 0 <= cc < self.world_map.shape[1]:
                            self.world_map.obstacle[rr, cc] = True
                            self.world_map.cost[rr, cc] = 1.00

        # 3. Dynamic Obstacle Models
        self.motion_models: List[BaseObstacleMotionModel] = [
            create_motion_model(spec) for spec in config.obstacles
        ]

        # 4. Observation Generator (strictly isolated from ground truth state)
        self.observation_generator = DynamicObservationGenerator(
            motion_models=self.motion_models,
            frame_id="odom",
            random_seed=config.random_seed,
        )

        # 5. Core Phase 13 Dynamic Planning Stack
        self.tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))
        self.predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.10))
        self.occupancy_layer = DynamicOccupancyLayer(DynamicOccupancyConfig(frame_id="odom", resolution_m=0.10))
        self.collision_checker = PredictiveCollisionChecker(
            PredictiveCollisionConfig(vehicle_length_m=0.65, vehicle_width_m=0.45, collision_margin_m=0.05)
        )
        self.global_planner = AStarPlanner()
        self.dwa_planner = DWAPlanner()
        self.behavior_machine = BehavioralStateMachine(BehavioralConfig())
        self.replanner = EventDrivenGlobalReplanner(GlobalReplannerConfig())
        self.footprint_checker = FootprintCollisionChecker(VehicleKinematics(length_m=0.65, width_m=0.45))

        # Internal Execution State
        self.active_global_path: Optional[Path] = None
        self.last_behavior: Optional[BehavioralDecision] = None
        self.goal_pose = GoalPose(
            x=config.goal_pose[0],
            y=config.goal_pose[1],
            yaw=config.goal_pose[2] if len(config.goal_pose) > 2 else None,
            xy_tolerance_m=getattr(config, "goal_tolerance_m", 0.25),
        )

        # Telemetry & Evaluation
        self.telemetry: List[Dict[str, Any]] = []
        self.collision_events: List[Dict[str, Any]] = []
        self.min_clearance_observed = float("inf")
        self.emergency_stop_count = 0
        self.wait_count = 0
        self.local_planner_cycles = 0
        self.global_replan_count = 0
        self.failed_global_replan_count = 0
        self.replan_count = 0  # legacy alias
        self.failed_replan_count = 0  # legacy alias
        self.stale_observations_rejected = 0
        self.travel_distance = 0.0
        self.planner_latencies: List[float] = []
        self.last_pose = (self.motion.state.x, self.motion.state.y)

    def step(self) -> Dict[str, Any]:
        """Execute a single deterministic simulation step forward by control_dt_s."""
        t_now = float(self.motion.state.timestamp)
        st = self.motion.state
        robot_state = RobotState(
            x=st.x, y=st.y, yaw=st.yaw, v=st.linear_x, omega=st.angular_z, timestamp=t_now
        )

        # 1. Check Goal Arrival
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
                "status": "GOAL_REACHED",
            }
            self.telemetry.append(step_record)
            return step_record

        # 2. Perception: Generate isolated sensor observations
        raw_detections = self.observation_generator.generate_observations(
            timestamp_s=t_now,
            robot_pose=(robot_state.x, robot_state.y, robot_state.yaw),
        )

        # Invariant K: Reject stale or out-of-order observations
        valid_detections = []
        for d in raw_detections:
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
            replan_res = self.replanner.check_and_replan(
                current_pose=robot_state.pose,
                goal=(self.goal_pose.x, self.goal_pose.y),
                base_map=self.world_map,
                behavioral_decision=self.last_behavior,
                dynamic_obstacles=confirmed_tracks,
                timestamp=t_now,
            )
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
            x=st.x, y=st.y, yaw=st.yaw, v=st.linear_x, omega=st.angular_z, timestamp=0.0
        )
        cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.dwa_planner.compute_trajectory(
            current_state=dwa_state,
            global_path=self.active_global_path,
            costmap=self.world_map,
            dynamic_layer=self.occupancy_layer,
            predicted_tracks=active_predictions,
        )

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
        self.planner_latencies.append(time.perf_counter() - t_plan_start)

        if decision.state == BehavioralState.WAIT:
            self.wait_count += 1

        # 11. Command Arbitration & Phase 12 Supervisory Safety Clamp
        cmd_v = float(decision.command_v)
        cmd_w = float(decision.command_omega)

        is_emergency_stop = False
        # If immediate proximity to predicted or static obstacle breached
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

        # 12. Kinematics Propagation
        self.motion.set_command(cmd_v, cmd_w)
        next_state = self.motion.step(self.config.control_dt_s)

        # Update travel distance
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
            # Exact SAT polygon collision check
            has_overlap = polygon_intersection_sat(ugv_corners, obs_corners)
            if has_overlap:
                step_collision = True
                self.collision_events.append({
                    "timestamp": t_now,
                    "obstacle_id": gt_obs.obstacle_id,
                    "ugv_pose": (next_state.x, next_state.y, next_state.yaw),
                    "obs_pose": (gt_obs.x, gt_obs.y, gt_obs.yaw),
                })

            # Calculate center-to-center distance minus bounding radii for clearance
            dist_centers = math.hypot(next_state.x - gt_obs.x, next_state.y - gt_obs.y)
            r_ugv = 0.395  # circumscribed radius
            r_obs = math.hypot(gt_obs.length_m / 2.0, gt_obs.width_m / 2.0)
            approx_clearance = max(0.0, dist_centers - (r_ugv + r_obs))
            if approx_clearance < step_min_clearance:
                step_min_clearance = approx_clearance

        if step_min_clearance < self.min_clearance_observed:
            self.min_clearance_observed = step_min_clearance

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
        step_record = {
            "step": len(self.telemetry),
            "sim_time": t_now,
            "robot_x": next_state.x,
            "robot_y": next_state.y,
            "robot_yaw": next_state.yaw,
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
            "tracked_details": tracked_summaries,
            "prediction_details": prediction_summaries,
            "blocked_duration_s": blocked_dur,
        }
        self.telemetry.append(step_record)
        return step_record

    def run(self) -> ScenarioRunMetrics:
        """Run the complete scenario until goal reached, collision, or timeout."""
        t_start_wall = time.perf_counter()
        max_steps = int(round(self.config.max_duration_s / self.config.control_dt_s))
        exit_reason = "HORIZON_BOUNDED_TIMEOUT"

        for _ in range(max_steps):
            rec = self.step()
            if rec.get("status") == "GOAL_REACHED":
                exit_reason = "GOAL_REACHED"
                break

            if rec.get("gt_collision", False):
                exit_reason = "COLLISION"
                break

        wall_duration = time.perf_counter() - t_start_wall

        # Calculate final metrics
        final_st = self.motion.state
        dist_to_goal = math.hypot(final_st.x - self.goal_pose.x, final_st.y - self.goal_pose.y)
        collision_count = len(self.collision_events)

        # Determine canonical mission and safety outcomes
        is_goal = (exit_reason == "GOAL_REACHED") or (dist_to_goal <= self.goal_pose.xy_tolerance_m)

        if collision_count > 0:
            safety_outcome = "COLLISION"
            mission_outcome = "INCOMPLETE"
            termination_reason = "COLLISION"
            mission_success = False
            exit_reason = "COLLISION"
        elif is_goal:
            safety_outcome = "SAFE"
            mission_outcome = "GOAL_REACHED"
            termination_reason = "GOAL_REACHED"
            mission_success = True
            exit_reason = "GOAL_REACHED"
        else:
            safety_outcome = "SAFE"
            mission_outcome = "INCOMPLETE"
            termination_reason = "HORIZON_BOUNDED_TIMEOUT"
            mission_success = False
            exit_reason = "HORIZON_BOUNDED_TIMEOUT"

        speeds = [t["robot_v"] for t in self.telemetry]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        mean_lat = float(np.mean(self.planner_latencies)) if self.planner_latencies else 0.0
        max_lat = float(np.max(self.planner_latencies)) if self.planner_latencies else 0.0
        rtf = (final_st.timestamp / wall_duration) if wall_duration > 0 else 0.0

        # Behavior counts
        proceed_count = sum(1 for t in self.telemetry if t.get("behavior_state") == "PROCEED")
        avoid_count = sum(1 for t in self.telemetry if t.get("behavior_state") == "AVOID")
        wait_count = sum(1 for t in self.telemetry if t.get("behavior_state") == "WAIT")
        stop_count = sum(1 for t in self.telemetry if t.get("behavior_state") == "STOP")
        deadlock_count = sum(1 for t in self.telemetry if t.get("deadlock", False))
        blocked_durations = [t.get("blocked_duration_s") for t in self.telemetry if t.get("blocked_duration_s") is not None]
        max_blocked = max(blocked_durations) if blocked_durations else None

        # Run post-run ground-truth metric evaluation
        evaluator = ScenarioEvaluator(self.motion_models)
        detailed = evaluator.evaluate(
            telemetry=self.telemetry,
            goal_pose=(self.goal_pose.x, self.goal_pose.y, self.goal_pose.yaw),
            dt_s=self.config.control_dt_s,
            wall_clock_duration_s=wall_duration,
            goal_tolerance_m=self.goal_pose.xy_tolerance_m,
            termination_reason=termination_reason,
            configured_duration_s=self.config.max_duration_s,
            local_planner_cycles=self.local_planner_cycles,
            global_replan_count=self.global_replan_count,
            failed_global_replan_count=self.failed_global_replan_count,
        )

        return ScenarioRunMetrics(
            scenario_id=self.config.scenario_id,
            scenario_name=self.config.name,
            mission_outcome=mission_outcome,
            safety_outcome=safety_outcome,
            termination_reason=termination_reason,
            mission_success=mission_success,
            configured_duration_s=self.config.max_duration_s,
            actual_simulation_time_s=final_st.timestamp,
            steps_executed=len(self.telemetry),
            exit_reason=exit_reason,
            goal_position_error_m=dist_to_goal,
            goal_tolerance_m=self.goal_pose.xy_tolerance_m,
            collision_count=collision_count,
            collision_detected=collision_count > 0,
            minimum_clearance_m=self.min_clearance_observed if self.min_clearance_observed != float("inf") else 0.0,
            local_planner_cycles=self.local_planner_cycles,
            global_replan_count=self.global_replan_count,
            failed_global_replan_count=self.failed_global_replan_count,
            proceed_count=proceed_count,
            avoid_count=avoid_count,
            wait_count=wait_count,
            stop_count=stop_count,
            deadlock_count=deadlock_count,
            blocked_duration_s=max_blocked,
            wall_clock_duration_s=wall_duration,
            real_time_factor=rtf,
            travel_distance_m=self.travel_distance,
            average_speed_m_s=avg_speed,
            max_speed_m_s=max_speed,
            planner_latency_mean_s=mean_lat,
            planner_latency_max_s=max_lat,
            emergency_stop_count=self.emergency_stop_count,
            stale_observations_rejected=self.stale_observations_rejected,
            # Legacy compatibility fields
            replan_count=self.global_replan_count,
            failed_replan_count=self.failed_global_replan_count,
            min_clearance_m=self.min_clearance_observed if self.min_clearance_observed != float("inf") else 0.0,
            total_time_s=final_st.timestamp,
            goal_distance_error_m=dist_to_goal,
            detailed_metrics=detailed,
            telemetry=self.telemetry,
            collision_details=self.collision_events,
        )
