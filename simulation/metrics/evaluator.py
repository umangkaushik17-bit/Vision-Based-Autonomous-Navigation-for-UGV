"""Comprehensive post-run metric evaluation and temporal alignment engine for SIL scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from simulation.scenarios.dynamic.data_model import GroundTruthObstacleState
from simulation.scenarios.dynamic.motion_models import BaseObstacleMotionModel


@dataclass
class QuantitativeScenarioMetrics:
    """Detailed multi-dimensional metrics evaluated post-run against ground truth."""

    # Mission
    mission_outcome: str = "INCOMPLETE"  # GOAL_REACHED, INCOMPLETE
    safety_outcome: str = "SAFE"        # SAFE, COLLISION
    termination_reason: str = "HORIZON_BOUNDED_TIMEOUT"  # GOAL_REACHED, HORIZON_BOUNDED_TIMEOUT, COLLISION
    mission_success: bool = False
    goal_position_error_m: float = 0.0
    goal_heading_error_rad: Optional[float] = None
    goal_tolerance_m: float = 0.25
    configured_duration_s: float = 0.0
    travel_time_s: float = 0.0
    travel_distance_m: float = 0.0
    average_speed_m_s: float = 0.0
    max_speed_m_s: float = 0.0
    stop_count: int = 0

    # Safety
    collision_count: int = 0
    min_clearance_m: float = float("inf")
    near_collision_count: int = 0
    unsafe_command_count: int = 0
    emergency_stop_count: int = 0
    stop_duration_s: float = 0.0
    wait_duration_s: float = 0.0

    # Tracking
    detection_count: int = 0
    confirmed_track_count: int = 0
    confirmation_latency_s: Optional[float] = None
    track_continuity: float = 1.0
    tracking_position_rmse_m: float = 0.0
    tracking_velocity_rmse_m_s: float = 0.0
    false_track_count: int = 0
    lifecycle_transitions: int = 0

    # Prediction (aligned at distinct horizons)
    prediction_rmse_0_25s: float = 0.0
    prediction_rmse_0_50s: float = 0.0
    prediction_rmse_1_00s: float = 0.0
    prediction_rmse_1_50s: float = 0.0
    prediction_rmse_2_00s: float = 0.0
    uncertainty_coverage_pct: float = 100.0
    mean_uncertainty_radius_m: float = 0.0
    unknown_horizon_count: int = 0

    # Planning
    local_planner_cycles: int = 0
    global_replan_count: int = 0
    failed_global_replan_count: int = 0
    planner_cycle_count: int = 0  # alias to local_planner_cycles
    replan_count: int = 0         # legacy alias to global_replan_count
    failed_replan_count: int = 0  # legacy alias to failed_global_replan_count
    planner_latency_mean_s: float = 0.0
    planner_latency_max_s: float = 0.0
    dynamic_collision_rejections: int = 0
    replan_latency_s: float = 0.0

    # Behavior
    proceed_count: int = 0
    avoid_count: int = 0
    wait_count: int = 0
    stop_count_behavior: int = 0
    behavior_transitions: int = 0
    deadlock_count: int = 0
    blocked_duration_s: Optional[float] = None

    # System
    wall_clock_duration_s: float = 0.0
    real_time_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": {
                "outcome": self.mission_outcome,
                "safety": self.safety_outcome,
                "termination_reason": self.termination_reason,
                "success": self.mission_success,
                "goal_position_error_m": round(self.goal_position_error_m, 4),
                "goal_heading_error_rad": round(self.goal_heading_error_rad, 4) if self.goal_heading_error_rad is not None else None,
                "goal_tolerance_m": round(self.goal_tolerance_m, 3),
                "configured_duration_s": round(self.configured_duration_s, 2),
                "travel_time_s": round(self.travel_time_s, 2),
                "travel_distance_m": round(self.travel_distance_m, 3),
                "average_speed_m_s": round(self.average_speed_m_s, 3),
                "max_speed_m_s": round(self.max_speed_m_s, 3),
                "stop_count": self.stop_count,
            },
            "safety": {
                "collision_count": self.collision_count,
                "min_clearance_m": round(self.min_clearance_m, 3) if self.min_clearance_m != float("inf") else -1.0,
                "near_collision_count": self.near_collision_count,
                "unsafe_command_count": self.unsafe_command_count,
                "emergency_stop_count": self.emergency_stop_count,
                "stop_duration_s": round(self.stop_duration_s, 2),
                "wait_duration_s": round(self.wait_duration_s, 2),
            },
            "tracking": {
                "detection_count": self.detection_count,
                "confirmed_track_count": self.confirmed_track_count,
                "confirmation_latency_s": round(self.confirmation_latency_s, 3) if self.confirmation_latency_s is not None else None,
                "track_continuity": round(self.track_continuity, 3),
                "position_rmse_m": round(self.tracking_position_rmse_m, 4),
                "velocity_rmse_m_s": round(self.tracking_velocity_rmse_m_s, 4),
                "false_track_count": self.false_track_count,
                "lifecycle_transitions": self.lifecycle_transitions,
            },
            "prediction": {
                "rmse_0_25s": round(self.prediction_rmse_0_25s, 4),
                "rmse_0_50s": round(self.prediction_rmse_0_50s, 4),
                "rmse_1_00s": round(self.prediction_rmse_1_00s, 4),
                "rmse_1_50s": round(self.prediction_rmse_1_50s, 4),
                "rmse_2_00s": round(self.prediction_rmse_2_00s, 4),
                "uncertainty_coverage_pct": round(self.uncertainty_coverage_pct, 1),
                "mean_uncertainty_radius_m": round(self.mean_uncertainty_radius_m, 3),
                "unknown_horizon_count": self.unknown_horizon_count,
            },
            "planning": {
                "local_planner_cycles": self.local_planner_cycles,
                "global_replan_count": self.global_replan_count,
                "failed_global_replan_count": self.failed_global_replan_count,
                "planner_cycle_count": self.planner_cycle_count,
                "replan_count": self.replan_count,
                "failed_replan_count": self.failed_replan_count,
                "latency_mean_s": round(self.planner_latency_mean_s, 4),
                "latency_max_s": round(self.planner_latency_max_s, 4),
                "dynamic_collision_rejections": self.dynamic_collision_rejections,
                "replan_latency_s": round(self.replan_latency_s, 4),
            },
            "behavior": {
                "proceed_count": self.proceed_count,
                "avoid_count": self.avoid_count,
                "wait_count": self.wait_count,
                "stop_count": self.stop_count_behavior,
                "transitions": self.behavior_transitions,
                "deadlock_count": self.deadlock_count,
                "blocked_duration_s": round(self.blocked_duration_s, 3) if self.blocked_duration_s is not None else None,
            },
            "system": {
                "wall_clock_duration_s": round(self.wall_clock_duration_s, 3),
                "real_time_factor": round(self.real_time_factor, 3),
            },
        }


class ScenarioEvaluator:
    """Post-run evaluator comparing executed navigation output against ground truth."""

    def __init__(self, motion_models: Sequence[BaseObstacleMotionModel]) -> None:
        self.motion_models = list(motion_models)

    def evaluate(
        self,
        telemetry: List[Dict[str, Any]],
        goal_pose: Tuple[float, ...],
        dt_s: float = 0.10,
        wall_clock_duration_s: float = 0.0,
        goal_tolerance_m: float = 0.25,
        termination_reason: Optional[str] = None,
        configured_duration_s: float = 0.0,
        local_planner_cycles: Optional[int] = None,
        global_replan_count: Optional[int] = None,
        failed_global_replan_count: Optional[int] = None,
    ) -> QuantitativeScenarioMetrics:
        """Calculate comprehensive metrics from recorded telemetry and ground truth."""
        metrics = QuantitativeScenarioMetrics()
        metrics.goal_tolerance_m = float(goal_tolerance_m)
        metrics.configured_duration_s = float(configured_duration_s)
        if not telemetry:
            return metrics

        # 1. Mission Metrics
        final_step = telemetry[-1]
        metrics.travel_time_s = float(final_step.get("sim_time", 0.0))
        gx, gy = float(goal_pose[0]), float(goal_pose[1])
        rx, ry = float(final_step["robot_x"]), float(final_step["robot_y"])
        metrics.goal_position_error_m = math.hypot(rx - gx, ry - gy)

        # 2. Safety Metrics (collisions computed early for outcome logic)
        metrics.collision_count = sum(1 for t in telemetry if t.get("gt_collision", False))

        is_goal = (termination_reason == "GOAL_REACHED") or (metrics.goal_position_error_m <= metrics.goal_tolerance_m)

        if metrics.collision_count > 0:
            metrics.safety_outcome = "COLLISION"
            metrics.mission_outcome = "INCOMPLETE"
            metrics.termination_reason = "COLLISION"
            metrics.mission_success = False
        elif is_goal:
            metrics.safety_outcome = "SAFE"
            metrics.mission_outcome = "GOAL_REACHED"
            metrics.termination_reason = "GOAL_REACHED"
            metrics.mission_success = True
        else:
            metrics.safety_outcome = "SAFE"
            metrics.mission_outcome = "INCOMPLETE"
            metrics.termination_reason = termination_reason or "HORIZON_BOUNDED_TIMEOUT"
            metrics.mission_success = False

        if len(goal_pose) > 2 and goal_pose[2] is not None:
            target_yaw = float(goal_pose[2])
            current_yaw = float(final_step.get("robot_yaw", 0.0))
            err_yaw = (current_yaw - target_yaw + math.pi) % (2 * math.pi) - math.pi
            metrics.goal_heading_error_rad = abs(err_yaw)

        # Distance & speed
        speeds = [float(t["robot_v"]) for t in telemetry]
        metrics.average_speed_m_s = float(np.mean(speeds)) if speeds else 0.0
        metrics.max_speed_m_s = float(np.max(speeds)) if speeds else 0.0

        dist = 0.0
        for i in range(1, len(telemetry)):
            dx = telemetry[i]["robot_x"] - telemetry[i - 1]["robot_x"]
            dy = telemetry[i]["robot_y"] - telemetry[i - 1]["robot_y"]
            dist += math.hypot(dx, dy)
        metrics.travel_distance_m = dist

        # Stop count (velocity < 0.02 m/s while moving)
        was_stopped = False
        for s in speeds:
            if s < 0.02 and not was_stopped:
                metrics.stop_count += 1
                was_stopped = True
            elif s >= 0.05:
                was_stopped = False

        # Clearances
        clearances = [float(t.get("min_clearance_m", float("inf"))) for t in telemetry if "min_clearance_m" in t]
        metrics.min_clearance_m = min(clearances) if clearances else float("inf")
        metrics.near_collision_count = sum(1 for c in clearances if 0.0 < c < 0.25)
        metrics.emergency_stop_count = sum(1 for t in telemetry if t.get("emergency_stop", False))
        metrics.stop_duration_s = sum(dt_s for t in telemetry if t.get("behavior_state") == "STOP")
        metrics.wait_duration_s = sum(dt_s for t in telemetry if t.get("behavior_state") == "WAIT")

        # 3. Behavior Metrics
        prev_state = None
        for t in telemetry:
            b_st = t.get("behavior_state", "PROCEED")
            if b_st == "PROCEED":
                metrics.proceed_count += 1
            elif b_st == "AVOID":
                metrics.avoid_count += 1
            elif b_st == "WAIT":
                metrics.wait_count += 1
            elif b_st == "STOP":
                metrics.stop_count_behavior += 1

            if prev_state is not None and b_st != prev_state:
                metrics.behavior_transitions += 1
            prev_state = b_st

        metrics.deadlock_count = sum(1 for t in telemetry if t.get("deadlock", False))
        blocked_durations = [t.get("blocked_duration_s") for t in telemetry if t.get("blocked_duration_s") is not None]
        metrics.blocked_duration_s = max(blocked_durations) if blocked_durations else None

        # 4. Planning Metrics
        metrics.local_planner_cycles = int(local_planner_cycles) if local_planner_cycles is not None else len(telemetry)
        metrics.global_replan_count = int(global_replan_count) if global_replan_count is not None else sum(1 for t in telemetry if t.get("replan_triggered", False))
        metrics.failed_global_replan_count = int(failed_global_replan_count) if failed_global_replan_count is not None else sum(1 for t in telemetry if t.get("failed_replan", False))
        metrics.planner_cycle_count = metrics.local_planner_cycles
        metrics.replan_count = metrics.global_replan_count
        metrics.failed_replan_count = metrics.failed_global_replan_count
        metrics.dynamic_collision_rejections = sum(
            1 for t in telemetry
            if t.get("dynamic_collision", False) or t.get("collision_status") in ("PREDICTED_COLLISION", "PREDICTED_HIGH_RISK")
        )

        # 5. System Timing & RTF
        metrics.wall_clock_duration_s = float(wall_clock_duration_s)
        metrics.real_time_factor = (metrics.travel_time_s / wall_clock_duration_s) if wall_clock_duration_s > 0 else 0.0

        # 5. Tracking Metrics & Temporal Ground-Truth Association
        pos_errors: List[float] = []
        vel_errors: List[float] = []
        first_detection_t: Optional[float] = None
        first_confirmed_t: Optional[float] = None
        track_active_steps = 0
        total_gt_active_steps = 0

        for t in telemetry:
            sim_t = float(t.get("sim_time", 0.0))
            active_gt = [m.evaluate(sim_t) for m in self.motion_models if m.evaluate(sim_t) is not None and m.evaluate(sim_t).is_active]
            if active_gt:
                total_gt_active_steps += 1

            tracked_details = t.get("tracked_details", [])
            if tracked_details:
                track_active_steps += 1
                if first_confirmed_t is None:
                    first_confirmed_t = sim_t

            # Compare tracked state to closest GT obstacle
            for trk in tracked_details:
                tx, ty = trk["x"], trk["y"]
                tvx, tvy = trk.get("vx", 0.0), trk.get("vy", 0.0)

                best_gt = None
                best_d = float("inf")
                for gt in active_gt:
                    d = math.hypot(tx - gt.x, ty - gt.y)
                    if d < best_d:
                        best_d = d
                        best_gt = gt

                if best_gt is not None and best_d < 1.5:  # Valid association gate
                    pos_errors.append(best_d)
                    v_err = math.hypot(tvx - best_gt.vx, tvy - best_gt.vy)
                    vel_errors.append(v_err)

        metrics.tracking_position_rmse_m = float(np.sqrt(np.mean(np.square(pos_errors)))) if pos_errors else 0.0
        metrics.tracking_velocity_rmse_m_s = float(np.sqrt(np.mean(np.square(vel_errors)))) if vel_errors else 0.0
        if first_confirmed_t is not None:
            metrics.confirmation_latency_s = max(0.0, first_confirmed_t)
        if total_gt_active_steps > 0:
            metrics.track_continuity = min(1.0, track_active_steps / total_gt_active_steps)

        # Lifecycle transitions across steps
        last_track_statuses: Dict[int, str] = {}
        for t in telemetry:
            for trk in t.get("tracked_details", []):
                tid = trk.get("track_id")
                st = trk.get("status")
                if tid in last_track_statuses and last_track_statuses[tid] != st:
                    metrics.lifecycle_transitions += 1
                last_track_statuses[tid] = st

        # Unknown horizon events
        metrics.unknown_horizon_count = sum(
            1 for t in telemetry
            if t.get("unknown_horizon", False) or t.get("collision_status") == "UNKNOWN_HORIZON"
        )

        # 6. Temporal Prediction Evaluation
        # Compare prediction waypoint at target_t = t_now + h with GT evaluated at target_t
        pred_errors_by_horizon: Dict[float, List[float]] = {0.25: [], 0.50: [], 1.00: [], 1.50: [], 2.00: []}
        coverage_checks: List[bool] = []
        uncertainty_radii: List[float] = []

        for t in telemetry:
            sim_t = float(t.get("sim_time", 0.0))
            pred_details = t.get("prediction_details", [])

            for pred in pred_details:
                waypoints = pred.get("waypoints", [])
                for wp in waypoints:
                    # wp: (wp_time, x, y, vx, vy, uncertainty_radius)
                    wp_t, px, py = float(wp[0]), float(wp[1]), float(wp[2])
                    unc_r = float(wp[5]) if len(wp) > 5 else 0.20
                    h = round(wp_t - sim_t, 2)

                    # Find corresponding ground truth at wp_t
                    for m in self.motion_models:
                        gt_at_h = m.evaluate(wp_t)
                        if gt_at_h is not None and gt_at_h.is_active:
                            err = math.hypot(px - gt_at_h.x, py - gt_at_h.y)
                            uncertainty_radii.append(unc_r)
                            coverage_checks.append(err <= (2.0 * unc_r + 0.10))

                            # Match to nearest nominal horizon bucket
                            for target_h in (0.25, 0.50, 1.00, 1.50, 2.00):
                                if abs(h - target_h) < 0.06:
                                    pred_errors_by_horizon[target_h].append(err)
                                    break

        if pred_errors_by_horizon[0.25]:
            metrics.prediction_rmse_0_25s = float(np.sqrt(np.mean(np.square(pred_errors_by_horizon[0.25]))))
        if pred_errors_by_horizon[0.50]:
            metrics.prediction_rmse_0_50s = float(np.sqrt(np.mean(np.square(pred_errors_by_horizon[0.50]))))
        if pred_errors_by_horizon[1.00]:
            metrics.prediction_rmse_1_00s = float(np.sqrt(np.mean(np.square(pred_errors_by_horizon[1.00]))))
        if pred_errors_by_horizon[1.50]:
            metrics.prediction_rmse_1_50s = float(np.sqrt(np.mean(np.square(pred_errors_by_horizon[1.50]))))
        if pred_errors_by_horizon[2.00]:
            metrics.prediction_rmse_2_00s = float(np.sqrt(np.mean(np.square(pred_errors_by_horizon[2.00]))))

        if coverage_checks:
            metrics.uncertainty_coverage_pct = float(np.mean(coverage_checks) * 100.0)
        if uncertainty_radii:
            metrics.mean_uncertainty_radius_m = float(np.mean(uncertainty_radii))

        return metrics
