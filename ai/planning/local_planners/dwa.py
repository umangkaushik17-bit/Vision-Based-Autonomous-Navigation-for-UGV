"""Dynamic Window Approach (DWA) real-time local trajectory planner for WorldTerrainMap and BEVTerrainCostmap."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.costs import TerrainAwareCostEvaluator
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionResult,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.dwa_filter import (
    DynamicBroadPhaseFilter,
    DynamicPlanningConfig,
    ObstacleSpatiotemporalHull,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.local_planners.base import LocalPlanner
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint


class DWAPlanner(LocalPlanner):
    """Dynamic Window Approach local trajectory planner with multi-term safety scoring."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(name="DWA", config=config)
        self.collision_checker = FootprintCollisionChecker(self.config.vehicle)
        self.cost_evaluator = TerrainAwareCostEvaluator(self.config)

        v_cfg = self.config.vehicle
        self.max_v = float(v_cfg.max_linear_velocity_mps)
        self.min_v = float(v_cfg.min_linear_velocity_mps)
        self.max_omega = float(v_cfg.max_angular_velocity_radps)
        self.max_accel_v = float(v_cfg.max_linear_acceleration_mps2)
        self.max_accel_omega = float(v_cfg.max_angular_acceleration_radps2)
        self.degraded_max_v = float(getattr(v_cfg, "degraded_max_linear_velocity_mps", 0.35))

        dwa_cfg = self.config.dwa
        self.pred_horizon = float(dwa_cfg.get("prediction_horizon_s", 1.50))
        self.sim_dt = float(dwa_cfg.get("simulation_dt_s", 0.10))
        self.v_samples = int(dwa_cfg.get("v_samples", 15))
        self.omega_samples = int(dwa_cfg.get("omega_samples", 25))

        self.w_path = float(dwa_cfg.get("path_cost_weight", 0.30))
        self.w_dist = float(dwa_cfg.get("goal_dist_cost_weight", 0.25))
        self.w_heading = float(dwa_cfg.get("heading_cost_weight", 0.15))
        self.w_clear = float(dwa_cfg.get("clearance_cost_weight", 0.20))
        self.w_vel = float(dwa_cfg.get("velocity_cost_weight", 0.15))
        self.w_terrain = float(dwa_cfg.get("terrain_cost_weight", 0.20))

        # Milestone 13.9: Dynamic predictive collision checker and broad-phase filter
        self.dynamic_checker = PredictiveCollisionChecker(vehicle=self.config.vehicle)
        self.broad_phase_filter = DynamicBroadPhaseFilter()
        self.r_circumscribed = float(getattr(self.config.vehicle, "circumscribed_radius_m", 0.40))

        # Milestone 13.10: Behavioral state machine (PROCEED / AVOID / WAIT / STOP)
        behavior_cfg = getattr(self.config, "behavior", None)
        self.behavioral_state_machine = BehavioralStateMachine(behavior_cfg)
        self.enable_behavioral_arbitration = bool(getattr(self.config, "enable_behavioral_arbitration", False))

    def _grid_coords(self, map_obj: Any, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert metric coordinate to (row, col) across WorldTerrainMap and BEVTerrainCostmap."""
        if hasattr(map_obj, "world_to_grid"):
            return map_obj.world_to_grid(x_m, y_m)
        return map_obj.meter_to_grid(x_m, y_m)

    def _compute_dynamic_window(
        self,
        state: RobotState,
        dt: float,
        max_v_limit: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """Calculate reachable velocity limits based on vehicle accelerations and localization mode."""
        eff_max_v = min(self.max_v, max_v_limit) if max_v_limit is not None else self.max_v

        v_low = max(self.min_v, state.v - self.max_accel_v * dt)
        v_high = min(eff_max_v, state.v + self.max_accel_v * dt)

        w_low = max(-self.max_omega, state.omega - self.max_accel_omega * dt)
        w_high = min(self.max_omega, state.omega + self.max_accel_omega * dt)

        return (float(v_low), float(v_high), float(w_low), float(w_high))

    def _find_local_target(
        self,
        state: RobotState,
        global_path: Path,
        lookahead_dist: float = 0.60
    ) -> Tuple[float, float]:
        """Find target waypoint along global path approximately lookahead_dist ahead."""
        if len(global_path) == 0:
            return (state.x + 1.0 * math.cos(state.yaw), state.y + 1.0 * math.sin(state.yaw))

        xy = global_path.waypoints_xy
        dists = np.hypot(xy[:, 0] - state.x, xy[:, 1] - state.y)
        closest_idx = int(np.argmin(dists))

        target_idx = closest_idx
        accum_dist = 0.0
        while target_idx < len(xy) - 1:
            segment_d = float(np.hypot(xy[target_idx+1, 0] - xy[target_idx, 0], xy[target_idx+1, 1] - xy[target_idx, 1]))
            accum_dist += segment_d
            target_idx += 1
            if accum_dist >= lookahead_dist:
                break

        return (float(xy[target_idx, 0]), float(xy[target_idx, 1]))

    def _predict_trajectory(
        self,
        v: float,
        omega: float,
        state: RobotState,
        costmap: Any
    ) -> Optional[Trajectory]:
        """Simulate forward trajectory using differential-drive kinematics and check footprint collisions."""
        steps = max(1, int(math.ceil(self.pred_horizon / self.sim_dt)))
        curr_x = state.x
        curr_y = state.y
        curr_yaw = state.yaw
        curr_t = state.timestamp

        points: List[TrajectoryPoint] = []
        min_clearance = float("inf")
        accum_terrain_cost = 0.0

        for _ in range(steps):
            # Midpoint integration for smooth circular arc
            theta_mid = curr_yaw + 0.5 * omega * self.sim_dt
            curr_x += v * math.cos(theta_mid) * self.sim_dt
            curr_y += v * math.sin(theta_mid) * self.sim_dt
            curr_yaw += omega * self.sim_dt
            curr_yaw = (curr_yaw + np.pi) % (2.0 * np.pi) - np.pi
            curr_t += self.sim_dt

            # 1. Footprint Collision check on WorldTerrainMap / BEVTerrainCostmap
            if self.collision_checker.is_footprint_collision(curr_x, curr_y, curr_yaw, costmap):
                return None  # Discard collision trajectory immediately

            r, c = self._grid_coords(costmap, curr_x, curr_y)
            c_terrain = self.cost_evaluator.evaluate_cell_cost(r, c, costmap)
            if np.isinf(c_terrain):
                return None  # Discard lethal cell trajectory

            accum_terrain_cost += c_terrain
            clr = self.collision_checker.get_clearance(curr_x, curr_y, costmap)
            min_clearance = min(min_clearance, clr)

            points.append(TrajectoryPoint(
                t=curr_t, x=curr_x, y=curr_y, yaw=curr_yaw,
                v=v, omega=omega, cost=c_terrain, clearance_m=clr
            ))

        avg_terrain = accum_terrain_cost / max(steps, 1)
        return Trajectory(
            points=points, is_valid=True,
            total_cost=avg_terrain, min_clearance_m=float(min_clearance)
        )

    @staticmethod
    def _point_to_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Perpendicular distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return float(math.hypot(px - x1, py - y1))
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return float(math.hypot(px - proj_x, py - proj_y))

    def compute_trajectory(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        is_degraded: bool = False,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Evaluate dynamic window, score candidate trajectories, and select safest command.

        Milestone 13.9:
        Integrates predictive dynamic collision checking and broad-phase pre-filtering:
        candidate generation -> static validity -> broad-phase filter -> exact predictive check -> dynamic scoring.
        """
        t_start = time.perf_counter()

        # Enforce degraded velocity cap (Phase 8/12 contract: v <= 0.35 m/s)
        max_v_limit = self.degraded_max_v if is_degraded else None

        local_target = self._find_local_target(current_state, global_path)
        d_curr = math.hypot(local_target[0] - current_state.x, local_target[1] - current_state.y)
        v_low, v_high, w_low, w_high = self._compute_dynamic_window(
            current_state, dt=self.sim_dt, max_v_limit=max_v_limit
        )

        v_candidates = np.linspace(v_low, v_high, self.v_samples)
        omega_candidates = np.linspace(w_low, w_high, self.omega_samples)

        # Configure dynamic layer and broad-phase filter
        has_dynamic = (dynamic_layer is not None) or (predicted_tracks is not None)
        dyn_cfg = dynamic_config or DynamicPlanningConfig()
        if not dyn_cfg.enabled:
            has_dynamic = False

        obstacle_hulls: List[ObstacleSpatiotemporalHull] = []
        if has_dynamic:
            obstacle_hulls = self.broad_phase_filter.build_obstacle_hulls(
                predicted_tracks, dynamic_layer, max_horizon_s=self.pred_horizon
            )

        statically_valid_candidates = 0
        broad_phase_cleared_count = 0
        expensive_collision_checks = 0
        dynamically_rejected_count = 0
        dynamically_penalized_count = 0
        unknown_horizon_candidates = 0

        valid_candidates: List[Tuple[float, Trajectory, float, float, Optional[PredictiveCollisionResult]]] = []

        for v in v_candidates:
            for w in omega_candidates:
                # 1. Forward simulation & static Phase 12 footprint collision check
                traj = self._predict_trajectory(float(v), float(w), current_state, costmap)
                if traj is None or len(traj) == 0:
                    continue

                statically_valid_candidates += 1

                # 2. Dynamic obstacle evaluation
                dyn_res: Optional[PredictiveCollisionResult] = None
                dyn_penalty = 0.0
                dyn_clearance = 1.0  # Normalized [0, 1] reward

                if has_dynamic:
                    # Cheap broad-phase pre-filter test
                    is_relevant = self.broad_phase_filter.is_candidate_relevant(
                        traj, obstacle_hulls, self.r_circumscribed
                    )

                    if not is_relevant:
                        # Broad-phase cleared: cannot collide or penetrate risk zone
                        broad_phase_cleared_count += 1
                        dyn_res = None
                        dyn_penalty = 0.0
                        dyn_clearance = 1.0
                    else:
                        # Exact predictive collision check (Milestone 13.8 contract)
                        dyn_res = self.dynamic_checker.check_trajectory(traj, dynamic_layer, predicted_tracks)
                        expensive_collision_checks += 1

                        # Physical collision strictly rejects the candidate
                        if dyn_res.status == CollisionStatus.PREDICTED_COLLISION or dyn_res.collision:
                            dynamically_rejected_count += 1
                            continue

                        elif dyn_res.status == CollisionStatus.PREDICTED_HIGH_RISK:
                            if dyn_cfg.reject_high_risk:
                                dynamically_rejected_count += 1
                                continue
                            dynamically_penalized_count += 1
                            dyn_penalty = dyn_cfg.high_risk_penalty + dyn_cfg.w_dynamic_risk * dyn_res.maximum_cost
                            dyn_clearance = min(dyn_res.minimum_clearance, 1.50) / 1.50

                        elif dyn_res.status == CollisionStatus.PREDICTED_MEDIUM_RISK:
                            dynamically_penalized_count += 1
                            dyn_penalty = dyn_cfg.medium_risk_penalty + dyn_cfg.w_dynamic_risk * dyn_res.maximum_cost
                            dyn_clearance = min(dyn_res.minimum_clearance, 1.50) / 1.50

                        elif dyn_res.status == CollisionStatus.PREDICTED_LOW_RISK:
                            dynamically_penalized_count += 1
                            dyn_penalty = dyn_cfg.low_risk_penalty + 0.5 * dyn_cfg.w_dynamic_risk * dyn_res.maximum_cost
                            dyn_clearance = min(dyn_res.minimum_clearance, 1.50) / 1.50

                        elif dyn_res.status == CollisionStatus.UNKNOWN_HORIZON or dyn_res.unknown_horizon_encountered:
                            unknown_horizon_candidates += 1
                            if dyn_cfg.unknown_horizon_policy == "REJECT":
                                dynamically_rejected_count += 1
                                continue
                            elif dyn_cfg.unknown_horizon_policy == "PENALIZE":
                                dynamically_penalized_count += 1
                                dyn_penalty = dyn_cfg.unknown_horizon_penalty
                                dyn_clearance = min(dyn_res.minimum_clearance, 1.50) / 1.50
                            else:  # "ALLOW"
                                dyn_penalty = 0.0
                                dyn_clearance = min(dyn_res.minimum_clearance, 1.50) / 1.50

                        else:  # SAFE
                            dyn_penalty = 0.0
                            dyn_clearance = min(dyn_res.minimum_clearance, 1.50) / 1.50

                end_pt = traj.points[-1]

                # 3. Phase 12 Static Multi-Objective Scoring
                # 3a. Path alignment: perpendicular cross-track error to nearest global path segment
                if len(global_path) >= 2:
                    xy = global_path.waypoints_xy
                    min_cte = min(
                        self._point_to_segment_dist(end_pt.x, end_pt.y, float(xy[k, 0]), float(xy[k, 1]), float(xy[k + 1, 0]), float(xy[k + 1, 1]))
                        for k in range(len(xy) - 1)
                    )
                    path_score = 1.0 / (1.0 + min_cte)
                elif len(global_path) == 1:
                    min_cte = float(math.hypot(global_path.points[0].x - end_pt.x, global_path.points[0].y - end_pt.y))
                    path_score = 1.0 / (1.0 + min_cte)
                else:
                    path_score = 1.0

                # 3b. Heading score (alignment to local target)
                target_yaw = math.atan2(local_target[1] - end_pt.y, local_target[0] - end_pt.x)
                heading_err = abs((target_yaw - end_pt.yaw + np.pi) % (2.0 * np.pi) - np.pi)
                heading_score = (math.pi - heading_err) / math.pi  # Normalized [0, 1]

                # 3c. Distance and forward progress to local target
                d_end = math.hypot(local_target[0] - end_pt.x, local_target[1] - end_pt.y)
                progress_score = (d_curr - d_end) / max(0.1, self.max_v * self.pred_horizon)
                dist_score = 0.5 * (1.0 / (1.0 + d_end)) + 0.5 * max(0.0, progress_score)

                # 3d. Static clearance score (safety reward)
                clearance_score = min(traj.min_clearance_m, 1.50) / 1.50

                # 3e. Velocity score (prefer forward motion aligned with path heading, penalize unnecessary reverse)
                vel_factor = max(0.0, math.cos(heading_err))
                vel_score = (float(v) / max(0.1, self.max_v) * vel_factor) if v >= 0 else -0.5 * abs(v) / self.max_v

                # 3f. Terrain score (lower terrain cost is preferred)
                terrain_score = 1.0 / (1.0 + traj.total_cost)

                # Composite score: static multi-objective minus dynamic penalty plus dynamic clearance
                total_score = (
                    self.w_path * path_score +
                    self.w_heading * heading_score +
                    self.w_dist * dist_score +
                    self.w_clear * clearance_score +
                    self.w_vel * vel_score +
                    self.w_terrain * terrain_score -
                    dyn_penalty +
                    (dyn_cfg.w_dynamic_clearance * dyn_clearance if has_dynamic else 0.0)
                )

                valid_candidates.append((total_score, traj, float(v), float(w), dyn_res))

        lat_ms = (time.perf_counter() - t_start) * 1000.0

        # If no valid collision-free trajectory exists: STOP SAFELY
        if not valid_candidates:
            zero_traj = Trajectory(
                points=[TrajectoryPoint(
                    t=current_state.timestamp + self.sim_dt,
                    x=current_state.x, y=current_state.y, yaw=current_state.yaw,
                    v=0.0, omega=0.0, cost=1.0, clearance_m=0.0
                )],
                is_valid=False, total_cost=1.0, min_clearance_m=0.0
            )
            zero_diag = {
                "trajectories_evaluated": self.v_samples * self.omega_samples,
                "statically_valid_candidates": statically_valid_candidates,
                "valid_trajectories": 0,
                "broad_phase_rejections": broad_phase_cleared_count,
                "broad_phase_cleared_count": broad_phase_cleared_count,
                "expensive_collision_checks": expensive_collision_checks,
                "number_of_expensive_collision_checker_invocations": expensive_collision_checks,
                "number_of_broad_phase_rejections": broad_phase_cleared_count,
                "number_of_candidates_evaluated": self.v_samples * self.omega_samples,
                "dynamically_rejected_candidates": dynamically_rejected_count,
                "dynamically_penalized_candidates": dynamically_penalized_count,
                "unknown_horizon_candidates": unknown_horizon_candidates,
                "dynamic_collision": True if dynamically_rejected_count > 0 else False,
                "dynamic_risk": "LETHAL" if dynamically_rejected_count > 0 else "FREE",
                "unknown_horizon": unknown_horizon_candidates > 0,
                "blocking_obstacle_id": None,
                "earliest_collision_time": None,
                "minimum_dynamic_clearance": 0.0,
                "maximum_dynamic_cost": 1.0 if dynamically_rejected_count > 0 else 0.0,
                "dwa_latency_ms": lat_ms,
                "emergency_stop_triggered": True,
            }
            decision = self.behavioral_state_machine.evaluate(
                current_state=current_state,
                best_trajectory=zero_traj,
                best_v=0.0,
                best_w=0.0,
                planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
                dwa_diagnostics=zero_diag,
                predicted_tracks=predicted_tracks,
                global_path=global_path,
                current_time_s=current_state.timestamp,
            )
            zero_diag["behavioral_state"] = decision.state.value
            zero_diag["behavioral_decision"] = decision.to_dict()
            zero_diag["deadlock_event"] = decision.deadlock_event
            cmd_v = decision.command_v if self.enable_behavioral_arbitration else 0.0
            cmd_w = decision.command_omega if self.enable_behavioral_arbitration else 0.0
            return zero_traj, cmd_v, cmd_w, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, zero_diag

        # Select candidate with highest safety-informed score
        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_traj, best_v, best_w, best_dyn_res = valid_candidates[0]

        diagnostics = {
            "trajectories_evaluated": self.v_samples * self.omega_samples,
            "statically_valid_candidates": statically_valid_candidates,
            "valid_trajectories": len(valid_candidates),
            "broad_phase_rejections": broad_phase_cleared_count,
            "broad_phase_cleared_count": broad_phase_cleared_count,
            "expensive_collision_checks": expensive_collision_checks,
            "number_of_expensive_collision_checker_invocations": expensive_collision_checks,
            "number_of_broad_phase_rejections": broad_phase_cleared_count,
            "number_of_candidates_evaluated": self.v_samples * self.omega_samples,
            "dynamically_rejected_candidates": dynamically_rejected_count,
            "dynamically_penalized_candidates": dynamically_penalized_count,
            "unknown_horizon_candidates": unknown_horizon_candidates,
            "dynamic_collision": best_dyn_res.collision if best_dyn_res else False,
            "dynamic_risk": (
                "UNKNOWN_HORIZON"
                if (best_dyn_res and (best_dyn_res.status == CollisionStatus.UNKNOWN_HORIZON or best_dyn_res.unknown_horizon_encountered))
                else (best_dyn_res.risk_classification.name if best_dyn_res else "FREE")
            ),
            "unknown_horizon": (
                (best_dyn_res.unknown_horizon_encountered or best_dyn_res.status == CollisionStatus.UNKNOWN_HORIZON)
                if best_dyn_res else False
            ),
            "blocking_obstacle_id": best_dyn_res.obstacle_id if best_dyn_res else None,
            "earliest_collision_time": best_dyn_res.collision_time if best_dyn_res else None,
            "minimum_dynamic_clearance": (
                best_dyn_res.minimum_clearance if best_dyn_res else float(self.dynamic_checker.config.max_clearance_search_m)
            ),
            "maximum_dynamic_cost": best_dyn_res.maximum_cost if best_dyn_res else 0.0,
            "best_score": round(best_score, 4),
            "desired_linear_v": round(best_v, 3),
            "desired_angular_w": round(best_w, 3),
            "min_clearance_m": round(best_traj.min_clearance_m, 3),
            "dwa_latency_ms": lat_ms,
            "emergency_stop_triggered": False,
        }

        decision = self.behavioral_state_machine.evaluate(
            current_state=current_state,
            best_trajectory=best_traj,
            best_v=best_v,
            best_w=best_w,
            planner_status=PlannerStatus.SUCCESS,
            dwa_diagnostics=diagnostics,
            predicted_tracks=predicted_tracks,
            global_path=global_path,
            current_time_s=current_state.timestamp,
        )
        diagnostics["behavioral_state"] = decision.state.value
        diagnostics["behavioral_decision"] = decision.to_dict()
        diagnostics["deadlock_event"] = decision.deadlock_event
        cmd_v = decision.command_v if self.enable_behavioral_arbitration else best_v
        cmd_w = decision.command_omega if self.enable_behavioral_arbitration else best_w

        return best_traj, cmd_v, cmd_w, PlannerStatus.SUCCESS, diagnostics

    def plan_with_behavior(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        target_v: Optional[float] = None,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
        behavioral_config: Optional[BehavioralConfig] = None,
    ) -> Tuple[Trajectory, float, float, PlannerStatus, Dict[str, Any], BehavioralDecision]:
        """Compute DWA trajectory and arbitrate final velocity command through BehavioralStateMachine.

        Returns:
            (trajectory, command_v, command_omega, planner_status, diagnostics, behavioral_decision)
        """
        if behavioral_config is not None:
            self.behavioral_state_machine.config = behavioral_config

        traj, raw_v, raw_w, status, diag = self.compute_trajectory(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )

        decision = self.behavioral_state_machine.evaluate(
            current_state=current_state,
            best_trajectory=traj,
            best_v=raw_v,
            best_w=raw_w,
            planner_status=status,
            dwa_diagnostics=diag,
            predicted_tracks=predicted_tracks,
            global_path=global_path,
            current_time_s=current_state.timestamp,
        )
        diag["behavioral_state"] = decision.state.value
        diag["behavioral_decision"] = decision.to_dict()
        diag["deadlock_event"] = decision.deadlock_event

        return traj, decision.command_v, decision.command_omega, status, diag, decision

    def plan(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        is_degraded: bool = False,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        """Canonical planning entry point (alias to compute_trajectory)."""
        return self.compute_trajectory(
            current_state=current_state,
            global_path=global_path,
            costmap=costmap,
            is_degraded=is_degraded,
            dynamic_layer=dynamic_layer,
            predicted_tracks=predicted_tracks,
            dynamic_config=dynamic_config,
        )
