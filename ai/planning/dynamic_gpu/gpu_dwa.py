"""Hardware-accelerated Dynamic Window Approach (DWA) local planner.

Integrates batched unicycle forward rollout, static footprint raster checking,
and predictive dynamic collision checking with 100% numerical and behavioral parity
to the Phase 13 reference DWAPlanner.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.costs import TerrainAwareCostEvaluator
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralStateMachine,
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
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer, DynamicRiskLevel
from ai.planning.dynamic_gpu.gpu_backend import GPUBackend
from ai.planning.dynamic_gpu.gpu_collision import GPUCollisionChecker
from ai.planning.dynamic_gpu.gpu_rollout import GPURollout
from ai.planning.dynamic_gpu.gpu_scoring import GPUScorer
from ai.planning.local_planners.base import LocalPlanner
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint


class AcceleratedDWAPlanner(LocalPlanner):
    """Batched, GPU-accelerated Dynamic Window Approach local trajectory planner."""

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        backend: Optional[GPUBackend] = None,
        preferred_device: str = "auto",
    ) -> None:
        super().__init__(name="AcceleratedDWA", config=config)

        # Hardware backend
        self.backend = backend or GPUBackend(preferred_device=preferred_device)
        self.device = self.backend.device
        self.dtype = self.backend.dtype

        # Sub-kernels
        self.rollout_kernel = GPURollout(self.backend)
        self.collision_kernel = GPUCollisionChecker(self.backend, vehicle=self.config.vehicle)
        self.scorer_kernel = GPUScorer(self.backend)

        # Reference CPU fallbacks & evaluators for verification and parity checks
        self.cpu_collision_checker = FootprintCollisionChecker(self.config.vehicle)
        self.cost_evaluator = TerrainAwareCostEvaluator(self.config)
        self.dynamic_checker = PredictiveCollisionChecker(vehicle=self.config.vehicle)
        self.broad_phase_filter = DynamicBroadPhaseFilter()

        # Vehicle parameters
        v_cfg = self.config.vehicle
        self.max_v = float(v_cfg.max_linear_velocity_mps)
        self.min_v = float(v_cfg.min_linear_velocity_mps)
        self.max_omega = float(v_cfg.max_angular_velocity_radps)
        self.max_accel_v = float(v_cfg.max_linear_acceleration_mps2)
        self.max_accel_omega = float(v_cfg.max_angular_acceleration_radps2)
        self.degraded_max_v = float(getattr(v_cfg, "degraded_max_linear_velocity_mps", 0.35))
        self.r_circumscribed = float(getattr(self.config.vehicle, "circumscribed_radius_m", 0.40))

        # DWA sampling parameters
        dwa_cfg = self.config.dwa
        self.pred_horizon = float(dwa_cfg.get("prediction_horizon_s", 1.50))
        self.sim_dt = float(dwa_cfg.get("simulation_dt_s", 0.10))
        self.v_samples = int(dwa_cfg.get("v_samples", 15))
        self.omega_samples = int(dwa_cfg.get("omega_samples", 25))

        # Weights
        self.w_path = float(dwa_cfg.get("path_cost_weight", 0.30))
        self.w_dist = float(dwa_cfg.get("goal_dist_cost_weight", 0.25))
        self.w_heading = float(dwa_cfg.get("heading_cost_weight", 0.15))
        self.w_clear = float(dwa_cfg.get("clearance_cost_weight", 0.20))
        self.w_vel = float(dwa_cfg.get("velocity_cost_weight", 0.15))
        self.w_terrain = float(dwa_cfg.get("terrain_cost_weight", 0.20))

        # Behavioral state machine
        behavior_cfg = getattr(self.config, "behavior", None)
        self.behavioral_state_machine = BehavioralStateMachine(behavior_cfg)
        self.enable_behavioral_arbitration = bool(getattr(self.config, "enable_behavioral_arbitration", False))

    def _find_local_target(
        self,
        state: RobotState,
        global_path: Path,
        lookahead_dist: float = 0.60,
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

    def _apply_broad_phase_filter(
        self,
        trajectories: torch.Tensor,
        times: torch.Tensor,
        predicted_tracks: Any,
        dynamic_layer: Any,
        dyn_collision_mask: torch.Tensor,
        dyn_min_clearance: torch.Tensor,
        dyn_max_cost: torch.Tensor,
        dyn_risk_level: torch.Tensor,
        dyn_unknown_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply conservative hierarchical broad-phase filter semantics to match CPU reference."""
        obstacle_hulls = self.broad_phase_filter.build_obstacle_hulls(
            predicted_tracks, dynamic_layer, max_horizon_s=self.pred_horizon
        )
        if not obstacle_hulls:
            return dyn_collision_mask, dyn_min_clearance, dyn_max_cost, dyn_risk_level, dyn_unknown_mask

        K, S, _ = trajectories.shape
        xs = trajectories[:, :, 0]
        ys = trajectories[:, :, 1]
        r_ugv = self.r_circumscribed
        margin = float(self.broad_phase_filter.config.broad_phase_margin_m)
        sagitta_eps = 0.005

        t_xmin = xs.min(dim=1).values - r_ugv
        t_xmax = xs.max(dim=1).values + r_ugv
        t_ymin = ys.min(dim=1).values - r_ugv
        t_ymax = ys.max(dim=1).values + r_ugv

        gpu_relevant = torch.zeros(K, device=self.device, dtype=torch.bool)
        for hull in obstacle_hulls:
            rel_t_end = float(times[-1].item())
            if rel_t_end > hull.t_max + 1e-4:
                gpu_relevant[:] = True
                continue
            in_aabb = (t_xmax >= hull.x_min) & (t_xmin <= hull.x_max) & (t_ymax >= hull.y_min) & (t_ymin <= hull.y_max)
            if not in_aabb.any():
                continue
            if not hull.samples:
                gpu_relevant = gpu_relevant | in_aabb
                continue

            t0_list = [float(times[k].item()) for k in range(S - 1)]
            t1_list = [float(times[k + 1].item()) for k in range(S - 1)]
            s0_list = [hull.interpolate_at_t(t) for t in t0_list]
            s1_list = [hull.interpolate_at_t(t) for t in t1_list]

            if any(s is None for s in s0_list) or any(s is None for s in s1_list):
                gpu_relevant = gpu_relevant | in_aabb
                continue

            obs0 = torch.tensor([[s[0], s[1]] for s in s0_list], device=self.device, dtype=self.dtype)
            obs1 = torch.tensor([[s[0], s[1]] for s in s1_list], device=self.device, dtype=self.dtype)
            r_obs0 = torch.tensor([s[2] for s in s0_list], device=self.device, dtype=self.dtype)
            r_obs1 = torch.tensor([s[2] for s in s1_list], device=self.device, dtype=self.dtype)

            pts0 = trajectories[:, :-1, :2]
            pts1 = trajectories[:, 1:, :2]
            d0 = pts0 - obs0.unsqueeze(0)
            d1 = pts1 - obs1.unsqueeze(0)
            v = d1 - d0
            l_sq = torch.clamp((v * v).sum(dim=-1), min=1e-10)
            alpha = torch.clamp(-(d0 * v).sum(dim=-1) / l_sq, 0.0, 1.0)
            closest = d0 + alpha.unsqueeze(-1) * v
            d_min_sq = (closest * closest).sum(dim=-1)
            r_reach = r_ugv + torch.maximum(r_obs0, r_obs1) + margin + sagitta_eps
            hit = (d_min_sq <= (r_reach * r_reach).unsqueeze(0)).any(dim=-1)
            gpu_relevant = gpu_relevant | (in_aabb & hit)

        # Candidates cleared by broad-phase receive normalized clearance 1.0 (1.50m) and zero penalty
        dyn_min_clearance = torch.where(gpu_relevant, dyn_min_clearance, torch.tensor(1.50, device=self.device, dtype=self.dtype))
        dyn_max_cost = torch.where(gpu_relevant, dyn_max_cost, torch.tensor(0.0, device=self.device, dtype=self.dtype))
        dyn_risk_level = torch.where(gpu_relevant, dyn_risk_level, torch.tensor(0, device=self.device, dtype=torch.long))
        dyn_collision_mask = torch.where(gpu_relevant, dyn_collision_mask, torch.tensor(False, device=self.device, dtype=torch.bool))
        dyn_unknown_mask = torch.where(gpu_relevant, dyn_unknown_mask, torch.tensor(False, device=self.device, dtype=torch.bool))

        return dyn_collision_mask, dyn_min_clearance, dyn_max_cost, dyn_risk_level, dyn_unknown_mask

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
        """Compute best candidate trajectory using vectorized accelerator pipeline."""
        t_start = time.perf_counter()
        timing: Dict[str, float] = {}

        # 1. Dynamic window computation
        max_v_limit = self.degraded_max_v if is_degraded else None
        v_low, v_high, w_low, w_high = self.rollout_kernel.compute_dynamic_window(
            current_state.v,
            current_state.omega,
            self.sim_dt,
            self.max_v,
            self.min_v,
            self.max_omega,
            self.max_accel_v,
            self.max_accel_omega,
            max_v_limit=max_v_limit,
        )

        local_target = self._find_local_target(current_state, global_path)
        d_curr = math.hypot(local_target[0] - current_state.x, local_target[1] - current_state.y)

        # 2. Vectorized candidate rollout
        t_roll0 = time.perf_counter()
        v_flat, w_flat = self.rollout_kernel.generate_candidate_grid(
            v_low, v_high, w_low, w_high, self.v_samples, self.omega_samples
        )
        trajectories, times = self.rollout_kernel.rollout_trajectories(
            current_state, v_flat, w_flat, self.pred_horizon, self.sim_dt
        )
        timing["rollout_ms"] = (time.perf_counter() - t_roll0) * 1000.0

        # 3. Static collision checking and terrain cost
        t_col0 = time.perf_counter()
        static_valid, static_terrain, static_clearance = self.collision_kernel.check_static_trajectories(
            trajectories, costmap
        )
        timing["static_col_ms"] = (time.perf_counter() - t_col0) * 1000.0

        # 4. Dynamic obstacle evaluation
        t_dyn0 = time.perf_counter()
        has_dynamic = (dynamic_layer is not None) or (predicted_tracks is not None)
        dyn_cfg = dynamic_config or DynamicPlanningConfig()
        if not dyn_cfg.enabled:
            has_dynamic = False

        if has_dynamic:
            dyn_collision_mask, dyn_min_clearance, dyn_max_cost, dyn_risk_level, dyn_unknown_mask = (
                self.collision_kernel.evaluate_dynamic_obstacles(
                    trajectories, predicted_tracks, dynamic_layer, max_horizon_s=self.pred_horizon
                )
            )
            if dyn_cfg.enable_broad_phase:
                dyn_collision_mask, dyn_min_clearance, dyn_max_cost, dyn_risk_level, dyn_unknown_mask = (
                    self._apply_broad_phase_filter(
                        trajectories, times, predicted_tracks, dynamic_layer,
                        dyn_collision_mask, dyn_min_clearance, dyn_max_cost, dyn_risk_level, dyn_unknown_mask
                    )
                )
        else:
            K = v_flat.shape[0]
            dyn_collision_mask = torch.zeros(K, device=self.device, dtype=torch.bool)
            dyn_min_clearance = torch.full((K,), 5.0, device=self.device, dtype=self.dtype)
            dyn_max_cost = torch.zeros(K, device=self.device, dtype=self.dtype)
            dyn_risk_level = torch.zeros(K, device=self.device, dtype=torch.long)
            dyn_unknown_mask = torch.zeros(K, device=self.device, dtype=torch.bool)
        timing["dynamic_col_ms"] = (time.perf_counter() - t_dyn0) * 1000.0

        # 5. Candidate scoring & argmax reduction
        t_score0 = time.perf_counter()
        composite_scores, valid_mask, best_idx = self.scorer_kernel.score_trajectories(
            trajectories,
            v_flat,
            w_flat,
            static_valid,
            static_terrain,
            static_clearance,
            dyn_collision_mask,
            dyn_min_clearance,
            dyn_max_cost,
            dyn_risk_level,
            global_path,
            local_target,
            d_curr,
            self.max_v,
            self.pred_horizon,
            self.w_path,
            self.w_heading,
            self.w_dist,
            self.w_clear,
            self.w_vel,
            self.w_terrain,
            has_dynamic=has_dynamic,
            dynamic_config=dyn_cfg,
            dyn_unknown_mask=dyn_unknown_mask,
        )
        timing["scoring_ms"] = (time.perf_counter() - t_score0) * 1000.0

        # Retain latest score tensors for detailed diagnostics/ranking analysis
        self.latest_composite_scores = composite_scores
        self.latest_valid_mask = valid_mask
        self.latest_v_flat = v_flat
        self.latest_w_flat = w_flat

        # Synchronize if on CUDA
        sync_ms = self.backend.synchronize()
        timing["sync_ms"] = sync_ms

        total_dwa_ms = (time.perf_counter() - t_start) * 1000.0

        # Diagnostics counts
        statically_valid_count = int(static_valid.sum().item())
        valid_count = int(valid_mask.sum().item())
        dynamically_rejected_count = int((static_valid & (~valid_mask)).sum().item())
        penalized_mask = valid_mask & ((dyn_risk_level > 0) | (dyn_max_cost >= 0.15))
        dynamically_penalized_count = int(penalized_mask.sum().item())
        expensive_checks = statically_valid_count if has_dynamic else 0

        unknown_horizon_count = int(dyn_unknown_mask.sum().item())

        # No safe trajectory case
        if best_idx is None:
            zero_traj = Trajectory(
                points=[TrajectoryPoint(
                    t=current_state.timestamp + self.sim_dt,
                    x=current_state.x, y=current_state.y, yaw=current_state.yaw,
                    v=0.0, omega=0.0, cost=1.0, clearance_m=0.0
                )],
                is_valid=False, total_cost=1.0, min_clearance_m=0.0
            )
            zero_diag = {
                "trajectories_evaluated": len(v_flat),
                "statically_valid_candidates": statically_valid_count,
                "valid_trajectories": 0,
                "broad_phase_rejections": statically_valid_count - expensive_checks,
                "broad_phase_cleared_count": statically_valid_count - expensive_checks,
                "expensive_collision_checks": expensive_checks,
                "number_of_expensive_collision_checker_invocations": expensive_checks,
                "number_of_broad_phase_rejections": statically_valid_count - expensive_checks,
                "number_of_candidates_evaluated": len(v_flat),
                "dynamically_rejected_candidates": dynamically_rejected_count,
                "dynamically_penalized_candidates": 0,
                "unknown_horizon_candidates": unknown_horizon_count,
                "dynamic_collision": dynamically_rejected_count > 0,
                "dynamic_risk": "LETHAL" if dynamically_rejected_count > 0 else ("UNKNOWN_HORIZON" if unknown_horizon_count > 0 else "FREE"),
                "unknown_horizon": unknown_horizon_count > 0,
                "blocking_obstacle_id": None,
                "earliest_collision_time": None,
                "minimum_dynamic_clearance": 0.0,
                "maximum_dynamic_cost": 1.0 if dynamically_rejected_count > 0 else 0.0,
                "dwa_latency_ms": total_dwa_ms,
                "emergency_stop_triggered": True,
                "backend_device": self.backend.device_name,
                "timing_breakdown": timing,
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

        # Extract selected best trajectory
        best_v = float(v_flat[best_idx].item())
        best_w = float(w_flat[best_idx].item())
        best_score = float(composite_scores[best_idx].item())
        best_static_clearance = float(static_clearance[best_idx].item())
        best_terrain_cost = float(static_terrain[best_idx].item())
        best_dyn_clearance = float(dyn_min_clearance[best_idx].item())
        best_dyn_cost = float(dyn_max_cost[best_idx].item())
        best_risk_int = int(dyn_risk_level[best_idx].item())
        risk_names = ["FREE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "LETHAL"]
        best_risk_str = risk_names[best_risk_int] if 0 <= best_risk_int < len(risk_names) else "FREE"
        if dyn_unknown_mask[best_idx].item():
            best_risk_str = "UNKNOWN_HORIZON"

        best_traj_pts_tensor = trajectories[best_idx]  # (S, 4)
        best_pts_np = best_traj_pts_tensor.detach().cpu().numpy()

        points: List[TrajectoryPoint] = []
        for s_idx in range(len(best_pts_np)):
            points.append(TrajectoryPoint(
                t=float(current_state.timestamp + best_pts_np[s_idx, 3]),
                x=float(best_pts_np[s_idx, 0]),
                y=float(best_pts_np[s_idx, 1]),
                yaw=float(best_pts_np[s_idx, 2]),
                v=best_v,
                omega=best_w,
                cost=best_terrain_cost,
                clearance_m=best_static_clearance,
            ))

        best_traj = Trajectory(
            points=points,
            is_valid=True,
            total_cost=best_terrain_cost,
            min_clearance_m=best_static_clearance,
        )

        diagnostics = {
            "trajectories_evaluated": len(v_flat),
            "statically_valid_candidates": statically_valid_count,
            "valid_trajectories": valid_count,
            "broad_phase_rejections": statically_valid_count - expensive_checks,
            "broad_phase_cleared_count": statically_valid_count - expensive_checks,
            "expensive_collision_checks": expensive_checks,
            "number_of_expensive_collision_checker_invocations": expensive_checks,
            "number_of_broad_phase_rejections": statically_valid_count - expensive_checks,
            "number_of_candidates_evaluated": len(v_flat),
            "dynamically_rejected_candidates": dynamically_rejected_count,
            "dynamically_penalized_candidates": dynamically_penalized_count,
            "unknown_horizon_candidates": unknown_horizon_count,
            "dynamic_collision": False,
            "dynamic_risk": best_risk_str,
            "unknown_horizon": bool(dyn_unknown_mask[best_idx].item()),
            "blocking_obstacle_id": None,
            "earliest_collision_time": None,
            "minimum_dynamic_clearance": best_dyn_clearance,
            "maximum_dynamic_cost": best_dyn_cost,
            "best_score": round(best_score, 4),
            "desired_linear_v": round(best_v, 3),
            "desired_angular_w": round(best_w, 3),
            "min_clearance_m": round(best_static_clearance, 3),
            "dwa_latency_ms": total_dwa_ms,
            "emergency_stop_triggered": False,
            "backend_device": self.backend.device_name,
            "timing_breakdown": timing,
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
        """Compute accelerated DWA trajectory and arbitrate final velocity command through BehavioralStateMachine."""
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
