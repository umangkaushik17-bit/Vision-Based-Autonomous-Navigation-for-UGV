"""Batched static and dynamic collision checking kernels using PyTorch tensors.

Implements exact Separating Axis Theorem (SAT) 2D polygon intersection,
convex polygon clearance calculation, and static footprint raster collision.
Preserves 100% mathematical parity with Phase 13 CPU collision models.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from ai.planning.base import VehicleKinematics
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionConfig,
    PredictiveCollisionResult,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer, DynamicRiskLevel
from ai.planning.dynamic_gpu.gpu_backend import GPUBackend


class GPUCollisionChecker:
    """Hardware-accelerated collision and clearance evaluator."""

    def __init__(
        self,
        backend: GPUBackend,
        vehicle: Optional[VehicleKinematics] = None,
        config: Optional[PredictiveCollisionConfig] = None,
    ) -> None:
        self.backend = backend
        self.device = backend.device
        self.dtype = backend.dtype

        self.vehicle = vehicle or VehicleKinematics()
        self.config = config or PredictiveCollisionConfig()

        self.length = float(self.vehicle.length_m)
        self.width = float(self.vehicle.width_m)
        self.margin = float(getattr(self.vehicle, "collision_margin_m", 0.05))
        self.r_circumscribed = math.hypot(self.length / 2.0 + self.margin, self.width / 2.0 + self.margin)

        # Precompute static footprint sample point offsets in body frame (P, 2)
        self.footprint_offsets = self._build_footprint_offsets(margin_m=self.margin)

    def _build_footprint_offsets(self, margin_m: float = 0.0) -> torch.Tensor:
        """Precompute vehicle footprint evaluation points in robot frame."""
        hl = (self.length / 2.0) + margin_m
        hw = (self.width / 2.0) + margin_m

        corners = [
            (hl, hw),
            (hl, -hw),
            (-hl, -hw),
            (-hl, hw),
        ]
        pts: List[Tuple[float, float]] = []

        # 1. Corners + perimeter samples (matching edge_step_m = 0.05)
        edge_step = 0.05
        for i in range(4):
            cx, cy = corners[i]
            pts.append((cx, cy))
            nx, ny = corners[(i + 1) % 4]
            edge_dist = math.hypot(nx - cx, ny - cy)
            n_sub = max(1, int(math.ceil(edge_dist / edge_step)))
            for k in range(1, n_sub):
                alpha = k / float(n_sub)
                pts.append((cx + alpha * (nx - cx), cy + alpha * (ny - cy)))

        # 2. Interior samples (7 x 5 grid matching CPU collision.py)
        dx_samples = np.linspace(-hl * 0.85, hl * 0.85, 7)
        dy_samples = np.linspace(-hw * 0.80, hw * 0.80, 5)
        for dx in dx_samples:
            for dy in dy_samples:
                pts.append((float(dx), float(dy)))

        return torch.tensor(pts, device=self.device, dtype=self.dtype)

    def check_static_trajectories(
        self,
        trajectories: torch.Tensor,
        costmap: Any,
        max_cost: float = 0.65,
        critical_hazard: float = 0.70,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate static footprint collision, terrain costs, and clearances across all trajectories.

        Args:
            trajectories: (K, S, 4) tensor [x, y, yaw, t]
            costmap: WorldTerrainMap or BEVTerrainCostmap
            max_cost: Lethal cost threshold
            critical_hazard: Lethal hazard threshold

        Returns:
            is_valid: (K,) bool tensor (True if collision-free at all steps)
            avg_terrain_cost: (K,) float tensor of accumulated terrain costs
            min_clearance: (K,) float tensor of minimum obstacle clearance (m)
        """
        K, S, _ = trajectories.shape
        device = self.device

        # Costmap metadata
        res = float(getattr(costmap, "resolution_m", getattr(costmap, "resolution", 0.10)))
        x_bounds = getattr(costmap, "x_bounds_m", getattr(costmap, "x_range_m", (-20.0, 20.0)))
        y_bounds = getattr(costmap, "y_bounds_m", getattr(costmap, "y_range_m", (-20.0, 20.0)))
        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        H, W = int(shape[0]), int(shape[1])

        # Extract map layers as tensors
        obs_tensor = torch.as_tensor(costmap.obstacle if hasattr(costmap, "obstacle") else np.zeros((H, W), dtype=bool), device=device)
        cost_tensor = torch.as_tensor(costmap.cost, device=device, dtype=self.dtype)
        has_hazard = hasattr(costmap, "hazard")
        hazard_tensor = torch.as_tensor(costmap.hazard, device=device, dtype=self.dtype) if has_hazard else None

        # Static footprint offsets (exact Phase 13 reference footprint without artificial expansion)
        static_offsets = self._build_footprint_offsets(margin_m=0.0)

        # Transform footprint points across all (K, S) poses
        # Shape: (K, S, 1) and (1, 1, P)
        x = trajectories[:, :, 0:1]  # (K, S, 1)
        y = trajectories[:, :, 1:2]  # (K, S, 1)
        yaw = trajectories[:, :, 2:3]  # (K, S, 1)

        cos_yaw = torch.cos(yaw)  # (K, S, 1)
        sin_yaw = torch.sin(yaw)  # (K, S, 1)

        dx = static_offsets[:, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, P)
        dy = static_offsets[:, 1].unsqueeze(0).unsqueeze(0)  # (1, 1, P)

        # Rotated world footprint positions: (K, S, P)
        px = x + dx * cos_yaw - dy * sin_yaw
        py = y + dx * sin_yaw + dy * cos_yaw

        # Convert to grid indices (row, col) matching WorldTerrainMap / BEVTerrainCostmap convention:
        # row = floor((x_max - x) / res), col = floor((y_max - y) / res)
        grid_r = torch.floor((x_max - px) / res).to(torch.long)
        grid_c = torch.floor((y_max - py) / res).to(torch.long)

        # 1. Bounds check
        in_bounds = (px >= x_min) & (px < x_max) & (py >= y_min) & (py < y_max) & (grid_r >= 0) & (grid_r < H) & (grid_c >= 0) & (grid_c < W)

        # Clamp indices for safe tensor lookup (out of bounds will be flagged by in_bounds mask)
        clamped_r = torch.clamp(grid_r, 0, H - 1)
        clamped_c = torch.clamp(grid_c, 0, W - 1)

        # 2. Obstacle & Hazard check
        point_obs = obs_tensor[clamped_r, clamped_c]
        point_cost = cost_tensor[clamped_r, clamped_c]

        point_collision = (~in_bounds) | point_obs | (point_cost > max_cost)
        if hazard_tensor is not None:
            point_hazard = hazard_tensor[clamped_r, clamped_c]
            point_collision = point_collision | (point_hazard >= critical_hazard)

        # Trajectory has collision if ANY footprint point at ANY step collides
        # (K, S, P) -> (K,)
        traj_collision = point_collision.any(dim=-1).any(dim=-1)
        is_valid = ~traj_collision

        # 3. Full Weighted Terrain Cost Evaluation matching TerrainAwareCostEvaluator
        trav_raw = costmap.traversability if hasattr(costmap, "traversability") else np.full((H, W), 0.5, dtype=np.float64)
        trav_tensor = torch.as_tensor(trav_raw, device=device, dtype=self.dtype)
        slope_raw = costmap.slope_deg if hasattr(costmap, "slope_deg") else (costmap.slope if hasattr(costmap, "slope") else np.zeros((H, W), dtype=np.float64))
        slope_tensor = torch.as_tensor(slope_raw, device=device, dtype=self.dtype)
        rough_raw = costmap.roughness if hasattr(costmap, "roughness") else np.zeros((H, W), dtype=np.float64)
        rough_tensor = torch.as_tensor(rough_raw, device=device, dtype=self.dtype)
        hvar_raw = costmap.height_var if hasattr(costmap, "height_var") else np.zeros((H, W), dtype=np.float64)
        hvar_tensor = torch.as_tensor(hvar_raw, device=device, dtype=self.dtype)

        haz_val = hazard_tensor if hazard_tensor is not None else torch.zeros((H, W), device=device, dtype=self.dtype)
        unk_tensor = torch.as_tensor(costmap.unknown, device=device, dtype=torch.bool) if hasattr(costmap, "unknown") else None
        unk_val = torch.where(unk_tensor, torch.tensor(0.80, device=device, dtype=self.dtype), torch.tensor(0.0, device=device, dtype=self.dtype)) if unk_tensor is not None else torch.zeros((H, W), device=device, dtype=self.dtype)

        terrain_cost_map = (
            1.50 * cost_tensor +
            0.80 * (1.0 - trav_tensor) +
            2.50 * haz_val +
            1.20 * torch.clamp(slope_tensor / 20.0, 0.0, 2.0) +
            0.60 * rough_tensor +
            1.00 * torch.clamp(hvar_tensor / 0.12, 0.0, 2.0) +
            1.00 * unk_val
        )
        lethal_mask = obs_tensor | (cost_tensor >= 0.85) | (haz_val >= critical_hazard)
        terrain_cost_map = torch.where(lethal_mask, torch.tensor(float("inf"), device=device, dtype=self.dtype), terrain_cost_map)

        center_x = trajectories[:, :, 0]  # (K, S)
        center_y = trajectories[:, :, 1]  # (K, S)
        center_r = torch.clamp(torch.floor((x_max - center_x) / res).to(torch.long), 0, H - 1)
        center_c = torch.clamp(torch.floor((y_max - center_y) / res).to(torch.long), 0, W - 1)

        center_costs = terrain_cost_map[center_r, center_c]  # (K, S)
        inf_mask = torch.isinf(center_costs).any(dim=-1)
        is_valid = is_valid & (~inf_mask)

        avg_terrain_cost = center_costs.mean(dim=-1)

        # 4. Approximate Static Clearance
        # If map has obstacles, compute distance to nearest obstacle cell
        if obs_tensor.any():
            obs_indices = obs_tensor.nonzero(as_tuple=False)  # (M, 2) [r, c]
            obs_x = x_max - (obs_indices[:, 0].to(self.dtype) + 0.5) * res  # (M,)
            obs_y = y_max - (obs_indices[:, 1].to(self.dtype) + 0.5) * res  # (M,)
            obs_pts = torch.stack([obs_x, obs_y], dim=-1)  # (M, 2)

            # Compute min distance for each trajectory center points
            pts_eval = trajectories[:, :, :2].reshape(-1, 2)  # (K*S, 2)
            dists = torch.cdist(pts_eval, obs_pts)  # (K*S, M)
            min_pts_dist = dists.min(dim=-1).values.reshape(K, S)  # (K, S)
            min_clearance = min_pts_dist.min(dim=-1).values  # (K,)
        else:
            min_clearance = torch.full((K,), 2.0, device=device, dtype=self.dtype)

        return is_valid, avg_terrain_cost, min_clearance

    def evaluate_dynamic_obstacles(
        self,
        trajectories: torch.Tensor,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]],
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        max_horizon_s: float = 1.50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate dynamic obstacle collision, clearance, and risk penalties across candidate trajectories.

        Args:
            trajectories: (K, S, 4) tensor [x, y, yaw, t]
            predicted_tracks: Sequence of dynamic obstacle trajectories
            dynamic_layer: Optional occupancy layer
            max_horizon_s: Planning horizon

        Returns:
            dyn_collision_mask: (K,) bool tensor (True if physical SAT collision occurs)
            dyn_min_clearance: (K,) float tensor of minimum dynamic clearance (m)
            dyn_max_cost: (K,) float tensor of maximum risk cost encountered
            dyn_risk_level: (K,) int tensor of highest risk level encountered
        """
        K, S, _ = trajectories.shape
        device = self.device

        # If no tracks or empty tracks, all candidates are safe
        tracks_list: List[PredictedObstacleTrajectory] = []
        if predicted_tracks is not None:
            if isinstance(predicted_tracks, PredictedTrajectoryArray):
                tracks_list = [t for t in predicted_tracks.predictions if t.current_state.status.upper() != "EXPIRED"]
            else:
                tracks_list = [t for t in predicted_tracks if t.current_state.status.upper() != "EXPIRED"]

        dyn_collision_mask = torch.zeros(K, device=device, dtype=torch.bool)
        dyn_min_clearance = torch.full((K,), float(self.config.max_clearance_search_m), device=device, dtype=self.dtype)
        dyn_max_cost = torch.zeros(K, device=device, dtype=self.dtype)
        dyn_risk_level = torch.zeros(K, device=device, dtype=torch.long)
        dyn_unknown_mask = torch.zeros(K, device=device, dtype=torch.bool)

        dyn_footprint_offsets = self._build_footprint_offsets(margin_m=self.margin)

        # 1. Dynamic Layer Raster Evaluation (if provided)
        if dynamic_layer is not None:
            res_dyn = float(dynamic_layer.config.resolution_m)
            x_max_dyn = float(dynamic_layer.config.x_bounds_m[1])
            y_max_dyn = float(dynamic_layer.config.y_bounds_m[1])
            H_dyn = dynamic_layer.n_rows
            W_dyn = dynamic_layer.n_cols

            for s in range(S):
                t_s = float(trajectories[0, s, 3].item())
                grid_slice = dynamic_layer.get_slice(t_s, conservative_union=True)

                x_s = trajectories[:, s, 0:1]  # (K, 1)
                y_s = trajectories[:, s, 1:2]  # (K, 1)
                yaw_s = trajectories[:, s, 2:3]  # (K, 1)

                cos_s = torch.cos(yaw_s)
                sin_s = torch.sin(yaw_s)

                px = x_s + dyn_footprint_offsets[:, 0].unsqueeze(0) * cos_s - dyn_footprint_offsets[:, 1].unsqueeze(0) * sin_s
                py = y_s + dyn_footprint_offsets[:, 0].unsqueeze(0) * sin_s + dyn_footprint_offsets[:, 1].unsqueeze(0) * cos_s

                # Exact DynamicOccupancyLayer coordinate convention:
                # row = floor((x_max - x) / res), col = floor((y_max - y) / res)
                grid_r = torch.floor((x_max_dyn - px) / res_dyn).to(torch.long)
                grid_c = torch.floor((y_max_dyn - py) / res_dyn).to(torch.long)

                in_b = (grid_r >= 0) & (grid_r < H_dyn) & (grid_c >= 0) & (grid_c < W_dyn)
                cr = torch.clamp(grid_r, 0, H_dyn - 1)
                cc = torch.clamp(grid_c, 0, W_dyn - 1)

                slice_cost = torch.as_tensor(grid_slice.cost, device=device, dtype=self.dtype)
                slice_risk = torch.as_tensor(grid_slice.risk_level, device=device, dtype=torch.long)
                slice_nom = torch.as_tensor(grid_slice.nominal_obstacle, device=device, dtype=torch.bool)

                pt_cost = slice_cost[cr, cc]
                pt_risk = slice_risk[cr, cc]
                pt_nom = slice_nom[cr, cc]

                pt_hit = (~in_b) | pt_nom | (pt_cost >= 1.0) | (pt_risk == 4)
                traj_hit = pt_hit.any(dim=-1)

                if not tracks_list:
                    dyn_collision_mask = dyn_collision_mask | traj_hit
                dyn_max_cost = torch.maximum(dyn_max_cost, pt_cost.max(dim=-1).values)
                dyn_risk_level = torch.maximum(dyn_risk_level, pt_risk.max(dim=-1).values)

        # 2. Exact Track Evaluation (SAT and Clearance)
        ugv_obbs = self._compute_ugv_obbs(trajectories)
        future_times = trajectories[0, :, 3]

        for trk in tracks_list:
            extracted = self._extract_track_tensors(trk, future_times)
            if extracted[0] is None:
                continue
            obs_obbs, obs_radii, obs_centers, obs_uncs, step_valid = extracted

            # Broad-phase distance filter: (K, S)
            # Distance from UGV center to obstacle center at each step
            ugv_centers = trajectories[:, :, :2]  # (K, S, 2)
            d_centers = torch.norm(ugv_centers - obs_centers.unsqueeze(0), dim=-1)  # (K, S)
            search_thresh = self.r_circumscribed + obs_radii.unsqueeze(0) + self.config.max_clearance_search_m
            broad_phase_mask = d_centers <= search_thresh  # (K, S) candidates needing exact check

            if (future_times > trk.horizon_s + 1e-4).any():
                dyn_unknown_mask = dyn_unknown_mask | broad_phase_mask.any(dim=1)

            if not broad_phase_mask.any():
                continue

            # Exact 8-axis SAT collision check where broad_phase_mask is True
            sat_collision, clearance, risk_cost, risk_lvl = self._evaluate_sat_and_clearance(
                ugv_obbs, obs_obbs, obs_uncs, broad_phase_mask, step_valid_mask=step_valid
            )

            # Update candidate summaries
            dyn_collision_mask = dyn_collision_mask | sat_collision
            dyn_min_clearance = torch.minimum(dyn_min_clearance, clearance)
            dyn_max_cost = torch.maximum(dyn_max_cost, risk_cost)
            dyn_risk_level = torch.maximum(dyn_risk_level, risk_lvl)

        return dyn_collision_mask, dyn_min_clearance, dyn_max_cost, dyn_risk_level, dyn_unknown_mask

    def _compute_ugv_obbs(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute oriented bounding box corners (K, S, 4, 2) for UGV poses."""
        K, S, _ = trajectories.shape
        hl = (self.length / 2.0) + self.margin
        hw = (self.width / 2.0) + self.margin

        x = trajectories[:, :, 0]  # (K, S)
        y = trajectories[:, :, 1]  # (K, S)
        yaw = trajectories[:, :, 2]  # (K, S)

        c = torch.cos(yaw)  # (K, S)
        s = torch.sin(yaw)  # (K, S)

        # 4 corners in local frame: (+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)
        c0_x = x + hl * c - hw * s
        c0_y = y + hl * s + hw * c
        c1_x = x + hl * c - (-hw) * s
        c1_y = y + hl * s + (-hw) * c
        c2_x = x + (-hl) * c - (-hw) * s
        c2_y = y + (-hl) * s + (-hw) * c
        c3_x = x + (-hl) * c - hw * s
        c3_y = y + (-hl) * s + hw * c

        c0 = torch.stack([c0_x, c0_y], dim=-1)  # (K, S, 2)
        c1 = torch.stack([c1_x, c1_y], dim=-1)
        c2 = torch.stack([c2_x, c2_y], dim=-1)
        c3 = torch.stack([c3_x, c3_y], dim=-1)

        obbs = torch.stack([c0, c1, c2, c3], dim=2)  # (K, S, 4, 2)
        return obbs

    def _extract_track_tensors(
        self,
        track: PredictedObstacleTrajectory,
        future_times: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Interpolate obstacle track waypoints to match planning time steps."""
        S = future_times.shape[0]
        obs_size = track.current_state.bbox_size
        hl = obs_size[0] / 2.0
        hw = obs_size[1] / 2.0
        r_obs = math.hypot(hl, hw)

        obbs_list = []
        centers_list = []
        uncs_list = []
        step_valid_list = []

        for s in range(S):
            t_s = float(future_times[s].item())
            is_valid_step = t_s <= (track.horizon_s + 1e-4)
            step_valid_list.append(is_valid_step)
            # Find matching waypoint or interpolate
            wp_data = self._interpolate_waypoint(track, t_s)
            if wp_data is None:
                return None, None, None, None, None

            ox, oy, oyaw, or_unc = wp_data
            c = math.cos(oyaw)
            s_val = math.sin(oyaw)

            c0 = [ox + hl * c - hw * s_val, oy + hl * s_val + hw * c]
            c1 = [ox + hl * c + hw * s_val, oy + hl * s_val - hw * c]
            c2 = [ox - hl * c + hw * s_val, oy - hl * s_val - hw * c]
            c3 = [ox - hl * c - hw * s_val, oy - hl * s_val + hw * c]

            obbs_list.append([c0, c1, c2, c3])
            centers_list.append([ox, oy])
            uncs_list.append(or_unc)

        obbs = torch.tensor(obbs_list, device=self.device, dtype=self.dtype)  # (S, 4, 2)
        centers = torch.tensor(centers_list, device=self.device, dtype=self.dtype)  # (S, 2)
        radii = torch.full((S,), r_obs, device=self.device, dtype=self.dtype)  # (S,)
        uncs = torch.tensor(uncs_list, device=self.device, dtype=self.dtype)  # (S,)
        step_valid = torch.tensor(step_valid_list, device=self.device, dtype=torch.bool)  # (S,)

        return obbs, radii, centers, uncs, step_valid

    def _interpolate_waypoint(
        self,
        track: PredictedObstacleTrajectory,
        target_t: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Find interpolated track position, heading, and uncertainty at target_t."""
        wps = track.waypoints
        if not wps:
            return None
        if target_t < -1e-6:
            return None

        # Clamp to track horizon for extrapolation beyond horizon
        query_t = min(float(target_t), float(track.horizon_s))

        yaw = float(track.current_state.bbox_yaw)

        def get_unc(w: Any) -> float:
            return float(getattr(w, "uncertainty_radius", getattr(w, "position_uncertainty_radius", 0.10)))

        for wp in wps:
            if abs(wp.time_offset - query_t) < 1e-4:
                return (float(wp.x), float(wp.y), yaw, get_unc(wp))

        if query_t <= wps[0].time_offset:
            return (float(wps[0].x), float(wps[0].y), yaw, get_unc(wps[0]))
        if query_t >= wps[-1].time_offset:
            return (float(wps[-1].x), float(wps[-1].y), yaw, get_unc(wps[-1]))

        for i in range(len(wps) - 1):
            w1 = wps[i]
            w2 = wps[i + 1]
            if w1.time_offset <= query_t <= w2.time_offset:
                dt = w2.time_offset - w1.time_offset
                if dt < 1e-6:
                    return (float(w1.x), float(w1.y), yaw, get_unc(w1))
                alpha = (query_t - w1.time_offset) / dt
                x = float(w1.x + alpha * (w2.x - w1.x))
                y = float(w1.y + alpha * (w2.y - w1.y))
                unc = float(get_unc(w1) + alpha * (get_unc(w2) - get_unc(w1)))
                return (x, y, yaw, unc)
        return None

    def _evaluate_sat_and_clearance(
        self,
        ugv_obbs: torch.Tensor,  # (K, S, 4, 2)
        obs_obbs: torch.Tensor,  # (S, 4, 2)
        obs_uncs: torch.Tensor,  # (S,)
        broad_phase_mask: torch.Tensor,  # (K, S)
        step_valid_mask: Optional[torch.Tensor] = None,  # (S,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform vectorized 8-axis SAT collision check and clearance calculation."""
        K, S, _, _ = ugv_obbs.shape
        device = self.device

        if step_valid_mask is not None:
            broad_phase_mask = broad_phase_mask & step_valid_mask.unsqueeze(0)

        # 1. Fully batched 8-axis SAT across all (K, S) candidates
        e1 = ugv_obbs[:, :, [1, 2, 3, 0]] - ugv_obbs  # (K, S, 4, 2)
        e2 = obs_obbs[:, [1, 2, 3, 0]] - obs_obbs    # (S, 4, 2)
        n1 = torch.stack([-e1[:, :, :, 1], e1[:, :, :, 0]], dim=-1)  # (K, S, 4, 2)
        n2 = torch.stack([-e2[:, :, 1], e2[:, :, 0]], dim=-1).unsqueeze(0).expand(K, -1, -1, -1)  # (K, S, 4, 2)
        axes = torch.cat([n1, n2], dim=2)  # (K, S, 8, 2)

        proj1 = torch.matmul(ugv_obbs, axes.transpose(-1, -2))  # (K, S, 4, 8)
        obs_exp = obs_obbs.unsqueeze(0).expand(K, -1, -1, -1)   # (K, S, 4, 2)
        proj2 = torch.matmul(obs_exp, axes.transpose(-1, -2))   # (K, S, 4, 8)

        min1, max1 = proj1.min(dim=2).values, proj1.max(dim=2).values  # (K, S, 8)
        min2, max2 = proj2.min(dim=2).values, proj2.max(dim=2).values  # (K, S, 8)
        separated = ((max1 < min2 - 1e-9) | (max2 < min1 - 1e-9)).any(dim=-1)  # (K, S)
        intersecting = (~separated) & broad_phase_mask  # (K, S)

        sat_collision = intersecting.any(dim=1)  # (K,) bool

        grid_clr = torch.full((K, S), float(self.config.max_clearance_search_m), device=device, dtype=self.dtype)
        grid_cost = torch.zeros((K, S), device=device, dtype=self.dtype)
        grid_risk = torch.zeros((K, S), device=device, dtype=torch.long)

        grid_clr = torch.where(intersecting, 0.0, grid_clr)
        grid_cost = torch.where(intersecting, 1.0, grid_cost)
        grid_risk = torch.where(intersecting, 4, grid_risk)

        # 2. Vectorized clearance on non-intersecting broad-phase pairs
        act_mask = broad_phase_mask & (~intersecting)
        if act_mask.any():
            act_ugv = ugv_obbs[act_mask]  # (N_act, 4, 2)
            act_obs = obs_exp[act_mask]   # (N_act, 4, 2)
            act_unc = obs_uncs.unsqueeze(0).expand(K, -1)[act_mask]  # (N_act,)

            p1_exp = act_ugv.unsqueeze(2)  # (N_act, 4, 1, 2)
            a2_exp = act_obs.unsqueeze(1)  # (N_act, 1, 4, 2)
            b2_exp = act_obs[:, [1, 2, 3, 0]].unsqueeze(1)  # (N_act, 1, 4, 2)
            ab = b2_exp - a2_exp
            ap = p1_exp - a2_exp
            ab_sq = torch.clamp(torch.sum(ab * ab, dim=-1, keepdim=True), min=1e-12)
            t = torch.clamp(torch.sum(ap * ab, dim=-1, keepdim=True) / ab_sq, 0.0, 1.0)
            closest = a2_exp + t * ab
            d1 = torch.norm(p1_exp - closest, dim=-1)  # (N_act, 4, 4)

            p2_exp = act_obs.unsqueeze(2)  # (N_act, 4, 1, 2)
            a1_exp = act_ugv.unsqueeze(1)  # (N_act, 1, 4, 2)
            b1_exp = act_ugv[:, [1, 2, 3, 0]].unsqueeze(1)  # (N_act, 1, 4, 2)
            ab1 = b1_exp - a1_exp
            ap1 = p2_exp - a1_exp
            ab1_sq = torch.clamp(torch.sum(ab1 * ab1, dim=-1, keepdim=True), min=1e-12)
            t1 = torch.clamp(torch.sum(ap1 * ab1, dim=-1, keepdim=True) / ab1_sq, 0.0, 1.0)
            closest1 = a1_exp + t1 * ab1
            d2 = torch.norm(p2_exp - closest1, dim=-1)  # (N_act, 4, 4)

            clr = torch.minimum(
                d1.reshape(act_ugv.shape[0], -1).min(dim=-1).values,
                d2.reshape(act_ugv.shape[0], -1).min(dim=-1).values
            )

            sigma = torch.clamp(act_unc / 3.0, min=1e-4)
            cost = torch.zeros_like(clr)
            risk = torch.zeros(clr.shape[0], device=device, dtype=torch.long)

            m1 = clr <= sigma
            cost = torch.where(m1, 0.95 - 0.25 * (clr / sigma), cost)
            risk = torch.where(m1, 3, risk)

            m2 = (~m1) & (clr <= 2.0 * sigma)
            cost = torch.where(m2, 0.70 - 0.30 * ((clr - sigma) / sigma), cost)
            risk = torch.where(m2, 2, risk)

            m3 = (~m1) & (~m2) & (clr <= act_unc)
            cost = torch.where(m3, 0.40 - 0.25 * ((clr - 2.0 * sigma) / sigma), cost)
            risk = torch.where(m3, 1, risk)

            grid_clr[act_mask] = clr
            grid_cost[act_mask] = cost
            grid_risk[act_mask] = risk

        min_clearance = grid_clr.min(dim=-1).values  # (K,)
        max_cost = grid_cost.max(dim=-1).values  # (K,)
        highest_risk = grid_risk.max(dim=-1).values  # (K,)

        return sat_collision, min_clearance, max_cost, highest_risk

    def _compute_clearance_tensor(self, poly1: torch.Tensor, poly2: torch.Tensor) -> float:
        """Compute exact minimum Euclidean clearance between two convex 4-gons."""
        # poly1: (4, 2), poly2: (4, 2)
        # 16 pairs: vertices of poly1 to edges of poly2
        p1 = poly1.unsqueeze(1)  # (4, 1, 2)
        a2 = poly2.unsqueeze(0)  # (1, 4, 2)
        b2 = poly2[[1, 2, 3, 0]].unsqueeze(0)  # (1, 4, 2)
        d1 = self._point_segment_dist_tensor(p1, a2, b2)

        # 16 pairs: vertices of poly2 to edges of poly1
        p2 = poly2.unsqueeze(1)  # (4, 1, 2)
        a1 = poly1.unsqueeze(0)  # (1, 4, 2)
        b1 = poly1[[1, 2, 3, 0]].unsqueeze(0)  # (1, 4, 2)
        d2 = self._point_segment_dist_tensor(p2, a1, b1)

        return float(min(d1.min().item(), d2.min().item()))

    def _point_segment_dist_tensor(
        self,
        p: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized point-to-segment distance."""
        ab = b - a
        ap = p - a
        seg_len_sq = (ab ** 2).sum(dim=-1)
        t = (ap * ab).sum(dim=-1) / torch.clamp(seg_len_sq, min=1e-12)
        t_clamped = torch.clamp(t, 0.0, 1.0).unsqueeze(-1)
        proj = a + t_clamped * ab
        dist = torch.norm(p - proj, dim=-1)
        return dist
