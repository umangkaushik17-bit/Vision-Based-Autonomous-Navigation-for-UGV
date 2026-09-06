"""Batched candidate trajectory multi-term scoring kernel using PyTorch tensors.

Computes exact multi-objective safety scores matching DWAPlanner and Milestone 13.9:
Path alignment, Heading alignment, Distance/Progress, Static Clearance, Velocity,
Terrain cost, Dynamic risk penalty, and Dynamic clearance reward.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch

from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic_gpu.gpu_backend import GPUBackend
from ai.planning.path import Path


class GPUScorer:
    """Vectorized trajectory evaluator and scoring reducer."""

    def __init__(self, backend: GPUBackend) -> None:
        self.backend = backend
        self.device = backend.device
        self.dtype = backend.dtype

    def score_trajectories(
        self,
        trajectories: torch.Tensor,
        v_candidates: torch.Tensor,
        w_candidates: torch.Tensor,
        static_valid_mask: torch.Tensor,
        static_terrain_cost: torch.Tensor,
        static_min_clearance: torch.Tensor,
        dyn_collision_mask: torch.Tensor,
        dyn_min_clearance: torch.Tensor,
        dyn_max_cost: torch.Tensor,
        dyn_risk_level: torch.Tensor,
        global_path: Path,
        local_target: Tuple[float, float],
        d_curr: float,
        max_v: float,
        pred_horizon: float,
        w_path: float = 0.30,
        w_heading: float = 0.15,
        w_dist: float = 0.25,
        w_clear: float = 0.20,
        w_vel: float = 0.15,
        w_terrain: float = 0.20,
        has_dynamic: bool = False,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Compute composite scores and identify best valid candidate.

        Returns:
            composite_scores: (K,) tensor of scores
            valid_mask: (K,) bool tensor of valid candidates
            best_idx: Index of highest scoring valid candidate, or None if empty
        """
        K, S, _ = trajectories.shape
        device = self.device
        dtype = self.dtype

        dyn_cfg = dynamic_config or DynamicPlanningConfig()

        end_x = trajectories[:, -1, 0]  # (K,)
        end_y = trajectories[:, -1, 1]  # (K,)
        end_yaw = trajectories[:, -1, 2]  # (K,)

        # 1. Path Alignment Score
        if len(global_path) >= 2:
            path_xy = torch.tensor(global_path.waypoints_xy, device=device, dtype=dtype)
            # Compute distance to all line segments: seg_a = path_xy[:-1], seg_b = path_xy[1:]
            seg_a = path_xy[:-1]  # (N-1, 2)
            seg_b = path_xy[1:]   # (N-1, 2)
            ab = seg_b - seg_a    # (N-1, 2)
            seg_len_sq = torch.clamp((ab ** 2).sum(dim=-1), min=1e-12)  # (N-1,)

            # For each candidate end point (K, 1, 2) vs segments (1, N-1, 2)
            p = torch.stack([end_x, end_y], dim=-1).unsqueeze(1)  # (K, 1, 2)
            ap = p - seg_a.unsqueeze(0)  # (K, N-1, 2)
            t = (ap * ab.unsqueeze(0)).sum(dim=-1) / seg_len_sq.unsqueeze(0)  # (K, N-1)
            t_clamped = torch.clamp(t, 0.0, 1.0).unsqueeze(-1)  # (K, N-1, 1)
            proj = seg_a.unsqueeze(0) + t_clamped * ab.unsqueeze(0)  # (K, N-1, 2)
            dists = torch.norm(p - proj, dim=-1)  # (K, N-1)
            min_cte = dists.min(dim=-1).values  # (K,)
            path_score = 1.0 / (1.0 + min_cte)
        elif len(global_path) == 1:
            px, py = float(global_path.points[0].x), float(global_path.points[0].y)
            min_cte = torch.hypot(end_x - px, end_y - py)
            path_score = 1.0 / (1.0 + min_cte)
        else:
            path_score = torch.ones(K, device=device, dtype=dtype)

        # 2. Heading Score (alignment to local target)
        tx, ty = float(local_target[0]), float(local_target[1])
        target_yaw = torch.atan2(ty - end_y, tx - end_x)
        heading_err = torch.abs(torch.remainder(target_yaw - end_yaw + math.pi, 2.0 * math.pi) - math.pi)
        heading_score = (math.pi - heading_err) / math.pi

        # 3. Distance & Forward Progress Score
        d_end = torch.hypot(tx - end_x, ty - end_y)
        progress = (d_curr - d_end) / max(0.1, max_v * pred_horizon)
        dist_score = 0.5 * (1.0 / (1.0 + d_end)) + 0.5 * torch.clamp(progress, min=0.0)

        # 4. Static Clearance Score
        clearance_score = torch.clamp(static_min_clearance, max=1.50) / 1.50

        # 5. Velocity Score
        vel_factor = torch.clamp(torch.cos(heading_err), min=0.0)
        pos_vel_score = (v_candidates / max(0.1, max_v)) * vel_factor
        neg_vel_score = -0.5 * torch.abs(v_candidates) / max_v
        vel_score = torch.where(v_candidates >= 0.0, pos_vel_score, neg_vel_score)

        # 6. Terrain Score
        terrain_score = 1.0 / (1.0 + static_terrain_cost)

        # 7. Dynamic Risk Penalties & Dynamic Clearance
        dyn_penalty = torch.zeros(K, device=device, dtype=dtype)
        dyn_clear_score = torch.zeros(K, device=device, dtype=dtype)
        dyn_rejected = torch.zeros(K, device=device, dtype=torch.bool)

        if has_dynamic:
            # Physical collision strictly rejects
            dyn_rejected = dyn_rejected | dyn_collision_mask

            # Risk level: 4=LETHAL, 3=HIGH, 2=MED, 1=LOW, 0=FREE
            high_mask = dyn_risk_level == 3
            med_mask = dyn_risk_level == 2
            low_mask = dyn_risk_level == 1

            if dyn_cfg.reject_high_risk:
                dyn_rejected = dyn_rejected | high_mask
            else:
                dyn_penalty = torch.where(
                    high_mask,
                    dyn_cfg.high_risk_penalty + dyn_cfg.w_dynamic_risk * dyn_max_cost,
                    dyn_penalty
                )

            dyn_penalty = torch.where(
                med_mask,
                dyn_cfg.medium_risk_penalty + dyn_cfg.w_dynamic_risk * dyn_max_cost,
                dyn_penalty
            )

            dyn_penalty = torch.where(
                low_mask,
                dyn_cfg.low_risk_penalty + 0.5 * dyn_cfg.w_dynamic_risk * dyn_max_cost,
                dyn_penalty
            )

            dyn_clear_score = torch.clamp(dyn_min_clearance, max=1.50) / 1.50

        # Composite score
        composite_scores = (
            w_path * path_score +
            w_heading * heading_score +
            w_dist * dist_score +
            w_clear * clearance_score +
            w_vel * vel_score +
            w_terrain * terrain_score -
            dyn_penalty +
            (dyn_cfg.w_dynamic_clearance * dyn_clear_score if has_dynamic else 0.0)
        )

        valid_mask = static_valid_mask & (~dyn_rejected)

        # Select best candidate
        if not valid_mask.any():
            return composite_scores, valid_mask, None

        # Mask invalid scores with -infinity for argmax selection
        masked_scores = torch.where(valid_mask, composite_scores, torch.tensor(-float("inf"), device=device, dtype=dtype))
        best_idx = int(torch.argmax(masked_scores).item())

        return composite_scores, valid_mask, best_idx
