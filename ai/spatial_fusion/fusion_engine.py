"""Core Spatial Fusion Engine.

Transforms local BEV terrain costmaps into the world coordinate frame using
synchronized 6-DoF localization poses, enforcing safety precedence, confidence
weighting, and zero-trust tracking state protection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ai.spatial_fusion.synchronization import SyncResult, TemporalSynchronizer
from ai.spatial_fusion.transforms import SpatialCameraExtrinsics, transform_points_3d
from ai.spatial_fusion.world_map import WorldTerrainMap


@dataclass
class SpatialFusionResult:
    """Outcome of attempting to fuse a local BEV costmap into the WorldTerrainMap."""

    success: bool
    fused_cells_count: int = 0
    rejected_cells_count: int = 0
    tracking_state: TrackingState = TrackingState.LOST
    localization_confidence: float = 0.0
    sync_delta_s: float = 0.0
    reason: str = "ok"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class SpatialFusionEngine:
    """Orchestrates temporal synchronization, SE(3) spatial transformation,
    safety-precedence fusion, and global terrain representation updates.
    """

    def __init__(
        self,
        world_map: Optional[WorldTerrainMap] = None,
        extrinsics: Optional[SpatialCameraExtrinsics] = None,
        synchronizer: Optional[TemporalSynchronizer] = None,
        degraded_confidence_scale: float = 0.50,
        hazard_critical_threshold: float = 0.70,
        hazard_high_threshold: float = 0.40,
    ) -> None:
        self.world_map = world_map if world_map is not None else WorldTerrainMap()
        self.extrinsics = extrinsics if extrinsics is not None else SpatialCameraExtrinsics()
        self.synchronizer = synchronizer if synchronizer is not None else TemporalSynchronizer()
        self.degraded_confidence_scale = float(degraded_confidence_scale)
        self.hazard_critical_threshold = float(hazard_critical_threshold)
        self.hazard_high_threshold = float(hazard_high_threshold)

        # Performance and event counters
        self.fusion_calls = 0
        self.fusion_successes = 0
        self.fused_cells_total = 0
        self.blocked_lost_count = 0
        self.blocked_gap_count = 0
        self.degraded_fusions_count = 0

    def add_localization(self, estimate: LocalizationEstimate) -> None:
        """Feed incoming localization estimate to the internal synchronizer."""
        self.synchronizer.add_localization_estimate(estimate)

    def fuse_local_costmap(
        self,
        local_costmap: BEVTerrainCostmap,
        timestamp: Optional[float] = None,
        direct_pose: Optional[np.ndarray] = None,
        direct_state: TrackingState = TrackingState.TRACKING_OK,
        direct_confidence: float = 1.0,
    ) -> SpatialFusionResult:
        """Fuse a local BEVTerrainCostmap into the WorldTerrainMap.
        
        If direct_pose is provided, uses it directly (for controlled benchmarks/unit tests).
        Otherwise, looks up the nearest synchronized pose from the TemporalSynchronizer.
        """
        self.fusion_calls += 1
        obs_time = timestamp if timestamp is not None else local_costmap.metadata.timestamp

        # 1. Pose Resolution & Synchronization
        if direct_pose is not None:
            # Direct pose injection mode (for explicit unit testing)
            if direct_state in (TrackingState.LOST, TrackingState.RELOCALIZING):
                self.blocked_lost_count += 1
                return SpatialFusionResult(
                    success=False,
                    tracking_state=direct_state,
                    reason=f"blocked_untrusted_direct_state_{direct_state.value}"
                )
            T_world_body = direct_pose.copy()
            tracking_state = direct_state
            loc_confidence = direct_confidence
            sync_dt = 0.0
        else:
            # Synchronized mode
            sync_res = self.synchronizer.synchronize(obs_time)
            if not sync_res.is_valid or sync_res.pose is None:
                if sync_res.tracking_state in (TrackingState.LOST, TrackingState.RELOCALIZING):
                    self.blocked_lost_count += 1
                else:
                    self.blocked_gap_count += 1
                return SpatialFusionResult(
                    success=False,
                    tracking_state=sync_res.tracking_state,
                    localization_confidence=sync_res.confidence,
                    sync_delta_s=sync_res.time_delta_s,
                    reason=sync_res.reason
                )
            T_world_body = sync_res.pose
            tracking_state = sync_res.tracking_state
            loc_confidence = sync_res.confidence
            sync_dt = sync_res.time_delta_s

        # 2. Extract Valid Local Metric Points
        H, W = local_costmap.shape
        r_indices, c_indices = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        r_flat = r_indices.flatten()
        c_flat = c_indices.flatten()

        # Vectorized local metric coordinates (x_forward, y_left)
        x_min, x_max = local_costmap.x_range_m
        y_min, y_max = local_costmap.y_range_m
        res = local_costmap.resolution_m

        x_body = x_max - (r_flat + 0.5) * res
        y_body = y_max - (c_flat + 0.5) * res
        z_body = np.zeros_like(x_body) # Local terrain base plane

        pts_body = np.stack([x_body, y_body, z_body], axis=-1)

        # 3. Transform Points to World Navigation Frame
        pts_world = transform_points_3d(T_world_body, pts_body)
        xs_w = pts_world[:, 0]
        ys_w = pts_world[:, 1]

        # 4. Map to World Grid Raster
        rows_w, cols_w = self.world_map.world_to_grid_vec(xs_w, ys_w)

        # 5. Filter Points within World Map Bounds
        in_bounds = (
            (rows_w >= 0) & (rows_w < self.world_map.n_rows) &
            (cols_w >= 0) & (cols_w < self.world_map.n_cols)
        )

        valid_r_l = r_flat[in_bounds]
        valid_c_l = c_flat[in_bounds]
        valid_r_w = rows_w[in_bounds]
        valid_c_w = cols_w[in_bounds]

        rejected_cells = int(len(r_flat) - np.sum(in_bounds))

        # 6. Safety Precedence and Confidence-Weighted Fusion
        # Compute effective observation confidence
        conf_scale = self.degraded_confidence_scale if tracking_state == TrackingState.DEGRADED else 1.0
        if tracking_state == TrackingState.DEGRADED:
            self.degraded_fusions_count += 1

        fused_cells_count = 0

        for rl, cl, rw, cw in zip(valid_r_l, valid_c_l, valid_r_w, valid_c_w):
            obs_unknown = bool(local_costmap.unknown[rl, cl])
            if obs_unknown:
                # Never convert unknown to free space, and do not overwrite existing known cells with unknown
                continue

            fused_cells_count += 1
            obs_cost = float(local_costmap.cost[rl, cl])
            obs_trav = float(local_costmap.traversability[rl, cl])
            obs_obst = bool(local_costmap.obstacle[rl, cl])
            obs_hazard = float(local_costmap.hazard[rl, cl])
            obs_conf = float(local_costmap.confidence[rl, cl]) * loc_confidence * conf_scale
            obs_slope = float(local_costmap.slope_deg[rl, cl])
            obs_rough = float(local_costmap.roughness[rl, cl])
            obs_hvar = float(local_costmap.height_var[rl, cl])

            # Existing world cell state
            curr_obst = self.world_map.obstacle[rw, cw]
            curr_hazard = self.world_map.hazard[rw, cw]
            curr_cost = self.world_map.cost[rw, cw]
            curr_trav = self.world_map.traversability[rw, cw]
            curr_conf = self.world_map.confidence[rw, cw]
            curr_count = self.world_map.observation_count[rw, cw]

            # SAFETY PRECEDENCE RULE (Requirement 15):
            # Critical / High Hazard & Lethal Obstacles MUST NOT BE DILUTED by averaging
            is_critical = (obs_hazard >= self.hazard_critical_threshold) or (curr_hazard >= self.hazard_critical_threshold)
            is_high_risk = is_critical or obs_obst or curr_obst or (obs_hazard >= self.hazard_high_threshold)

            # 1. Update Obstacle & Hazard: Preserves maximum observed severity
            new_obst = curr_obst or obs_obst
            new_hazard = max(curr_hazard, obs_hazard)

            if is_critical or is_high_risk:
                # Hard safety override: Cost takes the maximum, Traversability takes the minimum
                new_cost = max(curr_cost, obs_cost)
                new_trav = min(curr_trav, obs_trav) if curr_count > 0 else obs_trav
            else:
                # Safe / continuous terrain: Confidence-weighted running update
                if curr_count == 0 or self.world_map.unknown[rw, cw]:
                    new_cost = obs_cost
                    new_trav = obs_trav
                else:
                    weight_new = obs_conf / max(1e-4, curr_conf + obs_conf)
                    new_cost = (1.0 - weight_new) * curr_cost + weight_new * obs_cost
                    new_trav = (1.0 - weight_new) * curr_trav + weight_new * obs_trav

            # 2. Update Confidence & Geometry
            if curr_count == 0 or self.world_map.unknown[rw, cw]:
                new_conf = obs_conf
                new_slope = obs_slope
                new_rough = obs_rough
                new_hvar = obs_hvar
            else:
                # Cumulative saturating confidence
                new_conf = min(1.0, max(curr_conf, obs_conf) + 0.1 * min(curr_conf, obs_conf))
                new_slope = 0.5 * (curr_slope := self.world_map.slope[rw, cw]) + 0.5 * obs_slope
                new_rough = 0.5 * (curr_rough := self.world_map.roughness[rw, cw]) + 0.5 * obs_rough
                new_hvar = max(self.world_map.height_var[rw, cw], obs_hvar)

            # Write fused values back to world map
            self.world_map.cost[rw, cw] = new_cost
            self.world_map.traversability[rw, cw] = new_trav
            self.world_map.obstacle[rw, cw] = new_obst
            self.world_map.hazard[rw, cw] = new_hazard
            self.world_map.unknown[rw, cw] = False # Cell is now legitimately observed!
            self.world_map.confidence[rw, cw] = new_conf
            self.world_map.slope[rw, cw] = new_slope
            self.world_map.roughness[rw, cw] = new_rough
            self.world_map.height_var[rw, cw] = new_hvar
            self.world_map.observation_count[rw, cw] = curr_count + 1
            self.world_map.last_observed_timestamp[rw, cw] = obs_time

        self.fusion_successes += 1
        self.fused_cells_total += fused_cells_count

        return SpatialFusionResult(
            success=True,
            fused_cells_count=fused_cells_count,
            rejected_cells_count=rejected_cells,
            tracking_state=tracking_state,
            localization_confidence=loc_confidence,
            sync_delta_s=sync_dt,
            reason="ok",
            diagnostics={
                "in_bounds_cells": int(np.sum(in_bounds)),
                "total_bev_cells": len(r_flat),
                "degraded_scaled": (tracking_state == TrackingState.DEGRADED),
            }
        )
