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
        self._body_grid_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

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

        # 2. Extract Valid Local Metric Points (with geometry cache)
        H, W = local_costmap.shape
        x_min, x_max = local_costmap.x_range_m
        y_min, y_max = local_costmap.y_range_m
        res = local_costmap.resolution_m
        cache_key = (H, W, x_min, x_max, y_min, y_max, res)
        cached_grid = self._body_grid_cache.get(cache_key)
        if cached_grid is not None:
            r_flat, c_flat, pts_body = cached_grid
        else:
            r_indices, c_indices = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing="ij")
            r_flat = r_indices.ravel()
            c_flat = c_indices.ravel()
            x_body = x_max - (r_flat + 0.5) * res
            y_body = y_max - (c_flat + 0.5) * res
            z_body = np.zeros_like(x_body)
            pts_body = np.stack([x_body, y_body, z_body], axis=-1)
            self._body_grid_cache[cache_key] = (r_flat, c_flat, pts_body)

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

        # 6. Vectorized Safety Precedence and Confidence-Weighted Fusion (WS-2A)
        conf_scale = self.degraded_confidence_scale if tracking_state == TrackingState.DEGRADED else 1.0
        if tracking_state == TrackingState.DEGRADED:
            self.degraded_fusions_count += 1

        known_mask = ~local_costmap.unknown.ravel()[in_bounds]
        if not np.any(known_mask):
            self.fusion_successes += 1
            return SpatialFusionResult(
                success=True,
                fused_cells_count=0,
                rejected_cells_count=rejected_cells,
                tracking_state=tracking_state,
                localization_confidence=loc_confidence,
                sync_delta_s=sync_dt,
                reason="ok",
                diagnostics={"in_bounds_cells": int(np.sum(in_bounds))},
            )

        flat_l = (valid_r_l * W + valid_c_l)[known_mask]
        flat_w = (valid_r_w * self.world_map.n_cols + valid_c_w)[known_mask]
        fused_cells_count = int(len(flat_l))

        obs_cost = local_costmap.cost.ravel()[flat_l]
        obs_trav = local_costmap.traversability.ravel()[flat_l]
        obs_obst = local_costmap.obstacle.ravel()[flat_l]
        obs_hazard = local_costmap.hazard.ravel()[flat_l]
        obs_conf = local_costmap.confidence.ravel()[flat_l] * np.float32(loc_confidence * conf_scale)
        obs_slope = local_costmap.slope_deg.ravel()[flat_l]
        obs_rough = local_costmap.roughness.ravel()[flat_l]
        obs_hvar = local_costmap.height_var.ravel()[flat_l]

        wm_cost = self.world_map.cost.ravel()
        wm_trav = self.world_map.traversability.ravel()
        wm_obst = self.world_map.obstacle.ravel()
        wm_hazard = self.world_map.hazard.ravel()
        wm_unk = self.world_map.unknown.ravel()
        wm_conf = self.world_map.confidence.ravel()
        wm_slope = self.world_map.slope.ravel()
        wm_rough = self.world_map.roughness.ravel()
        wm_hvar = self.world_map.height_var.ravel()
        wm_count = self.world_map.observation_count.ravel()
        wm_ts = self.world_map.last_observed_timestamp.ravel()

        curr_obst = wm_obst[flat_w]
        curr_hazard = wm_hazard[flat_w]
        curr_cost = wm_cost[flat_w]
        curr_trav = wm_trav[flat_w]
        curr_conf = wm_conf[flat_w]
        curr_count = wm_count[flat_w]
        curr_unk = wm_unk[flat_w]
        curr_slope = wm_slope[flat_w]
        curr_rough = wm_rough[flat_w]
        curr_hvar = wm_hvar[flat_w]

        is_first = (curr_count == 0) | curr_unk
        is_critical = (obs_hazard >= self.hazard_critical_threshold) | (curr_hazard >= self.hazard_critical_threshold)
        is_high_risk = is_critical | obs_obst | curr_obst | (obs_hazard >= self.hazard_high_threshold)

        new_obst = curr_obst | obs_obst
        new_hazard = np.maximum(curr_hazard, obs_hazard)

        weight_new = obs_conf / np.maximum(np.float32(1e-4), curr_conf + obs_conf)
        blend_cost = (np.float32(1.0) - weight_new) * curr_cost + weight_new * obs_cost
        blend_trav = (np.float32(1.0) - weight_new) * curr_trav + weight_new * obs_trav

        safe_cost = np.where(is_first, obs_cost, blend_cost)
        safe_trav = np.where(is_first, obs_trav, blend_trav)

        risk_cost = np.maximum(curr_cost, obs_cost)
        risk_trav = np.where(curr_count > 0, np.minimum(curr_trav, obs_trav), obs_trav)

        new_cost = np.where(is_high_risk, risk_cost, safe_cost)
        new_trav = np.where(is_high_risk, risk_trav, safe_trav)

        sat_conf = np.minimum(np.float32(1.0), np.maximum(curr_conf, obs_conf) + np.float32(0.1) * np.minimum(curr_conf, obs_conf))
        new_conf = np.where(is_first, obs_conf, sat_conf)
        new_slope = np.where(is_first, obs_slope, np.float32(0.5) * curr_slope + np.float32(0.5) * obs_slope)
        new_rough = np.where(is_first, obs_rough, np.float32(0.5) * curr_rough + np.float32(0.5) * obs_rough)
        new_hvar = np.where(is_first, obs_hvar, np.maximum(curr_hvar, obs_hvar))

        wm_cost[flat_w] = new_cost
        wm_trav[flat_w] = new_trav
        wm_obst[flat_w] = new_obst
        wm_hazard[flat_w] = new_hazard
        wm_unk[flat_w] = False
        wm_conf[flat_w] = new_conf
        wm_slope[flat_w] = new_slope
        wm_rough[flat_w] = new_rough
        wm_hvar[flat_w] = new_hvar
        wm_count[flat_w] = curr_count + 1
        wm_ts[flat_w] = obs_time

        # Enforce strict safety precedence against any intra-frame rotated index collisions
        if np.any(is_high_risk):
            f_hr = flat_w[is_high_risk]
            wm_obst[f_hr] = new_obst[is_high_risk]
            wm_hazard[f_hr] = np.maximum(wm_hazard[f_hr], new_hazard[is_high_risk])
            wm_cost[f_hr] = np.maximum(wm_cost[f_hr], new_cost[is_high_risk])
            wm_trav[f_hr] = np.minimum(wm_trav[f_hr], new_trav[is_high_risk])

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
