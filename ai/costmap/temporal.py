"""Temporal costmap accumulation engine maintaining moving terrain persistence."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.costmap.motion_compensation import MotionCompensator


class TemporalCostmapAccumulator:
    """Accumulates consecutive BEV costmaps over time with ego-motion compensation."""

    def __init__(
        self,
        metadata: CostmapMetadata,
        static_decay_factor: float = 0.98,
        dynamic_decay_factor: float = 0.70,
        learning_rate: float = 0.40,
        min_observations_to_confirm: int = 2,
        max_age_frames: int = 30,
        dropout_tolerance_frames: int = 3
    ) -> None:
        self.metadata = metadata
        self.static_decay = static_decay_factor
        self.dynamic_decay = dynamic_decay_factor
        self.lr = learning_rate
        self.min_hits = min_observations_to_confirm
        self.max_age = max_age_frames
        self.dropout_tolerance = dropout_tolerance_frames

        self.compensator = MotionCompensator(metadata)
        self.accumulated_map: Optional[BEVTerrainCostmap] = None
        self.age_grid: Optional[np.ndarray] = None  # Tracks consecutive frames since last hit

    def reset(self) -> None:
        """Reset temporal state to blank."""
        self.accumulated_map = None
        self.age_grid = None

    def update(
        self,
        current_map: BEVTerrainCostmap,
        pose_delta: Optional[Tuple[float, float, float]] = None
    ) -> BEVTerrainCostmap:
        """Fuse incoming instantaneous costmap into the accumulated temporal representation.

        Args:
            current_map: Instantaneous BEVTerrainCostmap from the current camera frame.
            pose_delta: Optional SE(2) motion (dx_forward, dy_lateral, dyaw_rad) since last frame.

        Returns:
            Temporally accumulated, stabilized BEVTerrainCostmap.
        """
        # First frame initialization
        if self.accumulated_map is None:
            self.accumulated_map = current_map
            self.age_grid = np.zeros(current_map.shape, dtype=np.int32)
            return current_map

        # 1. Warp Prior Map to Current Robot Frame
        if pose_delta is not None:
            dx, dy, dyaw = pose_delta
            warped_prior = self.compensator.warp_costmap(self.accumulated_map, dx, dy, dyaw)
            warped_age = self.compensator.warp_layer(self.age_grid.astype(np.float32), dx, dy, dyaw, border_value=float(self.max_age), interpolation=0).astype(np.int32)
        else:
            warped_prior = self.accumulated_map
            warped_age = self.age_grid

        # 2. Identify Current Observed Cells
        is_observed_now = (current_map.observation_count > 0)

        # 3. Update Age Grid
        new_age = np.where(is_observed_now, 0, warped_age + 1)

        # 4. Temporal Fusion of Traversability & Costs
        # Where observed now AND previously observed: blend with learning rate
        both_seen = is_observed_now & (~warped_prior.unknown)
        only_current = is_observed_now & warped_prior.unknown
        dropout_cells = (~is_observed_now) & (~warped_prior.unknown) & (new_age <= self.dropout_tolerance)

        # Base arrays
        new_trav = warped_prior.traversability.copy()
        new_cost = warped_prior.cost.copy()
        new_conf = warped_prior.confidence.copy()
        new_hazard = warped_prior.hazard.copy()
        new_obs_count = warped_prior.observation_count.copy()

        # Update observed cells
        new_trav[both_seen] = (1.0 - self.lr) * warped_prior.traversability[both_seen] + self.lr * current_map.traversability[both_seen]
        new_trav[only_current] = current_map.traversability[only_current]

        new_conf[both_seen] = np.minimum(1.0, warped_prior.confidence[both_seen] + 0.15)
        new_conf[only_current] = current_map.confidence[only_current]
        new_conf[dropout_cells] *= self.static_decay

        # Update observation count
        new_obs_count[is_observed_now] += 1

        # 5. Hazard & Obstacle Persistence vs Dynamic Clearing
        # If current frame shows high drivability (>0.75) where obstacle was, clear obstacle
        active_clear = is_observed_now & (current_map.traversability > 0.70) & (~current_map.obstacle)
        new_hazard[active_clear] *= self.dynamic_decay

        # Blend new hazard observations
        new_hazard[is_observed_now] = np.maximum(new_hazard[is_observed_now] * self.dynamic_decay, current_map.hazard[is_observed_now])
        new_hazard[dropout_cells] *= self.static_decay

        # Hard obstacle mask: active if confirmed by multiple observations or instantaneous critical hazard
        confirmed_obstacle = (new_obs_count >= self.min_hits) & (new_hazard > 0.60)
        instant_critical = current_map.obstacle & (current_map.hazard >= 0.85)
        new_obstacle = (confirmed_obstacle | instant_critical) & (~active_clear)

        # 6. Terrain Geometry (Slope, Roughness, Height)
        # Static physical terrain updates conservatively
        new_slope = np.where(is_observed_now, np.maximum(warped_prior.slope_deg, current_map.slope_deg), warped_prior.slope_deg)
        new_rough = np.where(is_observed_now, (1.0 - self.lr) * warped_prior.roughness + self.lr * current_map.roughness, warped_prior.roughness)
        new_height = np.where(is_observed_now, np.maximum(warped_prior.height_var, current_map.height_var), warped_prior.height_var)

        # 7. Compute Fused Cost
        new_cost = (1.0 - self.lr) * warped_prior.cost + self.lr * current_map.cost
        new_cost[only_current] = current_map.cost[only_current]

        # Enforce Hard Overrides
        new_cost[new_obstacle] = 1.00
        new_cost[new_slope >= 28.0] = 1.00
        new_cost[new_height >= 0.20] = 1.00

        # Cells that haven't been seen for > max_age revert to UNKNOWN
        is_stale = (new_age > self.max_age)
        new_unknown = (warped_prior.unknown & ~is_observed_now) | is_stale
        new_cost[new_unknown] = 0.50
        new_conf[new_unknown] = 0.00

        # Construct and persist output
        result_map = BEVTerrainCostmap(
            cost=new_cost.astype(np.float32),
            traversability=new_trav.astype(np.float32),
            obstacle=new_obstacle,
            hazard=new_hazard.astype(np.float32),
            unknown=new_unknown,
            confidence=new_conf.astype(np.float32),
            slope_deg=new_slope.astype(np.float32),
            roughness=new_rough.astype(np.float32),
            height_var=new_height.astype(np.float32),
            observation_count=new_obs_count.astype(np.int32),
            metadata=current_map.metadata
        )

        self.accumulated_map = result_map
        self.age_grid = new_age
        return result_map
