"""ROS2 Spatial Fusion Adapter wrapping Phase 9 SpatialFusionEngine."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine, SpatialFusionResult
from ai.spatial_fusion.synchronization import TemporalSynchronizer
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.tf.tf_tree import TfTreeBuffer, TfLookupError, TfExtrapolationError


class SpatialFusionAdapter:
    """Adapts Phase 9 SpatialFusionEngine into ROS2 middleware workflows.

    Coordinates incoming local BEV costmaps, TF2/Localization states,
    and updates the persistent WorldTerrainMap.
    """

    def __init__(
        self,
        resolution_m: float = 0.10,
        x_bounds_m: tuple[float, float] = (-20.0, 20.0),
        y_bounds_m: tuple[float, float] = (-20.0, 20.0),
        degraded_confidence_scale: float = 0.50,
        max_sync_delta_s: float = 0.10,
    ):
        self.world_map = WorldTerrainMap(resolution_m=resolution_m, x_bounds_m=x_bounds_m, y_bounds_m=y_bounds_m)
        self.synchronizer = TemporalSynchronizer(max_sync_delta_s=max_sync_delta_s)
        self.engine = SpatialFusionEngine(
            world_map=self.world_map,
            synchronizer=self.synchronizer,
            degraded_confidence_scale=degraded_confidence_scale,
        )

    def add_localization_msg(self, loc_msg: LocalizationStatusMsg) -> None:
        """Translates LocalizationStatusMsg and feeds into Phase 9 synchronizer."""
        est = loc_msg.to_localization_estimate()
        self.engine.add_localization(est)

    def fuse_costmap_with_tf(
        self,
        bev: BEVTerrainCostmap,
        tf_buffer: TfTreeBuffer,
        tracking_state: TrackingState = TrackingState.TRACKING_OK,
        confidence: float = 1.0,
    ) -> Tuple[bool, Optional[WorldTerrainMapMsg], str]:
        """Fuses local costmap using TF lookup for odom -> base_link at observation timestamp."""
        t_obs = bev.metadata.timestamp

        # Zero-trust safety gate: If localization is LOST or RELOCALIZING, immediately reject
        if tracking_state in (TrackingState.LOST, TrackingState.RELOCALIZING):
            return False, None, f"fusion_blocked_by_state_{tracking_state.name}"

        # Query TF buffer for odom -> base_link
        try:
            T_odom_base = tf_buffer.lookup_transform_matrix(
                target_frame=TfTreeBuffer.FRAME_ODOM,
                source_frame=TfTreeBuffer.FRAME_BASE_LINK,
                time_s=t_obs,
            )
        except (TfLookupError, TfExtrapolationError) as e:
            return False, None, f"tf_lookup_failed_{str(e)}"

        # Execute Phase 9 fusion cycle
        res: SpatialFusionResult = self.engine.fuse_local_costmap(
            bev,
            timestamp=t_obs,
            direct_pose=T_odom_base,
            direct_state=tracking_state,
            direct_confidence=confidence,
        )

        if not res.success:
            return False, None, res.reason

        map_msg = WorldTerrainMapMsg.from_world_terrain_map(
            self.world_map,
            frame_id=TfTreeBuffer.FRAME_ODOM,
            timestamp_s=t_obs,
        )
        return True, map_msg, "ok"
