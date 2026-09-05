"""Spatial Fusion ROS2 Node wrapping Phase 9 SpatialFusionEngine."""

from __future__ import annotations

import sys
from typing import Optional, Tuple

from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ugv_navigation.adapters.spatial_fusion_adapter import SpatialFusionAdapter
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.tf.tf_tree import TfTreeBuffer

try:
    import rclpy
    from rclpy.node import Node
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class SpatialFusionNode(Node if RCLPY_AVAILABLE else object):
    """Subscribes to perception costmaps and localization status, looks up TF, and updates persistent WorldTerrainMap."""

    def __init__(
        self,
        tf_buffer: Optional[TfTreeBuffer] = None,
        adapter: Optional[SpatialFusionAdapter] = None,
    ):
        self.tf_buffer = tf_buffer or TfTreeBuffer()
        self.adapter = adapter or SpatialFusionAdapter()
        self.diagnostics = NodeDiagnosticsTracker("spatial_fusion_node")

        self._latest_loc_status: Optional[LocalizationStatusMsg] = None

        if RCLPY_AVAILABLE:
            super().__init__("spatial_fusion_node")
            self.get_logger().info("Spatial Fusion Node initialized.")

    def on_localization_status(self, loc_msg: LocalizationStatusMsg) -> None:
        """Handler for incoming localization status updates."""
        self._latest_loc_status = loc_msg
        self.adapter.add_localization_msg(loc_msg)

    def on_costmap_received(
        self,
        bev: BEVTerrainCostmap,
    ) -> Tuple[bool, Optional[WorldTerrainMapMsg], str]:
        """Executes spatial fusion cycle using the current TF buffer and tracking status."""
        state = TrackingState.TRACKING_OK
        conf = 1.0

        if self._latest_loc_status is not None:
            state = LocalizationStatusMsg.INV_STATE_MAPPING.get(self._latest_loc_status.tracking_state, TrackingState.LOST)
            conf = self._latest_loc_status.confidence

        success, map_msg, reason = self.adapter.fuse_costmap_with_tf(
            bev=bev,
            tf_buffer=self.tf_buffer,
            tracking_state=state,
            confidence=conf,
        )

        # Update diagnostics
        rejected = not success
        tf_failed = "tf_lookup_failed" in reason
        stale = "extrapolation" in reason or "stale" in reason
        self.diagnostics.record_cycle(
            latency_ms=25.0,
            tracking_state_name=state.name,
            confidence=conf,
            rejected=rejected,
            tf_failed=tf_failed,
            stale=stale,
        )

        return success, map_msg, reason


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. Spatial Fusion Node requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = SpatialFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
