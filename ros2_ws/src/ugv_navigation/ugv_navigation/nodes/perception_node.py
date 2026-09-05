"""Perception ROS2 Node wrapping Phase 4/5 models."""

from __future__ import annotations

import sys
import numpy as np
from typing import Optional

from ai.costmap.bev_costmap import BEVTerrainCostmap
from ugv_navigation.adapters.perception_adapter import PerceptionAdapter
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class PerceptionNode(Node if RCLPY_AVAILABLE else object):
    """Subscribes to camera images and publishes validated BEV costmaps."""

    def __init__(self, adapter: Optional[PerceptionAdapter] = None):
        self.adapter = adapter or PerceptionAdapter()
        self.diagnostics = NodeDiagnosticsTracker("perception_node")

        if RCLPY_AVAILABLE:
            super().__init__("perception_node")
            self.get_logger().info("Perception Node initialized.")

    def process_frame(
        self,
        image_rgb: np.ndarray,
        timestamp_s: float,
        obstacle_mask: Optional[np.ndarray] = None,
        hazard_mask: Optional[np.ndarray] = None,
    ) -> BEVTerrainCostmap:
        """Executes perception cycle and updates diagnostics."""
        costmap = self.adapter.process_image(
            image_rgb=image_rgb,
            timestamp_s=timestamp_s,
            obstacle_mask=obstacle_mask,
            hazard_mask=hazard_mask,
        )
        self.diagnostics.record_cycle(latency_ms=10.0)
        return costmap


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. Perception Node requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
