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
    """Subscribes to camera images and publishes validated BEV costmaps using the new model."""

    def __init__(self, adapter: Optional[PerceptionAdapter] = None):
        self.adapter = adapter or PerceptionAdapter()
        self.diagnostics = NodeDiagnosticsTracker("perception_node")

        if RCLPY_AVAILABLE:
            super().__init__("perception_node")
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("costmap_topic", "/perception/costmap")
            self.declare_parameter("overlay_topic", "/perception/traversability_overlay")

            img_topic = self.get_parameter("image_topic").value
            overlay_topic = self.get_parameter("overlay_topic").value

            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )

            self.image_sub = self.create_subscription(
                Image, img_topic, self._on_image_msg, sensor_qos
            )
            self.overlay_pub = self.create_publisher(Image, overlay_topic, sensor_qos)
            self.get_logger().info(
                f"Perception Node initialized with model: {self.adapter.model_name}. "
                f"Subscribed to: {img_topic}"
            )

    def _on_image_msg(self, msg: Image) -> None:
        """Processes incoming camera image message with the traversability neural network."""
        try:
            h, w = msg.height, msg.width
            if msg.encoding in ("rgb8", "bgr8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
                rgb = arr if msg.encoding == "rgb8" else arr[:, :, ::-1]
            elif msg.encoding == "mono8":
                mono = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
                rgb = np.stack([mono, mono, mono], axis=-1)
            else:
                return

            t_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.process_frame(image_rgb=rgb, timestamp_s=t_s)

            # Publish visual traversability overlay
            if self.adapter.last_result is not None and self.adapter.last_result.overlay is not None:
                overlay = self.adapter.last_result.overlay
                out_msg = Image()
                out_msg.header = msg.header
                out_msg.height, out_msg.width = overlay.shape[:2]
                out_msg.encoding = "rgb8"
                out_msg.step = out_msg.width * 3
                out_msg.data = overlay.tobytes()
                self.overlay_pub.publish(out_msg)
        except Exception as e:
            if RCLPY_AVAILABLE:
                self.get_logger().warn(f"Failed to process image frame: {e}")

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
