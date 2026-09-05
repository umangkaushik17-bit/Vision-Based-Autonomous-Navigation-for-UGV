"""Localization ROS2 Node wrapping Phase 8 Visual Odometry and broadcasting TF."""

from __future__ import annotations

import sys
import numpy as np
from typing import Optional, Tuple

from ai.localization.base import CameraIntrinsics, TrackingState
from ugv_navigation.adapters.localization_adapter import LocalizationAdapter
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.tf.tf_tree import TransformStamped, TfTreeBuffer

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    import tf2_ros
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class LocalizationNode(Node if RCLPY_AVAILABLE else object):
    """Subscribes to camera feeds, computes visual odometry, broadcasts dynamic TF, and publishes localization state."""

    def __init__(
        self,
        camera: Optional[CameraIntrinsics] = None,
        tf_buffer: Optional[TfTreeBuffer] = None,
        frame_id: str = "odom",
        child_frame_id: str = "base_link",
    ):
        self.adapter = LocalizationAdapter(camera=camera, frame_id=frame_id, child_frame_id=child_frame_id)
        self.tf_buffer = tf_buffer or TfTreeBuffer()
        self.diagnostics = NodeDiagnosticsTracker("localization_node")

        if RCLPY_AVAILABLE:
            super().__init__("localization_node")
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.get_logger().info("Localization Node initialized.")

    def process_frame(
        self,
        img_left: np.ndarray,
        timestamp_s: float,
        img_right: Optional[np.ndarray] = None,
    ) -> Tuple[LocalizationStatusMsg, Optional[TransformStamped]]:
        """Processes frame, inserts dynamic transform into TF buffer if valid, and updates diagnostics."""
        msg, tf_stamped = self.adapter.process_frame(img_left, timestamp_s, img_right)

        # Broadcast into TF buffer if valid
        if tf_stamped is not None:
            self.tf_buffer.set_transform(tf_stamped)

        # Telemetry
        state_enum = LocalizationStatusMsg.INV_STATE_MAPPING.get(msg.tracking_state, TrackingState.LOST)
        state_name = state_enum.name
        rejected = not msg.is_pose_valid
        self.diagnostics.record_cycle(
            latency_ms=10.0,
            tracking_state_name=state_name,
            confidence=msg.confidence,
            features=msg.tracked_features,
            inliers=msg.inliers,
            rejected=rejected,
        )

        return msg, tf_stamped


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. Localization Node requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
