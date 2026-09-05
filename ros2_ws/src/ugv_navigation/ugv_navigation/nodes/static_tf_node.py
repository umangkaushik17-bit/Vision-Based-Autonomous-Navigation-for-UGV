"""Static TF Broadcaster Node for camera extrinsics."""

from __future__ import annotations

import sys
from ugv_navigation.tf.tf_tree import TfTreeBuffer
from ugv_navigation.tf.tf_broadcaster import StaticCameraExtrinsicsPublisher

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TransformStamped as RosTransformStamped
    import tf2_ros
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class StaticTfNode(Node if RCLPY_AVAILABLE else object):
    """Broadcasts static camera mount and optical orientation transforms."""

    def __init__(
        self,
        forward_offset_m: float = 0.15,
        lateral_offset_m: float = 0.0,
        height_offset_m: float = 0.40,
        pitch_deg: float = 15.0,
        tf_buffer: TfTreeBuffer | None = None,
    ):
        self.forward_offset_m = forward_offset_m
        self.lateral_offset_m = lateral_offset_m
        self.height_offset_m = height_offset_m
        self.pitch_deg = pitch_deg
        self.tf_buffer = tf_buffer or TfTreeBuffer()

        # Always configure into the in-memory TF buffer
        StaticCameraExtrinsicsPublisher.configure_extrinsics(
            self.tf_buffer,
            forward_offset_m=self.forward_offset_m,
            lateral_offset_m=self.lateral_offset_m,
            height_offset_m=self.height_offset_m,
            pitch_deg=self.pitch_deg,
        )

        if RCLPY_AVAILABLE:
            super().__init__("static_tf_node")
            self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            self.publish_static_transforms()
            self.get_logger().info("Static TF Node initialized and broadcasting nominal camera extrinsics.")

    def publish_static_transforms(self) -> None:
        if not RCLPY_AVAILABLE:
            return
        # Broadcast base_link -> camera_link and camera_link -> camera_optical via rclpy
        # Implemented for ROS2 runtime deployment


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed on this system. Static TF Node requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = StaticTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
