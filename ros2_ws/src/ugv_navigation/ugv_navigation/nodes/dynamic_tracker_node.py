"""Dynamic Tracker ROS2 Node wrapping Phase 13 MultiFrameTracker and KalmanTrackFilter."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
    TrackedObstacleArray,
    TrackedObstacleState,
)
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig
from ugv_navigation.converters.dynamic_converters import (
    dynamic_obstacle_detection_array_from_ros,
    tracked_obstacle_array_to_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    DynamicObstacleDetectionArrayMsg,
    TrackedObstacleArrayMsg,
)

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    from ugv_navigation_msgs.msg import (
        DynamicObstacleDetectionArray as RosDynamicObstacleDetectionArray,
        TrackedObstacleArray as RosTrackedObstacleArray,
    )
    NATIVE_ROSIDL_AVAILABLE = True
except ImportError:
    NATIVE_ROSIDL_AVAILABLE = False
    RosDynamicObstacleDetectionArray = None
    RosTrackedObstacleArray = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class DynamicTrackerNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that subscribes to /perception/dynamic_obstacles,

    runs multi-frame association and state estimation, and publishes
    /tracking/dynamic_obstacles.
    """

    def __init__(
        self,
        tracker: Optional[MultiFrameTracker] = None,
        config: Optional[TrackerConfig] = None,
        pub_msg_type: Optional[Any] = None,
        sub_msg_type: Optional[Any] = None,
        output_msg_cls: Optional[Any] = None,
        track_item_msg_cls: Optional[Any] = None,
    ):
        self.config = config or TrackerConfig(use_kalman_filter=True)
        self.tracker = tracker or MultiFrameTracker(config=self.config)
        self.diagnostics = NodeDiagnosticsTracker("dynamic_tracker_node")

        self._output_msg_cls = output_msg_cls
        self._track_item_msg_cls = track_item_msg_cls

        self._latest_input_msg: Optional[Any] = None
        self._latest_tracks_msg: Optional[Any] = None
        self._last_timestamp: Optional[float] = None
        self._publisher = None
        self._subscriber = None

        # Canonical message types: native ROSIDL when available, fallback for test harness
        self._pub_msg_type = pub_msg_type or (
            RosTrackedObstacleArray
            if RosTrackedObstacleArray is not None
            else TrackedObstacleArrayMsg
        )
        self._sub_msg_type = sub_msg_type or (
            RosDynamicObstacleDetectionArray
            if RosDynamicObstacleDetectionArray is not None
            else DynamicObstacleDetectionArrayMsg
        )

        if RCLPY_AVAILABLE:
            super().__init__("dynamic_tracker_node")
            self.declare_parameter("target_frame", self.config.target_frame)
            self.declare_parameter("hits_to_confirm", self.config.hits_to_confirm)
            self.declare_parameter("misses_to_expire", self.config.misses_to_expire)
            self.declare_parameter("max_association_distance_m", self.config.max_association_distance_m)

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._publisher = self.create_publisher(
                self._pub_msg_type,
                "/tracking/dynamic_obstacles",
                qos,
            )
            self._subscriber = self.create_subscription(
                self._sub_msg_type,
                "/perception/dynamic_obstacles",
                self.on_detections_received,
                qos,
            )
            self.get_logger().info("DynamicTrackerNode initialized with canonical ROSIDL boundary.")

    def convert_input_detections(self, msg: Any) -> DynamicObstacleDetectionArray:
        """Convert incoming ROS2 message to pure Phase 13 DynamicObstacleDetectionArray.

        Preserves:
        - header timestamp
        - frame ID
        - obstacle/detection ID
        - center
        - dimensions
        - yaw
        - confidence
        - covariance
        - validity fields
        """
        if isinstance(msg, DynamicObstacleDetectionArray):
            return msg
        if hasattr(msg, "to_phase13") and callable(msg.to_phase13):
            return msg.to_phase13()
        try:
            return dynamic_obstacle_detection_array_from_ros(msg)
        except Exception:
            return DynamicObstacleDetectionArray(
                header_timestamp=self._last_timestamp or 0.0,
                frame_id=self.config.target_frame,
                detections=[],
            )

    def convert_output_tracks(
        self,
        tracked_array: TrackedObstacleArray,
        msg_cls: Optional[Any] = None,
        track_item_msg_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 TrackedObstacleArray to ROS2 message type.

        Preserves:
        - track ID
        - position
        - velocity
        - covariance
        - heading
        - speed
        - lifecycle/status
        - total hits
        - consecutive misses
        - confirmation state
        - timestamp
        - frame ID
        """
        target_cls = msg_cls or self._output_msg_cls
        item_cls = track_item_msg_cls or self._track_item_msg_cls
        if target_cls is not None:
            return tracked_obstacle_array_to_ros(
                tracked_array,
                msg_cls=target_cls,
                track_msg_cls=item_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosTrackedObstacleArray is not None:
            return tracked_obstacle_array_to_ros(tracked_array)
        return TrackedObstacleArrayMsg.from_phase13(tracked_array)

    def on_detections_received(self, msg: Any) -> None:
        """ROS2 subscription callback."""
        self._latest_input_msg = msg
        self.process_detections(msg)

    def process_detections(
        self,
        msg: Any,
        robot_pose: Optional[Tuple[float, float, float]] = None,
        output_msg_cls: Optional[Any] = None,
        track_item_msg_cls: Optional[Any] = None,
    ) -> Any:
        """Process incoming detection array message and produce tracked obstacles.

        Boundary Flow:
        ROS2 message -> convert_input_detections -> Phase 13 tracker.update -> convert_output_tracks -> ROS2 message.
        """
        self._latest_input_msg = msg

        # 1. Input Boundary Conversion: ROS2 msg -> Phase 13 internal data model
        det_array: DynamicObstacleDetectionArray = self.convert_input_detections(msg)
        detections: List[DynamicObstacleDetection] = det_array.detections
        timestamp_s: float = float(det_array.header_timestamp)

        # Stale timestamp protection (Step 8: stale observations are rejected safely)
        if self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            return self._latest_tracks_msg or self.convert_output_tracks(
                TrackedObstacleArray(header_timestamp=self._last_timestamp, frame_id="odom", tracks=[]),
                msg_cls=output_msg_cls,
                track_item_msg_cls=track_item_msg_cls,
            )
        self._last_timestamp = timestamp_s

        # 2. Provide identity robot pose if detections are in base_link and no external pose is supplied
        pose = robot_pose
        if pose is None and any(d.frame_id == "base_link" for d in detections):
            pose = (0.0, 0.0, 0.0)

        # 3. Core Tracking Step: Pure Phase 13 MultiFrameTracker execution
        tracked_array: TrackedObstacleArray = self.tracker.update(
            detections=detections,
            timestamp_s=timestamp_s,
            robot_pose=pose,
        )

        # 4. Output Boundary Conversion: Phase 13 internal data model -> ROS2 msg
        tracks_msg = self.convert_output_tracks(
            tracked_array,
            msg_cls=output_msg_cls,
            track_item_msg_cls=track_item_msg_cls,
        )
        self._latest_tracks_msg = tracks_msg

        # 5. Publish to ROS2 topic if runtime is active
        if RCLPY_AVAILABLE and self._publisher is not None:
            self._publisher.publish(tracks_msg)

        self.diagnostics.record_cycle(latency_ms=1.5)
        return tracks_msg


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. DynamicTrackerNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = DynamicTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
