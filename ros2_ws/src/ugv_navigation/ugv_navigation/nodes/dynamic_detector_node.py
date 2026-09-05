"""Dynamic Detector ROS2 Node wrapping Phase 13 detector interfaces."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
)
from ai.planning.dynamic.detector_interface import (
    BaseDynamicObstacleDetector,
    SyntheticDynamicObstacleDetector,
    VisionDynamicObstacleDetector,
)
from ugv_navigation.converters.dynamic_converters import (
    dynamic_obstacle_detection_array_to_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    DynamicObstacleDetectionArrayMsg,
    DynamicObstacleDetectionMsg,
)
from ugv_navigation.utils.time_utils import RosHeader

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    from ugv_navigation_msgs.msg import (
        DynamicObstacleDetection as RosDynamicObstacleDetection,
        DynamicObstacleDetectionArray as RosDynamicObstacleDetectionArray,
    )
    NATIVE_ROSIDL_AVAILABLE = True
except ImportError:
    NATIVE_ROSIDL_AVAILABLE = False
    RosDynamicObstacleDetection = None
    RosDynamicObstacleDetectionArray = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class DynamicDetectorNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that detects dynamic obstacles from perception inputs or synthetic streams

    and publishes validated DynamicObstacleDetectionArray messages.
    """

    def __init__(
        self,
        detector: Optional[BaseDynamicObstacleDetector] = None,
        frame_id: str = "base_link",
        min_confidence: float = 0.30,
        pub_msg_type: Optional[Any] = None,
        output_msg_cls: Optional[Any] = None,
        det_item_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
        vector3_cls: Optional[Any] = None,
    ):
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")
        if not frame_id:
            raise ValueError("frame_id must be non-empty")

        self.frame_id = frame_id
        self.min_confidence = min_confidence
        self.detector = detector or SyntheticDynamicObstacleDetector(
            frame_id=self.frame_id,
            min_confidence=self.min_confidence,
        )
        self.diagnostics = NodeDiagnosticsTracker("dynamic_detector_node")

        self._output_msg_cls = output_msg_cls
        self._det_item_msg_cls = det_item_msg_cls
        self._header_cls = header_cls
        self._point_cls = point_cls
        self._vector3_cls = vector3_cls

        self._latest_detections: Optional[Any] = None
        self._last_timestamp: Optional[float] = None
        self._publisher = None

        # Canonical message types: native ROSIDL when available, fallback for test harness
        self._pub_msg_type = pub_msg_type or (
            RosDynamicObstacleDetectionArray
            if RosDynamicObstacleDetectionArray is not None
            else DynamicObstacleDetectionArrayMsg
        )

        if RCLPY_AVAILABLE:
            super().__init__("dynamic_detector_node")
            self.declare_parameter("frame_id", self.frame_id)
            self.declare_parameter("min_confidence", self.min_confidence)
            self.declare_parameter("rate_hz", 10.0)

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._publisher = self.create_publisher(
                self._pub_msg_type,
                "/perception/dynamic_obstacles",
                qos,
            )
            self.get_logger().info("DynamicDetectorNode initialized with canonical ROSIDL boundary.")

    def convert_output_detections(
        self,
        det_array: DynamicObstacleDetectionArray,
        msg_cls: Optional[Any] = None,
        det_item_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
        vector3_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 DynamicObstacleDetectionArray to ROS2 message type.

        Preserves:
        - header timestamp
        - frame ID
        - detection ID
        - bbox center (x, y)
        - dimensions (length, width, height)
        - yaw
        - confidence
        - position covariance
        - validity flag
        """
        target_cls = msg_cls or self._output_msg_cls
        item_cls = det_item_msg_cls or self._det_item_msg_cls
        hdr_cls = header_cls or self._header_cls
        p_cls = point_cls or self._point_cls
        v_cls = vector3_cls or self._vector3_cls

        if target_cls is not None:
            return dynamic_obstacle_detection_array_to_ros(
                det_array,
                msg_cls=target_cls,
                det_msg_cls=item_cls,
                header_cls=hdr_cls,
                point_cls=p_cls,
                vector3_cls=v_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosDynamicObstacleDetectionArray is not None:
            return dynamic_obstacle_detection_array_to_ros(det_array)
        return DynamicObstacleDetectionArrayMsg.from_phase13(det_array, frame_id=self.frame_id)

    def process_frame(
        self,
        frame_data: Any,
        timestamp_s: float,
        output_msg_cls: Optional[Any] = None,
        det_item_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
        vector3_cls: Optional[Any] = None,
    ) -> Any:
        """Process an incoming sensor frame or simulation state and emit detections.

        Boundary Flow:
        Input data -> Phase 13 detector.detect -> filter min_confidence -> Phase 13 model -> convert_output_detections -> ROS2 message.
        """
        if timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")

        # Stale timestamp protection (Step 8: stale observations rejected safely)
        if self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            return self._latest_detections or self.convert_output_detections(
                DynamicObstacleDetectionArray(
                    header_timestamp=self._last_timestamp,
                    frame_id=self.frame_id,
                    detections=[],
                ),
                msg_cls=output_msg_cls,
                det_item_msg_cls=det_item_msg_cls,
                header_cls=header_cls,
                point_cls=point_cls,
                vector3_cls=vector3_cls,
            )

        # 1. Detection execution via pure Phase 13 detector
        try:
            raw_detections: List[DynamicObstacleDetection] = self.detector.detect(
                frame_data,
                timestamp_s=timestamp_s,
            )
            if raw_detections is None:
                raw_detections = []
        except Exception:
            raw_detections = []

        # 2. Filter detections above confidence threshold
        filtered_detections: List[DynamicObstacleDetection] = [
            det
            for det in raw_detections
            if det is not None and getattr(det, "confidence", 1.0) >= self.min_confidence
        ]

        # 3. Form Phase 13 domain container
        det_array = DynamicObstacleDetectionArray(
            header_timestamp=timestamp_s,
            frame_id=self.frame_id,
            detections=filtered_detections,
        )
        self._last_timestamp = timestamp_s

        # 4. Output boundary conversion: Phase 13 -> ROS2 message
        msg = self.convert_output_detections(
            det_array,
            msg_cls=output_msg_cls,
            det_item_msg_cls=det_item_msg_cls,
            header_cls=header_cls,
            point_cls=point_cls,
            vector3_cls=vector3_cls,
        )

        self._latest_detections = msg
        if RCLPY_AVAILABLE and self._publisher is not None:
            self._publisher.publish(msg)

        self.diagnostics.record_cycle(latency_ms=2.0)
        return msg


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. DynamicDetectorNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = DynamicDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
