"""Dynamic Predictor ROS2 Node wrapping Phase 13 MotionPredictor."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ugv_navigation.converters.dynamic_converters import (
    predicted_trajectory_array_to_ros,
    tracked_obstacle_array_from_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    from ugv_navigation_msgs.msg import (
        TrackedObstacleArray as RosTrackedObstacleArray,
    )
    try:
        from ugv_navigation_msgs.msg import (
            PredictedTrajectoryArray as RosPredictedTrajectoryArray,
        )
    except ImportError:
        from ugv_navigation_msgs.msg import (
            PredictedObstacleTrajectoryArray as RosPredictedTrajectoryArray,
        )
    NATIVE_ROSIDL_AVAILABLE = True
except ImportError:
    NATIVE_ROSIDL_AVAILABLE = False
    RosTrackedObstacleArray = None
    RosPredictedTrajectoryArray = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class DynamicPredictorNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that subscribes to /tracking/dynamic_obstacles,

    runs constant-velocity motion prediction with covariance propagation,
    and publishes /prediction/obstacle_trajectories.
    """

    def __init__(
        self,
        predictor: Optional[MotionPredictor] = None,
        config: Optional[PredictorConfig] = None,
        pub_msg_type: Optional[Any] = None,
        sub_msg_type: Optional[Any] = None,
        output_msg_cls: Optional[Any] = None,
        traj_item_msg_cls: Optional[Any] = None,
        wp_item_msg_cls: Optional[Any] = None,
    ):
        self.config = config or PredictorConfig()
        self.predictor = predictor or MotionPredictor(config=self.config)
        self.diagnostics = NodeDiagnosticsTracker("dynamic_predictor_node")

        self._output_msg_cls = output_msg_cls
        self._traj_item_msg_cls = traj_item_msg_cls
        self._wp_item_msg_cls = wp_item_msg_cls

        self._latest_tracks_msg: Optional[Any] = None
        self._latest_predictions_msg: Optional[Any] = None
        self._last_timestamp: Optional[float] = None
        self._publisher = None
        self._subscriber = None

        # Canonical message types: native ROSIDL when available, fallback for test harness
        self._pub_msg_type = pub_msg_type or (
            RosPredictedTrajectoryArray
            if RosPredictedTrajectoryArray is not None
            else PredictedTrajectoryArrayMsg
        )
        self._sub_msg_type = sub_msg_type or (
            RosTrackedObstacleArray
            if RosTrackedObstacleArray is not None
            else TrackedObstacleArrayMsg
        )

        if RCLPY_AVAILABLE:
            super().__init__("dynamic_predictor_node")
            self.declare_parameter("default_horizon_s", self.config.default_horizon_s)
            self.declare_parameter("default_dt_s", self.config.default_dt_s)
            self.declare_parameter("sigma_scale", self.config.sigma_scale)

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            self._publisher = self.create_publisher(
                self._pub_msg_type,
                "/prediction/obstacle_trajectories",
                qos,
            )
            self._subscriber = self.create_subscription(
                self._sub_msg_type,
                "/tracking/dynamic_obstacles",
                self.on_tracks_received,
                qos,
            )
            self.get_logger().info("DynamicPredictorNode initialized with canonical ROSIDL boundary.")

    def convert_input_tracks(self, msg: Any) -> TrackedObstacleArray:
        """Convert incoming ROS2 message to pure Phase 13 TrackedObstacleArray.

        Preserves:
        - header timestamp
        - frame ID
        - track ID
        - position (x, y)
        - velocity (vx, vy)
        - 4x4 covariance
        - bbox size and yaw
        - status (NEW, TRACKING, DEGRADED, LOST)
        - is_confirmed
        - total_hits
        - consecutive_misses
        """
        if msg is None:
            return TrackedObstacleArray(header_timestamp=0.0, frame_id="odom", tracks=[])
        if isinstance(msg, TrackedObstacleArray):
            return msg
        if hasattr(msg, "to_phase13") and callable(msg.to_phase13):
            return msg.to_phase13()
        try:
            return tracked_obstacle_array_from_ros(msg)
        except Exception:
            return TrackedObstacleArray(header_timestamp=0.0, frame_id="odom", tracks=[])

    def convert_output_predictions(
        self,
        pred_array: PredictedTrajectoryArray,
        msg_cls: Optional[Any] = None,
        traj_msg_cls: Optional[Any] = None,
        wp_msg_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 PredictedTrajectoryArray to ROS2 message type.

        Preserves:
        - header timestamp
        - frame ID
        - track ID
        - trajectory waypoints (target_time, relative_time_s, position, velocity, 2x2 covariance, uncertainty radius)
        - prediction horizon
        - obstacle bbox size and yaw
        - status
        - confidence
        """
        target_cls = msg_cls or self._output_msg_cls
        item_cls = traj_msg_cls or self._traj_item_msg_cls
        wp_cls = wp_msg_cls or self._wp_item_msg_cls
        if target_cls is not None:
            return predicted_trajectory_array_to_ros(
                pred_array,
                msg_cls=target_cls,
                traj_msg_cls=item_cls,
                wp_msg_cls=wp_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosPredictedTrajectoryArray is not None:
            return predicted_trajectory_array_to_ros(pred_array)
        return PredictedTrajectoryArrayMsg.from_phase13(pred_array)

    def on_tracks_received(self, msg: Any) -> None:
        """ROS2 subscription callback."""
        self._latest_tracks_msg = msg
        self.process_tracks(msg)

    def process_tracks(
        self,
        msg: Any,
        horizon_s: Optional[float] = None,
        dt_s: Optional[float] = None,
        output_msg_cls: Optional[Any] = None,
        traj_item_msg_cls: Optional[Any] = None,
        wp_item_msg_cls: Optional[Any] = None,
    ) -> Any:
        """Process incoming tracked obstacles and produce predicted future trajectories.

        Boundary Flow:
        ROS2 message -> convert_input_tracks -> Phase 13 predictor.predict_tracks -> convert_output_predictions -> ROS2 message.
        """
        self._latest_tracks_msg = msg

        # 1. Input Boundary Conversion: ROS2 msg -> Phase 13 internal data model
        tracked_array: TrackedObstacleArray = self.convert_input_tracks(msg)
        timestamp_s: float = float(tracked_array.header_timestamp)

        # Stale timestamp protection (TIMESTAMP RULES: reject or safely handle stale/out-of-order)
        if self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            return self._latest_predictions_msg or self.convert_output_predictions(
                PredictedTrajectoryArray(header_timestamp=self._last_timestamp, frame_id=tracked_array.frame_id, predictions=[]),
                msg_cls=output_msg_cls,
                traj_msg_cls=traj_item_msg_cls,
                wp_msg_cls=wp_item_msg_cls,
            )
        self._last_timestamp = timestamp_s

        # 2. Core Prediction Step: Pure Phase 13 MotionPredictor execution
        pred_array: PredictedTrajectoryArray = self.predictor.predict_tracks(
            tracks=tracked_array,
            horizon_s=horizon_s or self.config.default_horizon_s,
            dt_s=dt_s or self.config.default_dt_s,
        )

        # 3. Output Boundary Conversion: Phase 13 internal data model -> ROS2 msg
        pred_msg = self.convert_output_predictions(
            pred_array,
            msg_cls=output_msg_cls,
            traj_msg_cls=traj_item_msg_cls,
            wp_msg_cls=wp_item_msg_cls,
        )
        self._latest_predictions_msg = pred_msg

        # 4. Publish to ROS2 topic if runtime is active
        if RCLPY_AVAILABLE and self._publisher is not None:
            self._publisher.publish(pred_msg)

        self.diagnostics.record_cycle(latency_ms=1.0)
        return pred_msg


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. DynamicPredictorNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = DynamicPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
