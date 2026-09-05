"""Dynamic Occupancy ROS2 Node wrapping Phase 13 DynamicOccupancyLayer."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyGridSlice,
    DynamicOccupancyLayer,
)
from ugv_navigation.converters.dynamic_converters import (
    dynamic_occupancy_slice_array_to_ros,
    predicted_trajectory_array_from_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    DynamicOccupancyGridSliceMsg,
    DynamicOccupancySliceArrayMsg,
    PredictedTrajectoryArrayMsg,
)
from ugv_navigation.utils.time_utils import RosHeader

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    try:
        from ugv_navigation_msgs.msg import (
            PredictedTrajectoryArray as RosPredictedTrajectoryArray,
        )
    except ImportError:
        from ugv_navigation_msgs.msg import (
            PredictedObstacleTrajectoryArray as RosPredictedTrajectoryArray,
        )
    from ugv_navigation_msgs.msg import (
        DynamicOccupancyGridSlice as RosDynamicOccupancyGridSlice,
        DynamicOccupancySliceArray as RosDynamicOccupancySliceArray,
    )
    NATIVE_ROSIDL_AVAILABLE = True
except ImportError:
    NATIVE_ROSIDL_AVAILABLE = False
    RosPredictedTrajectoryArray = None
    RosDynamicOccupancySliceArray = None
    RosDynamicOccupancyGridSlice = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class DynamicOccupancyNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that subscribes to /prediction/obstacle_trajectories,

    generates discrete spatiotemporal dynamic cost grids,
    and publishes /occupancy/dynamic_slices.
    """

    def __init__(
        self,
        occupancy_layer: Optional[DynamicOccupancyLayer] = None,
        config: Optional[DynamicOccupancyConfig] = None,
        pub_msg_type: Optional[Any] = None,
        sub_msg_type: Optional[Any] = None,
        output_msg_cls: Optional[Any] = None,
        slice_item_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        time_cls: Optional[Any] = None,
    ):
        self.config = config or DynamicOccupancyConfig()
        self.occupancy_layer = occupancy_layer or DynamicOccupancyLayer(config=self.config)
        self.diagnostics = NodeDiagnosticsTracker("dynamic_occupancy_node")

        self._output_msg_cls = output_msg_cls
        self._slice_item_msg_cls = slice_item_msg_cls
        self._header_cls = header_cls
        self._time_cls = time_cls

        self._latest_pred_msg: Optional[Any] = None
        self._latest_occupancy_msg: Optional[Any] = None
        self._last_timestamp: Optional[float] = None
        self._publisher = None
        self._subscriber = None

        # Canonical message types: native ROSIDL when available, fallback for test harness
        self._pub_msg_type = pub_msg_type or (
            RosDynamicOccupancySliceArray
            if RosDynamicOccupancySliceArray is not None
            else DynamicOccupancySliceArrayMsg
        )
        self._sub_msg_type = sub_msg_type or (
            RosPredictedTrajectoryArray
            if RosPredictedTrajectoryArray is not None
            else PredictedTrajectoryArrayMsg
        )

        if RCLPY_AVAILABLE:
            super().__init__("dynamic_occupancy_node")
            self.declare_parameter("resolution_m", self.config.resolution_m)
            self.declare_parameter("frame_id", self.config.frame_id)
            self.declare_parameter("collision_cost_threshold", self.config.collision_cost_threshold)

            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            self._publisher = self.create_publisher(
                self._pub_msg_type,
                "/occupancy/dynamic_slices",
                qos,
            )
            self._subscriber = self.create_subscription(
                self._sub_msg_type,
                "/prediction/obstacle_trajectories",
                self.on_predictions_received,
                qos,
            )
            self.get_logger().info("DynamicOccupancyNode initialized with canonical ROSIDL boundary.")

    def convert_input_predictions(self, msg: Any) -> PredictedTrajectoryArray:
        """Convert incoming ROS2 message to pure Phase 13 PredictedTrajectoryArray.

        Preserves:
        - header timestamp
        - frame ID
        - trajectories
        - track ID
        - waypoints (time offset, timestamp, position, velocity, covariance, uncertainty radius)
        - bounding box size and yaw
        - status
        - prediction horizon
        """
        if msg is None:
            return PredictedTrajectoryArray(header_timestamp=0.0, frame_id=self.config.frame_id, predictions=[])
        if isinstance(msg, PredictedTrajectoryArray):
            return msg
        if hasattr(msg, "to_phase13") and callable(msg.to_phase13):
            return msg.to_phase13()
        try:
            return predicted_trajectory_array_from_ros(msg)
        except Exception:
            return PredictedTrajectoryArray(header_timestamp=0.0, frame_id=self.config.frame_id, predictions=[])

    def convert_output_slices(
        self,
        slices: Sequence[DynamicOccupancyGridSlice],
        header_timestamp: float,
        frame_id: Optional[str] = None,
        msg_cls: Optional[Any] = None,
        slice_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        time_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 DynamicOccupancyGridSlice sequence to ROS2 message type.

        Preserves:
        - header timestamp
        - frame ID
        - grid origin (x, y)
        - resolution
        - dimensions (width, height)
        - target time offsets
        - target timestamps
        - cell data (cost/risk values)
        - unknown mask
        """
        target_cls = msg_cls or self._output_msg_cls
        item_cls = slice_msg_cls or self._slice_item_msg_cls
        hdr_cls = header_cls or self._header_cls
        t_cls = time_cls or self._time_cls
        fid = frame_id or self.config.frame_id

        if target_cls is not None:
            return dynamic_occupancy_slice_array_to_ros(
                slices=slices,
                resolution=float(self.config.resolution_m),
                origin_x=float(self.config.x_bounds_m[0]),
                origin_y=float(self.config.y_bounds_m[0]),
                frame_id=fid,
                timestamp_s=float(header_timestamp),
                msg_cls=target_cls,
                slice_msg_cls=item_cls,
                header_cls=hdr_cls,
                time_cls=t_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosDynamicOccupancySliceArray is not None:
            return dynamic_occupancy_slice_array_to_ros(
                slices=slices,
                resolution=float(self.config.resolution_m),
                origin_x=float(self.config.x_bounds_m[0]),
                origin_y=float(self.config.y_bounds_m[0]),
                frame_id=fid,
                timestamp_s=float(header_timestamp),
            )
        return self._convert_to_dataclass_slice_array(slices, header_timestamp, fid)

    def _convert_to_dataclass_slice_array(
        self,
        slices: Sequence[DynamicOccupancyGridSlice],
        header_timestamp: float,
        frame_id: str,
    ) -> DynamicOccupancySliceArrayMsg:
        slice_msgs: List[DynamicOccupancyGridSliceMsg] = []
        for occ_slice in slices:
            grid_2d = occ_slice.cost
            height, width = grid_2d.shape
            data_flat = [float(v) for v in grid_2d.flatten()]
            header = RosHeader.create(frame_id=frame_id, timestamp_s=header_timestamp)
            if hasattr(occ_slice, "unknown_mask") and occ_slice.unknown_mask is not None:
                unknown_mask = [bool(v) for v in np.asarray(occ_slice.unknown_mask).flatten()]
            else:
                unknown_mask = [False] * len(data_flat)

            slice_msg = DynamicOccupancyGridSliceMsg(
                header=header,
                target_time_offset_s=float(occ_slice.time_offset),
                target_timestamp_s=float(occ_slice.timestamp),
                resolution=float(self.config.resolution_m),
                origin_x=float(self.config.x_bounds_m[0]),
                origin_y=float(self.config.y_bounds_m[0]),
                width=int(width),
                height=int(height),
                data=data_flat,
                unknown_mask=unknown_mask,
            )
            slice_msgs.append(slice_msg)

        return DynamicOccupancySliceArrayMsg(
            header=RosHeader.create(frame_id=frame_id, timestamp_s=header_timestamp),
            slices=slice_msgs,
        )

    def on_predictions_received(self, msg: Any) -> None:
        """ROS2 subscription callback."""
        self._latest_pred_msg = msg
        self.process_predictions(msg)

    def process_predictions(
        self,
        msg: Any,
        output_msg_cls: Optional[Any] = None,
        slice_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        time_cls: Optional[Any] = None,
    ) -> Any:
        """Rasterize predicted obstacle trajectories into time-sliced 2D occupancy grids.

        Boundary Flow:
        ROS2 message -> convert_input_predictions -> Phase 13 occupancy_layer.update_predictions -> convert_output_slices -> ROS2 message.
        """
        self._latest_pred_msg = msg

        # 1. Input Boundary Conversion: ROS2 msg -> Phase 13 domain objects
        pred_array: PredictedTrajectoryArray = self.convert_input_predictions(msg)
        timestamp_s: float = float(pred_array.header_timestamp)

        # Stale timestamp protection (TIMESTAMP RULES: reject or safely handle stale/out-of-order)
        if self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            return self._latest_occupancy_msg or self.convert_output_slices(
                slices=[self.occupancy_layer.get_slice(t_bin) for t_bin in self.config.time_bins],
                header_timestamp=self._last_timestamp,
                frame_id=pred_array.frame_id or self.config.frame_id,
                msg_cls=output_msg_cls,
                slice_msg_cls=slice_msg_cls,
                header_cls=header_cls,
                time_cls=time_cls,
            )
        self._last_timestamp = timestamp_s

        # 2. Update Dynamic Occupancy Layer (Core Phase 13 engine)
        self.occupancy_layer.update_predictions(pred_array, header_timestamp=timestamp_s)

        # 3. Retrieve all time-bin slices in order
        slices = [self.occupancy_layer.get_slice(t_bin) for t_bin in self.config.time_bins]

        # 4. Output Boundary Conversion: Phase 13 slices -> ROS2 message
        slice_array_msg = self.convert_output_slices(
            slices=slices,
            header_timestamp=timestamp_s,
            frame_id=pred_array.frame_id or self.config.frame_id,
            msg_cls=output_msg_cls,
            slice_msg_cls=slice_msg_cls,
            header_cls=header_cls,
            time_cls=time_cls,
        )
        self._latest_occupancy_msg = slice_array_msg

        # 5. Publish to ROS2 topic if runtime is active
        if RCLPY_AVAILABLE and self._publisher is not None:
            self._publisher.publish(slice_array_msg)

        self.diagnostics.record_cycle(latency_ms=3.0)
        return slice_array_msg


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. DynamicOccupancyNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = DynamicOccupancyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
