"""Behavioral State ROS2 Node wrapping Phase 13 BehavioralStateMachine (Milestone 13.12.10).

Translates incoming ROSIDL inputs -> Pure Phase 13 models -> BehavioralStateMachine.evaluate -> Pure Phase 13 BehavioralDecision -> ROSIDL message boundary.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
)
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory
from ugv_navigation.converters.dynamic_converters import (
    behavioral_decision_to_ros,
    localization_status_from_ros,
    predicted_trajectory_array_from_ros,
    tracked_obstacle_array_from_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    from ugv_navigation_msgs.msg import (
        BehavioralDecision as RosBehavioralDecision,
        LocalizationStatus as RosLocalizationStatus,
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
    RosBehavioralDecision = None
    RosLocalizationStatus = None
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


class BehavioralStateNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that arbitrates UGV behavioral states (PROCEED, AVOID, WAIT, STOP)

    from dynamic tracking, predictions, and candidate evaluation states.

    Architectural Boundary:
    ROSIDL inputs -> converter -> Pure Phase 13 models -> BehavioralStateMachine -> Pure Phase 13 BehavioralDecision -> converter -> ROSIDL message.
    """

    def __init__(
        self,
        sm: Optional[BehavioralStateMachine] = None,
        config: Optional[BehavioralConfig] = None,
        pub_behavior_type: Optional[Any] = None,
        behavior_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
    ):
        self.config = config or BehavioralConfig()
        self.sm = sm or BehavioralStateMachine(config=self.config)
        self.diagnostics = NodeDiagnosticsTracker("behavioral_state_node")

        self._behavior_msg_cls = behavior_msg_cls
        self._header_cls = header_cls
        self._pub_behavior_type = pub_behavior_type or (
            RosBehavioralDecision
            if RosBehavioralDecision is not None
            else BehavioralDecisionMsg
        )

        self._latest_decision_msg: Optional[Any] = None
        self._latest_tracks_msg: Optional[Any] = None
        self._latest_preds_msg: Optional[Any] = None
        self._latest_loc_msg: Optional[Any] = None
        self._last_timestamp: Optional[float] = None

        self._publisher = None
        self._tracks_sub = None
        self._preds_sub = None
        self._loc_sub = None

        if RCLPY_AVAILABLE:
            super().__init__("behavioral_state_node")
            self.declare_parameter("stop_distance_m", 0.80)
            self.declare_parameter("wait_distance_m", 1.80)
            self.declare_parameter("deadlock_time_threshold_s", 3.0)
            self.declare_parameter("wait_timeout_s", self.config.wait_timeout_s)
            self.declare_parameter("min_dwell_time_s", self.config.min_dwell_time_s)
            self.declare_parameter("deadlock_consecutive_stop_cycles", self.config.deadlock_consecutive_stop_cycles)

            state_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            pred_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )

            self._publisher = self.create_publisher(
                self._pub_behavior_type,
                "/behavior/tactical_decision",
                state_qos,
            )
            sub_tracks_type = RosTrackedObstacleArray if RosTrackedObstacleArray is not None else TrackedObstacleArrayMsg
            sub_preds_type = RosPredictedTrajectoryArray if RosPredictedTrajectoryArray is not None else PredictedTrajectoryArrayMsg
            sub_loc_type = RosLocalizationStatus if RosLocalizationStatus is not None else LocalizationStatusMsg

            self._tracks_sub = self.create_subscription(
                sub_tracks_type,
                "/tracking/dynamic_obstacles",
                self.on_tracks_received,
                sensor_qos,
            )
            self._preds_sub = self.create_subscription(
                sub_preds_type,
                "/prediction/obstacle_trajectories",
                self.on_predictions_received,
                pred_qos,
            )
            self._loc_sub = self.create_subscription(
                sub_loc_type,
                "/localization/status",
                self.on_localization_received,
                state_qos,
            )
            self.get_logger().info("BehavioralStateNode initialized with canonical ROSIDL boundary.")

    def convert_input_tracks(self, msg: Any) -> Optional[TrackedObstacleArray]:
        """Convert incoming ROS2 tracked obstacle array to pure Phase 13 model."""
        if msg is None:
            return None
        if isinstance(msg, TrackedObstacleArray):
            return msg
        if hasattr(msg, "to_phase13") and callable(msg.to_phase13):
            return msg.to_phase13()
        try:
            return tracked_obstacle_array_from_ros(msg)
        except Exception:
            return None

    def convert_input_predictions(self, msg: Any) -> Optional[PredictedTrajectoryArray]:
        """Convert incoming ROS2 predicted trajectory array to pure Phase 13 model."""
        if msg is None:
            return None
        if isinstance(msg, PredictedTrajectoryArray):
            return msg
        if hasattr(msg, "to_phase13") and callable(msg.to_phase13):
            return msg.to_phase13()
        try:
            return predicted_trajectory_array_from_ros(msg)
        except Exception:
            return None

    def convert_input_localization(self, msg: Any) -> Optional[RobotState]:
        """Convert incoming ROS2 localization status to pure Phase 13 RobotState."""
        if msg is None:
            return None
        if isinstance(msg, RobotState):
            return msg
        try:
            res = localization_status_from_ros(msg)
            pos = (0.0, 0.0, 0.0)
            if hasattr(msg, "pose") and hasattr(msg.pose, "pose") and hasattr(msg.pose.pose, "position"):
                pos = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(msg.pose.pose.position.z))
            elif hasattr(msg, "position"):
                pos = (float(msg.position.x), float(msg.position.y), float(getattr(msg.position, "z", 0.0)))
            elif "position" in res:
                pos = res["position"]

            yaw = 0.0
            if hasattr(msg, "yaw"):
                yaw = float(msg.yaw)
            elif "yaw" in res:
                yaw = float(res["yaw"])

            v = float(getattr(msg, "v_linear", 0.0))
            w = float(getattr(msg, "w_angular", 0.0))
            ts = float(res.get("timestamp", 0.0))

            return RobotState(
                x=float(pos[0]),
                y=float(pos[1]),
                yaw=yaw,
                v=v,
                omega=w,
                timestamp=ts,
            )
        except Exception:
            return None

    def convert_output_decision(
        self,
        dec: BehavioralDecision,
        msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        timestamp_s: Optional[float] = None,
    ) -> Any:
        """Convert Phase 13 BehavioralDecision to ROS2 message type.

        Preserves:
        - header timestamp and frame_id ('odom')
        - state: 'PROCEED', 'AVOID', 'WAIT', 'STOP'
        - command_v, command_omega
        - reason
        - blocking_obstacle_id
        - dynamic_collision
        - dynamic_risk
        - unknown_horizon
        - wait_elapsed_s, wait_timeout
        - deadlock, deadlock_event
        - transition, previous_state, state_dwell_time_s, confidence
        """
        if dec is None:
            return None
        target_cls = msg_cls or self._behavior_msg_cls
        hdr_cls = header_cls or self._header_cls
        ts = timestamp_s if timestamp_s is not None else 0.0

        if target_cls is not None:
            return behavioral_decision_to_ros(
                dec,
                frame_id="odom",
                timestamp_s=ts,
                msg_cls=target_cls,
                header_cls=hdr_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosBehavioralDecision is not None:
            return behavioral_decision_to_ros(dec, timestamp_s=ts)
        return BehavioralDecisionMsg.from_phase13(dec, timestamp_s=ts)

    def on_tracks_received(self, msg: Any) -> None:
        """Subscription callback for tracked obstacles."""
        self._latest_tracks_msg = msg

    def on_predictions_received(self, msg: Any) -> None:
        """Subscription callback for predicted trajectories."""
        self._latest_preds_msg = msg

    def on_localization_received(self, msg: Any) -> None:
        """Subscription callback for localization state."""
        self._latest_loc_msg = msg

    def evaluate_state(
        self,
        current_state: Optional[Union[RobotState, Tuple[float, float, float], Any]] = None,
        best_trajectory: Optional[Trajectory] = None,
        best_v: float = 0.0,
        best_w: float = 0.0,
        planner_status: PlannerStatus = PlannerStatus.PLANNING,
        dwa_diagnostics: Optional[Dict[str, Any]] = None,
        tracks_msg: Optional[Any] = None,
        predictions_msg: Optional[Any] = None,
        global_path: Optional[Path] = None,
        current_time_s: Optional[float] = None,
        output_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        # Backward-compatible keyword arguments
        ugv_pose: Optional[Tuple[float, float, float]] = None,
        ugv_twist: Optional[Tuple[float, float]] = None,
        timestamp_s: Optional[float] = None,
        best_trajectory_cost: Optional[float] = None,
        has_safe_trajectory: bool = True,
        obstacles: Optional[Any] = None,
        predictions: Optional[Any] = None,
    ) -> Any:
        """Run one complete behavioral state arbitration cycle.

        Boundary Flow:
        ROSIDL inputs -> convert_input_* -> Phase 13 domain objects -> sm.evaluate -> convert_output_decision -> ROSIDL.
        """
        # Resolve timestamp
        t_now = current_time_s if current_time_s is not None else (timestamp_s or 0.0)

        # Monotonic timestamp protection: cannot move backward in time
        if self._last_timestamp is not None and t_now < self._last_timestamp:
            stale_dec = BehavioralDecision(
                state=BehavioralState.STOP,
                command_v=0.0,
                command_omega=0.0,
                reason="Stale input: timestamp moved backward in time",
                previous_state=self.sm.current_state,
                transition="SAFETY_STOP",
            )
            return self.convert_output_decision(
                stale_dec,
                msg_cls=output_msg_cls,
                header_cls=header_cls,
                timestamp_s=self._last_timestamp,
            )
        self._last_timestamp = t_now

        # Resolve robot state
        robot_state: Optional[RobotState] = None
        if isinstance(current_state, RobotState):
            robot_state = current_state
        elif ugv_pose is not None:
            v_val = ugv_twist[0] if ugv_twist else 0.0
            w_val = ugv_twist[1] if ugv_twist else 0.0
            robot_state = RobotState(
                x=float(ugv_pose[0]),
                y=float(ugv_pose[1]),
                yaw=float(ugv_pose[2]),
                v=float(v_val),
                omega=float(w_val),
                timestamp=t_now,
            )
        elif self._latest_loc_msg is not None:
            robot_state = self.convert_input_localization(self._latest_loc_msg)
        else:
            robot_state = RobotState(timestamp=t_now)

        # Resolve predictions and tracks
        preds_in = predictions_msg if predictions_msg is not None else (predictions if predictions is not None else self._latest_preds_msg)
        converted_preds = self.convert_input_predictions(preds_in)

        tracks_in = tracks_msg if tracks_msg is not None else (obstacles if obstacles is not None else self._latest_tracks_msg)
        converted_tracks = self.convert_input_tracks(tracks_in)

        # Diagnostics & planner status
        diag = dwa_diagnostics.copy() if dwa_diagnostics is not None else {}
        status = planner_status
        if not has_safe_trajectory:
            diag["valid_trajectories"] = 0
            diag["emergency_stop_triggered"] = True
            status = PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY

        # Execute pure Phase 13 BehavioralStateMachine evaluation
        decision: BehavioralDecision = self.sm.evaluate(
            current_state=robot_state,
            best_trajectory=best_trajectory,
            best_v=best_v,
            best_w=best_w,
            planner_status=status,
            dwa_diagnostics=diag,
            predicted_tracks=converted_preds or (converted_tracks.tracks if converted_tracks else None),
            global_path=global_path,
            current_time_s=t_now,
        )

        # Output boundary conversion
        ros_msg = self.convert_output_decision(
            decision,
            msg_cls=output_msg_cls,
            header_cls=header_cls,
            timestamp_s=t_now,
        )
        self._latest_decision_msg = ros_msg

        if RCLPY_AVAILABLE and self._publisher is not None:
            self._publisher.publish(ros_msg)

        self.diagnostics.record_cycle(latency_ms=0.5)
        return ros_msg


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. BehavioralStateNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = BehavioralStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
