"""Dynamic Planner ROS2 Node integrating Predictive DWA, Behavioral State Machine,

Event-Driven Global Replanning, and Phase 12 Safety Clamping.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from ai.planning.base import PlannerStatus
from ai.planning.dynamic.behavioral_state import BehavioralDecision
from ai.planning.dynamic.data_model import (
    PredictedTrajectoryArray,
    TrackedObstacleArray,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanTrigger,
)
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.adapters.dynamic_navigation_adapter import (
    DynamicAdapterConfig,
    DynamicNavigationAdapter,
    DynamicPlanningResult,
)
from ugv_navigation.adapters.planner_adapter import PathMsg, PoseStampedMsg, TwistMsg
from ugv_navigation.converters.dynamic_converters import (
    behavioral_decision_to_ros,
    dynamic_replan_event_to_ros,
    localization_status_from_ros,
    predicted_trajectory_array_from_ros,
    tracked_obstacle_array_from_ros,
    world_terrain_map_from_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    DynamicReplanEventMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.tf.tf_tree import TfTreeBuffer
from ugv_navigation.utils.time_utils import RosHeader

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    from ugv_navigation_msgs.msg import (
        BehavioralDecision as RosBehavioralDecision,
        DynamicReplanEvent as RosDynamicReplanEvent,
        LocalizationStatus as RosLocalizationStatus,
        TrackedObstacleArray as RosTrackedObstacleArray,
        WorldTerrainMap as RosWorldTerrainMap,
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
    RosDynamicReplanEvent = None
    RosBehavioralDecision = None
    RosLocalizationStatus = None
    RosWorldTerrainMap = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class DynamicPlannerNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that coordinates dynamic-obstacle navigation:

    - Subscribes to localization, world terrain map, goal pose, tracked obstacles, and predictions.
    - Synchronously executes dynamic DWA trajectory evaluation.
    - Evaluates the Behavioral State Machine (PROCEED, AVOID, WAIT, STOP).
    - Triggers Event-Driven Global Replanning on persistent blockages/deadlocks.
    - Enforces Phase 12 emergency safety clamping (d < 0.35m -> v=0).
    - Publishes /cmd_vel, /planning/global_path, /planning/dynamic_trajectory, /behavior/tactical_decision, and /planning/dynamic_replan_event.
    """

    def __init__(
        self,
        adapter: Optional[DynamicNavigationAdapter] = None,
        tf_buffer: Optional[TfTreeBuffer] = None,
        config: Optional[DynamicAdapterConfig] = None,
        pub_cmd_type: Optional[Any] = None,
        pub_path_type: Optional[Any] = None,
        pub_traj_type: Optional[Any] = None,
        pub_behavior_type: Optional[Any] = None,
        pub_replan_type: Optional[Any] = None,
        replan_msg_cls: Optional[Any] = None,
        behavior_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ):
        self.config = config or DynamicAdapterConfig()
        self.adapter = adapter or DynamicNavigationAdapter(config=self.config)
        self.tf_buffer = tf_buffer or TfTreeBuffer()
        self.diagnostics = NodeDiagnosticsTracker("dynamic_planner_node")

        self._replan_msg_cls = replan_msg_cls
        self._behavior_msg_cls = behavior_msg_cls
        self._header_cls = header_cls
        self._point_cls = point_cls

        self._latest_map_msg: Optional[Any] = None
        self._latest_loc_msg: Optional[Any] = None
        self._latest_goal_msg: Optional[PoseStampedMsg] = None
        self._latest_tracks_msg: Optional[Any] = None
        self._latest_predictions_msg: Optional[Any] = None
        self._last_timestamp: Optional[float] = None

        self._last_result: Optional[DynamicPlanningResult] = None
        self._cmd_pub = None
        self._path_pub = None
        self._trajectory_pub = None
        self._behavior_pub = None
        self._replan_pub = None

        self._map_sub = None
        self._loc_sub = None
        self._goal_sub = None
        self._tracks_sub = None
        self._preds_sub = None

        # Canonical message types: native ROSIDL when available, fallback for test harness
        self._pub_cmd_type = pub_cmd_type or TwistMsg
        self._pub_path_type = pub_path_type or PathMsg
        self._pub_traj_type = pub_traj_type or PathMsg
        self._pub_behavior_type = pub_behavior_type or (
            RosBehavioralDecision
            if RosBehavioralDecision is not None
            else BehavioralDecisionMsg
        )
        self._pub_replan_type = pub_replan_type or (
            RosDynamicReplanEvent
            if RosDynamicReplanEvent is not None
            else DynamicReplanEventMsg
        )

        if RCLPY_AVAILABLE:
            super().__init__("dynamic_planner_node")
            self.declare_parameter("prediction_horizon_s", self.config.prediction_horizon_s)
            self.declare_parameter("replan_cooldown_s", self.config.replan_cooldown_s)
            self.declare_parameter("deadlock_time_threshold_s", self.config.deadlock_time_threshold_s)
            self.declare_parameter("stop_distance_m", self.config.stop_distance_m)
            self.declare_parameter("wait_distance_m", self.config.wait_distance_m)

            cmd_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            state_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            traj_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
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

            self._cmd_pub = self.create_publisher(self._pub_cmd_type, "/cmd_vel", cmd_qos)
            self._path_pub = self.create_publisher(self._pub_path_type, "/planning/global_path", state_qos)
            self._trajectory_pub = self.create_publisher(self._pub_traj_type, "/planning/dynamic_trajectory", traj_qos)
            self._behavior_pub = self.create_publisher(self._pub_behavior_type, "/behavior/tactical_decision", state_qos)
            self._replan_pub = self.create_publisher(self._pub_replan_type, "/planning/dynamic_replan_event", state_qos)

            sub_map_type = RosWorldTerrainMap if RosWorldTerrainMap is not None else WorldTerrainMapMsg
            sub_loc_type = RosLocalizationStatus if RosLocalizationStatus is not None else LocalizationStatusMsg
            sub_tracks_type = RosTrackedObstacleArray if RosTrackedObstacleArray is not None else TrackedObstacleArrayMsg
            sub_preds_type = RosPredictedTrajectoryArray if RosPredictedTrajectoryArray is not None else PredictedTrajectoryArrayMsg

            self._map_sub = self.create_subscription(
                sub_map_type, "/terrain/world_map", self.on_map_received, state_qos
            )
            self._loc_sub = self.create_subscription(
                sub_loc_type, "/localization/status", self.on_localization_status, state_qos
            )
            self._goal_sub = self.create_subscription(
                PoseStampedMsg, "/goal_pose", self.on_goal_received, state_qos
            )
            self._tracks_sub = self.create_subscription(
                sub_tracks_type, "/tracking/dynamic_obstacles", self.on_tracks_received, sensor_qos
            )
            self._preds_sub = self.create_subscription(
                sub_preds_type, "/prediction/obstacle_trajectories", self.on_predictions_received, pred_qos
            )

            self.get_logger().info("DynamicPlannerNode initialized with canonical ROSIDL boundary.")

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

    def convert_input_map(self, msg: Any) -> Optional[WorldTerrainMap]:
        """Convert incoming ROS2 terrain map to pure Phase 13 WorldTerrainMap."""
        if msg is None:
            return None
        if isinstance(msg, WorldTerrainMap):
            return msg
        if hasattr(msg, "to_world_terrain_map") and callable(msg.to_world_terrain_map):
            return msg.to_world_terrain_map()
        try:
            return world_terrain_map_from_ros(msg)
        except Exception:
            return None

    def convert_input_localization(self, msg: Any) -> Any:
        """Convert incoming ROS2 localization status."""
        if msg is None:
            return None
        if isinstance(msg, LocalizationStatusMsg):
            return msg
        try:
            res = localization_status_from_ros(msg)
            # Wrap as LocalizationStatusMsg for adapter consumption
            header = RosHeader.create(frame_id="odom", timestamp_s=float(res.get("timestamp", 0.0)))
            import math
            from ugv_navigation.messages.localization_status_msg import (
                Point3D,
                Pose,
                PoseWithCovariance,
                Quaternion,
                Twist,
                TwistWithCovariance,
            )
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

            pose_obj = Pose(
                position=Point3D(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=Quaternion(x=0.0, y=0.0, z=float(math.sin(yaw / 2.0)), w=float(math.cos(yaw / 2.0))),
            )
            pose = PoseWithCovariance(
                pose=pose_obj,
                covariance=[0.01] * 36,
            )
            twist_obj = Twist(
                linear=Point3D(x=float(res.get("v_linear", 0.0)), y=0.0, z=0.0),
                angular=Point3D(x=0.0, y=0.0, z=float(res.get("w_angular", 0.0))),
            )
            twist = TwistWithCovariance(
                twist=twist_obj,
                covariance=[0.01] * 36,
            )
            return LocalizationStatusMsg(
                header=header,
                child_frame_id="base_link",
                tracking_state=res.get("tracking_state", 0),
                confidence=res.get("confidence", 1.0),
                is_pose_valid=res.get("is_valid", True),
                allows_spatial_fusion=True,
                pose=pose,
                twist=twist,
            )
        except Exception:
            return msg

    def convert_output_replan_event(
        self,
        event_obj: Any,
        msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 DynamicReplanEvent to ROS2 message type."""
        if event_obj is None:
            return None
        target_cls = msg_cls or self._replan_msg_cls
        hdr_cls = header_cls or self._header_cls
        p_cls = point_cls or self._point_cls

        if isinstance(event_obj, DynamicReplanEvent):
            p13_event = event_obj
        elif hasattr(event_obj, "to_phase13") and callable(event_obj.to_phase13):
            p13_event = event_obj.to_phase13()
        elif hasattr(event_obj, "event_id"):
            trigger = (
                event_obj.trigger_type
                if isinstance(event_obj.trigger_type, DynamicReplanTrigger)
                else DynamicReplanTrigger(event_obj.trigger_type)
            )
            p13_event = DynamicReplanEvent(
                event_id=str(event_obj.event_id),
                trigger_type=trigger,
                timestamp=float(event_obj.header.stamp.to_sec()),
                current_pose=event_obj.current_pose,
                current_goal=event_obj.current_goal,
                current_global_path=None,
                blocking_obstacle_ids=list(event_obj.blocking_obstacle_ids),
                blocked_duration=float(event_obj.blocked_duration_s),
                deadlock_duration=float(event_obj.deadlock_duration_s),
                current_progress=float(event_obj.current_progress_m),
                reason=str(event_obj.reason),
                confidence=float(event_obj.confidence),
            )
        else:
            return event_obj

        if target_cls is not None:
            return dynamic_replan_event_to_ros(
                p13_event,
                frame_id="odom",
                msg_cls=target_cls,
                header_cls=hdr_cls,
                point_cls=p_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosDynamicReplanEvent is not None:
            return dynamic_replan_event_to_ros(p13_event)
        return DynamicReplanEventMsg.from_phase13(p13_event)

    def convert_output_behavior(
        self,
        dec_obj: Any,
        msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        timestamp_s: Optional[float] = None,
    ) -> Any:
        """Convert Phase 13 BehavioralDecision to ROS2 message type."""
        if dec_obj is None:
            return None
        target_cls = msg_cls or self._behavior_msg_cls
        hdr_cls = header_cls or self._header_cls

        if isinstance(dec_obj, BehavioralDecision):
            p13_dec = dec_obj
        elif hasattr(dec_obj, "to_phase13") and callable(dec_obj.to_phase13):
            p13_dec = dec_obj.to_phase13()
        else:
            p13_dec = dec_obj

        ts = timestamp_s
        if ts is None and hasattr(dec_obj, "header") and hasattr(dec_obj.header, "stamp"):
            ts = float(dec_obj.header.stamp.to_sec())

        if target_cls is not None:
            return behavioral_decision_to_ros(
                p13_dec,
                frame_id="odom",
                timestamp_s=ts,
                msg_cls=target_cls,
                header_cls=hdr_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosBehavioralDecision is not None:
            return behavioral_decision_to_ros(p13_dec, timestamp_s=ts)
        return BehavioralDecisionMsg.from_phase13(p13_dec, timestamp_s=ts or 0.0)

    def convert_output_trajectory(
        self,
        path_obj: Any,
        frame_id: str = "odom",
        timestamp_s: float = 0.0,
        msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13/Planner Path or PathMsg to ROS2 Path message type.

        Preserves:
        - header timestamp and frame_id
        - sequence of waypoints / poses
        """
        if path_obj is None:
            return None
        target_cls = msg_cls or (self._pub_traj_type if self._pub_traj_type is not PathMsg else None)
        if target_cls is not None:
            res = target_cls()
            if hasattr(res, "header"):
                hdr_cls = header_cls or self._header_cls or RosHeader
                if hasattr(hdr_cls, "create"):
                    res.header = hdr_cls.create(frame_id=frame_id, timestamp_s=timestamp_s)
                else:
                    res.header = hdr_cls()
                    if hasattr(res.header, "frame_id"):
                        res.header.frame_id = frame_id
            if hasattr(res, "poses") and hasattr(path_obj, "poses"):
                res.poses = list(path_obj.poses)
            return res
        return path_obj

    def on_map_received(self, map_msg: Any) -> None:
        """Callback for WorldTerrainMap / costmap updates."""
        self._latest_map_msg = map_msg

    def on_localization_status(self, loc_msg: Any) -> None:
        """Callback for localization state updates."""
        self._latest_loc_msg = loc_msg

    def on_goal_received(self, goal_msg: PoseStampedMsg) -> None:
        """Callback for target navigation goal."""
        self._latest_goal_msg = goal_msg

    def on_tracks_received(self, tracks_msg: Any) -> None:
        """Callback for tracked dynamic obstacle state."""
        self._latest_tracks_msg = tracks_msg

    def on_predictions_received(self, pred_msg: Any) -> None:
        """Callback for predicted dynamic obstacle trajectories."""
        self._latest_predictions_msg = pred_msg

    def process_cycle(
        self,
        sim_time_s: Optional[float] = None,
        dt_s: float = 0.10,
        output_replan_cls: Optional[Any] = None,
        output_behavior_cls: Optional[Any] = None,
        output_traj_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ) -> DynamicPlanningResult:
        """Execute one complete dynamic planning cycle."""
        # 1. Convert input boundaries
        converted_tracks = self.convert_input_tracks(self._latest_tracks_msg)
        converted_preds = self.convert_input_predictions(self._latest_predictions_msg)
        converted_map = self.convert_input_map(self._latest_map_msg)
        converted_loc = self.convert_input_localization(self._latest_loc_msg)

        # Monotonic timestamp & stale protection: cannot move backward in time
        current_time = sim_time_s
        if current_time is None and converted_loc is not None and hasattr(converted_loc, "header") and hasattr(converted_loc.header, "stamp"):
            current_time = float(converted_loc.header.stamp.to_sec())
        if current_time is not None:
            if self._last_timestamp is not None and current_time < self._last_timestamp:
                zero_cmd = TwistMsg.create(0.0, 0.0)
                empty_path = PathMsg.from_planner_path(None)
                stale_dec = BehavioralDecisionMsg.from_phase13(
                    BehavioralDecision(
                        state=BehavioralState.STOP,
                        command_v=0.0,
                        command_omega=0.0,
                        reason="Stale input: timestamp moved backward in time",
                    )
                )
                return DynamicPlanningResult(
                    cmd_vel=zero_cmd,
                    global_path=empty_path,
                    behavior=stale_dec,
                    replan_event=None,
                    status=PlannerStatus.LOCALIZATION_INVALID,
                    diagnostics={"error": "Stale timestamp moved backward in time"},
                    cycle_latency_ms=0.0,
                )
            self._last_timestamp = current_time

        # Check TF tree availability
        tf_avail = True
        try:
            if len(self.tf_buffer._frames) == 0:
                tf_avail = True
        except Exception:
            tf_avail = True

        # Prepare adapter inputs: adapter expects either Phase 13 or adapter message types
        tracks_input = (
            TrackedObstacleArrayMsg.from_phase13(converted_tracks, frame_id=converted_tracks.frame_id)
            if converted_tracks is not None
            else self._latest_tracks_msg
        )
        preds_input = (
            PredictedTrajectoryArrayMsg.from_phase13(converted_preds, frame_id=converted_preds.frame_id)
            if converted_preds is not None
            else self._latest_predictions_msg
        )
        map_input = converted_map if converted_map is not None else self._latest_map_msg
        loc_input = converted_loc if converted_loc is not None else self._latest_loc_msg

        # 2. Execute planning cycle via DynamicNavigationAdapter
        result: DynamicPlanningResult = self.adapter.step(
            loc_msg=loc_input,
            map_msg=map_input,
            goal_msg=self._latest_goal_msg,
            tracks_msg=tracks_input,
            predictions_msg=preds_input,
            sim_time_s=sim_time_s,
            dt_s=dt_s,
            tf_available=tf_avail,
        )

        self._last_result = result

        # 3. Output boundary conversions
        ros_behavior = self.convert_output_behavior(
            result.behavior,
            msg_cls=output_behavior_cls,
            header_cls=header_cls,
            timestamp_s=current_time,
        )
        ros_replan = (
            self.convert_output_replan_event(
                result.replan_event,
                msg_cls=output_replan_cls,
                header_cls=header_cls,
                point_cls=point_cls,
            )
            if result.replan_event is not None
            else None
        )
        ros_traj = self.convert_output_trajectory(
            result.global_path,
            frame_id="odom",
            timestamp_s=current_time or 0.0,
            msg_cls=output_traj_cls,
            header_cls=header_cls,
        )

        # 4. Publish to ROS2 topics if runtime is active
        if RCLPY_AVAILABLE:
            if self._cmd_pub is not None:
                self._cmd_pub.publish(result.cmd_vel)
            if self._path_pub is not None:
                self._path_pub.publish(result.global_path)
            if self._trajectory_pub is not None:
                self._trajectory_pub.publish(ros_traj or result.global_path)
            if self._behavior_pub is not None:
                self._behavior_pub.publish(ros_behavior)
            if self._replan_pub is not None and ros_replan is not None:
                self._replan_pub.publish(ros_replan)

        # 5. Record diagnostics
        tracking_name = (
            LocalizationStatusMsg.INV_STATE_MAPPING.get(loc_input.tracking_state, "LOST").name
            if loc_input and hasattr(LocalizationStatusMsg, "INV_STATE_MAPPING") and hasattr(loc_input, "tracking_state")
            else "UNKNOWN"
        )
        self.diagnostics.record_cycle(
            latency_ms=result.cycle_latency_ms,
            tracking_state_name=tracking_name,
            confidence=float(loc_input.confidence) if loc_input and hasattr(loc_input, "confidence") else 0.0,
            rejected=(result.status in (
                PlannerStatus.EMERGENCY_STOP,
                PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
                PlannerStatus.NO_PATH,
                PlannerStatus.LOCALIZATION_INVALID,
                PlannerStatus.MAP_STALE,
            )),
            tf_failed=not tf_avail,
            stale=(result.status == PlannerStatus.MAP_STALE),
        )

        return result


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. DynamicPlannerNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = DynamicPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
