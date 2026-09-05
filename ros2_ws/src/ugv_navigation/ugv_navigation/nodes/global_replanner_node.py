"""Global Replanner ROS2 Node wrapping Phase 13 EventDrivenGlobalReplanner (Milestone 13.12.11).

Translates incoming ROSIDL inputs -> Pure Phase 13 models -> EventDrivenGlobalReplanner -> Pure Phase 13 DynamicReplanResult / DynamicReplanEvent -> ROSIDL message boundary.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralDecision,
    BehavioralState,
    DeadlockEvent,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.path import Path, PathPoint
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.adapters.planner_adapter import PathMsg, PoseStampedMsg
from ugv_navigation.converters.dynamic_converters import (
    behavioral_decision_from_ros,
    dynamic_replan_event_to_ros,
    localization_status_from_ros,
    predicted_trajectory_array_from_ros,
    tracked_obstacle_array_from_ros,
    world_terrain_map_from_ros,
)
from ugv_navigation.diagnostics.node_diagnostics import NodeDiagnosticsTracker
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    DynamicOccupancySliceArrayMsg,
    DynamicReplanEventMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.utils.time_utils import RosHeader

# Native ROSIDL message types (isolated behind ROS boundary)
try:
    from ugv_navigation_msgs.msg import (
        BehavioralDecision as RosBehavioralDecision,
        DynamicOccupancySliceArray as RosDynamicOccupancySliceArray,
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
    RosDynamicReplanEvent = None
    RosBehavioralDecision = None
    RosDynamicOccupancySliceArray = None
    RosLocalizationStatus = None
    RosTrackedObstacleArray = None
    RosPredictedTrajectoryArray = None
    RosWorldTerrainMap = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class GlobalReplannerNode(Node if RCLPY_AVAILABLE else object):
    """ROS2 node that triggers event-driven A* global replanning upon persistent

    behavioral blockage, deadlock, path invalidation, or dynamic obstruction.

    Architectural Boundary:
    ROSIDL inputs -> converter -> Phase 13 domain objects -> EventDrivenGlobalReplanner -> Phase 13 result/event -> converter -> ROSIDL messages.
    """

    def __init__(
        self,
        replanner: Optional[EventDrivenGlobalReplanner] = None,
        config: Optional[GlobalReplannerConfig] = None,
        pub_path_type: Optional[Any] = None,
        pub_event_type: Optional[Any] = None,
        event_msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ):
        self.config = config or GlobalReplannerConfig()
        self.replanner = replanner or EventDrivenGlobalReplanner(config=self.config)
        self.diagnostics = NodeDiagnosticsTracker("global_replanner_node")

        self._event_msg_cls = event_msg_cls
        self._header_cls = header_cls
        self._point_cls = point_cls

        self._pub_path_type = pub_path_type or PathMsg
        self._pub_event_type = pub_event_type or (
            RosDynamicReplanEvent
            if RosDynamicReplanEvent is not None
            else DynamicReplanEventMsg
        )

        self._latest_map_msg: Optional[Any] = None
        self._latest_loc_msg: Optional[Any] = None
        self._latest_goal_msg: Optional[PoseStampedMsg] = None
        self._latest_behavior_msg: Optional[Any] = None
        self._latest_tracks_msg: Optional[Any] = None
        self._latest_preds_msg: Optional[Any] = None
        self._last_timestamp: Optional[float] = None

        self._path_pub = None
        self._event_pub = None
        self._map_sub = None
        self._loc_sub = None
        self._goal_sub = None
        self._occupancy_sub = None
        self._behavior_sub = None
        self._tracks_sub = None
        self._preds_sub = None

        if RCLPY_AVAILABLE:
            super().__init__("global_replanner_node")
            self.declare_parameter("replan_cooldown_s", self.config.cooldown_s)
            self.declare_parameter("deadlock_threshold_s", self.config.deadlock_timeout_s)

            state_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            occ_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )

            self._path_pub = self.create_publisher(self._pub_path_type, "/planning/global_path", state_qos)
            self._event_pub = self.create_publisher(self._pub_event_type, "/planning/dynamic_replan_event", state_qos)

            sub_map_type = RosWorldTerrainMap if RosWorldTerrainMap is not None else WorldTerrainMapMsg
            sub_loc_type = RosLocalizationStatus if RosLocalizationStatus is not None else LocalizationStatusMsg
            sub_beh_type = RosBehavioralDecision if RosBehavioralDecision is not None else BehavioralDecisionMsg
            sub_occ_type = RosDynamicOccupancySliceArray if RosDynamicOccupancySliceArray is not None else DynamicOccupancySliceArrayMsg
            sub_tracks_type = RosTrackedObstacleArray if RosTrackedObstacleArray is not None else TrackedObstacleArrayMsg
            sub_preds_type = RosPredictedTrajectoryArray if RosPredictedTrajectoryArray is not None else PredictedTrajectoryArrayMsg

            self._map_sub = self.create_subscription(
                sub_map_type, "/terrain/world_map", self.on_map_received, state_qos
            )
            self._loc_sub = self.create_subscription(
                sub_loc_type, "/localization/status", self.on_localization_received, state_qos
            )
            self._goal_sub = self.create_subscription(
                PoseStampedMsg, "/goal_pose", self.on_goal_received, state_qos
            )
            self._occupancy_sub = self.create_subscription(
                sub_occ_type, "/occupancy/dynamic_slices", lambda msg: None, occ_qos
            )
            self._behavior_sub = self.create_subscription(
                sub_beh_type, "/behavior/tactical_decision", self.on_behavior_received, state_qos
            )
            self._tracks_sub = self.create_subscription(
                sub_tracks_type, "/tracking/dynamic_obstacles", self.on_tracks_received, sensor_qos
            )
            self._preds_sub = self.create_subscription(
                sub_preds_type, "/prediction/obstacle_trajectories", self.on_predictions_received, occ_qos
            )

            self.get_logger().info("GlobalReplannerNode initialized with canonical ROSIDL boundary.")

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

    def convert_input_behavior(self, msg: Any) -> Optional[BehavioralDecision]:
        """Convert incoming ROS2 behavioral decision to pure Phase 13 BehavioralDecision."""
        if msg is None:
            return None
        if isinstance(msg, BehavioralDecision):
            return msg
        if hasattr(msg, "to_phase13") and callable(msg.to_phase13):
            return msg.to_phase13()
        try:
            return behavioral_decision_from_ros(msg)
        except Exception:
            return None

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

    def convert_input_localization(self, msg: Any) -> Tuple[float, float, float]:
        """Extract (x, y, yaw) from incoming ROS2 localization status."""
        if msg is None:
            return (0.0, 0.0, 0.0)
        try:
            if hasattr(msg, "pose") and hasattr(msg.pose, "pose") and hasattr(msg.pose.pose, "position"):
                px = float(msg.pose.pose.position.x)
                py = float(msg.pose.pose.position.y)
                yaw = float(getattr(msg, "yaw", 0.0))
                return (px, py, yaw)
            if hasattr(msg, "position"):
                return (float(msg.position.x), float(msg.position.y), float(getattr(msg, "yaw", 0.0)))
            res = localization_status_from_ros(msg)
            pos = res.get("position", (0.0, 0.0, 0.0))
            return (float(pos[0]), float(pos[1]), float(res.get("yaw", 0.0)))
        except Exception:
            return (0.0, 0.0, 0.0)

    def convert_output_replan_event(
        self,
        event: DynamicReplanEvent,
        success: bool = True,
        is_path_stale: bool = False,
        old_path_length_m: float = 0.0,
        new_path_length_m: float = 0.0,
        path_deviation_m: float = 0.0,
        planning_latency_ms: float = 0.0,
        msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 DynamicReplanEvent to ROS2 message type.

        Preserves:
        - header timestamp and frame_id ('odom')
        - event_id
        - trigger_type: 'DYNAMIC_DEADLOCK', 'PERSISTENT_BLOCKAGE', 'INSUFFICIENT_PROGRESS', 'PATH_INVALIDATED', 'MANUAL'
        - current_pose, current_yaw, current_goal
        - blocking_obstacle_ids
        - blocked_duration_s, deadlock_duration_s, current_progress_m
        - reason, confidence
        - success, is_path_stale, old_path_length_m, new_path_length_m, path_deviation_m, planning_latency_ms
        """
        if event is None:
            return None
        target_cls = msg_cls or self._event_msg_cls
        hdr_cls = header_cls or self._header_cls
        p_cls = point_cls or self._point_cls

        if target_cls is not None:
            return dynamic_replan_event_to_ros(
                event,
                success=success,
                is_path_stale=is_path_stale,
                old_path_length_m=old_path_length_m,
                new_path_length_m=new_path_length_m,
                path_deviation_m=path_deviation_m,
                planning_latency_ms=planning_latency_ms,
                frame_id="odom",
                msg_cls=target_cls,
                header_cls=hdr_cls,
                point_cls=p_cls,
            )
        if NATIVE_ROSIDL_AVAILABLE and RosDynamicReplanEvent is not None:
            return dynamic_replan_event_to_ros(
                event,
                success=success,
                is_path_stale=is_path_stale,
                old_path_length_m=old_path_length_m,
                new_path_length_m=new_path_length_m,
                path_deviation_m=path_deviation_m,
                planning_latency_ms=planning_latency_ms,
            )
        return DynamicReplanEventMsg.from_phase13(event)

    def convert_output_path(
        self,
        path: Optional[Path],
        frame_id: str = "odom",
        timestamp_s: float = 0.0,
        msg_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
    ) -> Any:
        """Convert Phase 13 Path to ROS2 Path message type."""
        target_cls = msg_cls or (self._pub_path_type if self._pub_path_type is not PathMsg else None)
        path_msg = PathMsg.from_planner_path(path, frame_id=frame_id, timestamp_s=timestamp_s)
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
            if hasattr(res, "poses"):
                res.poses = list(path_msg.poses)
            return res
        return path_msg

    def on_map_received(self, msg: Any) -> None:
        self._latest_map_msg = msg

    def on_localization_received(self, msg: Any) -> None:
        self._latest_loc_msg = msg

    def on_goal_received(self, msg: PoseStampedMsg) -> None:
        self._latest_goal_msg = msg

    def on_behavior_received(self, msg: Any) -> None:
        self._latest_behavior_msg = msg

    def on_tracks_received(self, msg: Any) -> None:
        self._latest_tracks_msg = msg

    def on_predictions_received(self, msg: Any) -> None:
        self._latest_preds_msg = msg

    def check_and_replan(
        self,
        current_pose: Optional[Tuple[float, float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        base_map: Optional[Any] = None,
        behavioral_decision: Optional[Union[BehavioralDecisionMsg, BehavioralDecision, Any]] = None,
        dynamic_obstacles: Optional[Any] = None,
        timestamp: Optional[float] = None,
        output_event_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ) -> Tuple[DynamicReplanResult, Optional[Any]]:
        """Run periodic trigger condition evaluation and execute replanning if required.

        Boundary Flow:
        ROSIDL inputs -> convert_input_* -> Phase 13 replanner.check_and_replan -> convert_output_* -> ROSIDL.
        """
        # Resolve timestamp
        t_now = timestamp if timestamp is not None else 0.0

        # Monotonic timestamp protection
        if self._last_timestamp is not None and t_now < self._last_timestamp:
            stale_result = DynamicReplanResult(
                replan_triggered=False,
                trigger_type=DynamicReplanTrigger.NONE,
                success=False,
                status=PlannerStatus.MAP_STALE,
                path=self.replanner.current_global_path,
                is_path_stale=True,
                failure_reason="Stale input: timestamp moved backward in time",
            )
            return stale_result, None
        self._last_timestamp = t_now

        # Resolve inputs
        pose = current_pose if current_pose is not None else self.convert_input_localization(self._latest_loc_msg)
        goal_pt = goal
        if goal_pt is None and self._latest_goal_msg is not None:
            goal_pt = (float(self._latest_goal_msg.pose.position.x), float(self._latest_goal_msg.pose.position.y))
        if goal_pt is None:
            goal_pt = (0.0, 0.0)

        map_obj = self.convert_input_map(base_map if base_map is not None else self._latest_map_msg)
        beh_obj = self.convert_input_behavior(behavioral_decision if behavioral_decision is not None else self._latest_behavior_msg)

        # Resolve obstacles (predictions or tracks)
        obs_seq = None
        if dynamic_obstacles is not None:
            if isinstance(dynamic_obstacles, (list, tuple)):
                obs_seq = dynamic_obstacles
            elif hasattr(dynamic_obstacles, "predictions"):
                obs_seq = dynamic_obstacles.predictions
            elif hasattr(dynamic_obstacles, "tracks"):
                obs_seq = dynamic_obstacles.tracks
            else:
                converted_preds = self.convert_input_predictions(dynamic_obstacles)
                obs_seq = converted_preds.predictions if converted_preds else None
        elif self._latest_preds_msg is not None:
            converted_preds = self.convert_input_predictions(self._latest_preds_msg)
            obs_seq = converted_preds.predictions if converted_preds else None
        elif self._latest_tracks_msg is not None:
            converted_tracks = self.convert_input_tracks(self._latest_tracks_msg)
            obs_seq = converted_tracks.tracks if converted_tracks else None

        # Execute pure Phase 13 EventDrivenGlobalReplanner logic
        replan_res: DynamicReplanResult = self.replanner.check_and_replan(
            current_pose=pose,
            goal=goal_pt,
            base_map=map_obj,
            behavioral_decision=beh_obj,
            dynamic_obstacles=obs_seq,
            timestamp=t_now,
        )

        ros_event = None
        if replan_res.event is not None:
            m_old = replan_res.metrics.old_path_length if replan_res.metrics is not None else 0.0
            m_new = replan_res.metrics.new_path_length if replan_res.metrics is not None else 0.0
            m_dev = replan_res.metrics.path_deviation if replan_res.metrics is not None else 0.0
            ros_event = self.convert_output_replan_event(
                replan_res.event,
                success=replan_res.success,
                is_path_stale=replan_res.is_path_stale,
                old_path_length_m=m_old,
                new_path_length_m=m_new,
                path_deviation_m=m_dev,
                planning_latency_ms=replan_res.total_latency_ms,
                msg_cls=output_event_cls,
                header_cls=header_cls,
                point_cls=point_cls,
            )

        # Output publishing if active
        if RCLPY_AVAILABLE:
            if ros_event is not None and self._event_pub is not None:
                self._event_pub.publish(ros_event)
            if replan_res.success and replan_res.path is not None and self._path_pub is not None:
                ros_path = self.convert_output_path(replan_res.path, timestamp_s=t_now)
                self._path_pub.publish(ros_path)

        self.diagnostics.record_cycle(latency_ms=replan_res.total_latency_ms)
        return replan_res, ros_event

    def manual_replan(
        self,
        current_pose: Tuple[float, float, float],
        goal: Tuple[float, float],
        base_map: Any,
        dynamic_obstacles: Optional[Any] = None,
        reason: str = "Operator manual trigger",
        timestamp: float = 0.0,
        output_event_cls: Optional[Any] = None,
        header_cls: Optional[Any] = None,
        point_cls: Optional[Any] = None,
    ) -> Tuple[DynamicReplanResult, Optional[Any]]:
        """Trigger an immediate MANUAL global replan."""
        map_obj = self.convert_input_map(base_map)
        obs_seq = dynamic_obstacles
        if hasattr(obs_seq, "predictions"):
            obs_seq = obs_seq.predictions

        manual_event = DynamicReplanEvent(
            trigger_type=DynamicReplanTrigger.MANUAL,
            timestamp=timestamp,
            current_pose=current_pose,
            current_goal=goal,
            current_global_path=self.replanner.current_global_path,
            blocking_obstacle_ids=[],
            blocked_duration=0.0,
            deadlock_duration=0.0,
            current_progress=0.0,
            reason=reason,
            confidence=1.0,
            event_id=f"manual_{int(timestamp * 1000)}",
        )

        res = self.replanner.execute_replan(
            current_pose=current_pose,
            goal=goal,
            base_map=map_obj,
            dynamic_obstacles=obs_seq,
            event=manual_event,
        )

        m_old = res.metrics.old_path_length if res.metrics is not None else 0.0
        m_new = res.metrics.new_path_length if res.metrics is not None else 0.0
        m_dev = res.metrics.path_deviation if res.metrics is not None else 0.0
        ros_event = self.convert_output_replan_event(
            manual_event,
            success=res.success,
            is_path_stale=res.is_path_stale,
            old_path_length_m=m_old,
            new_path_length_m=m_new,
            path_deviation_m=m_dev,
            planning_latency_ms=res.total_latency_ms,
            msg_cls=output_event_cls,
            header_cls=header_cls,
            point_cls=point_cls,
        )

        if RCLPY_AVAILABLE:
            if ros_event is not None and self._event_pub is not None:
                self._event_pub.publish(ros_event)
            if res.success and res.path is not None and self._path_pub is not None:
                ros_path = self.convert_output_path(res.path, timestamp_s=timestamp)
                self._path_pub.publish(ros_path)

        return res, ros_event

    def evaluate_and_replan(
        self,
        costmap_2d: Any,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        current_path: Optional[List[Tuple[float, float]]] = None,
        behavior_decision: Optional[Union[BehavioralDecisionMsg, BehavioralDecision]] = None,
        timestamp_s: float = 0.0,
        obstacles: Optional[Any] = None,
        predictions: Optional[Any] = None,
    ) -> Tuple[Optional[Path], Optional[Any]]:
        """Backward-compatible wrapper executing global replan check."""
        goal_2d = (float(goal_pose[0]), float(goal_pose[1]))
        res, ros_event = self.check_and_replan(
            current_pose=start_pose,
            goal=goal_2d,
            base_map=costmap_2d,
            behavioral_decision=behavior_decision,
            dynamic_obstacles=predictions or obstacles,
            timestamp=timestamp_s,
        )
        return res.path, ros_event


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. GlobalReplannerNode requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = GlobalReplannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
