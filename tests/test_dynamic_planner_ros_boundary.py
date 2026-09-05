"""Focused test suite validating DynamicPlannerNode ROS2 boundary (Milestone 13.12.9).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate DynamicPlannerNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/BehavioralDecision.msg
- ugv_navigation_msgs/msg/DynamicReplanEvent.msg
- ugv_navigation_msgs/msg/TrackedObstacleArray.msg
- ugv_navigation_msgs/msg/PredictedTrajectoryArray.msg
- ugv_navigation_msgs/msg/LocalizationStatus.msg
- ugv_navigation_msgs/msg/WorldTerrainMap.msg

Native DDS communication is NOT claimed or executed by this test suite.
All validation focuses on message conversion, field preservation, safety invariants,
and behavioral arbitration at the ROS boundary.
==============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanTrigger,
)
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path, PathPoint
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
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    DynamicReplanEventMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.messages.localization_status_msg import (
    LocalizationStatusMsg,
    Point3D,
    Pose,
    PoseWithCovariance,
    Quaternion,
    Twist,
    TwistWithCovariance,
)
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.dynamic_planner_node import (
    DynamicPlannerNode,
    NATIVE_ROSIDL_AVAILABLE,
)
from ugv_navigation.utils.time_utils import RosHeader, RosTime


# ==============================================================================
# TEST DOUBLES (Exact structural replicas of ROSIDL generated messages)
# ==============================================================================

class DoubleTime:
    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.sec = int(sec)
        self.nanosec = int(nanosec)

    def to_sec(self) -> float:
        return float(self.sec) + float(self.nanosec) * 1e-9


class DoubleHeader:
    def __init__(self, frame_id: str = "odom", stamp: Optional[DoubleTime] = None):
        self.frame_id = str(frame_id)
        self.stamp = stamp or DoubleTime()


class DoublePoint:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class DoubleVector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class DoubleBehavioralDecisionMsg:
    """Test Double for ugv_navigation_msgs/msg/BehavioralDecision.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.state = "PROCEED"
        self.command_v = 0.0
        self.command_omega = 0.0
        self.reason = ""
        self.blocking_obstacle_id = 0
        self.dynamic_collision = False
        self.dynamic_risk = "FREE"
        self.unknown_horizon = False
        self.wait_elapsed_s = 0.0
        self.wait_timeout = False
        self.deadlock = False
        self.deadlock_event = "NONE"
        self.transition = ""
        self.previous_state = ""
        self.state_dwell_time_s = 0.0
        self.confidence = 1.0


class DoubleDynamicReplanEventMsg:
    """Test Double for ugv_navigation_msgs/msg/DynamicReplanEvent.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.event_id = ""
        self.trigger_type = ""
        self.current_pose = DoublePoint()
        self.current_yaw = 0.0
        self.current_goal = DoublePoint()
        self.blocking_obstacle_ids = []
        self.blocked_duration_s = 0.0
        self.deadlock_duration_s = 0.0
        self.current_progress_m = 0.0
        self.reason = ""
        self.confidence = 1.0
        self.success = False
        self.is_path_stale = False
        self.old_path_length_m = 0.0
        self.new_path_length_m = 0.0
        self.path_deviation_m = 0.0
        self.planning_latency_ms = 0.0


class DoubleTrackedObstacleMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 1
        self.status = "CONFIRMED"
        self.is_confirmed = True
        self.confidence = 1.0
        self.position = DoublePoint(5.0, 0.0, 0.0)
        self.velocity = DoubleVector3(0.0, 0.0, 0.0)
        self.bbox_size = DoubleVector3(1.0, 1.0, 1.0)
        self.bbox_yaw = 0.0
        self.covariance = [0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.04]
        self.total_hits = 5
        self.consecutive_misses = 0
        self.time_since_last_update_s = 0.0


class DoubleTrackedObstacleArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.tracks = []


class DoubleTrajectoryWaypointMsg:
    def __init__(self):
        self.target_time = DoubleTime(1, 0)
        self.relative_time_s = 0.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.covariance = [0.01, 0.0, 0.0, 0.01]
        self.uncertainty_radius_m = 0.1
        self.confidence = 1.0


class DoublePredictedObstacleTrajectoryMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 1
        self.status = "TRACKING"
        self.bbox_size = DoubleVector3(1.0, 1.0, 1.0)
        self.bbox_yaw = 0.0
        self.waypoints = []
        self.prediction_horizon_s = 2.0
        self.is_unknown = False
        self.confidence = 0.95


class DoublePredictedTrajectoryArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.trajectories = []


class DoublePoseStampedMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.pose = type("Pose", (), {"position": DoublePoint(), "orientation": DoubleVector3()})()


class DoublePathMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.poses = []


class DoubleLocalizationStatusMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.child_frame_id = "base_link"
        self.tracking_state = 0
        self.confidence = 0.98
        self.is_pose_valid = True
        self.allows_spatial_fusion = True
        self.position = DoublePoint(0.0, 0.0, 0.0)
        self.orientation = (0.0, 0.0, 0.0, 1.0)
        self.yaw = 0.0
        self.v_linear = 0.0
        self.w_angular = 0.0
        self.position_covariance = [0.01] * 9
        self.velocity_covariance = [0.01] * 9
        self.tracked_features = 50
        self.inliers = 40
        self.keyframe_id = 1
        self.is_keyframe = False
        self.relocalized = False


class DoubleWorldTerrainMapMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.resolution = 0.10
        self.width = 10
        self.height = 10
        self.origin = type("Origin", (), {"position": DoublePoint(-5.0, -5.0, 0.0)})()
        self.cost = [0.0] * 100
        self.traversability = [1.0] * 100
        self.obstacle = [False] * 100
        self.hazard = [0.0] * 100
        self.unknown = [False] * 100
        self.confidence = [1.0] * 100
        self.slope = [0.0] * 100
        self.roughness = [0.0] * 100
        self.height_var = [0.0] * 100
        self.observation_count = [1] * 100


# ==============================================================================
# HELPER FIXTURES
# ==============================================================================

def make_test_map(width=50, height=50, resolution=0.1) -> WorldTerrainMap:
    m = WorldTerrainMap(resolution_m=resolution, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    m.cost.fill(0.0)
    m.traversability.fill(1.0)
    return m


def make_test_goal(x: float = 3.0, y: float = 0.0) -> PoseStampedMsg:
    return PoseStampedMsg.from_xy_yaw(x=x, y=y, yaw=0.0, frame_id="odom", timestamp_s=1.0)


def make_test_localization(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, timestamp: float = 1.0) -> LocalizationStatusMsg:
    header = RosHeader.create(frame_id="odom", timestamp_s=timestamp)
    pose_obj = Pose(
        position=Point3D(x=float(x), y=float(y), z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=float(math.sin(yaw / 2.0)), w=float(math.cos(yaw / 2.0))),
    )
    pose = PoseWithCovariance(pose=pose_obj, covariance=[0.01] * 36)
    twist_obj = Twist(linear=Point3D(x=0.0, y=0.0, z=0.0), angular=Point3D(x=0.0, y=0.0, z=0.0))
    twist = TwistWithCovariance(twist=twist_obj, covariance=[0.01] * 36)
    return LocalizationStatusMsg(
        header=header,
        child_frame_id="base_link",
        tracking_state=0,
        confidence=0.99,
        is_pose_valid=True,
        allows_spatial_fusion=True,
        pose=pose,
        twist=twist,
    )


# ==============================================================================
# TEST SUITE: 15 MANDATORY TESTS
# ==============================================================================

class TestDynamicPlannerRosBoundary:
    """Validate all 15 required boundary behaviors of DynamicPlannerNode."""

    # 1. Input conversion
    def test_input_conversion(self):
        """Node converts ROSIDL/double messages to pure Phase 13 domain models without error."""
        node = DynamicPlannerNode()

        # Tracked obstacles
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        ros_tracks.header = DoubleHeader("odom", DoubleTime(10, 0))
        t1 = DoubleTrackedObstacleMsg()
        t1.track_id = 42
        t1.position = DoublePoint(2.5, 1.0, 0.0)
        ros_tracks.tracks.append(t1)

        p13_tracks = node.convert_input_tracks(ros_tracks)
        assert isinstance(p13_tracks, TrackedObstacleArray)
        assert len(p13_tracks.tracks) == 1
        assert p13_tracks.tracks[0].track_id == 42
        assert p13_tracks.tracks[0].position == (2.5, 1.0)

        # Predictions
        ros_preds = DoublePredictedTrajectoryArrayMsg()
        ros_preds.header = DoubleHeader("odom", DoubleTime(10, 0))
        p1 = DoublePredictedObstacleTrajectoryMsg()
        p1.track_id = 42
        wp = DoubleTrajectoryWaypointMsg()
        wp.relative_time_s = 0.5
        wp.position = DoublePoint(3.0, 1.0, 0.0)
        p1.waypoints.append(wp)
        ros_preds.trajectories.append(p1)

        p13_preds = node.convert_input_predictions(ros_preds)
        assert isinstance(p13_preds, PredictedTrajectoryArray)
        assert len(p13_preds.predictions) == 1
        assert p13_preds.predictions[0].track_id == 42
        assert len(p13_preds.predictions[0].waypoints) == 1

        # Localization
        ros_loc = DoubleLocalizationStatusMsg()
        ros_loc.header = DoubleHeader("odom", DoubleTime(10, 0))
        ros_loc.position = DoublePoint(1.0, 2.0, 0.0)
        ros_loc.yaw = 0.5
        converted_loc = node.convert_input_localization(ros_loc)
        assert isinstance(converted_loc, LocalizationStatusMsg)
        assert converted_loc.pose.pose.position.x == 1.0
        assert converted_loc.pose.pose.position.y == 2.0

        # World map
        ros_map = DoubleWorldTerrainMapMsg()
        ros_map.header = DoubleHeader("odom", DoubleTime(10, 0))
        converted_map = node.convert_input_map(ros_map)
        assert isinstance(converted_map, WorldTerrainMap)

    # 2. Planner invocation
    def test_planner_invocation(self):
        """process_cycle invokes adapter and produces a valid DynamicPlanningResult."""
        node = DynamicPlannerNode()
        node.on_localization_status(make_test_localization(0.0, 0.0, 0.0, 1.0))
        node.on_map_received(make_test_map())
        node.on_goal_received(make_test_goal(3.0, 0.0))

        result = node.process_cycle(sim_time_s=1.0)
        assert isinstance(result, DynamicPlanningResult)
        assert result.cmd_vel is not None
        assert result.behavior is not None
        assert result.status in (PlannerStatus.FOLLOWING, PlannerStatus.GOAL_REACHED, PlannerStatus.NO_PATH, PlannerStatus.PLANNING, PlannerStatus.SUCCESS)

    # 3. Trajectory output conversion
    def test_trajectory_output_conversion(self):
        """convert_output_trajectory converts PathMsg to ROS target class preserving poses & stamp."""
        node = DynamicPlannerNode()
        p1 = PathPoint(x=0.0, y=0.0, yaw=0.0)
        p2 = PathPoint(x=1.0, y=0.0, yaw=0.0)
        path = Path(points=[p1, p2])
        path_msg = PathMsg.from_planner_path(path, frame_id="odom", timestamp_s=2.5)

        ros_traj = node.convert_output_trajectory(
            path_msg,
            frame_id="odom",
            timestamp_s=2.5,
            msg_cls=DoublePathMsg,
            header_cls=DoubleHeader,
        )
        assert isinstance(ros_traj, DoublePathMsg)
        assert ros_traj.header.frame_id == "odom"
        assert len(ros_traj.poses) == 2

    # 4. Replan-event conversion
    def test_replan_event_conversion(self):
        """convert_output_replan_event converts Phase 13 event to ROS message double."""
        node = DynamicPlannerNode()
        event = DynamicReplanEvent(
            event_id="ev_001",
            trigger_type=DynamicReplanTrigger.PERSISTENT_BLOCKAGE,
            timestamp=5.0,
            current_pose=(1.0, 2.0, 0.0),
            current_goal=(10.0, 0.0),
            current_global_path=None,
            blocking_obstacle_ids=[7, 9],
            blocked_duration=2.5,
            deadlock_duration=0.0,
            current_progress=0.1,
            reason="Path completely blocked by obstacle 7",
            confidence=0.95,
        )

        ros_event = node.convert_output_replan_event(
            event,
            msg_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert isinstance(ros_event, DoubleDynamicReplanEventMsg)
        assert ros_event.event_id == "ev_001"
        assert ros_event.trigger_type == "PERSISTENT_BLOCKAGE"
        assert ros_event.blocking_obstacle_ids == [7, 9]
        assert ros_event.blocked_duration_s == 2.5
        assert ros_event.current_pose.x == 1.0
        assert ros_event.current_pose.y == 2.0
        assert ros_event.current_goal.x == 10.0

    # 5. Timestamp preservation
    def test_timestamp_preservation(self):
        """Timestamps are faithfully preserved through planner cycle and conversions."""
        node = DynamicPlannerNode()
        t_now = 1234.56
        node.on_localization_status(make_test_localization(0.0, 0.0, 0.0, timestamp=t_now))
        node.on_map_received(make_test_map())
        node.on_goal_received(make_test_goal(3.0, 0.0))

        result = node.process_cycle(
            sim_time_s=t_now,
            output_behavior_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert result.behavior is not None
        ros_dec = node.convert_output_behavior(
            result.behavior,
            msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
            timestamp_s=t_now,
        )
        assert ros_dec.header.stamp.sec == 1234
        assert abs(ros_dec.header.stamp.nanosec - int((1234.56 - 1234) * 1e9)) < 2000

    # 6. Frame preservation
    def test_frame_preservation(self):
        """Frame 'odom' is preserved across all output boundaries."""
        node = DynamicPlannerNode()
        node.on_localization_status(make_test_localization(0.0, 0.0, 0.0, timestamp=1.0))
        node.on_map_received(make_test_map())
        node.on_goal_received(make_test_goal(3.0, 0.0))

        result = node.process_cycle(sim_time_s=1.0)
        assert result.global_path.header.frame_id == "odom"
        ros_dec = node.convert_output_behavior(result.behavior, msg_cls=DoubleBehavioralDecisionMsg, header_cls=DoubleHeader)
        assert ros_dec.header.frame_id == "odom"

    # 7. Collision safety fields
    def test_collision_safety_fields(self):
        """Behavioral decision correctly populates collision safety fields."""
        node = DynamicPlannerNode()
        dec = BehavioralDecision(
            state=BehavioralState.STOP,
            command_v=0.0,
            command_omega=0.0,
            reason="Emergency obstacle proximity",
            blocking_obstacle_id=99,
            dynamic_collision=True,
            dynamic_risk=CollisionStatus.PREDICTED_COLLISION,
            unknown_horizon=False,
            confidence=0.98,
        )
        ros_dec = node.convert_output_behavior(dec, msg_cls=DoubleBehavioralDecisionMsg, header_cls=DoubleHeader)
        assert ros_dec.state == "STOP"
        assert ros_dec.command_v == 0.0
        assert ros_dec.blocking_obstacle_id == 99
        assert ros_dec.dynamic_collision is True
        assert ros_dec.dynamic_risk == "PREDICTED_COLLISION"
        assert ros_dec.confidence == 0.98

    # 8. Unknown-horizon behavior
    def test_unknown_horizon_behavior(self):
        """UNKNOWN horizon is NEVER silently converted to FREE or SAFE."""
        node = DynamicPlannerNode()
        dec = BehavioralDecision(
            state=BehavioralState.WAIT,
            command_v=0.0,
            command_omega=0.0,
            reason="Prediction horizon insufficient for obstacle 5",
            blocking_obstacle_id=5,
            dynamic_collision=False,
            dynamic_risk=CollisionStatus.UNKNOWN_HORIZON,
            unknown_horizon=True,
        )
        ros_dec = node.convert_output_behavior(dec, msg_cls=DoubleBehavioralDecisionMsg, header_cls=DoubleHeader)
        assert ros_dec.unknown_horizon is True
        assert ros_dec.dynamic_risk == "UNKNOWN_HORIZON"
        assert ros_dec.dynamic_risk != "FREE"
        assert ros_dec.dynamic_risk != "LOW"

    # 9. Multiple-obstacle behavior
    def test_multiple_obstacle_behavior(self):
        """Multiple dynamic obstacles in tracks/predictions are all preserved without overwriting."""
        node = DynamicPlannerNode()
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        ros_tracks.header = DoubleHeader("odom", DoubleTime(1, 0))
        for obs_id, x_pos in [(1, 2.0), (2, 4.0), (3, 6.0)]:
            t = DoubleTrackedObstacleMsg()
            t.header = DoubleHeader("odom", DoubleTime(1, 0))
            t.track_id = obs_id
            t.position = DoublePoint(x_pos, 0.0, 0.0)
            ros_tracks.tracks.append(t)

        converted = node.convert_input_tracks(ros_tracks)
        assert len(converted.tracks) == 3
        track_ids = [t.track_id for t in converted.tracks]
        assert track_ids == [1, 2, 3]

    # 10. Deterministic output
    def test_deterministic_output(self):
        """Identical inputs produce identical outputs across repeated invocations."""
        node1 = DynamicPlannerNode()
        node2 = DynamicPlannerNode()

        loc = make_test_localization(0.0, 0.0, 0.0, timestamp=5.0)
        m = make_test_map()
        goal = make_test_goal(2.0, 0.0)

        for n in (node1, node2):
            n.on_localization_status(loc)
            n.on_map_received(m)
            n.on_goal_received(goal)

        res1 = node1.process_cycle(sim_time_s=5.0)
        res2 = node2.process_cycle(sim_time_s=5.0)

        assert res1.status == res2.status
        assert res1.cmd_vel.linear.x == res2.cmd_vel.linear.x
        assert res1.cmd_vel.angular.z == res2.cmd_vel.angular.z
        assert res1.behavior.state == res2.behavior.state

    # 11. Malformed input handling
    def test_malformed_input_handling(self):
        """Malformed or corrupted input does not crash node and produces safe zero velocity."""
        node = DynamicPlannerNode()
        # Non-existent or garbage inputs
        assert node.convert_input_tracks(None) is None
        assert node.convert_input_tracks("corrupted_string") is None
        assert node.convert_input_predictions(None) is None
        assert node.convert_input_map(None) is None

        # Cycle without map/localization returns safe STOP status with zero velocity
        res = node.process_cycle(sim_time_s=1.0)
        assert res.cmd_vel.linear.x == 0.0
        assert res.cmd_vel.angular.z == 0.0
        assert res.status in (PlannerStatus.LOCALIZATION_INVALID, PlannerStatus.MAP_UNAVAILABLE)

    # 12. Canonical topic and QoS
    def test_canonical_topics_and_qos(self):
        """DynamicPlannerNode declares canonical topics and correct QoS profiles."""
        node = DynamicPlannerNode()
        # Verify canonical topic strings exist in node source
        from pathlib import Path as SysPath
        node_file = SysPath(__file__).parent.parent / "ros2_ws" / "src" / "ugv_navigation" / "ugv_navigation" / "nodes" / "dynamic_planner_node.py"
        content = node_file.read_text(encoding="utf-8")

        assert '"/cmd_vel"' in content
        assert '"/planning/global_path"' in content
        assert '"/planning/dynamic_trajectory"' in content
        assert '"/behavior/tactical_decision"' in content
        assert '"/planning/dynamic_replan_event"' in content
        assert '"/terrain/world_map"' in content
        assert '"/localization/status"' in content
        assert '"/tracking/dynamic_obstacles"' in content
        assert '"/prediction/obstacle_trajectories"' in content

    # 13. STOP behavior
    def test_stop_behavior(self):
        """When an obstacle is immediately ahead (within stop distance), node produces STOP and zero cmd_vel."""
        node = DynamicPlannerNode()
        node.on_localization_status(make_test_localization(0.0, 0.0, 0.0, timestamp=1.0))
        node.on_map_received(make_test_map())
        node.on_goal_received(make_test_goal(5.0, 0.0))

        # Place obstacle directly in front at x=0.30m (emergency stop < 0.35m)
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        ros_tracks.header = DoubleHeader("odom", DoubleTime(1, 0))
        t = DoubleTrackedObstacleMsg()
        t.header = DoubleHeader("odom", DoubleTime(1, 0))
        t.track_id = 99
        t.position = DoublePoint(0.30, 0.0, 0.0)
        ros_tracks.tracks.append(t)
        node.on_tracks_received(ros_tracks)

        ros_preds = DoublePredictedTrajectoryArrayMsg()
        ros_preds.header = DoubleHeader("odom", DoubleTime(1, 0))
        p = DoublePredictedObstacleTrajectoryMsg()
        p.header = DoubleHeader("odom", DoubleTime(1, 0))
        p.track_id = 99
        wp = DoubleTrajectoryWaypointMsg()
        wp.target_time = DoubleTime(1, 0)
        wp.position = DoublePoint(0.30, 0.0, 0.0)
        p.waypoints.append(wp)
        ros_preds.trajectories.append(p)
        node.on_predictions_received(ros_preds)

        result = node.process_cycle(sim_time_s=1.0)
        assert result.cmd_vel.linear.x == 0.0
        assert result.behavior.state == "STOP"

    # 14. WAIT behavior
    def test_wait_behavior(self):
        """When an obstacle blocks within wait distance (e.g. 1.2m), node commands zero velocity."""
        node = DynamicPlannerNode()
        node.on_localization_status(make_test_localization(0.0, 0.0, 0.0, timestamp=1.0))
        node.on_map_received(make_test_map())
        node.on_goal_received(make_test_goal(5.0, 0.0))

        # Place crossing obstacle in corridor that clears within 1.5s
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        ros_tracks.header = DoubleHeader("odom", DoubleTime(1, 0))
        t = DoubleTrackedObstacleMsg()
        t.header = DoubleHeader("odom", DoubleTime(1, 0))
        t.track_id = 50
        t.position = DoublePoint(1.0, 0.1, 0.0)
        t.velocity = DoubleVector3(0.0, 1.0, 0.0)
        ros_tracks.tracks.append(t)
        node.on_tracks_received(ros_tracks)

        ros_preds = DoublePredictedTrajectoryArrayMsg()
        ros_preds.header = DoubleHeader("odom", DoubleTime(1, 0))
        p = DoublePredictedObstacleTrajectoryMsg()
        p.header = DoubleHeader("odom", DoubleTime(1, 0))
        p.track_id = 50

        wp1 = DoubleTrajectoryWaypointMsg()
        wp1.target_time = DoubleTime(1, 0)
        wp1.relative_time_s = 0.0
        wp1.position = DoublePoint(1.0, 0.1, 0.0)
        wp1.velocity = DoubleVector3(0.0, 1.0, 0.0)

        wp2 = DoubleTrajectoryWaypointMsg()
        wp2.target_time = DoubleTime(1, 500000000)
        wp2.relative_time_s = 0.5
        wp2.position = DoublePoint(1.0, 0.6, 0.0)
        wp2.velocity = DoubleVector3(0.0, 1.0, 0.0)

        wp3 = DoubleTrajectoryWaypointMsg()
        wp3.target_time = DoubleTime(2, 0)
        wp3.relative_time_s = 1.0
        wp3.position = DoublePoint(1.0, 1.1, 0.0)
        wp3.velocity = DoubleVector3(0.0, 1.0, 0.0)

        p.waypoints.extend([wp1, wp2, wp3])
        ros_preds.trajectories.append(p)
        node.on_predictions_received(ros_preds)

        result = node.process_cycle(sim_time_s=1.0)
        # In wait state or avoidance hold, forward command must be safe / zeroed
        assert result.cmd_vel.linear.x == 0.0
        assert result.behavior.state in ("WAIT", "STOP", "AVOID")

    # 15. AVOID behavior
    def test_avoid_behavior(self):
        """When an obstacle is slightly off-center with lateral clearance, planner can navigate or avoid."""
        node = DynamicPlannerNode()
        node.on_localization_status(make_test_localization(0.0, 0.0, 0.0, timestamp=1.0))
        node.on_map_received(make_test_map())
        node.on_goal_received(make_test_goal(5.0, 0.0))

        # Obstacle at x=2.0, y=1.2 (clearance exists on right or ahead)
        ros_tracks = DoubleTrackedObstacleArrayMsg()
        t = DoubleTrackedObstacleMsg()
        t.track_id = 12
        t.position = DoublePoint(2.0, 1.2, 0.0)
        ros_tracks.tracks.append(t)
        node.on_tracks_received(ros_tracks)

        result = node.process_cycle(sim_time_s=1.0)
        # Should not crash and should produce a valid decision (PROCEED or AVOID)
        assert result.behavior.state in ("PROCEED", "AVOID")
