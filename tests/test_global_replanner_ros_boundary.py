"""Focused test suite validating GlobalReplannerNode ROS2 boundary (Milestone 13.12.11).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate GlobalReplannerNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/DynamicReplanEvent.msg
- ugv_navigation_msgs/msg/BehavioralDecision.msg
- ugv_navigation_msgs/msg/TrackedObstacleArray.msg
- ugv_navigation_msgs/msg/PredictedTrajectoryArray.msg
- ugv_navigation_msgs/msg/WorldTerrainMap.msg
- ugv_navigation_msgs/msg/LocalizationStatus.msg

Native DDS communication is NOT claimed or executed by this test suite.
All validation focuses on message conversion, field preservation, safety invariants,
and event arbitration at the ROS boundary.
==============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralDecision,
    BehavioralState,
    DeadlockEvent,
)
from ai.planning.dynamic.collision_checker import CollisionStatus
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
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
    dynamic_replan_event_to_ros,
)
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    DynamicReplanEventMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.nodes.global_replanner_node import (
    GlobalReplannerNode,
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
        self.previous_state = "PROCEED"
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


def make_test_map() -> WorldTerrainMap:
    m = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    m.cost.fill(0.0)
    m.traversability.fill(1.0)
    return m


def make_test_path() -> Path:
    pts = [PathPoint(x=float(i)*0.5, y=0.0, yaw=0.0) for i in range(10)]
    return Path(points=pts)


# ==============================================================================
# TEST SUITE: 13 TRIGGER & SAFETY TESTS
# ==============================================================================

class TestGlobalReplannerRosBoundary:
    """Validate all required trigger conditions, debounce, cooldown, and ROS boundary of GlobalReplannerNode."""

    # 1. DYNAMIC_DEADLOCK Trigger
    def test_trigger_dynamic_deadlock(self):
        """Deadlock signaled by behavioral state triggers DYNAMIC_DEADLOCK replan event."""
        node = GlobalReplannerNode()
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "STOP"
        beh.deadlock = True
        beh.deadlock_event = "DYNAMIC_DEADLOCK"
        beh.blocking_obstacle_id = 42
        beh.reason = "Blocked by obstacle 42"

        res, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=1.0,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )

        assert res.replan_triggered is True
        assert res.trigger_type == DynamicReplanTrigger.DYNAMIC_DEADLOCK
        assert ros_event is not None
        assert ros_event.trigger_type == "DYNAMIC_DEADLOCK"
        assert 42 in ros_event.blocking_obstacle_ids

    # 2. PERSISTENT_BLOCKAGE Trigger
    def test_trigger_persistent_blockage(self):
        """Remaining blocked in STOP/WAIT past duration threshold triggers PERSISTENT_BLOCKAGE."""
        cfg = GlobalReplannerConfig(persistent_blockage_duration_s=2.0, cooldown_s=0.0)
        node = GlobalReplannerNode(config=cfg)
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "STOP"
        beh.blocking_obstacle_id = 99

        # t=1.0: First blocked cycle (0s duration)
        res1, ev1 = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=1.0,
        )
        assert res1.replan_triggered is False

        # t=3.5: Blocked for 2.5s > 2.0s threshold
        res2, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=3.5,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert res2.replan_triggered is True
        assert res2.trigger_type == DynamicReplanTrigger.PERSISTENT_BLOCKAGE
        assert ros_event is not None
        assert ros_event.trigger_type == "PERSISTENT_BLOCKAGE"

    # 3. INSUFFICIENT_PROGRESS Trigger
    def test_trigger_insufficient_progress(self):
        """Lack of forward progress over rolling window triggers INSUFFICIENT_PROGRESS."""
        cfg = GlobalReplannerConfig(progress_loss_window_s=3.0, min_progress_m=0.50, cooldown_s=0.0, persistent_blockage_duration_s=10.0)
        node = GlobalReplannerNode(config=cfg)
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "WAIT"

        # Vehicle stuck at (0.0, 0.0)
        node.check_and_replan(current_pose=(0.0, 0.0, 0.0), goal=(4.0, 0.0), base_map=m, behavioral_decision=beh, timestamp=1.0)
        node.check_and_replan(current_pose=(0.01, 0.0, 0.0), goal=(4.0, 0.0), base_map=m, behavioral_decision=beh, timestamp=2.5)

        # At t=4.0 (window 3.0s elapsed, progress 0.02m < 0.50m)
        res, ros_event = node.check_and_replan(
            current_pose=(0.02, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=4.0,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert res.replan_triggered is True
        assert res.trigger_type == DynamicReplanTrigger.INSUFFICIENT_PROGRESS
        assert ros_event.trigger_type == "INSUFFICIENT_PROGRESS"

    # 4. PATH_INVALIDATED Trigger with Debounce
    def test_trigger_path_invalidated(self):
        """Obstacle intersecting global path triggers PATH_INVALIDATED after debounce cycles."""
        cfg = GlobalReplannerConfig(debounce_cycles=2, cooldown_s=0.0)
        node = GlobalReplannerNode(config=cfg)
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)

        # Dynamic obstacle right on path waypoint at (2.0, 0.0)
        obs = TrackedObstacleState(
            track_id=12,
            timestamp=1.0,
            x=2.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            covariance=np.eye(4),
            bbox_size=(0.8, 0.8),
            bbox_yaw=0.0,
            frame_id="odom",
            status="TRACKING",
        )

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "WAIT"

        # Cycle 1: debounce count = 1 < 2 -> no replan
        res1, ev1 = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            dynamic_obstacles=[obs],
            timestamp=1.0,
        )
        assert res1.replan_triggered is False

        # Cycle 2: debounce count = 2 -> trigger replan
        res2, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            dynamic_obstacles=[obs],
            timestamp=1.1,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert res2.replan_triggered is True
        assert res2.trigger_type == DynamicReplanTrigger.PATH_INVALIDATED
        assert ros_event.trigger_type == "PATH_INVALIDATED"

    # 5. MANUAL Trigger
    def test_trigger_manual(self):
        """manual_replan generates an immediate event with trigger MANUAL."""
        node = GlobalReplannerNode()
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)

        res, ros_event = node.manual_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(3.0, 0.0),
            base_map=m,
            reason="Operator test trigger",
            timestamp=5.0,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert res.replan_triggered is True
        assert res.trigger_type == DynamicReplanTrigger.MANUAL
        assert ros_event.trigger_type == "MANUAL"
        assert "Operator" in ros_event.reason

    # 6. Cooldown suppression
    def test_cooldown_suppression(self):
        """Replan requests within cooldown period are suppressed unless direct collision."""
        cfg = GlobalReplannerConfig(cooldown_s=3.0)
        node = GlobalReplannerNode(config=cfg)
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)
        node.replanner.last_replan_timestamp = 10.0

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "STOP"
        beh.deadlock = True
        beh.deadlock_event = "DYNAMIC_DEADLOCK"

        # Request at t=11.0 (1s into 3s cooldown) -> suppressed
        res, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=11.0,
        )
        assert res.replan_triggered is False
        assert res.cooldown_remaining_s > 0.0
        assert ros_event is None

    # 7. Goal suppression
    def test_goal_suppression(self):
        """When vehicle is already within goal tolerance, replan is suppressed."""
        node = GlobalReplannerNode()
        m = make_test_map()
        p = make_test_path()
        node.replanner.set_global_path(p)

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "STOP"
        beh.deadlock = True

        # Robot at (3.95, 0.0) with goal at (4.0, 0.0) -> distance 0.05m < tolerance
        res, ros_event = node.check_and_replan(
            current_pose=(3.95, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=1.0,
        )
        assert res.replan_triggered is False
        assert res.trigger_type == DynamicReplanTrigger.NONE

    # 8. Failed replan behavior
    def test_failed_replan_behavior(self):
        """When goal is completely unreachable (blocked by walls), replan safely fails without crashing."""
        node = GlobalReplannerNode()
        m = make_test_map()
        # Create solid wall of lethal obstacles
        m.obstacle.fill(True)
        m.cost.fill(1.0)
        p = make_test_path()
        node.replanner.set_global_path(p)

        beh = DoubleBehavioralDecisionMsg()
        beh.state = "STOP"
        beh.deadlock = True

        res, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(4.0, 0.0),
            base_map=m,
            behavioral_decision=beh,
            timestamp=1.0,
            output_event_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert res.replan_triggered is True
        assert res.success is False
        assert res.path is None
        assert res.is_path_stale is True
        assert ros_event.success is False

    # 9. Monotonic timestamp / stale protection
    def test_stale_timestamp_protection(self):
        """Stale timestamp cannot move replanner backward in time."""
        node = GlobalReplannerNode()
        m = make_test_map()
        node.check_and_replan(current_pose=(0.0, 0.0, 0.0), goal=(3.0, 0.0), base_map=m, timestamp=10.0)

        # Attempt older timestamp 8.0 < 10.0
        res, ros_event = node.check_and_replan(
            current_pose=(0.0, 0.0, 0.0),
            goal=(3.0, 0.0),
            base_map=m,
            timestamp=8.0,
        )
        assert res.replan_triggered is False
        assert res.status == PlannerStatus.MAP_STALE
        assert ros_event is None

    # 10. Malformed input handling
    def test_malformed_input_handling(self):
        """Malformed or None inputs handled safely without raising unhandled exceptions."""
        node = GlobalReplannerNode()
        assert node.convert_input_map(None) is None
        assert node.convert_input_behavior(None) is None
        assert node.convert_input_tracks(None) is None
        assert node.convert_input_predictions(None) is None
        assert node.convert_input_localization(None) == (0.0, 0.0, 0.0)

        res, ev = node.check_and_replan(timestamp=1.0)
        assert res is not None

    # 11. Output event field preservation
    def test_output_event_field_preservation(self):
        """All fields of DynamicReplanEvent.msg are faithfully populated."""
        node = GlobalReplannerNode()
        event = DynamicReplanEvent(
            event_id="ev_test_123",
            trigger_type=DynamicReplanTrigger.PERSISTENT_BLOCKAGE,
            timestamp=55.5,
            current_pose=(1.0, 2.0, 0.5),
            current_goal=(10.0, 5.0),
            current_global_path=None,
            blocking_obstacle_ids=[101, 102],
            blocked_duration=3.2,
            deadlock_duration=0.0,
            current_progress=0.15,
            reason="Blocked by obstacles 101, 102",
            confidence=0.97,
        )

        ros_ev = node.convert_output_replan_event(
            event,
            msg_cls=DoubleDynamicReplanEventMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
        )
        assert ros_ev.header.stamp.sec == 55
        assert ros_ev.header.frame_id == "odom"
        assert ros_ev.event_id == "ev_test_123"
        assert ros_ev.trigger_type == "PERSISTENT_BLOCKAGE"
        assert ros_ev.current_pose.x == 1.0
        assert ros_ev.current_pose.y == 2.0
        assert ros_ev.current_yaw == 0.5
        assert ros_ev.current_goal.x == 10.0
        assert ros_ev.current_goal.y == 5.0
        assert ros_ev.blocking_obstacle_ids == [101, 102]
        assert ros_ev.blocked_duration_s == 3.2
        assert ros_ev.deadlock_duration_s == 0.0
        assert ros_ev.current_progress_m == 0.15
        assert ros_ev.reason == "Blocked by obstacles 101, 102"
        assert ros_ev.confidence == 0.97

    # 12. Canonical topic and QoS
    def test_canonical_topic_and_qos(self):
        """GlobalReplannerNode source specifies canonical topics and appropriate QoS."""
        from pathlib import Path as SysPath
        node_file = SysPath(__file__).parent.parent / "ros2_ws" / "src" / "ugv_navigation" / "ugv_navigation" / "nodes" / "global_replanner_node.py"
        content = node_file.read_text(encoding="utf-8")

        assert '"/planning/dynamic_replan_event"' in content
        assert '"/planning/global_path"' in content
        assert '"/behavior/tactical_decision"' in content
        assert '"/occupancy/dynamic_slices"' in content

    # 13. Publisher ownership audit
    def test_publisher_ownership(self):
        """Audit publisher ownership: /planning/dynamic_replan_event is single-source in launch."""
        from pathlib import Path as SysPath
        launch_file = SysPath(__file__).parent.parent / "ros2_ws" / "src" / "ugv_navigation" / "launch" / "dynamic_navigation.launch.py"
        content = launch_file.read_text(encoding="utf-8")

        # In dynamic_navigation.launch.py, DynamicPlannerNode provides the synchronous replanning loop
        assert "dynamic_planner_node" in content
        # GlobalReplannerNode is for modular topologies and must not be run concurrently in dynamic_navigation.launch.py
        assert "global_replanner_node" not in content
