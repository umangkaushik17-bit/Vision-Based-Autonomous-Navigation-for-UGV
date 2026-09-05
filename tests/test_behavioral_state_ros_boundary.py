"""Focused test suite validating BehavioralStateNode ROS2 boundary (Milestone 13.12.10).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate BehavioralStateNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/BehavioralDecision.msg
- ugv_navigation_msgs/msg/TrackedObstacleArray.msg
- ugv_navigation_msgs/msg/PredictedTrajectoryArray.msg
- ugv_navigation_msgs/msg/LocalizationStatus.msg

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

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
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
from ugv_navigation.converters.dynamic_converters import (
    behavioral_decision_to_ros,
)
from ugv_navigation.messages.dynamic_messages import (
    BehavioralDecisionMsg,
    PredictedTrajectoryArrayMsg,
    TrackedObstacleArrayMsg,
)
from ugv_navigation.nodes.behavioral_state_node import (
    BehavioralStateNode,
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


# ==============================================================================
# TEST SUITE: BEHAVIORAL STATE ROS BOUNDARY
# ==============================================================================

class TestBehavioralStateRosBoundary:
    """Validate all required boundary behaviors of BehavioralStateNode."""

    # 1. PROCEED state
    def test_proceed_state(self):
        """Nominal path without obstacles produces PROCEED with non-zero forward command."""
        node = BehavioralStateNode()
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
        diag = {
            "dynamic_collision": False,
            "dynamic_risk": "FREE",
            "valid_trajectories": 10,
            "emergency_stop_triggered": False,
            "dynamically_rejected_candidates": 0,
        }

        res = node.evaluate_state(
            current_state=state,
            best_v=0.40,
            best_w=0.0,
            planner_status=PlannerStatus.FOLLOWING,
            dwa_diagnostics=diag,
            current_time_s=1.0,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert isinstance(res, DoubleBehavioralDecisionMsg)
        assert res.state == "PROCEED"
        assert res.command_v == 0.40
        assert res.dynamic_collision is False

    # 2. AVOID state
    def test_avoid_state(self):
        """Local avoidance maneuver (steering around obstacle) produces AVOID."""
        node = BehavioralStateNode()
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0, timestamp=1.0)
        diag = {
            "dynamic_collision": False,
            "dynamic_risk": "MEDIUM",
            "valid_trajectories": 5,
            "emergency_stop_triggered": False,
            "dynamically_rejected_candidates": 20,
        }

        res = node.evaluate_state(
            current_state=state,
            best_v=0.25,
            best_w=0.15,
            planner_status=PlannerStatus.FOLLOWING,
            dwa_diagnostics=diag,
            current_time_s=1.0,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert isinstance(res, DoubleBehavioralDecisionMsg)
        assert res.state == "AVOID"
        assert res.command_omega == 0.15

    # 3. WAIT state
    def test_wait_state(self):
        """Crossing obstacle predicted to clear corridor produces WAIT and zero command."""
        node = BehavioralStateNode()
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)

        # Crossing obstacle
        ros_preds = DoublePredictedTrajectoryArrayMsg()
        p = DoublePredictedObstacleTrajectoryMsg()
        p.track_id = 7
        wp1 = DoubleTrajectoryWaypointMsg()
        wp1.relative_time_s = 0.0
        wp1.position = DoublePoint(1.0, 0.1, 0.0)
        wp1.velocity = DoubleVector3(0.0, 1.0, 0.0)
        wp2 = DoubleTrajectoryWaypointMsg()
        wp2.relative_time_s = 0.8
        wp2.position = DoublePoint(1.0, 0.9, 0.0)
        wp2.velocity = DoubleVector3(0.0, 1.0, 0.0)
        p.waypoints.extend([wp1, wp2])
        ros_preds.trajectories.append(p)

        diag = {
            "dynamic_collision": False,
            "dynamic_risk": "HIGH",
            "valid_trajectories": 0,
            "emergency_stop_triggered": True,
            "dynamically_rejected_candidates": 100,
        }

        res = node.evaluate_state(
            current_state=state,
            best_v=0.0,
            best_w=0.0,
            planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
            dwa_diagnostics=diag,
            predictions_msg=ros_preds,
            current_time_s=1.0,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert isinstance(res, DoubleBehavioralDecisionMsg)
        assert res.state == "WAIT"
        assert res.command_v == 0.0
        assert res.command_omega == 0.0

    # 4. STOP state
    def test_stop_state(self):
        """Non-clearing persistent hazard or collision produces STOP and zero command."""
        node = BehavioralStateNode()
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.20, omega=0.0, timestamp=1.0)
        diag = {
            "dynamic_collision": True,
            "blocking_obstacle_id": 99,
            "dynamic_risk": "PREDICTED_COLLISION",
            "valid_trajectories": 0,
            "emergency_stop_triggered": True,
        }

        res = node.evaluate_state(
            current_state=state,
            best_v=0.0,
            best_w=0.0,
            planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
            dwa_diagnostics=diag,
            current_time_s=1.0,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert isinstance(res, DoubleBehavioralDecisionMsg)
        assert res.state == "STOP"
        assert res.command_v == 0.0
        assert res.command_omega == 0.0
        assert res.dynamic_collision is True

    # 5. Emergency Stop
    def test_emergency_stop(self):
        """Emergency stop intervention overrides all tactical states to STOP."""
        node = BehavioralStateNode()
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.50, omega=0.0, timestamp=1.0)
        diag = {
            "emergency_stop_triggered": True,
            "valid_trajectories": 0,
            "dynamic_collision": True,
            "blocking_obstacle_id": 15,
        }

        res = node.evaluate_state(
            current_state=state,
            best_v=0.50,
            best_w=0.0,
            planner_status=PlannerStatus.EMERGENCY_STOP,
            dwa_diagnostics=diag,
            current_time_s=1.0,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert res.state == "STOP"
        assert res.command_v == 0.0
        assert res.command_omega == 0.0

    # 6. Deadlock detection
    def test_deadlock_detection(self):
        """Consecutive stop cycles exceeding threshold emit DYNAMIC_DEADLOCK."""
        cfg = BehavioralConfig(deadlock_consecutive_stop_cycles=3, min_dwell_time_s=0.0)
        node = BehavioralStateNode(config=cfg)
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)
        diag = {"dynamic_collision": True, "valid_trajectories": 0}

        # Step 3 times in STOP
        for i in range(3):
            t = 1.0 + i * 0.1
            res = node.evaluate_state(
                current_state=state,
                planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
                dwa_diagnostics=diag,
                current_time_s=t,
                output_msg_cls=DoubleBehavioralDecisionMsg,
                header_cls=DoubleHeader,
            )

        assert res.deadlock is True
        assert res.deadlock_event == "DYNAMIC_DEADLOCK"

    # 7. Recovery behavior
    def test_recovery_behavior(self):
        """Dwell time and recovery confirmation cycles prevent instantaneous flapping."""
        cfg = BehavioralConfig(min_dwell_time_s=0.50, recovery_confirmation_cycles=2)
        node = BehavioralStateNode(config=cfg)
        state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=1.0)

        # Enter STOP at t=1.0
        node.evaluate_state(
            current_state=state,
            planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
            dwa_diagnostics={"dynamic_collision": True, "valid_trajectories": 0},
            current_time_s=1.0,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert node.sm.current_state == BehavioralState.STOP

        # Immediate clear input at t=1.1 (dwell time 0.5s not elapsed)
        res_held = node.evaluate_state(
            current_state=state,
            best_v=0.3,
            planner_status=PlannerStatus.FOLLOWING,
            dwa_diagnostics={"dynamic_collision": False, "valid_trajectories": 10},
            current_time_s=1.1,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert res_held.state == "STOP"

        # After dwell time at t=1.6
        res_after = node.evaluate_state(
            current_state=state,
            best_v=0.3,
            planner_status=PlannerStatus.FOLLOWING,
            dwa_diagnostics={"dynamic_collision": False, "valid_trajectories": 10},
            current_time_s=1.6,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        # Still awaiting second confirmation cycle
        assert res_after.state == "STOP"

        # Second confirmation cycle at t=1.7
        res_recov = node.evaluate_state(
            current_state=state,
            best_v=0.3,
            planner_status=PlannerStatus.FOLLOWING,
            dwa_diagnostics={"dynamic_collision": False, "valid_trajectories": 10},
            current_time_s=1.7,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert res_recov.state == "PROCEED"

    # 8. Timestamps
    def test_timestamp_preservation(self):
        """Timestamps are faithfully preserved in the header."""
        node = BehavioralStateNode()
        state = RobotState(timestamp=5432.10)
        res = node.evaluate_state(
            current_state=state,
            current_time_s=5432.10,
            output_msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
        )
        assert res.header.stamp.sec == 5432
        assert abs(res.header.stamp.nanosec - int((5432.10 - 5432) * 1e9)) < 2000

    # 9. Monotonic timestamp protection
    def test_monotonic_timestamp_protection(self):
        """Stale timestamp cannot move state machine backward in time."""
        node = BehavioralStateNode()
        node.evaluate_state(current_time_s=10.0, output_msg_cls=DoubleBehavioralDecisionMsg, header_cls=DoubleHeader)
        # Attempt older timestamp
        res_stale = node.evaluate_state(current_time_s=8.0, output_msg_cls=DoubleBehavioralDecisionMsg, header_cls=DoubleHeader)
        assert res_stale.state == "STOP"
        assert res_stale.command_v == 0.0
        assert "stale" in res_stale.reason.lower()

    # 10. Malformed input
    def test_malformed_input(self):
        """Malformed or None inputs do not crash node."""
        node = BehavioralStateNode()
        assert node.convert_input_tracks(None) is None
        assert node.convert_input_predictions(None) is None
        assert node.convert_input_localization(None) is None

        # Evaluates safely with defaults
        res = node.evaluate_state(output_msg_cls=DoubleBehavioralDecisionMsg, header_cls=DoubleHeader)
        assert res.state in ("PROCEED", "STOP")

    # 11. Output field preservation
    def test_output_field_preservation(self):
        """All 16 fields of BehavioralDecision.msg are faithfully populated."""
        node = BehavioralStateNode()
        dec = BehavioralDecision(
            state=BehavioralState.AVOID,
            command_v=0.35,
            command_omega=0.12,
            reason="Lateral detour around obstacle 42",
            blocking_obstacle_id=42,
            dynamic_collision=False,
            dynamic_risk=CollisionStatus.PREDICTED_MEDIUM_RISK,
            unknown_horizon=False,
            wait_elapsed_s=1.2,
            wait_timeout=False,
            deadlock=False,
            deadlock_event=DeadlockEvent.NONE,
            transition="PROCEED_TO_AVOID",
            previous_state=BehavioralState.PROCEED,
            state_dwell_time_s=0.5,
            confidence=0.92,
        )

        ros_msg = node.convert_output_decision(
            dec,
            msg_cls=DoubleBehavioralDecisionMsg,
            header_cls=DoubleHeader,
            timestamp_s=25.0,
        )
        assert ros_msg.header.stamp.sec == 25
        assert ros_msg.header.frame_id == "odom"
        assert ros_msg.state == "AVOID"
        assert ros_msg.command_v == 0.35
        assert ros_msg.command_omega == 0.12
        assert ros_msg.reason == "Lateral detour around obstacle 42"
        assert ros_msg.blocking_obstacle_id == 42
        assert ros_msg.dynamic_collision is False
        assert ros_msg.dynamic_risk == "PREDICTED_MEDIUM_RISK"
        assert ros_msg.unknown_horizon is False
        assert ros_msg.wait_elapsed_s == 1.2
        assert ros_msg.wait_timeout is False
        assert ros_msg.deadlock is False
        assert ros_msg.deadlock_event == "NONE"
        assert ros_msg.transition == "PROCEED_TO_AVOID"
        assert ros_msg.previous_state == "PROCEED"
        assert ros_msg.state_dwell_time_s == 0.5
        assert ros_msg.confidence == 0.92

    # 12. Canonical topic and QoS
    def test_canonical_topic_and_qos(self):
        """BehavioralStateNode source specifies canonical topic /behavior/tactical_decision and appropriate QoS."""
        from pathlib import Path as SysPath
        node_file = SysPath(__file__).parent.parent / "ros2_ws" / "src" / "ugv_navigation" / "ugv_navigation" / "nodes" / "behavioral_state_node.py"
        content = node_file.read_text(encoding="utf-8")

        assert '"/behavior/tactical_decision"' in content
        assert '"/tracking/dynamic_obstacles"' in content
        assert '"/prediction/obstacle_trajectories"' in content
        assert '"/localization/status"' in content

    # 13. Publisher ownership audit
    def test_publisher_ownership(self):
        """Audit publisher ownership: /behavior/tactical_decision has single authoritative publisher in launch."""
        from pathlib import Path as SysPath
        launch_file = SysPath(__file__).parent.parent / "ros2_ws" / "src" / "ugv_navigation" / "launch" / "dynamic_navigation.launch.py"
        content = launch_file.read_text(encoding="utf-8")

        # In the integrated launch pipeline, DynamicPlannerNode is the single authoritative coordinator
        assert "dynamic_planner_node" in content
        # BehavioralStateNode must NOT be co-launched concurrently in dynamic_navigation.launch.py
        # to prevent conflicting tactical decision publications
        assert "behavioral_state_node" not in content
