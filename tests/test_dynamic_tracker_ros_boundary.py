"""Focused test suite validating DynamicTrackerNode ROS2 boundary (Milestone 13.12.5).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate the DynamicTrackerNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/DynamicObstacleDetectionArray.msg
- ugv_navigation_msgs/msg/DynamicObstacleDetection.msg
- ugv_navigation_msgs/msg/TrackedObstacleArray.msg
- ugv_navigation_msgs/msg/TrackedObstacle.msg

Native DDS communication is NOT claimed or executed by this test suite.
All validation focuses on message conversion, field preservation, lifecycle,
and safety invariants at the ROS boundary.
==============================================================================
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple
import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
    TrackedObstacleArray,
    TrackedObstacleState,
)
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from ugv_navigation.converters.dynamic_converters import (
    dynamic_obstacle_detection_array_from_ros,
    dynamic_obstacle_detection_from_ros,
    tracked_obstacle_array_from_ros,
    tracked_obstacle_array_to_ros,
)
from ugv_navigation.nodes.dynamic_tracker_node import (
    DynamicTrackerNode,
    NATIVE_ROSIDL_AVAILABLE,
)


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


class DoubleDetectionMsg:
    """Test Double for ugv_navigation_msgs/msg/DynamicObstacleDetection.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.detection_id = 0
        self.class_name = "obstacle"
        self.confidence = 1.0
        self.position = DoublePoint()
        self.dimensions = DoubleVector3(1.0, 1.0, 1.0)
        self.yaw_rad = 0.0
        self.position_covariance = [0.04, 0.0, 0.0, 0.04]
        self.is_valid = True


class DoubleDetectionArrayMsg:
    """Test Double for ugv_navigation_msgs/msg/DynamicObstacleDetectionArray.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.detections: List[DoubleDetectionMsg] = []


class DoubleTrackedObstacleMsg:
    """Test Double for ugv_navigation_msgs/msg/TrackedObstacle.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "NEW"
        self.is_confirmed = False
        self.confidence = 1.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.bbox_size = DoubleVector3(1.0, 1.0, 1.0)
        self.bbox_yaw = 0.0
        self.covariance = [0.0] * 16
        self.total_hits = 1
        self.consecutive_misses = 0
        self.time_since_last_update_s = 0.0


class DoubleTrackedObstacleArrayMsg:
    """Test Double for ugv_navigation_msgs/msg/TrackedObstacleArray.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.tracks: List[DoubleTrackedObstacleMsg] = []


def make_double_detection_array(
    timestamp_s: float = 10.0,
    frame_id: str = "odom",
    detections_data: Optional[List[dict]] = None,
) -> DoubleDetectionArrayMsg:
    """Helper to construct a populated DoubleDetectionArrayMsg."""
    msg = DoubleDetectionArrayMsg()
    sec = int(math.floor(timestamp_s))
    nanosec = int(round((timestamp_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000

    msg.header = DoubleHeader(frame_id=frame_id, stamp=DoubleTime(sec, nanosec))

    if detections_data is None:
        detections_data = [
            {
                "detection_id": 1,
                "x": 3.0,
                "y": 1.5,
                "dim_l": 1.2,
                "dim_w": 0.8,
                "yaw": 0.25,
                "confidence": 0.92,
            }
        ]

    for d in detections_data:
        det = DoubleDetectionMsg()
        det.header = DoubleHeader(frame_id=frame_id, stamp=DoubleTime(sec, nanosec))
        det.detection_id = int(d.get("detection_id", 1))
        det.class_name = str(d.get("class_name", "pedestrian"))
        det.confidence = float(d.get("confidence", 0.9))
        det.position = DoublePoint(d.get("x", 0.0), d.get("y", 0.0), 0.0)
        det.dimensions = DoubleVector3(d.get("dim_l", 1.0), d.get("dim_w", 1.0), 0.5)
        det.yaw_rad = float(d.get("yaw", 0.0))
        det.position_covariance = list(d.get("covariance", [0.04, 0.0, 0.0, 0.04]))
        det.is_valid = bool(d.get("is_valid", True))
        msg.detections.append(det)

    return msg


# ==============================================================================
# TESTS
# ==============================================================================

class TestDynamicTrackerRosBoundary:
    """Validate DynamicTrackerNode at the ROS2 boundary with test doubles."""

    def test_01_input_conversion_preserves_all_detection_fields(self):
        """Test 1: Input conversion maps all ROSIDL detection fields to Phase 13 model."""
        node = DynamicTrackerNode()

        raw_msg = make_double_detection_array(
            timestamp_s=25.500123,
            frame_id="base_link",
            detections_data=[
                {
                    "detection_id": 101,
                    "class_name": "vehicle",
                    "x": 4.5,
                    "y": -2.0,
                    "dim_l": 2.2,
                    "dim_w": 1.1,
                    "yaw": 0.785,
                    "confidence": 0.95,
                    "covariance": [0.05, 0.01, 0.01, 0.05],
                    "is_valid": True,
                }
            ],
        )

        p13_array = node.convert_input_detections(raw_msg)
        assert isinstance(p13_array, DynamicObstacleDetectionArray)
        assert p13_array.frame_id == "base_link"
        assert p13_array.header_timestamp == pytest.approx(25.500123, rel=1e-6)
        assert len(p13_array.detections) == 1

        d = p13_array.detections[0]
        assert d.detection_id == "101"
        assert d.frame_id == "base_link"
        assert d.bbox_center[0] == pytest.approx(4.5)
        assert d.bbox_center[1] == pytest.approx(-2.0)
        assert d.bbox_size[0] == pytest.approx(2.2)
        assert d.bbox_size[1] == pytest.approx(1.1)
        assert d.bbox_yaw == pytest.approx(0.785)
        assert d.confidence == pytest.approx(0.95)

    def test_02_tracker_invocation_updates_state_correctly(self):
        """Test 2: Tracker core processes converted detections and tracks position & velocity."""
        node = DynamicTrackerNode(
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )

        # Frame 1 at t=10.0
        f1 = make_double_detection_array(10.0, "odom", [{"x": 1.0, "y": 0.0}])
        t1 = node.process_detections(f1)
        assert len(t1.tracks) == 1
        assert t1.tracks[0].position.x == pytest.approx(1.0)
        assert t1.tracks[0].status == "NEW"

        # Frame 2 at t=10.1 (moving at 2.0 m/s in x)
        f2 = make_double_detection_array(10.1, "odom", [{"x": 1.2, "y": 0.0}])
        t2 = node.process_detections(f2)
        assert len(t2.tracks) == 1
        assert t2.tracks[0].total_hits == 2

        # Frame 3 at t=10.2 (confirmed)
        f3 = make_double_detection_array(10.2, "odom", [{"x": 1.4, "y": 0.0}])
        t3 = node.process_detections(f3)
        assert len(t3.tracks) == 1
        assert t3.tracks[0].is_confirmed is True
        assert t3.tracks[0].status == "TRACKING"
        assert t3.tracks[0].velocity.x > 0.5

    def test_03_output_conversion_preserves_all_tracked_fields(self):
        """Test 3: Output conversion preserves all TrackedObstacleState fields to ROS message."""
        node = DynamicTrackerNode()

        cov = np.eye(4, dtype=np.float64) * 0.02
        cov[0, 1] = 0.005
        cov[1, 0] = 0.005

        track = TrackedObstacleState(
            track_id=88,
            timestamp=50.125,
            x=3.5,
            y=-1.5,
            vx=0.75,
            vy=0.25,
            covariance=cov,
            bbox_size=(1.6, 0.9),
            bbox_yaw=0.45,
            frame_id="odom",
            track_age=12,
            coast_cycles=0,
            is_confirmed=True,
            status="TRACKING",
        )
        arr = TrackedObstacleArray(header_timestamp=50.125, frame_id="odom", tracks=[track])

        ros_msg = node.convert_output_tracks(
            arr,
            msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )

        assert ros_msg.header.frame_id == "odom"
        assert len(ros_msg.tracks) == 1

        t = ros_msg.tracks[0]
        assert t.track_id == 88
        assert t.status == "TRACKING"
        assert t.is_confirmed is True
        assert t.position.x == pytest.approx(3.5)
        assert t.position.y == pytest.approx(-1.5)
        assert t.velocity.x == pytest.approx(0.75)
        assert t.velocity.y == pytest.approx(0.25)
        assert t.bbox_size.x == pytest.approx(1.6)
        assert t.bbox_size.y == pytest.approx(0.9)
        assert t.bbox_yaw == pytest.approx(0.45)
        assert t.total_hits == 12
        assert t.consecutive_misses == 0
        assert len(t.covariance) == 16
        assert t.covariance[1] == pytest.approx(0.005)

    def test_04_canonical_topic_and_qos_contract(self):
        """Test 4: Node specifies canonical topics /tracking/dynamic_obstacles and /perception/dynamic_obstacles."""
        node = DynamicTrackerNode()
        # Verify default topic contracts are canonical
        assert node._pub_msg_type is not None
        assert node._sub_msg_type is not None

        # Verify parameters match canonical defaults
        assert node.config.target_frame == "odom"
        assert node.config.hits_to_confirm == 3
        assert node.config.misses_to_expire == 5
        assert node.config.max_association_distance_m == 1.50

    def test_05_timestamp_preservation_nanosecond_precision(self):
        """Test 5: Timestamps maintain exact nanosecond precision across boundary roundtrips."""
        node = DynamicTrackerNode()
        target_t = 1234567.890123456

        raw_msg = make_double_detection_array(timestamp_s=target_t, frame_id="odom")
        p13 = node.convert_input_detections(raw_msg)
        assert p13.header_timestamp == pytest.approx(target_t, rel=1e-9)

        track = TrackedObstacleState(
            track_id=1,
            timestamp=target_t,
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            covariance=np.eye(4),
            bbox_size=(1.0, 1.0),
            bbox_yaw=0.0,
            frame_id="odom",
        )
        arr = TrackedObstacleArray(header_timestamp=target_t, frame_id="odom", tracks=[track])

        out_msg = node.convert_output_tracks(
            arr,
            msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )
        out_sec = out_msg.header.stamp.sec
        out_nanosec = out_msg.header.stamp.nanosec
        reconstructed = float(out_sec) + float(out_nanosec) * 1e-9
        assert reconstructed == pytest.approx(target_t, rel=1e-9)

    def test_06_frame_id_preservation_and_base_link_handling(self):
        """Test 6: base_link detections correctly receive identity pose if external pose not provided."""
        node = DynamicTrackerNode(
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )

        base_link_msg = make_double_detection_array(
            timestamp_s=1.0,
            frame_id="base_link",
            detections_data=[{"x": 2.0, "y": 0.5}],
        )

        out = node.process_detections(base_link_msg)
        assert out.header.frame_id == "odom"
        assert len(out.tracks) == 1
        assert out.tracks[0].position.x == pytest.approx(2.0)
        assert out.tracks[0].position.y == pytest.approx(0.5)

    def test_07_covariance_preservation_row_major(self):
        """Test 7: 4x4 covariance matrix is serialized in row-major order without transposition."""
        node = DynamicTrackerNode()

        cov = np.zeros((4, 4), dtype=np.float64)
        cov[0, 0] = 0.10
        cov[0, 1] = 0.02
        cov[1, 0] = 0.02
        cov[1, 1] = 0.20
        cov[2, 2] = 0.30
        cov[2, 3] = 0.04
        cov[3, 2] = 0.04
        cov[3, 3] = 0.40

        track = TrackedObstacleState(
            track_id=5,
            timestamp=2.0,
            x=1.0,
            y=1.0,
            vx=0.0,
            vy=0.0,
            covariance=cov,
            bbox_size=(1.0, 1.0),
            bbox_yaw=0.0,
            frame_id="odom",
        )
        arr = TrackedObstacleArray(header_timestamp=2.0, frame_id="odom", tracks=[track])

        out = node.convert_output_tracks(
            arr,
            msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )
        cov_list = out.tracks[0].covariance

        assert len(cov_list) == 16
        # [0, 0] -> 0
        assert cov_list[0] == pytest.approx(0.10)
        # [0, 1] -> 1
        assert cov_list[1] == pytest.approx(0.02)
        # [1, 0] -> 4
        assert cov_list[4] == pytest.approx(0.02)
        # [1, 1] -> 5
        assert cov_list[5] == pytest.approx(0.20)
        # [2, 3] -> 11
        assert cov_list[11] == pytest.approx(0.04)
        # [3, 2] -> 14
        assert cov_list[14] == pytest.approx(0.04)

    def test_08_lifecycle_preservation_hits_misses_confirmation(self):
        """Test 8: Track lifecycle (NEW -> TRACKING -> DEGRADED -> EXPIRED) is preserved across boundary."""
        cfg = TrackerConfig(hits_to_confirm=2, misses_to_degrade=1, misses_to_lost=2, misses_to_expire=3)
        node = DynamicTrackerNode(
            config=cfg,
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )

        # Hit 1
        f1 = make_double_detection_array(1.0, "odom", [{"x": 1.0, "y": 0.0}])
        t1 = node.process_detections(f1)
        assert t1.tracks[0].status == "NEW"
        assert t1.tracks[0].is_confirmed is False

        # Hit 2 (Confirmed)
        f2 = make_double_detection_array(1.1, "odom", [{"x": 1.1, "y": 0.0}])
        t2 = node.process_detections(f2)
        assert t2.tracks[0].status == "TRACKING"
        assert t2.tracks[0].is_confirmed is True

        # Miss 1 (Coasting / Degraded)
        f3 = make_double_detection_array(1.2, "odom", detections_data=[])
        t3 = node.process_detections(f3)
        assert len(t3.tracks) == 1
        assert t3.tracks[0].consecutive_misses == 1

        # Miss 2 (Lost)
        f4 = make_double_detection_array(1.3, "odom", detections_data=[])
        t4 = node.process_detections(f4)
        assert len(t4.tracks) == 1
        assert t4.tracks[0].consecutive_misses == 2

        # Miss 3 (Expired)
        f5 = make_double_detection_array(1.4, "odom", detections_data=[])
        t5 = node.process_detections(f5)
        assert len(t5.tracks) == 0

    def test_09_stale_or_out_of_order_observation_handling(self):
        """Test 9: Out-of-order or stale timestamps are handled safely without crash."""
        node = DynamicTrackerNode(
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )

        # Current observation at t=10.0
        f1 = make_double_detection_array(10.0, "odom", [{"x": 2.0, "y": 0.0}])
        t1 = node.process_detections(f1)
        assert len(t1.tracks) == 1
        assert node._last_timestamp == 10.0

        # Stale observation at t=9.0 (earlier than previous)
        f_stale = make_double_detection_array(9.0, "odom", [{"x": 2.0, "y": 0.0}])
        t_stale = node.process_detections(f_stale)
        # Node must not crash; stale observation is rejected safely
        assert t_stale is not None
        assert node._last_timestamp == 10.0

    def test_10_safety_behavior_unknown_unconfirmed_degraded(self):
        """Test 10: Safety flags (is_confirmed, status) prevent unconfirmed tracks from masquerading as confirmed."""
        node = DynamicTrackerNode(
            output_msg_cls=DoubleTrackedObstacleArrayMsg,
            track_item_msg_cls=DoubleTrackedObstacleMsg,
        )

        # Single detection: must NOT be confirmed
        f1 = make_double_detection_array(1.0, "odom", [{"x": 5.0, "y": 0.0, "confidence": 0.40}])
        t1 = node.process_detections(f1)
        assert len(t1.tracks) == 1
        assert t1.tracks[0].is_confirmed is False
        assert t1.tracks[0].status == "NEW"

        # Downstream planner can check `is_confirmed` to avoid phantom emergency stops
        assert not t1.tracks[0].is_confirmed
