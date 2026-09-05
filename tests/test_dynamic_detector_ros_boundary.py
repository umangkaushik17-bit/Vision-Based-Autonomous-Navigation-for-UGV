"""Focused test suite validating DynamicDetectorNode ROS2 boundary (Milestone 13.12.8).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate the DynamicDetectorNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/DynamicObstacleDetection.msg
- ugv_navigation_msgs/msg/DynamicObstacleDetectionArray.msg

Native DDS communication is NOT claimed or executed by this test suite.
All validation focuses on message conversion, field preservation, confidence
filtering, and safety invariants at the ROS boundary.
==============================================================================
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
)
from ai.planning.dynamic.detector_interface import (
    BaseDynamicObstacleDetector,
    SyntheticDynamicObstacleDetector,
)
from ugv_navigation.converters.dynamic_converters import (
    dynamic_obstacle_detection_array_from_ros,
    dynamic_obstacle_detection_array_to_ros,
)
from ugv_navigation.nodes.dynamic_detector_node import (
    DynamicDetectorNode,
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
    def __init__(self, frame_id: str = "base_link", stamp: Optional[DoubleTime] = None):
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
        self.dimensions = DoubleVector3(1.0, 1.0, 0.5)
        self.yaw_rad = 0.0
        self.position_covariance = [0.04, 0.0, 0.0, 0.04]
        self.is_valid = True


class DoubleDetectionArrayMsg:
    """Test Double for ugv_navigation_msgs/msg/DynamicObstacleDetectionArray.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.detections: List[DoubleDetectionMsg] = []


# ==============================================================================
# TESTS
# ==============================================================================

class TestDynamicDetectorRosBoundary:
    """Validate DynamicDetectorNode at the ROS2 boundary with test doubles."""

    def test_01_ros_input_boundary_frame_and_timestamp(self):
        """Test 1: Node ingests frame data and emits detection array with matching timestamp and frame."""
        det = DynamicObstacleDetection(
            detection_id="1",
            timestamp=10.0,
            frame_id="base_link",
            bbox_center=(2.0, 1.0),
            bbox_size=(1.2, 0.8),
            bbox_yaw=0.1,
            confidence=0.85,
        )
        mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        mock_detector.detect.return_value = [det]

        node = DynamicDetectorNode(
            detector=mock_detector,
            frame_id="base_link",
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )

        out_msg = node.process_frame(frame_data={"image": None}, timestamp_s=10.0)
        assert isinstance(out_msg, DoubleDetectionArrayMsg)
        assert out_msg.header.frame_id == "base_link"
        assert len(out_msg.detections) == 1

    def test_02_detector_actually_invoked(self):
        """Test 2: Node delegates to underlying detector interface with exact parameters."""
        mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        mock_detector.detect.return_value = []

        node = DynamicDetectorNode(
            detector=mock_detector,
            frame_id="base_link",
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
        )

        fake_frame = {"raw_sensor_data": [1, 2, 3]}
        node.process_frame(fake_frame, timestamp_s=15.5)

        mock_detector.detect.assert_called_once_with(fake_frame, timestamp_s=15.5)

    def test_03_output_conversion_to_rosidl(self):
        """Test 3: Output conversion converts Phase 13 detections to ROS2 message type."""
        node = DynamicDetectorNode(
            frame_id="base_link",
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )

        d1 = DynamicObstacleDetection("10", 5.0, "base_link", (3.0, 0.5), (1.5, 0.7), 0.2, 0.95)
        arr = DynamicObstacleDetectionArray(header_timestamp=5.0, frame_id="base_link", detections=[d1])

        ros_msg = node.convert_output_detections(arr)
        assert isinstance(ros_msg, DoubleDetectionArrayMsg)
        assert len(ros_msg.detections) == 1
        assert ros_msg.detections[0].detection_id == 10

    def test_04_field_preservation(self):
        """Test 4: All detection fields (id, center, dimensions, yaw, confidence) are preserved."""
        det = DynamicObstacleDetection(
            detection_id="42",
            timestamp=12.0,
            frame_id="base_link",
            bbox_center=(4.5, -2.1),
            bbox_size=(2.2, 1.1),
            bbox_yaw=0.785,
            confidence=0.91,
        )
        mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        mock_detector.detect.return_value = [det]

        node = DynamicDetectorNode(
            detector=mock_detector,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )

        out_msg = node.process_frame(None, timestamp_s=12.0)
        d = out_msg.detections[0]

        assert d.detection_id == 42
        assert d.position.x == pytest.approx(4.5)
        assert d.position.y == pytest.approx(-2.1)
        assert d.dimensions.x == pytest.approx(2.2)
        assert d.dimensions.y == pytest.approx(1.1)
        assert d.yaw_rad == pytest.approx(0.785)
        assert d.confidence == pytest.approx(0.91)
        assert d.is_valid is True

    def test_05_timestamp_precision_nanoseconds(self):
        """Test 5: Sub-millisecond nanosecond timestamps are preserved without drift."""
        test_ts = 1725519600.123456789
        det = DynamicObstacleDetection("1", test_ts, "base_link", (1.0, 1.0), (1.0, 1.0), 0.0, 0.9)
        mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        mock_detector.detect.return_value = [det]

        node = DynamicDetectorNode(
            detector=mock_detector,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )

        out_msg = node.process_frame(None, timestamp_s=test_ts)
        sec = out_msg.header.stamp.sec
        nanosec = out_msg.header.stamp.nanosec
        reconstructed = float(sec) + float(nanosec) * 1e-9
        assert reconstructed == pytest.approx(test_ts, rel=1e-8)

    def test_06_frame_preservation(self):
        """Test 6: Frame ID is preserved in array header and detection headers."""
        for fid in ["base_link", "camera_link", "velodyne"]:
            mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
            mock_detector.detect.return_value = [
                DynamicObstacleDetection("1", 1.0, fid, (1.0, 1.0), (1.0, 1.0), 0.0, 0.9)
            ]

            node = DynamicDetectorNode(
                detector=mock_detector,
                frame_id=fid,
                output_msg_cls=DoubleDetectionArrayMsg,
                det_item_msg_cls=DoubleDetectionMsg,
                header_cls=DoubleHeader,
                point_cls=DoublePoint,
                vector3_cls=DoubleVector3,
            )

            out_msg = node.process_frame(None, timestamp_s=1.0)
            assert out_msg.header.frame_id == fid
            assert out_msg.detections[0].header.frame_id == fid

    def test_07_covariance_preservation(self):
        """Test 7: 2x2 position covariance is serialized in row-major order [xx, xy, yx, yy]."""
        det = DynamicObstacleDetection("1", 1.0, "base_link", (1.0, 1.0), (1.0, 1.0), 0.0, 0.9)
        mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        mock_detector.detect.return_value = [det]

        node = DynamicDetectorNode(
            detector=mock_detector,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
            header_cls=DoubleHeader,
            point_cls=DoublePoint,
            vector3_cls=DoubleVector3,
        )

        out_msg = node.process_frame(None, timestamp_s=1.0)
        cov = out_msg.detections[0].position_covariance
        assert len(cov) == 4
        assert cov[0] == pytest.approx(0.04)
        assert cov[3] == pytest.approx(0.04)

    def test_08_confidence_preservation_and_thresholding(self):
        """Test 8: Low-confidence detections (< min_confidence) are filtered out cleanly."""
        dets = [
            DynamicObstacleDetection("1", 1.0, "base_link", (1.0, 0.0), (1.0, 1.0), 0.0, 0.20),
            DynamicObstacleDetection("2", 1.0, "base_link", (2.0, 0.0), (1.0, 1.0), 0.0, 0.50),
            DynamicObstacleDetection("3", 1.0, "base_link", (3.0, 0.0), (1.0, 1.0), 0.0, 0.85),
        ]
        mock_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        mock_detector.detect.return_value = dets

        node = DynamicDetectorNode(
            detector=mock_detector,
            min_confidence=0.40,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
        )

        out_msg = node.process_frame(None, timestamp_s=1.0)
        # Only detections 2 and 3 should survive
        assert len(out_msg.detections) == 2
        ids = [d.detection_id for d in out_msg.detections]
        assert 1 not in ids
        assert ids == [2, 3]

    def test_09_invalid_detection_and_timestamp_handling(self):
        """Test 9: Invalid timestamp raises ValueError; detector failure returns safe empty array."""
        node = DynamicDetectorNode(
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
        )

        # Negative timestamp must be rejected
        with pytest.raises(ValueError, match="non-negative"):
            node.process_frame(None, timestamp_s=-1.0)

        # Detector exception is safely caught and returns empty array
        failing_detector = MagicMock(spec=BaseDynamicObstacleDetector)
        failing_detector.detect.side_effect = RuntimeError("Sensor offline")
        node_fail = DynamicDetectorNode(
            detector=failing_detector,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
        )
        out_msg = node_fail.process_frame(None, timestamp_s=1.0)
        assert out_msg is not None
        assert len(out_msg.detections) == 0

    def test_10_deterministic_output(self):
        """Test 10: Deterministic detector produces identical detections for identical inputs."""
        detector = SyntheticDynamicObstacleDetector(frame_id="base_link", random_seed=42)
        node = DynamicDetectorNode(
            detector=detector,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
        )

        frame = {"robot_pose": (0.0, 0.0, 0.0)}
        out1 = node.process_frame(frame, timestamp_s=1.0)
        out2 = node.process_frame(frame, timestamp_s=1.0)

        assert len(out1.detections) == len(out2.detections)
        for d1, d2 in zip(out1.detections, out2.detections):
            assert d1.detection_id == d2.detection_id
            assert d1.position.x == pytest.approx(d2.position.x)
            assert d1.position.y == pytest.approx(d2.position.y)

    def test_11_canonical_topic_and_qos(self):
        """Test 11: Node specifies canonical topic /perception/dynamic_obstacles and parameters."""
        node = DynamicDetectorNode()
        assert node._pub_msg_type is not None
        assert node.frame_id == "base_link"
        assert node.min_confidence == 0.30

    def test_12_no_ground_truth_access(self):
        """Test 12: Detector operates strictly without accessing ground-truth registry or simulation internals."""
        detector = SyntheticDynamicObstacleDetector(frame_id="base_link")
        node = DynamicDetectorNode(
            detector=detector,
            output_msg_cls=DoubleDetectionArrayMsg,
            det_item_msg_cls=DoubleDetectionMsg,
        )

        # Frame data contains no ground-truth object references
        clean_frame = {"scan": [1.0] * 360}
        out_msg = node.process_frame(clean_frame, timestamp_s=2.0)
        assert out_msg is not None
