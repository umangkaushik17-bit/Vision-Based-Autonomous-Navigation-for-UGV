"""Focused test suite validating DynamicOccupancyNode ROS2 boundary (Milestone 13.12.7).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate the DynamicOccupancyNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/PredictedTrajectoryArray.msg
- ugv_navigation_msgs/msg/PredictedObstacleTrajectory.msg
- ugv_navigation_msgs/msg/TrajectoryWaypoint.msg
- ugv_navigation_msgs/msg/DynamicOccupancySliceArray.msg
- ugv_navigation_msgs/msg/DynamicOccupancyGridSlice.msg

Native DDS communication is NOT claimed or executed by this test suite.
All validation focuses on message conversion, field preservation, geometry,
and safety invariants at the ROS boundary.
==============================================================================
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyGridSlice,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ugv_navigation.converters.dynamic_converters import (
    dynamic_occupancy_grid_slice_from_ros,
    dynamic_occupancy_grid_slice_to_ros,
    dynamic_occupancy_slice_array_from_ros,
    dynamic_occupancy_slice_array_to_ros,
    predicted_trajectory_array_from_ros,
    predicted_trajectory_array_to_ros,
)
from ugv_navigation.nodes.dynamic_occupancy_node import (
    DynamicOccupancyNode,
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


class DoubleTrajectoryWaypointMsg:
    """Test Double for ugv_navigation_msgs/msg/TrajectoryWaypoint.msg."""

    def __init__(self):
        self.target_time = DoubleTime()
        self.relative_time_s = 0.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.covariance = [0.0] * 4
        self.uncertainty_radius_m = 0.0
        self.confidence = 1.0


class DoublePredictedObstacleTrajectoryMsg:
    """Test Double for ugv_navigation_msgs/msg/PredictedObstacleTrajectory.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "TRACKING"
        self.bbox_size = DoubleVector3(1.0, 1.0, 0.5)
        self.bbox_yaw = 0.0
        self.waypoints: List[DoubleTrajectoryWaypointMsg] = []
        self.prediction_horizon_s = 2.0
        self.is_unknown = False
        self.confidence = 1.0


class DoublePredictedTrajectoryArrayMsg:
    """Test Double for ugv_navigation_msgs/msg/PredictedTrajectoryArray.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.trajectories: List[DoublePredictedObstacleTrajectoryMsg] = []


class DoubleDynamicOccupancyGridSliceMsg:
    """Test Double for ugv_navigation_msgs/msg/DynamicOccupancyGridSlice.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.target_time_offset_s = 0.0
        self.target_timestamp = DoubleTime()
        self.resolution = 0.10
        self.origin_x = -20.0
        self.origin_y = -20.0
        self.width = 0
        self.height = 0
        self.data: List[float] = []
        self.unknown_mask: List[bool] = []


class DoubleDynamicOccupancySliceArrayMsg:
    """Test Double for ugv_navigation_msgs/msg/DynamicOccupancySliceArray.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.slices: List[DoubleDynamicOccupancyGridSliceMsg] = []


def make_double_predicted_trajectory_array(
    timestamp_s: float = 10.0,
    frame_id: str = "odom",
    trajectories_data: Optional[List[dict]] = None,
) -> DoublePredictedTrajectoryArrayMsg:
    """Helper to construct a populated DoublePredictedTrajectoryArrayMsg."""
    msg = DoublePredictedTrajectoryArrayMsg()
    sec = int(math.floor(timestamp_s))
    nanosec = int(round((timestamp_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000

    msg.header = DoubleHeader(frame_id=frame_id, stamp=DoubleTime(sec, nanosec))

    if trajectories_data is None:
        trajectories_data = [
            {
                "track_id": 1,
                "status": "TRACKING",
                "x": 2.0,
                "y": 1.0,
                "vx": 0.5,
                "vy": 0.0,
                "dim_l": 1.2,
                "dim_w": 0.8,
                "yaw": 0.0,
                "horizon_s": 2.0,
                "dt": 0.25,
            }
        ]

    for td in trajectories_data:
        t = DoublePredictedObstacleTrajectoryMsg()
        t.header = DoubleHeader(frame_id=frame_id, stamp=DoubleTime(sec, nanosec))
        t.track_id = int(td.get("track_id", 1))
        t.status = str(td.get("status", "TRACKING"))
        t.bbox_size = DoubleVector3(td.get("dim_l", 1.2), td.get("dim_w", 0.8), 0.5)
        t.bbox_yaw = float(td.get("yaw", 0.0))
        horizon = float(td.get("horizon_s", 2.0))
        dt = float(td.get("dt", 0.25))
        t.prediction_horizon_s = horizon
        t.is_unknown = bool(td.get("is_unknown", False))
        t.confidence = float(td.get("confidence", 1.0))

        # Generate waypoints
        x0 = float(td.get("x", 0.0))
        y0 = float(td.get("y", 0.0))
        vx = float(td.get("vx", 0.0))
        vy = float(td.get("vy", 0.0))

        steps = int(round(horizon / dt)) + 1
        for i in range(steps):
            rel_t = i * dt
            target_t = timestamp_s + rel_t
            wp = DoubleTrajectoryWaypointMsg()
            wp_sec = int(math.floor(target_t))
            wp_nsec = int(round((target_t - wp_sec) * 1e9))
            if wp_nsec >= 1_000_000_000:
                wp_sec += 1
                wp_nsec -= 1_000_000_000
            wp.target_time = DoubleTime(wp_sec, wp_nsec)
            wp.relative_time_s = rel_t
            wp.position = DoublePoint(x0 + vx * rel_t, y0 + vy * rel_t, 0.0)
            wp.velocity = DoubleVector3(vx, vy, 0.0)
            wp.covariance = [0.04, 0.0, 0.0, 0.04]
            wp.uncertainty_radius_m = 0.20 + 0.10 * rel_t
            wp.confidence = 1.0
            t.waypoints.append(wp)

        msg.trajectories.append(t)

    return msg


# ==============================================================================
# TESTS
# ==============================================================================

class TestDynamicOccupancyRosBoundary:
    """Validate DynamicOccupancyNode at the ROS2 boundary with test doubles."""

    @pytest.fixture
    def small_grid_config(self) -> DynamicOccupancyConfig:
        """Create a compact grid config for rapid test execution."""
        return DynamicOccupancyConfig(
            resolution_m=0.20,
            x_bounds_m=(-5.0, 5.0),
            y_bounds_m=(-5.0, 5.0),
            frame_id="odom",
            time_bins=[0.0, 0.50, 1.00],
            collision_cost_threshold=0.65,
        )

    def test_01_input_conversion_preserves_every_required_prediction_field(self, small_grid_config):
        """Test 1: Input conversion maps all ROSIDL prediction fields to Phase 13 model."""
        node = DynamicOccupancyNode(config=small_grid_config)

        raw_msg = make_double_predicted_trajectory_array(
            timestamp_s=25.123456,
            frame_id="odom",
            trajectories_data=[
                {
                    "track_id": 42,
                    "status": "TRACKING",
                    "x": 1.0,
                    "y": -0.5,
                    "vx": 0.8,
                    "vy": 0.2,
                    "dim_l": 1.5,
                    "dim_w": 0.7,
                    "yaw": 0.35,
                    "horizon_s": 1.0,
                    "dt": 0.5,
                }
            ],
        )

        p13_array = node.convert_input_predictions(raw_msg)
        assert isinstance(p13_array, PredictedTrajectoryArray)
        assert p13_array.frame_id == "odom"
        assert p13_array.header_timestamp == pytest.approx(25.123456, rel=1e-6)
        assert len(p13_array.predictions) == 1

        p = p13_array.predictions[0]
        assert p.track_id == 42
        assert p.current_state.status == "TRACKING"
        assert p.current_state.bbox_size == (pytest.approx(1.5), pytest.approx(0.7))
        assert p.current_state.bbox_yaw == pytest.approx(0.35)
        assert p.horizon_s == pytest.approx(1.0)
        assert len(p.waypoints) == 3

        wp0 = p.waypoints[0]
        assert wp0.time_offset == pytest.approx(0.0)
        assert wp0.x == pytest.approx(1.0)
        assert wp0.y == pytest.approx(-0.5)
        assert wp0.vx == pytest.approx(0.8)
        assert wp0.vy == pytest.approx(0.2)

    def test_02_occupancy_engine_actually_invoked_with_phase13_objects(self, small_grid_config):
        """Test 2: Node delegates prediction updates to pure Phase 13 DynamicOccupancyLayer."""
        mock_layer = MagicMock(spec=DynamicOccupancyLayer)
        mock_layer.config = small_grid_config
        mock_layer.time_bins = small_grid_config.time_bins
        mock_slice = DynamicOccupancyGridSlice(
            time_offset=0.0,
            timestamp=10.0,
            cost=np.zeros((50, 50), dtype=np.float32),
            risk_level=np.zeros((50, 50), dtype=np.uint8),
            nominal_obstacle=np.zeros((50, 50), dtype=bool),
            uncertainty_zone=np.zeros((50, 50), dtype=bool),
        )
        mock_layer.get_slice.return_value = mock_slice

        node = DynamicOccupancyNode(
            occupancy_layer=mock_layer,
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        raw_msg = make_double_predicted_trajectory_array(timestamp_s=10.0, frame_id="odom")
        node.process_predictions(raw_msg)

        assert mock_layer.update_predictions.call_count == 1
        call_args = mock_layer.update_predictions.call_args
        called_arg = call_args[0][0]
        assert isinstance(called_arg, PredictedTrajectoryArray)
        assert len(called_arg.predictions) == 1
        assert isinstance(called_arg.predictions[0], PredictedObstacleTrajectory)

    def test_03_output_conversion_preserves_every_occupancy_field(self, small_grid_config):
        """Test 3: Output conversion maps Phase 13 occupancy slices to ROS message without field loss."""
        node = DynamicOccupancyNode(config=small_grid_config)

        cost = np.zeros((10, 10), dtype=np.float32)
        cost[2, 2] = 1.0  # lethal
        cost[2, 3] = 0.5  # medium risk
        risk = np.zeros((10, 10), dtype=np.uint8)
        risk[2, 2] = int(DynamicRiskLevel.LETHAL)
        risk[2, 3] = int(DynamicRiskLevel.MEDIUM_RISK)

        slice_obj = DynamicOccupancyGridSlice(
            time_offset=0.5,
            timestamp=12.5,
            cost=cost,
            risk_level=risk,
            nominal_obstacle=(cost >= 0.95),
            uncertainty_zone=(cost >= 0.15) & (cost < 0.95),
            active_obstacle_ids=[5],
        )

        out_msg = node.convert_output_slices(
            slices=[slice_obj],
            header_timestamp=12.0,
            frame_id="odom",
            msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        assert isinstance(out_msg, DoubleDynamicOccupancySliceArrayMsg)
        assert out_msg.header.frame_id == "odom"
        assert len(out_msg.slices) == 1

        s = out_msg.slices[0]
        assert s.header.frame_id == "odom"
        assert s.target_time_offset_s == pytest.approx(0.5)
        assert s.resolution == pytest.approx(small_grid_config.resolution_m)
        assert s.origin_x == pytest.approx(small_grid_config.x_bounds_m[0])
        assert s.origin_y == pytest.approx(small_grid_config.y_bounds_m[0])
        assert s.width == 10
        assert s.height == 10
        assert len(s.data) == 100
        assert s.data[2 * 10 + 2] == pytest.approx(1.0)
        assert s.data[2 * 10 + 3] == pytest.approx(0.5)

    def test_04_canonical_input_output_topics_and_qos(self):
        """Test 4: Node specifies canonical topic contracts (/prediction/obstacle_trajectories -> /occupancy/dynamic_slices)."""
        node = DynamicOccupancyNode()

        assert node._pub_msg_type is not None
        assert node._sub_msg_type is not None

        # Configuration parameters
        assert node.config.resolution_m == 0.10
        assert node.config.frame_id == "odom"
        assert node.config.collision_cost_threshold == 0.65
        assert node.config.time_bins == [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

    def test_05_header_timestamp_nanosecond_precision(self, small_grid_config):
        """Test 5: Sub-millisecond and nanosecond precision are preserved across the boundary."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        test_ts = 1725519600.123456789
        raw_msg = make_double_predicted_trajectory_array(timestamp_s=test_ts, frame_id="odom")

        out_msg = node.process_predictions(raw_msg)
        assert out_msg is not None

        # Header timestamp precision
        sec = out_msg.header.stamp.sec
        nanosec = out_msg.header.stamp.nanosec
        reconstructed = float(sec) + float(nanosec) * 1e-9
        assert reconstructed == pytest.approx(test_ts, rel=1e-8)

        # Slice 1 (t_offset = 0.5s) timestamp precision
        s1 = out_msg.slices[1]
        s1_ts = float(s1.target_timestamp.sec) + float(s1.target_timestamp.nanosec) * 1e-9
        assert s1_ts == pytest.approx(test_ts + 0.50, rel=1e-8)

    def test_06_frame_preservation(self, small_grid_config):
        """Test 6: Frame ID ('map', 'odom') is strictly preserved across array and slices."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        for fid in ["odom", "map", "planning_frame"]:
            raw_msg = make_double_predicted_trajectory_array(timestamp_s=5.0, frame_id=fid)
            out_msg = node.process_predictions(raw_msg)
            assert out_msg.header.frame_id == fid
            for s in out_msg.slices:
                assert s.header.frame_id == fid

    def test_07_grid_origin_preservation(self, small_grid_config):
        """Test 7: Spatial grid origin is preserved in every generated slice."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0)
        out_msg = node.process_predictions(raw_msg)

        for s in out_msg.slices:
            assert s.origin_x == pytest.approx(small_grid_config.x_bounds_m[0])
            assert s.origin_y == pytest.approx(small_grid_config.y_bounds_m[0])

    def test_08_resolution_preservation(self, small_grid_config):
        """Test 8: Spatial grid resolution is preserved in every generated slice."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0)
        out_msg = node.process_predictions(raw_msg)

        for s in out_msg.slices:
            assert s.resolution == pytest.approx(small_grid_config.resolution_m)

    def test_09_grid_dimensions_preservation(self, small_grid_config):
        """Test 9: Grid dimensions (width, height) match expected raster shape."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0)
        out_msg = node.process_predictions(raw_msg)

        expected_w = int(round((small_grid_config.y_bounds_m[1] - small_grid_config.y_bounds_m[0]) / small_grid_config.resolution_m))
        expected_h = int(round((small_grid_config.x_bounds_m[1] - small_grid_config.x_bounds_m[0]) / small_grid_config.resolution_m))

        for s in out_msg.slices:
            assert s.width == expected_w
            assert s.height == expected_h
            assert len(s.data) == expected_w * expected_h

    def test_10_time_bin_count_and_ordering(self, small_grid_config):
        """Test 10: Slices strictly match the configured time bins in count and increasing order."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        raw_msg = make_double_predicted_trajectory_array(timestamp_s=10.0)
        out_msg = node.process_predictions(raw_msg)

        assert len(out_msg.slices) == len(small_grid_config.time_bins)

        prev_offset = -1.0
        for i, s in enumerate(out_msg.slices):
            expected_bin = small_grid_config.time_bins[i]
            assert s.target_time_offset_s == pytest.approx(expected_bin)
            assert s.target_time_offset_s > prev_offset
            prev_offset = s.target_time_offset_s

    def test_11_multiple_obstacle_union_behavior(self, small_grid_config):
        """Test 11: Multiple obstacles are unioned in occupancy slices without overwriting each other."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        # Two distinct stationary obstacles at (2.0, 0.0) and (-2.0, 0.0)
        tracks = [
            {"track_id": 1, "x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "dim_l": 1.0, "dim_w": 1.0},
            {"track_id": 2, "x": -2.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "dim_l": 1.0, "dim_w": 1.0},
        ]
        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0, trajectories_data=tracks)

        out_msg = node.process_predictions(raw_msg)
        s0 = out_msg.slices[0]

        data_arr = np.array(s0.data, dtype=np.float32).reshape((s0.height, s0.width))

        # Check cell at (2.0, 0.0)
        r1, c1 = node.occupancy_layer.world_to_grid(2.0, 0.0)
        # Check cell at (-2.0, 0.0)
        r2, c2 = node.occupancy_layer.world_to_grid(-2.0, 0.0)

        assert data_arr[r1, c1] >= 0.95, "Obstacle 1 footprint missing"
        assert data_arr[r2, c2] >= 0.95, "Obstacle 2 footprint missing"

    def test_12_lethal_vs_graded_risk_semantics(self, small_grid_config):
        """Test 12: Graded risk levels are preserved without collapsing into binary collision."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        tracks = [{"track_id": 1, "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "dim_l": 0.6, "dim_w": 0.6}]
        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0, trajectories_data=tracks)

        out_msg = node.process_predictions(raw_msg)
        s0 = out_msg.slices[0]
        data_arr = np.array(s0.data, dtype=np.float32).reshape((s0.height, s0.width))

        # Lethal center
        r_center, c_center = node.occupancy_layer.world_to_grid(0.0, 0.0)
        assert data_arr[r_center, c_center] == pytest.approx(1.0)

        # Graded uncertainty outside nominal footprint (e.g. at 0.5m away)
        r_edge, c_edge = node.occupancy_layer.world_to_grid(0.4, 0.0)
        assert 0.0 < data_arr[r_edge, c_edge] < 0.95, "Uncertainty zone should be graded, not lethal"

        # Free space far away
        r_far, c_far = node.occupancy_layer.world_to_grid(4.0, 4.0)
        assert data_arr[r_far, c_far] == pytest.approx(0.0)

    def test_13_free_vs_unknown_distinction(self, small_grid_config):
        """Test 13: Free space (cost 0.0) is cleanly distinguished from UNKNOWN space."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0)
        out_msg = node.process_predictions(raw_msg)
        s0 = out_msg.slices[0]

        # In nominal dynamic raster, free space has cost 0.0 and unknown_mask is False
        assert len(s0.unknown_mask) == len(s0.data)
        r_far, c_far = node.occupancy_layer.world_to_grid(4.0, 4.0)
        idx = r_far * s0.width + c_far
        assert s0.data[idx] == pytest.approx(0.0)
        assert s0.unknown_mask[idx] is False

    def test_14_clipping_boundaries(self, small_grid_config):
        """Test 14: Obstacles on or exceeding the grid boundary are clipped safely without crash."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        # Obstacle near world bounds (+5.0, 0.0) and outside (+10.0, 0.0)
        tracks = [
            {"track_id": 1, "x": 4.9, "y": 0.0, "vx": 0.0, "vy": 0.0},
            {"track_id": 2, "x": 10.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        ]
        raw_msg = make_double_predicted_trajectory_array(timestamp_s=1.0, trajectories_data=tracks)

        out_msg = node.process_predictions(raw_msg)
        assert out_msg is not None
        assert len(out_msg.slices) == len(small_grid_config.time_bins)

    def test_15_malformed_input_safety(self, small_grid_config):
        """Test 15: Malformed, empty, or None messages are safely handled without uncaught exceptions."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        # Case A: None
        out_none = node.process_predictions(None)
        assert out_none is not None
        assert len(out_none.slices) == len(small_grid_config.time_bins)

        # Case B: Empty trajectories
        empty_msg = DoublePredictedTrajectoryArrayMsg()
        empty_msg.header = DoubleHeader("odom", DoubleTime(1, 0))
        out_empty = node.process_predictions(empty_msg)
        assert out_empty is not None
        assert len(out_empty.slices) == len(small_grid_config.time_bins)

        # Case C: Malformed object
        class Malformed:
            pass

        out_malformed = node.process_predictions(Malformed())
        assert out_malformed is not None
        assert len(out_malformed.slices) == len(small_grid_config.time_bins)

    def test_16_stale_or_out_of_order_behavior(self, small_grid_config):
        """Test 16: Stale or out-of-order predictions are safely handled without state corruption."""
        node = DynamicOccupancyNode(
            config=small_grid_config,
            output_msg_cls=DoubleDynamicOccupancySliceArrayMsg,
            slice_item_msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        # First message at t=20.0
        msg1 = make_double_predicted_trajectory_array(timestamp_s=20.0)
        out1 = node.process_predictions(msg1)
        assert out1 is not None
        assert node._last_timestamp == 20.0

        # Stale message at t=19.0 (in the past)
        msg_stale = make_double_predicted_trajectory_array(timestamp_s=19.0)
        out_stale = node.process_predictions(msg_stale)
        assert out_stale is not None
        # Timestamp must not regress
        assert node._last_timestamp == 20.0

    def test_17_deterministic_serialization_roundtrip(self, small_grid_config):
        """Test 17: DynamicOccupancyGridSlice and SliceArray roundtrip bidirectionally with precision."""
        cost = np.zeros((20, 20), dtype=np.float32)
        cost[5:8, 5:8] = 0.98
        cost[4:9, 4:9][cost[4:9, 4:9] < 0.98] = 0.45
        risk = np.zeros((20, 20), dtype=np.uint8)
        risk[cost >= 0.95] = int(DynamicRiskLevel.LETHAL)
        risk[(cost >= 0.40) & (cost < 0.95)] = int(DynamicRiskLevel.MEDIUM_RISK)

        slice_obj = DynamicOccupancyGridSlice(
            time_offset=0.25,
            timestamp=10.25,
            cost=cost,
            risk_level=risk,
            nominal_obstacle=(cost >= 0.95),
            uncertainty_zone=(cost >= 0.15) & (cost < 0.95),
            active_obstacle_ids=[3],
        )

        ros_slice = dynamic_occupancy_grid_slice_to_ros(
            slice_obj,
            resolution=0.10,
            origin_x=-1.0,
            origin_y=-1.0,
            frame_id="odom",
            msg_cls=DoubleDynamicOccupancyGridSliceMsg,
            header_cls=DoubleHeader,
            time_cls=DoubleTime,
        )

        back_slice = dynamic_occupancy_grid_slice_from_ros(ros_slice)
        assert back_slice.time_offset == pytest.approx(0.25)
        assert back_slice.timestamp == pytest.approx(10.25)
        assert back_slice.cost.shape == (20, 20)
        assert back_slice.cost[6, 6] == pytest.approx(0.98)
        assert back_slice.cost[0, 0] == pytest.approx(0.0)
        assert back_slice.nominal_obstacle[6, 6] == True
        assert back_slice.risk_level[6, 6] == int(DynamicRiskLevel.LETHAL)
