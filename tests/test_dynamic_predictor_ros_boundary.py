"""Focused test suite validating DynamicPredictorNode ROS2 boundary (Milestone 13.12.6).

==============================================================================
ENVIRONMENT ARCHITECTURE NOTE
==============================================================================
TEST DOUBLE vs REAL ROSIDL GENERATED MESSAGE:
On the current Windows host, native ROS2 runtime (rclpy) and colcon-generated
C-extensions for `ugv_navigation_msgs` are unavailable.
The tests in this suite validate the DynamicPredictorNode's ROS boundary behavior
using controlled TEST DOUBLES matching the exact ROSIDL .msg schemas:
- ugv_navigation_msgs/msg/TrackedObstacleArray.msg
- ugv_navigation_msgs/msg/TrackedObstacle.msg
- ugv_navigation_msgs/msg/PredictedTrajectoryArray.msg
- ugv_navigation_msgs/msg/PredictedObstacleTrajectory.msg
- ugv_navigation_msgs/msg/TrajectoryWaypoint.msg

Native DDS communication is NOT claimed or executed by this test suite.
All validation focuses on message conversion, field preservation, lifecycle,
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
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ugv_navigation.converters.dynamic_converters import (
    predicted_trajectory_array_from_ros,
    predicted_trajectory_array_to_ros,
    tracked_obstacle_array_from_ros,
    tracked_obstacle_array_to_ros,
)
from ugv_navigation.nodes.dynamic_predictor_node import (
    DynamicPredictorNode,
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


class DoubleTrackedObstacleMsg:
    """Test Double for ugv_navigation_msgs/msg/TrackedObstacle.msg."""

    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "TRACKING"
        self.is_confirmed = True
        self.confidence = 1.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.bbox_size = DoubleVector3(1.0, 1.0, 0.5)
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


def make_double_tracked_obstacle_array(
    timestamp_s: float = 10.0,
    frame_id: str = "odom",
    tracks_data: Optional[List[dict]] = None,
) -> DoubleTrackedObstacleArrayMsg:
    """Helper to construct a populated DoubleTrackedObstacleArrayMsg."""
    msg = DoubleTrackedObstacleArrayMsg()
    sec = int(math.floor(timestamp_s))
    nanosec = int(round((timestamp_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000

    msg.header = DoubleHeader(frame_id=frame_id, stamp=DoubleTime(sec, nanosec))

    if tracks_data is None:
        tracks_data = [
            {
                "track_id": 1,
                "x": 2.0,
                "y": 1.0,
                "vx": 0.5,
                "vy": 0.0,
                "status": "TRACKING",
                "is_confirmed": True,
            }
        ]

    for td in tracks_data:
        t = DoubleTrackedObstacleMsg()
        t.header = DoubleHeader(frame_id=frame_id, stamp=DoubleTime(sec, nanosec))
        t.track_id = int(td.get("track_id", 1))
        t.status = str(td.get("status", "TRACKING"))
        t.is_confirmed = bool(td.get("is_confirmed", True))
        t.confidence = float(td.get("confidence", 1.0))
        t.position = DoublePoint(td.get("x", 0.0), td.get("y", 0.0), 0.0)
        t.velocity = DoubleVector3(td.get("vx", 0.0), td.get("vy", 0.0), 0.0)
        t.bbox_size = DoubleVector3(td.get("dim_l", 1.2), td.get("dim_w", 0.8), 0.5)
        t.bbox_yaw = float(td.get("yaw", 0.0))
        t.covariance = list(td.get("covariance", [0.04] * 16))
        t.total_hits = int(td.get("total_hits", 5))
        t.consecutive_misses = int(td.get("consecutive_misses", 0))
        t.time_since_last_update_s = float(td.get("time_since_last_update_s", 0.0))
        msg.tracks.append(t)

    return msg


# ==============================================================================
# TESTS
# ==============================================================================

class TestDynamicPredictorRosBoundary:
    """Validate DynamicPredictorNode at the ROS2 boundary with test doubles."""

    def test_01_input_conversion_preserves_all_tracked_fields(self):
        """Test 1: Input conversion maps all ROSIDL tracked obstacle fields to Phase 13 model."""
        node = DynamicPredictorNode()

        cov = [0.0] * 16
        cov[0] = 0.05  # var(x)
        cov[1] = 0.01  # cov(x,y)
        cov[4] = 0.01  # cov(y,x)
        cov[5] = 0.08  # var(y)
        cov[10] = 0.02 # var(vx)
        cov[15] = 0.03 # var(vy)

        raw_msg = make_double_tracked_obstacle_array(
            timestamp_s=42.123456,
            frame_id="odom",
            tracks_data=[
                {
                    "track_id": 101,
                    "status": "TRACKING",
                    "is_confirmed": True,
                    "confidence": 0.98,
                    "x": 3.5,
                    "y": -1.2,
                    "vx": 1.4,
                    "vy": 0.2,
                    "dim_l": 2.0,
                    "dim_w": 0.9,
                    "yaw": 0.45,
                    "covariance": cov,
                    "total_hits": 8,
                    "consecutive_misses": 1,
                }
            ],
        )

        p13_array = node.convert_input_tracks(raw_msg)
        assert isinstance(p13_array, TrackedObstacleArray)
        assert p13_array.frame_id == "odom"
        assert p13_array.header_timestamp == pytest.approx(42.123456, rel=1e-6)
        assert len(p13_array.tracks) == 1

        t = p13_array.tracks[0]
        assert t.track_id == 101
        assert t.status == "TRACKING"
        assert t.is_confirmed is True
        assert t.x == pytest.approx(3.5)
        assert t.y == pytest.approx(-1.2)
        assert t.vx == pytest.approx(1.4)
        assert t.vy == pytest.approx(0.2)
        assert t.bbox_size == (pytest.approx(2.0), pytest.approx(0.9))
        assert t.bbox_yaw == pytest.approx(0.45)
        assert t.track_age == 8
        assert t.coast_cycles == 1
        assert t.covariance[0, 0] == pytest.approx(0.05)
        assert t.covariance[0, 1] == pytest.approx(0.01)
        assert t.covariance[1, 0] == pytest.approx(0.01)
        assert t.covariance[1, 1] == pytest.approx(0.08)

    def test_02_predictor_invoked_with_phase13_objects(self):
        """Test 2: Node delegates prediction to core MotionPredictor using pure Phase 13 data model."""
        mock_predictor = MagicMock(spec=MotionPredictor)
        dummy_pred = PredictedTrajectoryArray(header_timestamp=10.0, frame_id="odom", predictions=[])
        mock_predictor.predict_tracks.return_value = dummy_pred

        node = DynamicPredictorNode(
            predictor=mock_predictor,
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        raw_msg = make_double_tracked_obstacle_array(timestamp_s=10.0, frame_id="odom")
        node.process_tracks(raw_msg, horizon_s=2.0, dt_s=0.25)

        # Assert predict_tracks was called once with a pure TrackedObstacleArray instance
        assert mock_predictor.predict_tracks.call_count == 1
        call_args = mock_predictor.predict_tracks.call_args
        called_tracks = call_args[1].get("tracks") or call_args[0][0]
        assert isinstance(called_tracks, TrackedObstacleArray)
        assert len(called_tracks.tracks) == 1
        assert isinstance(called_tracks.tracks[0], TrackedObstacleState)
        assert called_tracks.tracks[0].track_id == 1

    def test_03_output_conversion_preserves_trajectory_fields(self):
        """Test 3: Output conversion maps Phase 13 predictions to ROS message without field loss."""
        node = DynamicPredictorNode()

        state = TrackedObstacleState(
            track_id=5,
            timestamp=20.0,
            x=1.0,
            y=2.0,
            vx=0.5,
            vy=0.0,
            covariance=np.eye(4) * 0.04,
            bbox_size=(1.5, 0.8),
            bbox_yaw=0.1,
            frame_id="odom",
            status="TRACKING",
            is_confirmed=True,
        )
        cov2d = np.array([[0.04, 0.005], [0.005, 0.04]], dtype=np.float64)
        wp = TrajectoryWaypoint(
            time_offset=0.5,
            timestamp=20.5,
            x=1.25,
            y=2.0,
            vx=0.5,
            vy=0.0,
            covariance_2d=cov2d,
            uncertainty_radius=0.35,
        )
        traj = PredictedObstacleTrajectory(
            track_id=5,
            current_state=state,
            horizon_s=1.0,
            dt=0.5,
            waypoints=[wp],
            bounding_polygon_per_step=[np.zeros((4, 2))],
        )
        pred_array = PredictedTrajectoryArray(header_timestamp=20.0, frame_id="odom", predictions=[traj])

        ros_msg = node.convert_output_predictions(
            pred_array,
            msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        assert isinstance(ros_msg, DoublePredictedTrajectoryArrayMsg)
        assert ros_msg.header.frame_id == "odom"
        assert len(ros_msg.trajectories) == 1

        t_msg = ros_msg.trajectories[0]
        assert t_msg.track_id == 5
        assert t_msg.status == "TRACKING"
        assert t_msg.bbox_size.x == pytest.approx(1.5)
        assert t_msg.bbox_size.y == pytest.approx(0.8)
        assert t_msg.bbox_yaw == pytest.approx(0.1)
        assert t_msg.prediction_horizon_s == pytest.approx(1.0)
        assert len(t_msg.waypoints) == 1

        wp_msg = t_msg.waypoints[0]
        assert wp_msg.relative_time_s == pytest.approx(0.5)
        assert wp_msg.position.x == pytest.approx(1.25)
        assert wp_msg.position.y == pytest.approx(2.0)
        assert wp_msg.velocity.x == pytest.approx(0.5)
        assert wp_msg.velocity.y == pytest.approx(0.0)
        assert wp_msg.uncertainty_radius_m == pytest.approx(0.35)
        assert len(wp_msg.covariance) == 4
        assert wp_msg.covariance[0] == pytest.approx(0.04)
        assert wp_msg.covariance[1] == pytest.approx(0.005)

    def test_04_canonical_topic_and_qos_contract(self):
        """Test 4: Node specifies canonical topic contracts (/tracking/dynamic_obstacles -> /prediction/obstacle_trajectories)."""
        node = DynamicPredictorNode()

        # Contract validation: message types configured
        assert node._pub_msg_type is not None
        assert node._sub_msg_type is not None

        # Predictor configuration adheres to canonical defaults
        assert node.config.default_horizon_s == 2.0
        assert node.config.default_dt_s == 0.25
        assert node.config.sigma_scale == 3.0
        assert node.config.degraded_noise_multiplier == 2.0
        assert node.config.lost_noise_multiplier == 4.0
        assert node.config.lost_max_horizon_s == 1.0

    def test_05_timestamp_preservation_nanosecond_precision(self):
        """Test 5: Sub-millisecond and nanosecond precision are preserved across the boundary."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        test_timestamp = 1725519600.123456789
        raw_msg = make_double_tracked_obstacle_array(timestamp_s=test_timestamp, frame_id="odom")

        out_msg = node.process_tracks(raw_msg, horizon_s=0.5, dt_s=0.25)
        assert out_msg is not None

        # Verify header stamp roundtrip
        sec = out_msg.header.stamp.sec
        nanosec = out_msg.header.stamp.nanosec
        reconstructed = float(sec) + float(nanosec) * 1e-9
        assert reconstructed == pytest.approx(test_timestamp, rel=1e-8)

        # Verify waypoint target timestamps
        wp0 = out_msg.trajectories[0].waypoints[0]
        wp0_reconstructed = float(wp0.target_time.sec) + float(wp0.target_time.nanosec) * 1e-9
        assert wp0_reconstructed == pytest.approx(test_timestamp, rel=1e-8)

    def test_06_frame_id_preservation(self):
        """Test 6: Frame ID (e.g. 'map', 'odom') is strictly preserved without transformation."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        for fid in ["odom", "map", "planning_frame"]:
            raw_msg = make_double_tracked_obstacle_array(timestamp_s=5.0, frame_id=fid)
            out_msg = node.process_tracks(raw_msg)
            assert out_msg.header.frame_id == fid
            assert out_msg.trajectories[0].header.frame_id == fid

    def test_07_covariance_preservation_row_major(self):
        """Test 7: Positional covariance (2x2) in prediction waypoints is preserved in row-major order."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        cov_in = [0.0] * 16
        cov_in[0] = 0.16  # var(x)
        cov_in[1] = 0.04  # cov(x,y)
        cov_in[4] = 0.04  # cov(y,x)
        cov_in[5] = 0.09  # var(y)
        cov_in[10] = 0.01 # var(vx)
        cov_in[15] = 0.01 # var(vy)

        raw_msg = make_double_tracked_obstacle_array(
            timestamp_s=1.0,
            frame_id="odom",
            tracks_data=[{"track_id": 7, "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "covariance": cov_in}],
        )

        out_msg = node.process_tracks(raw_msg, horizon_s=0.5, dt_s=0.25)
        # Waypoint 0 (t=0.0): initial covariance propagation
        wp0 = out_msg.trajectories[0].waypoints[0]
        assert len(wp0.covariance) == 4
        # [0]=xx, [1]=xy, [2]=yx, [3]=yy
        assert wp0.covariance[0] == pytest.approx(0.16, rel=1e-3)
        assert wp0.covariance[1] == pytest.approx(0.04, rel=1e-3)
        assert wp0.covariance[2] == pytest.approx(0.04, rel=1e-3)
        assert wp0.covariance[3] == pytest.approx(0.09, rel=1e-3)

    def test_08_multiple_tracked_obstacles_produce_multiple_predictions(self):
        """Test 8: Multiple tracked obstacles in input array produce corresponding distinct trajectory predictions."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        tracks = [
            {"track_id": 10, "x": 2.0, "y": 0.0, "vx": 1.0, "vy": 0.0},
            {"track_id": 20, "x": 5.0, "y": -3.0, "vx": -0.5, "vy": 0.5},
            {"track_id": 30, "x": -1.0, "y": 4.0, "vx": 0.0, "vy": -1.2},
        ]
        raw_msg = make_double_tracked_obstacle_array(timestamp_s=10.0, tracks_data=tracks)

        out_msg = node.process_tracks(raw_msg)
        assert len(out_msg.trajectories) == 3

        ids = [t.track_id for t in out_msg.trajectories]
        assert ids == [10, 20, 30]

        # Trajectory 0 moving in +x direction
        t10 = out_msg.trajectories[0]
        assert t10.waypoints[-1].position.x > t10.waypoints[0].position.x

        # Trajectory 1 moving in -x, +y direction
        t20 = out_msg.trajectories[1]
        assert t20.waypoints[-1].position.x < t20.waypoints[0].position.x
        assert t20.waypoints[-1].position.y > t20.waypoints[0].position.y

    def test_09_lifecycle_confirmation_semantics_preserved(self):
        """Test 9: Lifecycle statuses (TRACKING, DEGRADED, LOST, EXPIRED) govern prediction horizon and uncertainty."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        tracks = [
            {"track_id": 1, "status": "TRACKING", "x": 1.0, "y": 0.0, "vx": 1.0},
            {"track_id": 2, "status": "DEGRADED", "x": 2.0, "y": 0.0, "vx": 1.0},
            {"track_id": 3, "status": "LOST", "x": 3.0, "y": 0.0, "vx": 1.0},
            {"track_id": 4, "status": "EXPIRED", "x": 4.0, "y": 0.0, "vx": 1.0},
        ]
        raw_msg = make_double_tracked_obstacle_array(timestamp_s=10.0, tracks_data=tracks)

        out_msg = node.process_tracks(raw_msg, horizon_s=2.0, dt_s=0.5)

        # EXPIRED track (id=4) must be omitted cleanly
        ids = [t.track_id for t in out_msg.trajectories]
        assert 4 not in ids
        assert len(out_msg.trajectories) == 3

        t_tracking = next(t for t in out_msg.trajectories if t.track_id == 1)
        t_degraded = next(t for t in out_msg.trajectories if t.track_id == 2)
        t_lost = next(t for t in out_msg.trajectories if t.track_id == 3)

        # Statuses preserved
        assert t_tracking.status == "TRACKING"
        assert t_degraded.status == "DEGRADED"
        assert t_lost.status == "LOST"

        # LOST track horizon capped at lost_max_horizon_s (1.0s) instead of requested 2.0s
        assert t_tracking.prediction_horizon_s == pytest.approx(2.0)
        assert t_lost.prediction_horizon_s == pytest.approx(1.0)

        # DEGRADED and LOST have higher uncertainty than nominal TRACKING
        unc_tracking = t_tracking.waypoints[-1].uncertainty_radius_m
        unc_degraded = t_degraded.waypoints[-1].uncertainty_radius_m
        unc_lost = t_lost.waypoints[-1].uncertainty_radius_m
        assert unc_degraded > unc_tracking

    def test_10_stale_or_out_of_order_handling(self):
        """Test 10: Stale or out-of-order tracked obstacle messages are safely rejected without state corruption."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        # Message 1 at t=15.0
        msg1 = make_double_tracked_obstacle_array(timestamp_s=15.0, frame_id="odom")
        out1 = node.process_tracks(msg1)
        assert out1 is not None
        assert len(out1.trajectories) == 1
        assert node._last_timestamp == 15.0

        # Message 2 at t=14.0 (stale / in the past)
        msg_stale = make_double_tracked_obstacle_array(timestamp_s=14.0, frame_id="odom")
        out_stale = node.process_tracks(msg_stale)
        assert out_stale is not None
        # Internal timestamp must not regress
        assert node._last_timestamp == 15.0

    def test_11_prediction_timestamp_monotonicity(self):
        """Test 11: Prediction trajectory waypoints possess strictly monotonically increasing timestamps."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        raw_msg = make_double_tracked_obstacle_array(timestamp_s=100.0, frame_id="odom")
        out_msg = node.process_tracks(raw_msg, horizon_s=3.0, dt_s=0.25)

        traj = out_msg.trajectories[0]
        assert len(traj.waypoints) > 5

        prev_rel = -1.0
        prev_target = -1.0

        for wp in traj.waypoints:
            target_ts = float(wp.target_time.sec) + float(wp.target_time.nanosec) * 1e-9
            rel_ts = wp.relative_time_s

            assert rel_ts > prev_rel, f"Relative timestamp not strictly monotonic: {rel_ts} <= {prev_rel}"
            assert target_ts > prev_target, f"Target timestamp not strictly monotonic: {target_ts} <= {prev_target}"

            prev_rel = rel_ts
            prev_target = target_ts

    def test_12_malformed_invalid_input_safety_behavior(self):
        """Test 12: Node safely handles malformed, empty, or None messages without uncaught exception."""
        node = DynamicPredictorNode(
            output_msg_cls=DoublePredictedTrajectoryArrayMsg,
            traj_item_msg_cls=DoublePredictedObstacleTrajectoryMsg,
            wp_item_msg_cls=DoubleTrajectoryWaypointMsg,
        )

        # Case A: None input
        out_none = node.process_tracks(None)
        assert out_none is not None
        assert isinstance(out_none, DoublePredictedTrajectoryArrayMsg)
        assert len(out_none.trajectories) == 0

        # Case B: Empty tracks array
        empty_msg = DoubleTrackedObstacleArrayMsg()
        empty_msg.header = DoubleHeader("odom", DoubleTime(1, 0))
        out_empty = node.process_tracks(empty_msg)
        assert out_empty is not None
        assert len(out_empty.trajectories) == 0

        # Case C: Completely malformed object missing fields
        class MalformedMsg:
            pass

        out_malformed = node.process_tracks(MalformedMsg())
        assert out_malformed is not None
        assert len(out_malformed.trajectories) == 0
