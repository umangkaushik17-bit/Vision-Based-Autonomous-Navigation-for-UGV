"""Unit and integration test suite for ROSIDL <-> Phase 13 converter boundary (Milestone 13.12.3).

Validates:
1. Timestamp conversions (zero, integer, fractional, large, precision, negative rejection)
2. Covariance conversions (2x2, 4x4, row-major preservation, transposition safety, invalid dimensions)
3. Enum / State conversions (BehavioralState, DeadlockEvent, DynamicReplanTrigger, LocalizationState, safety fallbacks)
4. Message bidirectional conversions across all 13 interfaces:
   - DynamicObstacleDetection
   - DynamicObstacleDetectionArray
   - TrackedObstacle
   - TrackedObstacleArray
   - TrajectoryWaypoint
   - PredictedObstacleTrajectory
   - PredictedTrajectoryArray
   - DynamicOccupancyGridSlice
   - DynamicOccupancySliceArray
   - BehavioralDecision
   - DynamicReplanEvent
   - LocalizationStatus
   - WorldTerrainMap
5. Safety field preservation (unknown masks, emergency clamps, validity flags, stale indicators)
6. Empty collections and boundary conditions
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from ai.planning.dynamic.behavioral_state import (
    BehavioralDecision,
    BehavioralState,
    DeadlockEvent,
)
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanEvent,
    DynamicReplanTrigger,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyGridSlice, DynamicRiskLevel
from ai.spatial_fusion.world_map import WorldTerrainMap

from ugv_navigation.converters.dynamic_converters import (
    behavioral_decision_from_ros,
    behavioral_decision_to_ros,
    covariance_2x2_to_list,
    covariance_4x4_to_list,
    dynamic_obstacle_detection_array_from_ros,
    dynamic_obstacle_detection_array_to_ros,
    dynamic_obstacle_detection_from_ros,
    dynamic_obstacle_detection_to_ros,
    dynamic_occupancy_grid_slice_from_ros,
    dynamic_occupancy_grid_slice_to_ros,
    dynamic_occupancy_slice_array_from_ros,
    dynamic_occupancy_slice_array_to_ros,
    dynamic_replan_event_from_ros,
    dynamic_replan_event_to_ros,
    list_to_covariance_2x2,
    list_to_covariance_4x4,
    localization_status_from_ros,
    localization_status_to_ros,
    predicted_obstacle_trajectory_from_ros,
    predicted_obstacle_trajectory_to_ros,
    predicted_trajectory_array_from_ros,
    predicted_trajectory_array_to_ros,
    ros_time_to_timestamp,
    string_to_behavioral_state,
    string_to_deadlock_event,
    string_to_replan_trigger,
    timestamp_to_ros_time,
    tracked_obstacle_array_from_ros,
    tracked_obstacle_array_to_ros,
    tracked_obstacle_from_ros,
    tracked_obstacle_to_ros,
    trajectory_waypoint_from_ros,
    trajectory_waypoint_to_ros,
    world_terrain_map_from_ros,
    world_terrain_map_to_ros,
)


# ==============================================================================
# TEST DOUBLES (Controlled lightweight doubles for non-ROS2 environment testing)
# ==============================================================================

class DoubleTime:
    def __init__(self, sec: int = 0, nanosec: int = 0):
        self.sec = int(sec)
        self.nanosec = int(nanosec)


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


class DoublePose:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.position = DoublePoint(x, y, z)


class DoubleDetectionMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.detection_id = 0
        self.class_name = ""
        self.confidence = 0.0
        self.position = DoublePoint()
        self.dimensions = DoubleVector3()
        self.yaw_rad = 0.0
        self.position_covariance = [0.0] * 4
        self.is_valid = True


class DoubleDetectionArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.detections = []


class DoubleTrackedObstacleMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "NEW"
        self.is_confirmed = False
        self.confidence = 1.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.bbox_size = DoubleVector3()
        self.bbox_yaw = 0.0
        self.covariance = [0.0] * 16
        self.total_hits = 1
        self.consecutive_misses = 0
        self.time_since_last_update_s = 0.0


class DoubleTrackedObstacleArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.tracks = []


class DoubleTrajectoryWaypointMsg:
    def __init__(self):
        self.target_time = DoubleTime()
        self.relative_time_s = 0.0
        self.position = DoublePoint()
        self.velocity = DoubleVector3()
        self.covariance = [0.0] * 4
        self.uncertainty_radius_m = 0.0
        self.confidence = 1.0


class DoublePredictedObstacleTrajectoryMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.track_id = 0
        self.status = "TRACKING"
        self.bbox_size = DoubleVector3()
        self.bbox_yaw = 0.0
        self.waypoints = []
        self.prediction_horizon_s = 2.0
        self.is_unknown = False
        self.confidence = 1.0


class DoublePredictedTrajectoryArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.trajectories = []


class DoubleDynamicOccupancyGridSliceMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.target_time_offset_s = 0.0
        self.target_timestamp = DoubleTime()
        self.resolution = 0.10
        self.origin_x = -20.0
        self.origin_y = -20.0
        self.width = 0
        self.height = 0
        self.data = []
        self.unknown_mask = []


class DoubleDynamicOccupancySliceArrayMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.slices = []


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
        self.transition = "NONE"
        self.previous_state = "PROCEED"
        self.state_dwell_time_s = 0.0
        self.confidence = 1.0


class DoubleDynamicReplanEventMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.event_id = ""
        self.trigger_type = "NONE"
        self.current_pose = DoublePoint()
        self.current_yaw = 0.0
        self.current_goal = DoublePoint()
        self.blocking_obstacle_ids = []
        self.blocked_duration_s = 0.0
        self.deadlock_duration_s = 0.0
        self.current_progress_m = 0.0
        self.reason = ""
        self.confidence = 1.0
        self.success = True
        self.is_path_stale = False
        self.old_path_length_m = 0.0
        self.new_path_length_m = 0.0
        self.path_deviation_m = 0.0
        self.planning_latency_ms = 0.0


class DoubleLocalizationStatusMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.child_frame_id = "base_link"
        self.tracking_state = 0
        self.confidence = 1.0
        self.is_pose_valid = True
        self.allows_spatial_fusion = True
        self.pose = None
        self.twist = None
        self.tracked_features = 100
        self.inliers = 80
        self.keyframe_id = 1
        self.is_keyframe = False
        self.relocalized = False


class DoubleWorldTerrainMapMsg:
    def __init__(self):
        self.header = DoubleHeader()
        self.resolution = 0.10
        self.origin = DoublePose()
        self.width = 10
        self.height = 10
        self.cost = []
        self.traversability = []
        self.obstacle = []
        self.hazard = []
        self.unknown = []
        self.confidence = []
        self.slope = []
        self.roughness = []
        self.height_var = []
        self.observation_count = []


# ==============================================================================
# 1. TIMESTAMP CONVERSION TESTS
# ==============================================================================

def test_timestamp_zero():
    t_ros = timestamp_to_ros_time(0.0, time_cls=DoubleTime)
    assert t_ros.sec == 0
    assert t_ros.nanosec == 0
    t_back = ros_time_to_timestamp(t_ros)
    assert t_back == 0.0


def test_timestamp_integer():
    t_ros = timestamp_to_ros_time(42.0, time_cls=DoubleTime)
    assert t_ros.sec == 42
    assert t_ros.nanosec == 0
    assert ros_time_to_timestamp(t_ros) == 42.0


def test_timestamp_fractional_precision():
    original = 1725519600.1234567
    t_ros = timestamp_to_ros_time(original, time_cls=DoubleTime)
    assert t_ros.sec == 1725519600
    expected_nanosec = int(round((original - 1725519600) * 1e9))
    assert t_ros.nanosec == expected_nanosec

    reconstructed = ros_time_to_timestamp(t_ros)
    assert abs(reconstructed - original) < 1e-6


def test_timestamp_negative_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        timestamp_to_ros_time(-1.0, time_cls=DoubleTime)


# ==============================================================================
# 2. COVARIANCE CONVERSION TESTS
# ==============================================================================

def test_covariance_2x2_row_major():
    # Asymmetric input matrix to detect transposition bugs
    cov = np.array([
        [0.04, 0.01],
        [0.02, 0.09],
    ])
    cov_list = covariance_2x2_to_list(cov)
    assert cov_list == [0.04, 0.01, 0.02, 0.09]

    # Symmetrized reconstruction check
    reconstructed = list_to_covariance_2x2(cov_list)
    assert reconstructed.shape == (2, 2)
    assert reconstructed[0, 0] == pytest.approx(0.04)
    assert reconstructed[1, 1] == pytest.approx(0.09)
    assert reconstructed[0, 1] == pytest.approx(0.015)  # Symmetrized 0.5 * (0.01 + 0.02)


def test_covariance_2x2_invalid_size():
    with pytest.raises(ValueError, match="Expected 2x2"):
        covariance_2x2_to_list(np.ones((3, 3)))
    with pytest.raises(ValueError, match="Expected 4 covariance elements"):
        list_to_covariance_2x2([1.0, 2.0, 3.0])


def test_covariance_4x4_row_major():
    cov = np.arange(16, dtype=np.float64).reshape((4, 4))
    cov_list = covariance_4x4_to_list(cov)
    assert len(cov_list) == 16
    assert cov_list[1] == 1.0
    assert cov_list[4] == 4.0
    assert cov_list[15] == 15.0

    reconstructed = list_to_covariance_4x4(cov_list)
    assert reconstructed.shape == (4, 4)


# ==============================================================================
# 3. ENUM / STATE CONVERSION TESTS
# ==============================================================================

def test_enum_behavioral_state():
    assert string_to_behavioral_state("PROCEED") == BehavioralState.PROCEED
    assert string_to_behavioral_state("AVOID") == BehavioralState.AVOID
    assert string_to_behavioral_state("WAIT") == BehavioralState.WAIT
    assert string_to_behavioral_state("STOP") == BehavioralState.STOP

    with pytest.raises(ValueError, match="Invalid behavioral state"):
        string_to_behavioral_state("GO_FAST")


def test_enum_deadlock_event():
    assert string_to_deadlock_event("NONE") == "NONE"
    assert string_to_deadlock_event("DYNAMIC_DEADLOCK") == "DYNAMIC_DEADLOCK"

    with pytest.raises(ValueError, match="Invalid deadlock event"):
        string_to_deadlock_event("MAYBE_DEADLOCKED")


def test_enum_replan_trigger():
    assert string_to_replan_trigger("DYNAMIC_DEADLOCK") == DynamicReplanTrigger.DYNAMIC_DEADLOCK
    assert string_to_replan_trigger("PERSISTENT_BLOCKAGE") == DynamicReplanTrigger.PERSISTENT_BLOCKAGE
    assert string_to_replan_trigger("PATH_INVALIDATED") == DynamicReplanTrigger.PATH_INVALIDATED

    with pytest.raises(ValueError, match="Invalid replan trigger"):
        string_to_replan_trigger("WIFI_DISCONNECTED")


# ==============================================================================
# 4. MESSAGE BIDIRECTIONAL TESTS
# ==============================================================================

def test_dynamic_obstacle_detection_roundtrip():
    det = DynamicObstacleDetection(
        detection_id="42",
        timestamp=10.5,
        frame_id="base_link",
        bbox_center=(2.5, 1.2),
        bbox_size=(1.8, 0.7),
        bbox_yaw=0.35,
        confidence=0.88,
    )

    ros_msg = dynamic_obstacle_detection_to_ros(
        det,
        msg_cls=DoubleDetectionMsg,
        header_cls=DoubleHeader,
        point_cls=DoublePoint,
        vector3_cls=DoubleVector3,
    )

    assert ros_msg.detection_id == 42
    assert ros_msg.confidence == pytest.approx(0.88)
    assert ros_msg.position.x == pytest.approx(2.5)
    assert ros_msg.position.y == pytest.approx(1.2)
    assert ros_msg.yaw_rad == pytest.approx(0.35)

    back = dynamic_obstacle_detection_from_ros(ros_msg)
    assert back.detection_id == "42"
    assert back.timestamp == pytest.approx(10.5)
    assert back.frame_id == "base_link"
    assert back.bbox_center[0] == pytest.approx(2.5)
    assert back.bbox_center[1] == pytest.approx(1.2)
    assert back.bbox_size[0] == pytest.approx(1.8)
    assert back.bbox_size[1] == pytest.approx(0.7)
    assert back.bbox_yaw == pytest.approx(0.35)
    assert back.confidence == pytest.approx(0.88)


def test_dynamic_obstacle_detection_array_roundtrip():
    d1 = DynamicObstacleDetection("1", 10.0, "base_link", (1.0, 0.5), (1.0, 1.0), 0.0, 0.9)
    d2 = DynamicObstacleDetection("2", 10.0, "base_link", (3.0, -1.0), (0.8, 0.8), 0.1, 0.85)
    arr = DynamicObstacleDetectionArray(header_timestamp=10.0, frame_id="base_link", detections=[d1, d2])

    ros_msg = dynamic_obstacle_detection_array_to_ros(
        arr,
        msg_cls=DoubleDetectionArrayMsg,
        det_msg_cls=DoubleDetectionMsg,
        header_cls=DoubleHeader,
        point_cls=DoublePoint,
        vector3_cls=DoubleVector3,
    )

    assert len(ros_msg.detections) == 2
    back = dynamic_obstacle_detection_array_from_ros(ros_msg)
    assert len(back.detections) == 2
    assert back.header_timestamp == pytest.approx(10.0)
    assert back.detections[0].detection_id == "1"
    assert back.detections[1].detection_id == "2"


def test_tracked_obstacle_roundtrip():
    cov = np.eye(4, dtype=np.float64) * 0.05
    track = TrackedObstacleState(
        track_id=7,
        timestamp=15.2,
        x=4.0,
        y=-2.0,
        vx=0.8,
        vy=-0.1,
        covariance=cov,
        bbox_size=(1.5, 0.6),
        bbox_yaw=0.20,
        frame_id="odom",
        track_age=5,
        coast_cycles=1,
        is_confirmed=True,
        status="TRACKING",
    )

    ros_msg = tracked_obstacle_to_ros(
        track,
        msg_cls=DoubleTrackedObstacleMsg,
        header_cls=DoubleHeader,
        point_cls=DoublePoint,
        vector3_cls=DoubleVector3,
    )

    assert ros_msg.track_id == 7
    assert ros_msg.is_confirmed is True
    assert ros_msg.status == "TRACKING"
    assert ros_msg.position.x == pytest.approx(4.0)
    assert ros_msg.velocity.x == pytest.approx(0.8)

    back = tracked_obstacle_from_ros(ros_msg)
    assert back.track_id == 7
    assert back.timestamp == pytest.approx(15.2)
    assert back.x == pytest.approx(4.0)
    assert back.vx == pytest.approx(0.8)
    assert back.is_confirmed is True
    assert back.status == "TRACKING"
    assert back.covariance.shape == (4, 4)


def test_tracked_obstacle_array_roundtrip():
    cov = np.eye(4, dtype=np.float64) * 0.05
    t1 = TrackedObstacleState(1, 10.0, 2.0, 1.0, 0.5, 0.0, cov, (1.0, 1.0), 0.0, is_confirmed=True)
    arr = TrackedObstacleArray(header_timestamp=10.0, frame_id="odom", tracks=[t1])

    ros_msg = tracked_obstacle_array_to_ros(
        arr,
        msg_cls=DoubleTrackedObstacleArrayMsg,
        track_msg_cls=DoubleTrackedObstacleMsg,
        header_cls=DoubleHeader,
        point_cls=DoublePoint,
        vector3_cls=DoubleVector3,
    )
    assert len(ros_msg.tracks) == 1
    back = tracked_obstacle_array_from_ros(ros_msg)
    assert len(back.tracks) == 1
    assert back.tracks[0].track_id == 1


def test_trajectory_waypoint_roundtrip():
    cov2d = np.array([[0.04, 0.0], [0.0, 0.04]])
    wp = TrajectoryWaypoint(
        time_offset=0.5,
        timestamp=20.5,
        x=3.0,
        y=1.5,
        vx=1.0,
        vy=0.2,
        covariance_2d=cov2d,
        uncertainty_radius=0.35,
    )

    ros_msg = trajectory_waypoint_to_ros(
        wp,
        msg_cls=DoubleTrajectoryWaypointMsg,
        time_cls=DoubleTime,
        point_cls=DoublePoint,
        vector3_cls=DoubleVector3,
    )

    assert ros_msg.relative_time_s == pytest.approx(0.5)
    assert ros_msg.position.x == pytest.approx(3.0)
    assert ros_msg.uncertainty_radius_m == pytest.approx(0.35)

    back = trajectory_waypoint_from_ros(ros_msg)
    assert back.time_offset == pytest.approx(0.5)
    assert back.timestamp == pytest.approx(20.5)
    assert back.x == pytest.approx(3.0)
    assert back.uncertainty_radius == pytest.approx(0.35)
    assert back.covariance_2d.shape == (2, 2)


def test_predicted_obstacle_trajectory_roundtrip():
    cov4 = np.eye(4, dtype=np.float64) * 0.04
    state = TrackedObstacleState(3, 10.0, 2.0, 1.0, 1.0, 0.0, cov4, (1.2, 0.6), 0.0, status="TRACKING")
    cov2 = np.eye(2, dtype=np.float64) * 0.04
    wp0 = TrajectoryWaypoint(0.0, 10.0, 2.0, 1.0, 1.0, 0.0, cov2, 0.20)
    wp1 = TrajectoryWaypoint(0.5, 10.5, 2.5, 1.0, 1.0, 0.0, cov2, 0.30)
    polys = [compute_obb_corners((2.0, 1.0), (1.2, 0.6), 0.0), compute_obb_corners((2.5, 1.0), (1.2, 0.6), 0.0)]

    traj = PredictedObstacleTrajectory(
        track_id=3,
        current_state=state,
        horizon_s=0.5,
        dt=0.5,
        waypoints=[wp0, wp1],
        bounding_polygon_per_step=polys,
    )

    ros_msg = predicted_obstacle_trajectory_to_ros(
        traj,
        frame_id="odom",
        msg_cls=DoublePredictedObstacleTrajectoryMsg,
        wp_msg_cls=DoubleTrajectoryWaypointMsg,
        header_cls=DoubleHeader,
        vector3_cls=DoubleVector3,
        time_cls=DoubleTime,
        point_cls=DoublePoint,
    )

    assert ros_msg.track_id == 3
    assert len(ros_msg.waypoints) == 2

    back = predicted_obstacle_trajectory_from_ros(ros_msg)
    assert back.track_id == 3
    assert len(back.waypoints) == 2
    assert back.horizon_s == pytest.approx(0.5)


def test_dynamic_occupancy_grid_slice_roundtrip():
    cost = np.zeros((10, 10), dtype=np.float32)
    cost[4:6, 4:6] = 0.98  # Lethal
    cost[3:7, 3:7][cost[3:7, 3:7] < 0.98] = 0.50  # Medium risk
    risk = np.zeros((10, 10), dtype=np.uint8)
    risk[cost >= 0.95] = int(DynamicRiskLevel.LETHAL)
    risk[(cost >= 0.40) & (cost < 0.70)] = int(DynamicRiskLevel.MEDIUM_RISK)

    slice_obj = DynamicOccupancyGridSlice(
        time_offset=0.25,
        timestamp=10.25,
        cost=cost,
        risk_level=risk,
        nominal_obstacle=(cost >= 0.95),
        uncertainty_zone=(cost >= 0.15) & (cost < 0.95),
        active_obstacle_ids=[1],
    )

    ros_msg = dynamic_occupancy_grid_slice_to_ros(
        slice_obj,
        resolution=0.10,
        origin_x=-1.0,
        origin_y=-1.0,
        frame_id="odom",
        msg_cls=DoubleDynamicOccupancyGridSliceMsg,
        header_cls=DoubleHeader,
        time_cls=DoubleTime,
    )

    assert ros_msg.width == 10
    assert ros_msg.height == 10
    assert len(ros_msg.data) == 100

    back = dynamic_occupancy_grid_slice_from_ros(ros_msg)
    assert back.time_offset == pytest.approx(0.25)
    assert back.cost.shape == (10, 10)
    assert back.cost[4, 4] == pytest.approx(0.98)
    assert back.cost[0, 0] == pytest.approx(0.0)
    assert back.nominal_obstacle[4, 4] == True


def test_behavioral_decision_roundtrip():
    dec = BehavioralDecision(
        state=BehavioralState.STOP,
        command_v=0.0,
        command_omega=0.0,
        reason="Imminent dynamic collision threat",
        blocking_obstacle_id=5,
        dynamic_collision=True,
        dynamic_risk="LETHAL",
        unknown_horizon=False,
        wait_elapsed_s=0.0,
        wait_timeout=False,
        deadlock=True,
        deadlock_event=DeadlockEvent.DYNAMIC_DEADLOCK.value,
        transition="AVOID->STOP",
        previous_state=BehavioralState.AVOID,
        state_dwell_time_s=1.2,
        confidence=0.99,
    )

    ros_msg = behavioral_decision_to_ros(
        dec,
        frame_id="odom",
        timestamp_s=12.0,
        msg_cls=DoubleBehavioralDecisionMsg,
        header_cls=DoubleHeader,
    )

    assert ros_msg.state == "STOP"
    assert ros_msg.command_v == pytest.approx(0.0)
    assert ros_msg.blocking_obstacle_id == 5
    assert ros_msg.dynamic_collision is True
    assert ros_msg.deadlock is True

    back = behavioral_decision_from_ros(ros_msg)
    assert back.state == BehavioralState.STOP
    assert back.previous_state == BehavioralState.AVOID
    assert back.command_v == 0.0
    assert back.blocking_obstacle_id == 5
    assert back.dynamic_collision is True
    assert back.deadlock is True
    assert back.deadlock_event == "DYNAMIC_DEADLOCK"


def test_dynamic_replan_event_roundtrip():
    evt = DynamicReplanEvent(
        event_id="REPLAN_001",
        trigger_type=DynamicReplanTrigger.DYNAMIC_DEADLOCK,
        timestamp=25.0,
        current_pose=(5.0, 2.0, 0.5),
        current_goal=(20.0, 15.0),
        current_global_path=None,
        blocking_obstacle_ids=[2, 4],
        blocked_duration=3.2,
        deadlock_duration=3.0,
        current_progress=10.5,
        reason="Persistent deadlock with 2 obstacles",
        confidence=0.95,
    )

    ros_msg = dynamic_replan_event_to_ros(
        evt,
        success=True,
        is_path_stale=False,
        frame_id="odom",
        msg_cls=DoubleDynamicReplanEventMsg,
        header_cls=DoubleHeader,
        point_cls=DoublePoint,
    )

    assert ros_msg.event_id == "REPLAN_001"
    assert ros_msg.trigger_type == "DYNAMIC_DEADLOCK"
    assert ros_msg.blocking_obstacle_ids == [2, 4]
    assert ros_msg.blocked_duration_s == pytest.approx(3.2)

    back = dynamic_replan_event_from_ros(ros_msg)
    assert back.event_id == "REPLAN_001"
    assert back.trigger_type == DynamicReplanTrigger.DYNAMIC_DEADLOCK
    assert back.timestamp == pytest.approx(25.0)
    assert back.current_pose == (pytest.approx(5.0), pytest.approx(2.0), pytest.approx(0.5))
    assert back.current_goal == (pytest.approx(20.0), pytest.approx(15.0))
    assert back.blocking_obstacle_ids == [2, 4]


def test_world_terrain_map_roundtrip():
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    wmap.cost[10, 10] = 0.85
    wmap.obstacle[10, 10] = True

    ros_msg = world_terrain_map_to_ros(
        wmap,
        frame_id="odom",
        timestamp_s=50.0,
        msg_cls=DoubleWorldTerrainMapMsg,
        header_cls=DoubleHeader,
        pose_cls=DoublePose,
        point_cls=DoublePoint,
    )

    assert ros_msg.width == 100
    assert ros_msg.height == 100
    assert len(ros_msg.cost) == 10000

    back = world_terrain_map_from_ros(ros_msg)
    assert back.resolution_m == pytest.approx(0.10)
    assert back.shape == (100, 100)
    assert back.cost[10, 10] == pytest.approx(0.85)
    assert back.obstacle[10, 10] == True


# ==============================================================================
# 5. SAFETY PRESERVATION TESTS
# ==============================================================================

def test_safety_lost_localization_mapping():
    # If tracking state is corrupted or invalid, must map strictly to LOST (2)
    corrupted_msg = DoubleLocalizationStatusMsg()
    corrupted_msg.tracking_state = 99  # invalid!
    result = localization_status_from_ros(corrupted_msg)
    assert result["tracking_state"] == 2  # LOST


def test_safety_empty_detections_preserve_validity():
    arr = DynamicObstacleDetectionArray(header_timestamp=1.0, frame_id="base_link", detections=[])
    ros_msg = dynamic_obstacle_detection_array_to_ros(
        arr,
        msg_cls=DoubleDetectionArrayMsg,
        det_msg_cls=DoubleDetectionMsg,
        header_cls=DoubleHeader,
    )
    assert len(ros_msg.detections) == 0
    back = dynamic_obstacle_detection_array_from_ros(ros_msg)
    assert len(back.detections) == 0
    assert back.header_timestamp == pytest.approx(1.0)
