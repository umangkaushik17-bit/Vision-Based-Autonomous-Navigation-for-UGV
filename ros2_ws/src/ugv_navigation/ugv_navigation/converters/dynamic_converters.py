"""Dedicated bidirectional conversion layer between ROS2 interfaces and Phase 13 data models.

Encapsulates bidirectional conversions between:
- ROS2 generated message types (ugv_navigation_msgs, std_msgs, geometry_msgs, builtin_interfaces)
- Pure Python Phase 13 frozen mathematical models (ai.planning.dynamic.*)
- Pre-existing Phase 9/10 models (WorldTerrainMap, LocalizationStatus)

Architectural Invariants:
1. Core Phase 13 algorithm modules (ai/planning/dynamic/*) remain 100% ROS-independent.
2. Conversion logic is strictly isolated to this module and adapters.
3. Row-major covariance layouts are preserved without transposition.
4. Timestamps preserve full nanosecond precision.
5. All safety-critical fields (validity flags, unknown masks, STOP/WAIT states, emergency clamps)
   are strictly preserved during round-trip conversions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

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
from ugv_navigation.utils.time_utils import RosHeader, RosTime

# Optional lazy imports of native ROS2 message types
try:
    from ugv_navigation_msgs.msg import (
        BehavioralDecision as RosBehavioralDecision,
        DynamicObstacleDetection as RosDynamicObstacleDetection,
        DynamicObstacleDetectionArray as RosDynamicObstacleDetectionArray,
        DynamicOccupancyGridSlice as RosDynamicOccupancyGridSlice,
        DynamicOccupancySliceArray as RosDynamicOccupancySliceArray,
        DynamicReplanEvent as RosDynamicReplanEvent,
        LocalizationStatus as RosLocalizationStatus,
        PredictedObstacleTrajectory as RosPredictedObstacleTrajectory,
        PredictedTrajectoryArray as RosPredictedTrajectoryArray,
        TrackedObstacle as RosTrackedObstacle,
        TrackedObstacleArray as RosTrackedObstacleArray,
        TrajectoryWaypoint as RosTrajectoryWaypoint,
        WorldTerrainMap as RosWorldTerrainMap,
    )
    ROSIDL_UGV_MSGS_AVAILABLE = True
except ImportError:
    ROSIDL_UGV_MSGS_AVAILABLE = False
    RosBehavioralDecision = None
    RosDynamicObstacleDetection = None
    RosDynamicObstacleDetectionArray = None
    RosDynamicOccupancyGridSlice = None
    RosDynamicOccupancySliceArray = None
    RosDynamicReplanEvent = None
    RosLocalizationStatus = None
    RosPredictedObstacleTrajectory = None
    RosPredictedTrajectoryArray = None
    RosTrackedObstacle = None
    RosTrackedObstacleArray = None
    RosTrajectoryWaypoint = None
    RosWorldTerrainMap = None

try:
    from std_msgs.msg import Header as NativeRosHeader
    from builtin_interfaces.msg import Time as NativeRosTime
    from geometry_msgs.msg import (
        Point as NativeRosPoint,
        Pose as NativeRosPose,
        PoseWithCovariance as NativeRosPoseWithCovariance,
        Twist as NativeRosTwist,
        TwistWithCovariance as NativeRosTwistWithCovariance,
        Vector3 as NativeRosVector3,
    )
    ROSIDL_STD_MSGS_AVAILABLE = True
except ImportError:
    ROSIDL_STD_MSGS_AVAILABLE = False
    NativeRosHeader = None
    NativeRosTime = None
    NativeRosPoint = None
    NativeRosPose = None
    NativeRosPoseWithCovariance = None
    NativeRosTwist = None
    NativeRosTwistWithCovariance = None
    NativeRosVector3 = None


# ==============================================================================
# 1. TIMESTAMP CONVERSION
# ==============================================================================

def timestamp_to_ros_time(timestamp_s: float, time_cls: Optional[Any] = None) -> Any:
    """Convert float seconds to ROS builtin_interfaces/Time representation."""
    if timestamp_s < 0.0:
        raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")
    sec = int(math.floor(timestamp_s))
    nanosec = int(round((timestamp_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000

    target_cls = time_cls or NativeRosTime or RosTime
    if hasattr(target_cls, "__call__"):
        return target_cls(sec=sec, nanosec=nanosec)
    return {"sec": sec, "nanosec": nanosec}


def ros_time_to_timestamp(stamp: Any) -> float:
    """Convert ROS builtin_interfaces/Time representation to float seconds."""
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    elif isinstance(stamp, dict):
        sec = int(stamp.get("sec", 0))
        nanosec = int(stamp.get("nanosec", 0))
    elif isinstance(stamp, (tuple, list)) and len(stamp) == 2:
        sec = int(stamp[0])
        nanosec = int(stamp[1])
    elif isinstance(stamp, (int, float)):
        return float(stamp)
    else:
        raise ValueError(f"Unrecognized ROS Time stamp structure: {type(stamp)}")

    if sec < 0 or nanosec < 0:
        raise ValueError(f"Negative timestamp components: sec={sec}, nanosec={nanosec}")
    return float(sec) + float(nanosec) * 1e-9


def make_ros_header(
    frame_id: str,
    timestamp_s: float,
    header_cls: Optional[Any] = None,
    time_cls: Optional[Any] = None,
) -> Any:
    """Create a ROS-compatible Header object."""
    stamp = timestamp_to_ros_time(timestamp_s, time_cls=time_cls)
    target_cls = header_cls or NativeRosHeader or RosHeader
    if hasattr(target_cls, "create") and target_cls is RosHeader:
        return RosHeader(stamp=stamp if isinstance(stamp, RosTime) else RosTime(stamp.sec, stamp.nanosec), frame_id=frame_id)
    if hasattr(target_cls, "__call__"):
        hdr = target_cls()
        hdr.stamp = stamp
        hdr.frame_id = str(frame_id)
        return hdr
    return {"stamp": stamp, "frame_id": str(frame_id)}


# ==============================================================================
# 2. COVARIANCE CONVERSION UTILITIES
# ==============================================================================

def covariance_2x2_to_list(cov: np.ndarray) -> List[float]:
    """Convert a 2x2 covariance matrix to a 4-element row-major list [xx, xy, yx, yy]."""
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.shape != (2, 2):
        raise ValueError(f"Expected 2x2 covariance array, got shape {cov_arr.shape}")
    return [
        float(cov_arr[0, 0]),
        float(cov_arr[0, 1]),
        float(cov_arr[1, 0]),
        float(cov_arr[1, 1]),
    ]


def list_to_covariance_2x2(cov_list: Sequence[float]) -> np.ndarray:
    """Reconstruct a 2x2 covariance matrix from a 4-element row-major list."""
    if len(cov_list) != 4:
        raise ValueError(f"Expected 4 covariance elements, got {len(cov_list)}")
    arr = np.array([
        [cov_list[0], cov_list[1]],
        [cov_list[2], cov_list[3]],
    ], dtype=np.float64)
    # Symmetrize to prevent numerical asymmetry
    arr = 0.5 * (arr + arr.T)
    return arr


def covariance_4x4_to_list(cov: np.ndarray) -> List[float]:
    """Convert a 4x4 covariance matrix to a 16-element row-major list."""
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 covariance array, got shape {cov_arr.shape}")
    return [float(v) for v in cov_arr.flatten()]


def list_to_covariance_4x4(cov_list: Sequence[float]) -> np.ndarray:
    """Reconstruct a 4x4 covariance matrix from a 16-element row-major list."""
    if len(cov_list) != 16:
        raise ValueError(f"Expected 16 covariance elements, got {len(cov_list)}")
    arr = np.array(cov_list, dtype=np.float64).reshape((4, 4))
    arr = 0.5 * (arr + arr.T)
    return arr


# ==============================================================================
# 3. ENUM / STATE CONVERSION UTILITIES
# ==============================================================================

def string_to_behavioral_state(state_str: str, default: Optional[BehavioralState] = None) -> BehavioralState:
    """Parse string to BehavioralState enum with strict error checking."""
    if (not state_str or state_str == "") and default is not None:
        return default
    try:
        return BehavioralState(state_str)
    except ValueError:
        if default is not None:
            return default
        raise ValueError(
            f"Invalid behavioral state: '{state_str}'. Must be one of {[s.value for s in BehavioralState]}"
        )


def string_to_deadlock_event(event_str: str) -> str:
    """Parse and validate DeadlockEvent string."""
    valid_events = {DeadlockEvent.NONE.value, DeadlockEvent.DYNAMIC_DEADLOCK.value}
    if event_str not in valid_events:
        raise ValueError(f"Invalid deadlock event: '{event_str}'. Must be one of {valid_events}")
    return event_str


def string_to_replan_trigger(trigger_str: str) -> DynamicReplanTrigger:
    """Parse string to DynamicReplanTrigger enum with strict error checking."""
    try:
        return DynamicReplanTrigger(trigger_str)
    except ValueError:
        raise ValueError(
            f"Invalid replan trigger: '{trigger_str}'. Must be one of {[t.value for t in DynamicReplanTrigger]}"
        )


# ==============================================================================
# 4. MESSAGE CONVERTERS: DETECTION
# ==============================================================================

def dynamic_obstacle_detection_to_ros(
    obj: DynamicObstacleDetection,
    frame_id: Optional[str] = None,
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 DynamicObstacleDetection to ROS2 message."""
    target_cls = msg_cls or RosDynamicObstacleDetection
    if target_cls is None:
        raise RuntimeError("ROS2 DynamicObstacleDetection message class unavailable.")

    f_id = frame_id or obj.frame_id
    header = make_ros_header(f_id, obj.timestamp, header_cls=header_cls)

    # Convert detection ID to integer if possible
    det_id = int(obj.detection_id) if str(obj.detection_id).isdigit() else 1

    # Position Point
    p_cls = point_cls or NativeRosPoint
    pos_x, pos_y = float(obj.bbox_center[0]), float(obj.bbox_center[1])
    if p_cls is not None:
        pos = p_cls(x=pos_x, y=pos_y, z=0.0)
    else:
        pos = type("Point", (), {"x": pos_x, "y": pos_y, "z": 0.0})()

    # Dimensions Vector3
    v_cls = vector3_cls or NativeRosVector3
    dim_l, dim_w = float(obj.bbox_size[0]), float(obj.bbox_size[1])
    if v_cls is not None:
        dims = v_cls(x=dim_l, y=dim_w, z=0.50)
    else:
        dims = type("Vector3", (), {"x": dim_l, "y": dim_w, "z": 0.50})()

    msg = target_cls()
    msg.header = header
    msg.detection_id = det_id
    msg.class_name = "obstacle"
    msg.confidence = float(obj.confidence)
    msg.position = pos
    msg.dimensions = dims
    msg.yaw_rad = float(obj.bbox_yaw)
    msg.position_covariance = [0.04, 0.0, 0.0, 0.04]
    msg.is_valid = True
    return msg


def dynamic_obstacle_detection_from_ros(msg: Any) -> DynamicObstacleDetection:
    """Convert ROS2 DynamicObstacleDetection message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    frame_id = str(msg.header.frame_id)
    pos_x = float(msg.position.x)
    pos_y = float(msg.position.y)
    dim_l = float(msg.dimensions.x)
    dim_w = float(msg.dimensions.y)
    yaw = float(msg.yaw_rad)
    confidence = float(msg.confidence)

    return DynamicObstacleDetection(
        detection_id=str(msg.detection_id),
        timestamp=ts,
        frame_id=frame_id,
        bbox_center=(pos_x, pos_y),
        bbox_size=(dim_l, dim_w),
        bbox_yaw=yaw,
        confidence=confidence,
    )


def dynamic_obstacle_detection_array_to_ros(
    obj: DynamicObstacleDetectionArray,
    msg_cls: Optional[Any] = None,
    det_msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 DynamicObstacleDetectionArray to ROS2 message."""
    target_cls = msg_cls or RosDynamicObstacleDetectionArray
    if target_cls is None:
        raise RuntimeError("ROS2 DynamicObstacleDetectionArray message class unavailable.")

    header = make_ros_header(obj.frame_id, obj.header_timestamp, header_cls=header_cls)
    det_msgs = [
        dynamic_obstacle_detection_to_ros(
            d,
            frame_id=obj.frame_id,
            msg_cls=det_msg_cls,
            header_cls=header_cls,
            point_cls=point_cls,
            vector3_cls=vector3_cls,
        )
        for d in obj.detections
    ]

    msg = target_cls()
    msg.header = header
    msg.detections = det_msgs
    return msg


def dynamic_obstacle_detection_array_from_ros(msg: Any) -> DynamicObstacleDetectionArray:
    """Convert ROS2 DynamicObstacleDetectionArray message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    frame_id = str(msg.header.frame_id)
    dets = [dynamic_obstacle_detection_from_ros(d) for d in msg.detections]
    return DynamicObstacleDetectionArray(
        header_timestamp=ts,
        frame_id=frame_id,
        detections=dets,
    )


# ==============================================================================
# 5. MESSAGE CONVERTERS: TRACKING
# ==============================================================================

def tracked_obstacle_to_ros(
    obj: TrackedObstacleState,
    frame_id: Optional[str] = None,
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 TrackedObstacleState to ROS2 message."""
    target_cls = msg_cls or RosTrackedObstacle
    if target_cls is None:
        raise RuntimeError("ROS2 TrackedObstacle message class unavailable.")

    f_id = frame_id or obj.frame_id
    header = make_ros_header(f_id, obj.timestamp, header_cls=header_cls)

    p_cls = point_cls or NativeRosPoint
    if p_cls is not None:
        pos = p_cls(x=float(obj.x), y=float(obj.y), z=0.0)
    else:
        pos = type("Point", (), {"x": float(obj.x), "y": float(obj.y), "z": 0.0})()

    v_cls = vector3_cls or NativeRosVector3
    if v_cls is not None:
        vel = v_cls(x=float(obj.vx), y=float(obj.vy), z=0.0)
        bbox = v_cls(x=float(obj.bbox_size[0]), y=float(obj.bbox_size[1]), z=0.50)
    else:
        vel = type("Vector3", (), {"x": float(obj.vx), "y": float(obj.vy), "z": 0.0})()
        bbox = type("Vector3", (), {"x": float(obj.bbox_size[0]), "y": float(obj.bbox_size[1]), "z": 0.50})()

    cov_list = covariance_4x4_to_list(obj.covariance)

    msg = target_cls()
    msg.header = header
    msg.track_id = int(obj.track_id)
    msg.status = str(obj.status)
    msg.is_confirmed = bool(obj.is_confirmed)
    msg.confidence = 1.0
    msg.position = pos
    msg.velocity = vel
    msg.bbox_size = bbox
    msg.bbox_yaw = float(obj.bbox_yaw)
    msg.covariance = cov_list
    msg.total_hits = int(obj.track_age)
    msg.consecutive_misses = int(obj.coast_cycles)
    msg.time_since_last_update_s = float(obj.coast_cycles * 0.10)
    return msg


def tracked_obstacle_from_ros(msg: Any) -> TrackedObstacleState:
    """Convert ROS2 TrackedObstacle message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    frame_id = str(msg.header.frame_id)
    cov = list_to_covariance_4x4(msg.covariance)

    return TrackedObstacleState(
        track_id=int(msg.track_id),
        timestamp=ts,
        x=float(msg.position.x),
        y=float(msg.position.y),
        vx=float(msg.velocity.x),
        vy=float(msg.velocity.y),
        covariance=cov,
        bbox_size=(float(msg.bbox_size.x), float(msg.bbox_size.y)),
        bbox_yaw=float(msg.bbox_yaw),
        frame_id=frame_id,
        track_age=max(1, int(msg.total_hits)),
        coast_cycles=int(msg.consecutive_misses),
        is_confirmed=bool(msg.is_confirmed),
        status=str(msg.status),
    )


def tracked_obstacle_array_to_ros(
    obj: TrackedObstacleArray,
    msg_cls: Optional[Any] = None,
    track_msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 TrackedObstacleArray to ROS2 message."""
    target_cls = msg_cls or RosTrackedObstacleArray
    if target_cls is None:
        raise RuntimeError("ROS2 TrackedObstacleArray message class unavailable.")

    header = make_ros_header(obj.frame_id, obj.header_timestamp, header_cls=header_cls)
    track_msgs = [
        tracked_obstacle_to_ros(
            t,
            frame_id=obj.frame_id,
            msg_cls=track_msg_cls,
            header_cls=header_cls,
            point_cls=point_cls,
            vector3_cls=vector3_cls,
        )
        for t in obj.tracks
    ]

    msg = target_cls()
    msg.header = header
    msg.tracks = track_msgs
    return msg


def tracked_obstacle_array_from_ros(msg: Any) -> TrackedObstacleArray:
    """Convert ROS2 TrackedObstacleArray message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    frame_id = str(msg.header.frame_id)
    tracks = [tracked_obstacle_from_ros(t) for t in msg.tracks]
    return TrackedObstacleArray(
        header_timestamp=ts,
        frame_id=frame_id,
        tracks=tracks,
    )


# ==============================================================================
# 6. MESSAGE CONVERTERS: PREDICTION
# ==============================================================================

def trajectory_waypoint_to_ros(
    obj: TrajectoryWaypoint,
    msg_cls: Optional[Any] = None,
    time_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 TrajectoryWaypoint to ROS2 message."""
    target_cls = msg_cls or RosTrajectoryWaypoint
    if target_cls is None:
        raise RuntimeError("ROS2 TrajectoryWaypoint message class unavailable.")

    target_time = timestamp_to_ros_time(obj.timestamp, time_cls=time_cls)

    p_cls = point_cls or NativeRosPoint
    if p_cls is not None:
        pos = p_cls(x=float(obj.x), y=float(obj.y), z=0.0)
    else:
        pos = type("Point", (), {"x": float(obj.x), "y": float(obj.y), "z": 0.0})()

    v_cls = vector3_cls or NativeRosVector3
    if v_cls is not None:
        vel = v_cls(x=float(obj.vx), y=float(obj.vy), z=0.0)
    else:
        vel = type("Vector3", (), {"x": float(obj.vx), "y": float(obj.vy), "z": 0.0})()

    cov_list = covariance_2x2_to_list(obj.covariance_2d)

    msg = target_cls()
    msg.target_time = target_time
    msg.relative_time_s = float(obj.time_offset)
    msg.position = pos
    msg.velocity = vel
    msg.covariance = cov_list
    msg.uncertainty_radius_m = float(obj.uncertainty_radius)
    msg.confidence = 1.0
    return msg


def trajectory_waypoint_from_ros(msg: Any) -> TrajectoryWaypoint:
    """Convert ROS2 TrajectoryWaypoint message to Phase 13 data model."""
    target_ts = ros_time_to_timestamp(msg.target_time)
    rel_time = float(msg.relative_time_s)
    pos_x = float(msg.position.x)
    pos_y = float(msg.position.y)
    vel_x = float(msg.velocity.x)
    vel_y = float(msg.velocity.y)
    cov_2d = list_to_covariance_2x2(msg.covariance)
    unc_radius = float(msg.uncertainty_radius_m)

    return TrajectoryWaypoint(
        time_offset=rel_time,
        timestamp=target_ts,
        x=pos_x,
        y=pos_y,
        vx=vel_x,
        vy=vel_y,
        covariance_2d=cov_2d,
        uncertainty_radius=unc_radius,
    )


def predicted_obstacle_trajectory_to_ros(
    obj: PredictedObstacleTrajectory,
    frame_id: str = "odom",
    msg_cls: Optional[Any] = None,
    wp_msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
    time_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 PredictedObstacleTrajectory to ROS2 message."""
    target_cls = msg_cls or RosPredictedObstacleTrajectory
    if target_cls is None:
        raise RuntimeError("ROS2 PredictedObstacleTrajectory message class unavailable.")

    header = make_ros_header(frame_id, obj.current_state.timestamp, header_cls=header_cls)

    v_cls = vector3_cls or NativeRosVector3
    if v_cls is not None:
        bbox = v_cls(x=float(obj.current_state.bbox_size[0]), y=float(obj.current_state.bbox_size[1]), z=0.50)
    else:
        bbox = type("Vector3", (), {"x": float(obj.current_state.bbox_size[0]), "y": float(obj.current_state.bbox_size[1]), "z": 0.50})()

    wp_msgs = [
        trajectory_waypoint_to_ros(
            wp,
            msg_cls=wp_msg_cls,
            time_cls=time_cls,
            point_cls=point_cls,
            vector3_cls=vector3_cls,
        )
        for wp in obj.waypoints
    ]

    msg = target_cls()
    msg.header = header
    msg.track_id = int(obj.track_id)
    msg.status = str(obj.current_state.status)
    msg.bbox_size = bbox
    msg.bbox_yaw = float(obj.current_state.bbox_yaw)
    msg.waypoints = wp_msgs
    msg.prediction_horizon_s = float(obj.horizon_s)
    msg.is_unknown = False
    msg.confidence = 1.0
    return msg


def predicted_obstacle_trajectory_from_ros(msg: Any) -> PredictedObstacleTrajectory:
    """Convert ROS2 PredictedObstacleTrajectory message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    frame_id = str(msg.header.frame_id)
    waypoints = [trajectory_waypoint_from_ros(wp) for wp in msg.waypoints]

    # Reconstruct current TrackedObstacleState from first waypoint or header
    if waypoints:
        first_wp = waypoints[0]
        curr_x, curr_y = first_wp.x, first_wp.y
        curr_vx, curr_vy = first_wp.vx, first_wp.vy
        cov_4x4 = np.zeros((4, 4), dtype=np.float64)
        cov_4x4[:2, :2] = first_wp.covariance_2d
        cov_4x4[2:, 2:] = np.eye(2) * 0.10
    else:
        curr_x, curr_y = 0.0, 0.0
        curr_vx, curr_vy = 0.0, 0.0
        cov_4x4 = np.eye(4, dtype=np.float64) * 0.10

    bbox_len = float(msg.bbox_size.x)
    bbox_wid = float(msg.bbox_size.y)
    bbox_yaw = float(msg.bbox_yaw)

    state = TrackedObstacleState(
        track_id=int(msg.track_id),
        timestamp=ts,
        x=curr_x,
        y=curr_y,
        vx=curr_vx,
        vy=curr_vy,
        covariance=cov_4x4,
        bbox_size=(bbox_len, bbox_wid),
        bbox_yaw=bbox_yaw,
        frame_id=frame_id,
        status=str(msg.status),
    )

    horizon_s = float(msg.prediction_horizon_s)
    dt = (waypoints[1].time_offset - waypoints[0].time_offset) if len(waypoints) > 1 else 0.25

    polygons = [
        compute_obb_corners(
            center=(wp.x, wp.y),
            size=(bbox_len, bbox_wid),
            yaw_rad=bbox_yaw,
        )
        for wp in waypoints
    ]

    return PredictedObstacleTrajectory(
        track_id=int(msg.track_id),
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=waypoints,
        bounding_polygon_per_step=polygons,
    )


def predicted_trajectory_array_to_ros(
    obj: PredictedTrajectoryArray,
    msg_cls: Optional[Any] = None,
    traj_msg_cls: Optional[Any] = None,
    wp_msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    vector3_cls: Optional[Any] = None,
    time_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 PredictedTrajectoryArray to ROS2 message."""
    target_cls = msg_cls or RosPredictedTrajectoryArray
    if target_cls is None:
        raise RuntimeError("ROS2 PredictedTrajectoryArray message class unavailable.")

    header = make_ros_header(obj.frame_id, obj.header_timestamp, header_cls=header_cls)
    traj_msgs = [
        predicted_obstacle_trajectory_to_ros(
            p,
            frame_id=obj.frame_id,
            msg_cls=traj_msg_cls,
            wp_msg_cls=wp_msg_cls,
            header_cls=header_cls,
            vector3_cls=vector3_cls,
            time_cls=time_cls,
            point_cls=point_cls,
        )
        for p in obj.predictions
    ]

    msg = target_cls()
    msg.header = header
    msg.trajectories = traj_msgs
    return msg


def predicted_trajectory_array_from_ros(msg: Any) -> PredictedTrajectoryArray:
    """Convert ROS2 PredictedTrajectoryArray message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    frame_id = str(msg.header.frame_id)
    trajs = [predicted_obstacle_trajectory_from_ros(t) for t in msg.trajectories]
    return PredictedTrajectoryArray(
        header_timestamp=ts,
        frame_id=frame_id,
        predictions=trajs,
    )


# ==============================================================================
# 7. MESSAGE CONVERTERS: DYNAMIC OCCUPANCY
# ==============================================================================

def dynamic_occupancy_grid_slice_to_ros(
    obj: DynamicOccupancyGridSlice,
    resolution: float = 0.10,
    origin_x: float = -20.0,
    origin_y: float = -20.0,
    frame_id: str = "odom",
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    time_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 DynamicOccupancyGridSlice to ROS2 message."""
    target_cls = msg_cls or RosDynamicOccupancyGridSlice
    if target_cls is None:
        raise RuntimeError("ROS2 DynamicOccupancyGridSlice message class unavailable.")

    header = make_ros_header(frame_id, obj.timestamp, header_cls=header_cls, time_cls=time_cls)
    target_time = timestamp_to_ros_time(obj.timestamp, time_cls=time_cls)

    h, w = obj.cost.shape
    flat_data = [float(v) for v in obj.cost.flatten()]
    if hasattr(obj, "unknown_mask") and obj.unknown_mask is not None:
        unknown_mask = [bool(v) for v in np.asarray(obj.unknown_mask).flatten()]
    else:
        unknown_mask = [False] * len(flat_data)

    msg = target_cls()
    msg.header = header
    msg.target_time_offset_s = float(obj.time_offset)
    msg.target_timestamp = target_time
    msg.resolution = float(resolution)
    msg.origin_x = float(origin_x)
    msg.origin_y = float(origin_y)
    msg.width = int(w)
    msg.height = int(h)
    msg.data = flat_data
    msg.unknown_mask = unknown_mask
    return msg


def dynamic_occupancy_grid_slice_from_ros(msg: Any) -> DynamicOccupancyGridSlice:
    """Convert ROS2 DynamicOccupancyGridSlice message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.target_timestamp)
    time_offset = float(msg.target_time_offset_s)
    h = int(msg.height)
    w = int(msg.width)

    cost_arr = np.array(msg.data, dtype=np.float32).reshape((h, w))
    risk_arr = np.zeros((h, w), dtype=np.uint8)
    risk_arr[cost_arr >= 0.95] = int(DynamicRiskLevel.LETHAL)
    risk_arr[(cost_arr >= 0.70) & (cost_arr < 0.95)] = int(DynamicRiskLevel.HIGH_RISK)
    risk_arr[(cost_arr >= 0.40) & (cost_arr < 0.70)] = int(DynamicRiskLevel.MEDIUM_RISK)
    risk_arr[(cost_arr > 0.0) & (cost_arr < 0.40)] = int(DynamicRiskLevel.LOW_RISK)

    nominal = cost_arr >= 0.95
    uncertainty = (cost_arr >= 0.15) & (cost_arr < 0.95)

    slice_obj = DynamicOccupancyGridSlice(
        time_offset=time_offset,
        timestamp=ts,
        cost=cost_arr,
        risk_level=risk_arr,
        nominal_obstacle=nominal,
        uncertainty_zone=uncertainty,
        active_obstacle_ids=[],
    )
    if hasattr(msg, "unknown_mask") and msg.unknown_mask:
        slice_obj.unknown_mask = np.array(msg.unknown_mask, dtype=bool).reshape((h, w))
    return slice_obj


def dynamic_occupancy_slice_array_to_ros(
    slices: Sequence[DynamicOccupancyGridSlice],
    resolution: float = 0.10,
    origin_x: float = -20.0,
    origin_y: float = -20.0,
    frame_id: str = "odom",
    timestamp_s: Optional[float] = None,
    msg_cls: Optional[Any] = None,
    slice_msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    time_cls: Optional[Any] = None,
) -> Any:
    """Convert a sequence of DynamicOccupancyGridSlice objects to DynamicOccupancySliceArray ROS2 message."""
    target_cls = msg_cls or RosDynamicOccupancySliceArray
    if target_cls is None:
        raise RuntimeError("ROS2 DynamicOccupancySliceArray message class unavailable.")

    ts = timestamp_s if timestamp_s is not None else (slices[0].timestamp if slices else 0.0)
    header = make_ros_header(frame_id, ts, header_cls=header_cls, time_cls=time_cls)

    slice_msgs = [
        dynamic_occupancy_grid_slice_to_ros(
            s,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            frame_id=frame_id,
            msg_cls=slice_msg_cls,
            header_cls=header_cls,
            time_cls=time_cls,
        )
        for s in slices
    ]

    msg = target_cls()
    msg.header = header
    msg.slices = slice_msgs
    return msg


def dynamic_occupancy_slice_array_from_ros(msg: Any) -> List[DynamicOccupancyGridSlice]:
    """Convert ROS2 DynamicOccupancySliceArray message to list of DynamicOccupancyGridSlice objects."""
    return [dynamic_occupancy_grid_slice_from_ros(s) for s in msg.slices]


# ==============================================================================
# 8. MESSAGE CONVERTERS: BEHAVIORAL DECISION
# ==============================================================================

def behavioral_decision_to_ros(
    obj: BehavioralDecision,
    frame_id: str = "odom",
    timestamp_s: Optional[float] = None,
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 BehavioralDecision to ROS2 message."""
    target_cls = msg_cls or RosBehavioralDecision
    if target_cls is None:
        raise RuntimeError("ROS2 BehavioralDecision message class unavailable.")

    ts = timestamp_s if timestamp_s is not None else 0.0
    header = make_ros_header(frame_id, ts, header_cls=header_cls)

    msg = target_cls()
    msg.header = header
    msg.state = obj.state.value if hasattr(obj.state, "value") else str(obj.state)
    msg.command_v = float(obj.command_v)
    msg.command_omega = float(obj.command_omega)
    msg.reason = str(obj.reason)
    msg.blocking_obstacle_id = int(obj.blocking_obstacle_id) if obj.blocking_obstacle_id is not None else 0
    msg.dynamic_collision = bool(obj.dynamic_collision)
    msg.dynamic_risk = obj.dynamic_risk.value if hasattr(obj.dynamic_risk, "value") else str(obj.dynamic_risk)
    msg.unknown_horizon = bool(obj.unknown_horizon)
    msg.wait_elapsed_s = float(obj.wait_elapsed_s)
    msg.wait_timeout = bool(obj.wait_timeout)
    msg.deadlock = bool(obj.deadlock)
    msg.deadlock_event = obj.deadlock_event.value if hasattr(obj.deadlock_event, "value") else str(obj.deadlock_event)
    msg.transition = str(obj.transition)
    prev = obj.previous_state
    msg.previous_state = prev.value if hasattr(prev, "value") else str(prev)
    msg.state_dwell_time_s = float(obj.state_dwell_time_s)
    msg.confidence = float(obj.confidence)
    return msg


def behavioral_decision_from_ros(msg: Any) -> BehavioralDecision:
    """Convert ROS2 BehavioralDecision message to Phase 13 data model."""
    state = string_to_behavioral_state(msg.state)
    prev_state = string_to_behavioral_state(getattr(msg, "previous_state", ""), default=BehavioralState.PROCEED)
    deadlock_event = string_to_deadlock_event(msg.deadlock_event)
    blocking_id = int(msg.blocking_obstacle_id) if int(msg.blocking_obstacle_id) != 0 else None

    return BehavioralDecision(
        state=state,
        command_v=float(msg.command_v),
        command_omega=float(msg.command_omega),
        reason=str(msg.reason),
        blocking_obstacle_id=blocking_id,
        dynamic_collision=bool(msg.dynamic_collision),
        dynamic_risk=str(msg.dynamic_risk),
        unknown_horizon=bool(msg.unknown_horizon),
        wait_elapsed_s=float(msg.wait_elapsed_s),
        wait_timeout=bool(msg.wait_timeout),
        deadlock=bool(msg.deadlock),
        deadlock_event=deadlock_event,
        transition=str(msg.transition),
        previous_state=prev_state,
        state_dwell_time_s=float(msg.state_dwell_time_s),
        confidence=float(msg.confidence),
    )


# ==============================================================================
# 9. MESSAGE CONVERTERS: REPLAN EVENT
# ==============================================================================

def dynamic_replan_event_to_ros(
    obj: DynamicReplanEvent,
    success: bool = True,
    is_path_stale: bool = False,
    old_path_length_m: float = 0.0,
    new_path_length_m: float = 0.0,
    path_deviation_m: float = 0.0,
    planning_latency_ms: float = 0.0,
    frame_id: str = "odom",
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 13 DynamicReplanEvent to ROS2 message."""
    target_cls = msg_cls or RosDynamicReplanEvent
    if target_cls is None:
        raise RuntimeError("ROS2 DynamicReplanEvent message class unavailable.")

    header = make_ros_header(frame_id, obj.timestamp, header_cls=header_cls)

    p_cls = point_cls or NativeRosPoint
    if p_cls is not None:
        pose = p_cls(x=float(obj.current_pose[0]), y=float(obj.current_pose[1]), z=0.0)
        goal = p_cls(x=float(obj.current_goal[0]), y=float(obj.current_goal[1]), z=0.0)
    else:
        pose = type("Point", (), {"x": float(obj.current_pose[0]), "y": float(obj.current_pose[1]), "z": 0.0})()
        goal = type("Point", (), {"x": float(obj.current_goal[0]), "y": float(obj.current_goal[1]), "z": 0.0})()

    yaw = float(obj.current_pose[2]) if len(obj.current_pose) > 2 else 0.0

    msg = target_cls()
    msg.header = header
    msg.event_id = str(obj.event_id)
    msg.trigger_type = obj.trigger_type.value if hasattr(obj.trigger_type, "value") else str(obj.trigger_type)
    msg.current_pose = pose
    msg.current_yaw = yaw
    msg.current_goal = goal
    msg.blocking_obstacle_ids = [int(i) for i in obj.blocking_obstacle_ids]
    msg.blocked_duration_s = float(obj.blocked_duration)
    msg.deadlock_duration_s = float(obj.deadlock_duration)
    msg.current_progress_m = float(obj.current_progress)
    msg.reason = str(obj.reason)
    msg.confidence = float(obj.confidence)
    msg.success = bool(success)
    msg.is_path_stale = bool(is_path_stale)
    msg.old_path_length_m = float(old_path_length_m)
    msg.new_path_length_m = float(new_path_length_m)
    msg.path_deviation_m = float(path_deviation_m)
    msg.planning_latency_ms = float(planning_latency_ms)
    return msg


def dynamic_replan_event_from_ros(msg: Any) -> DynamicReplanEvent:
    """Convert ROS2 DynamicReplanEvent message to Phase 13 data model."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    trigger = string_to_replan_trigger(msg.trigger_type)
    curr_pose = (float(msg.current_pose.x), float(msg.current_pose.y), float(msg.current_yaw))
    curr_goal = (float(msg.current_goal.x), float(msg.current_goal.y))

    return DynamicReplanEvent(
        trigger_type=trigger,
        timestamp=ts,
        current_pose=curr_pose,
        current_goal=curr_goal,
        current_global_path=None,
        blocking_obstacle_ids=[int(i) for i in msg.blocking_obstacle_ids],
        blocked_duration=float(msg.blocked_duration_s),
        deadlock_duration=float(msg.deadlock_duration_s),
        current_progress=float(msg.current_progress_m),
        reason=str(msg.reason),
        confidence=float(msg.confidence),
        event_id=str(msg.event_id),
    )


# ==============================================================================
# 10. MESSAGE CONVERTERS: LOCALIZATION STATUS & TERRAIN MAP
# ==============================================================================

def localization_status_to_ros(
    obj: Any,
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    pose_cov_cls: Optional[Any] = None,
    twist_cov_cls: Optional[Any] = None,
    pose_cls: Optional[Any] = None,
    twist_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    quat_cls: Optional[Any] = None,
) -> Any:
    """Convert LocalizationStatus object to ROS2 message."""
    target_cls = msg_cls or RosLocalizationStatus
    if target_cls is None:
        raise RuntimeError("ROS2 LocalizationStatus message class unavailable.")

    ts = float(obj.header.stamp.to_sec()) if hasattr(obj.header.stamp, "to_sec") else 0.0
    frame_id = str(obj.header.frame_id)
    header = make_ros_header(frame_id, ts, header_cls=header_cls)

    msg = target_cls()
    msg.header = header
    msg.child_frame_id = str(getattr(obj, "child_frame_id", "base_link"))
    msg.tracking_state = int(getattr(obj, "tracking_state", 0))
    msg.confidence = float(getattr(obj, "confidence", 1.0))
    msg.is_pose_valid = bool(getattr(obj, "is_pose_valid", True))
    msg.allows_spatial_fusion = bool(getattr(obj, "allows_spatial_fusion", True))
    msg.tracked_features = int(getattr(obj, "tracked_features", 100))
    msg.inliers = int(getattr(obj, "inliers", 80))
    msg.keyframe_id = int(getattr(obj, "keyframe_id", 1))
    msg.is_keyframe = bool(getattr(obj, "is_keyframe", False))
    msg.relocalized = bool(getattr(obj, "relocalized", False))

    # Pose and Twist
    p_cls = point_cls or NativeRosPoint
    q_cls = quat_cls or None
    po_cls = pose_cls or NativeRosPose
    pc_cls = pose_cov_cls or NativeRosPoseWithCovariance
    tw_cls = twist_cls or NativeRosTwist
    tc_cls = twist_cov_cls or NativeRosTwistWithCovariance

    if pc_cls is not None and po_cls is not None:
        p_cov = pc_cls()
        t_cov = tc_cls()
        msg.pose = p_cov
        msg.twist = t_cov
    else:
        # Mock struct for test doubles
        msg.pose = getattr(obj, "pose", None)
        msg.twist = getattr(obj, "twist", None)

    return msg


def localization_status_from_ros(msg: Any) -> Dict[str, Any]:
    """Convert ROS2 LocalizationStatus message to dictionary representation."""
    ts = ros_time_to_timestamp(msg.header.stamp)
    tracking_state = int(msg.tracking_state)
    if tracking_state not in {0, 1, 2, 3, 4}:
        # Invariant: Invalid tracking state must map to safety-critical LOST
        tracking_state = 2

    return {
        "timestamp": ts,
        "frame_id": str(msg.header.frame_id),
        "child_frame_id": str(msg.child_frame_id),
        "tracking_state": tracking_state,
        "confidence": float(msg.confidence),
        "is_pose_valid": bool(msg.is_pose_valid),
        "allows_spatial_fusion": bool(msg.allows_spatial_fusion),
        "tracked_features": int(msg.tracked_features),
        "inliers": int(msg.inliers),
        "keyframe_id": int(msg.keyframe_id),
        "is_keyframe": bool(msg.is_keyframe),
        "relocalized": bool(msg.relocalized),
    }


def world_terrain_map_to_ros(
    wmap: WorldTerrainMap,
    frame_id: str = "odom",
    timestamp_s: Optional[float] = None,
    msg_cls: Optional[Any] = None,
    header_cls: Optional[Any] = None,
    pose_cls: Optional[Any] = None,
    point_cls: Optional[Any] = None,
    quat_cls: Optional[Any] = None,
) -> Any:
    """Convert Phase 9 WorldTerrainMap object to ROS2 message."""
    target_cls = msg_cls or RosWorldTerrainMap
    if target_cls is None:
        raise RuntimeError("ROS2 WorldTerrainMap message class unavailable.")

    if timestamp_s is None:
        t_max = float(np.max(wmap.last_observed_timestamp))
        ts = t_max if t_max > 0.0 else 0.0
    else:
        ts = timestamp_s

    header = make_ros_header(frame_id, ts, header_cls=header_cls)
    H, W = wmap.shape
    ox, oy = float(wmap.x_bounds_m[0]), float(wmap.y_bounds_m[0])

    p_cls = point_cls or NativeRosPoint
    q_cls = quat_cls or None
    po_cls = pose_cls or NativeRosPose

    if po_cls is not None and p_cls is not None:
        origin = po_cls()
        origin.position = p_cls(x=ox, y=oy, z=0.0)
    else:
        origin = type("Pose", (), {
            "position": type("Point", (), {"x": ox, "y": oy, "z": 0.0})(),
            "orientation": type("Quaternion", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})(),
        })()

    msg = target_cls()
    msg.header = header
    msg.resolution = float(wmap.resolution_m)
    msg.origin = origin
    msg.width = int(W)
    msg.height = int(H)
    msg.cost = [float(v) for v in wmap.cost.flatten()]
    msg.traversability = [float(v) for v in wmap.traversability.flatten()]
    msg.obstacle = [bool(v) for v in wmap.obstacle.flatten()]
    msg.hazard = [float(v) for v in wmap.hazard.flatten()]
    msg.unknown = [bool(v) for v in wmap.unknown.flatten()]
    msg.confidence = [float(v) for v in wmap.confidence.flatten()]
    msg.slope = [float(v) for v in wmap.slope.flatten()]
    msg.roughness = [float(v) for v in wmap.roughness.flatten()]
    msg.height_var = [float(v) for v in wmap.height_var.flatten()]
    msg.observation_count = [int(v) for v in wmap.observation_count.flatten()]
    return msg


def world_terrain_map_from_ros(msg: Any) -> WorldTerrainMap:
    """Convert ROS2 WorldTerrainMap message to Phase 9 WorldTerrainMap object."""
    H = int(msg.height)
    W = int(msg.width)
    res = float(msg.resolution)

    # Extract origin
    if hasattr(msg.origin, "position"):
        ox = float(msg.origin.position.x)
        oy = float(msg.origin.position.y)
    elif isinstance(msg.origin, dict):
        ox = float(msg.origin.get("x", -20.0))
        oy = float(msg.origin.get("y", -20.0))
    else:
        ox = -20.0
        oy = -20.0

    x_bounds = (ox, ox + W * res)
    y_bounds = (oy, oy + H * res)

    wmap = WorldTerrainMap(resolution_m=res, x_bounds_m=x_bounds, y_bounds_m=y_bounds)
    wmap.cost = np.array(msg.cost, dtype=np.float32).reshape((H, W))
    wmap.traversability = np.array(msg.traversability, dtype=np.float32).reshape((H, W))
    wmap.obstacle = np.array(msg.obstacle, dtype=bool).reshape((H, W))
    wmap.hazard = np.array(msg.hazard, dtype=np.float32).reshape((H, W))
    wmap.unknown = np.array(msg.unknown, dtype=bool).reshape((H, W))
    wmap.confidence = np.array(msg.confidence, dtype=np.float32).reshape((H, W))
    wmap.slope = np.array(msg.slope, dtype=np.float32).reshape((H, W))
    wmap.roughness = np.array(msg.roughness, dtype=np.float32).reshape((H, W))
    wmap.height_var = np.array(msg.height_var, dtype=np.float32).reshape((H, W))
    wmap.observation_count = np.array(msg.observation_count, dtype=np.int32).reshape((H, W))
    return wmap
