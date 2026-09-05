"""Python dataclasses and serialization codecs for Phase 13 Dynamic Obstacle ROS2 messages.

Supports dual-mode execution:
1. Standard ROS2 message transport when rclpy/ros2 is available.
2. Fast, standalone in-memory message transport and verification in pure Python environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ai.planning.base import PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import BehavioralDecision, BehavioralState, BehavioralStateMachine, DeadlockEvent
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
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
    PathChangeMetrics,
)
from ai.planning.path import Path, PathPoint
from ugv_navigation.utils.time_utils import RosHeader, RosTime


# ==============================================================================
# Header Codec Utilities (do not assume RosHeader has .to_dict() / .from_dict())
# ==============================================================================

def serialize_ros_header(header: RosHeader) -> Dict[str, Any]:
    """Serialize RosHeader to dictionary using its concrete fields."""
    return {
        "stamp": {
            "sec": int(header.stamp.sec),
            "nanosec": int(header.stamp.nanosec),
        },
        "frame_id": str(header.frame_id),
    }


def deserialize_ros_header(d: Dict[str, Any]) -> RosHeader:
    """Deserialize RosHeader from dictionary using concrete fields."""
    stamp_d = d.get("stamp", {}) if isinstance(d, dict) else {}
    sec = int(stamp_d.get("sec", 0))
    nanosec = int(stamp_d.get("nanosec", 0))
    frame_id = str(d.get("frame_id", "")) if isinstance(d, dict) else ""
    return RosHeader(stamp=RosTime(sec=sec, nanosec=nanosec), frame_id=frame_id)


# Provide backwards-compatible methods on RosHeader if absent in environment
if not hasattr(RosHeader, "to_dict"):
    RosHeader.to_dict = serialize_ros_header  # type: ignore[attr-defined]

if not hasattr(RosHeader, "from_dict"):
    RosHeader.from_dict = classmethod(lambda cls, d: deserialize_ros_header(d))  # type: ignore[attr-defined]


# ==============================================================================
# 1. DYNAMIC OBSTACLE DETECTION MESSAGES
# ==============================================================================

@dataclass
class DynamicObstacleDetectionMsg:
    """In-memory representation of DynamicObstacleDetection.msg."""

    header: RosHeader
    detection_id: int
    class_name: str
    confidence: float
    position: Tuple[float, float, float]       # (x, y, z)
    dimensions: Tuple[float, float, float]     # (length, width, height)
    yaw_rad: float
    position_covariance: List[float] = field(default_factory=lambda: [0.04, 0.0, 0.0, 0.04])
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "detection_id": int(self.detection_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "position": [round(float(v), 4) for v in self.position],
            "dimensions": [round(float(v), 4) for v in self.dimensions],
            "yaw_rad": round(float(self.yaw_rad), 4),
            "position_covariance": [round(float(v), 6) for v in self.position_covariance],
            "is_valid": bool(self.is_valid),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicObstacleDetectionMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            detection_id=int(d["detection_id"]),
            class_name=str(d.get("class_name", "obstacle")),
            confidence=float(d.get("confidence", 1.0)),
            position=tuple(float(v) for v in d["position"]),
            dimensions=tuple(float(v) for v in d["dimensions"]),
            yaw_rad=float(d.get("yaw_rad", 0.0)),
            position_covariance=[float(v) for v in d.get("position_covariance", [0.04, 0.0, 0.0, 0.04])],
            is_valid=bool(d.get("is_valid", True)),
        )

    @classmethod
    def from_phase13(cls, det: DynamicObstacleDetection, frame_id: str = "odom") -> DynamicObstacleDetectionMsg:
        header = RosHeader.create(frame_id=frame_id, timestamp_s=float(det.timestamp))
        cov_list = [0.04, 0.0, 0.0, 0.04]
        return cls(
            header=header,
            detection_id=int(det.detection_id) if str(det.detection_id).isdigit() else 1,
            class_name="obstacle",
            confidence=float(det.confidence),
            position=(float(det.bbox_center[0]), float(det.bbox_center[1]), 0.0),
            dimensions=(float(det.bbox_size[0]), float(det.bbox_size[1]), 0.50),
            yaw_rad=float(det.bbox_yaw),
            position_covariance=cov_list,
            is_valid=True,
        )

    def to_phase13(self) -> DynamicObstacleDetection:
        return DynamicObstacleDetection(
            detection_id=str(self.detection_id),
            timestamp=float(self.header.stamp.to_sec()),
            frame_id=self.header.frame_id,
            bbox_center=(float(self.position[0]), float(self.position[1])),
            bbox_size=(float(self.dimensions[0]), float(self.dimensions[1])),
            bbox_yaw=float(self.yaw_rad),
            confidence=float(self.confidence),
        )



@dataclass
class DynamicObstacleDetectionArrayMsg:
    """In-memory representation of DynamicObstacleDetectionArray.msg."""

    header: RosHeader
    detections: List[DynamicObstacleDetectionMsg] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicObstacleDetectionArrayMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            detections=[DynamicObstacleDetectionMsg.from_dict(item) for item in d.get("detections", [])],
        )

    @classmethod
    def from_phase13(cls, arr: DynamicObstacleDetectionArray, frame_id: str = "odom") -> DynamicObstacleDetectionArrayMsg:
        ts = getattr(arr, "header_timestamp", getattr(arr, "timestamp", 0.0))
        f_id = getattr(arr, "frame_id", frame_id)
        header = RosHeader.create(frame_id=f_id, timestamp_s=float(ts))
        dets = [DynamicObstacleDetectionMsg.from_phase13(d, frame_id=f_id) for d in arr.detections]
        return cls(header=header, detections=dets)

    def to_phase13(self) -> DynamicObstacleDetectionArray:
        dets = [d.to_phase13() for d in self.detections]
        return DynamicObstacleDetectionArray(
            header_timestamp=float(self.header.stamp.to_sec()),
            frame_id=self.header.frame_id,
            detections=dets,
        )


# ==============================================================================
# 2. TRACKED OBSTACLE MESSAGES
# ==============================================================================

@dataclass
class TrackedObstacleMsg:
    """In-memory representation of TrackedObstacle.msg."""

    header: RosHeader
    track_id: int
    status: str
    is_confirmed: bool
    confidence: float
    position: Tuple[float, float, float]       # (x, y, z)
    velocity: Tuple[float, float, float]       # (vx, vy, vz)
    bbox_size: Tuple[float, float, float]      # (length, width, height)
    bbox_yaw: float
    covariance: List[float] = field(default_factory=lambda: [0.0] * 16)
    total_hits: int = 1
    consecutive_misses: int = 0
    time_since_last_update_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "track_id": int(self.track_id),
            "status": str(self.status),
            "is_confirmed": bool(self.is_confirmed),
            "confidence": round(float(self.confidence), 4),
            "position": [round(float(v), 4) for v in self.position],
            "velocity": [round(float(v), 4) for v in self.velocity],
            "bbox_size": [round(float(v), 4) for v in self.bbox_size],
            "bbox_yaw": round(float(self.bbox_yaw), 4),
            "covariance": [round(float(v), 6) for v in self.covariance],
            "total_hits": int(self.total_hits),
            "consecutive_misses": int(self.consecutive_misses),
            "time_since_last_update_s": round(float(self.time_since_last_update_s), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrackedObstacleMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            track_id=int(d["track_id"]),
            status=str(d.get("status", "TRACKING")),
            is_confirmed=bool(d.get("is_confirmed", True)),
            confidence=float(d.get("confidence", 1.0)),
            position=tuple(float(v) for v in d["position"]),
            velocity=tuple(float(v) for v in d["velocity"]),
            bbox_size=tuple(float(v) for v in d["bbox_size"]),
            bbox_yaw=float(d.get("bbox_yaw", 0.0)),
            covariance=[float(v) for v in d.get("covariance", [0.0] * 16)],
            total_hits=int(d.get("total_hits", 1)),
            consecutive_misses=int(d.get("consecutive_misses", 0)),
            time_since_last_update_s=float(d.get("time_since_last_update_s", 0.0)),
        )

    @classmethod
    def from_phase13(cls, track: TrackedObstacleState, frame_id: str = "odom") -> TrackedObstacleMsg:
        header = RosHeader.create(frame_id=frame_id, timestamp_s=float(track.timestamp))
        cov_list = track.covariance.flatten().tolist() if track.covariance is not None else [0.0] * 16
        if len(cov_list) < 16:
            cov_16 = [0.0] * 16
            for i, v in enumerate(cov_list):
                cov_16[i] = v
            cov_list = cov_16

        return cls(
            header=header,
            track_id=int(track.track_id),
            status=track.status.value if hasattr(track.status, "value") else str(track.status),
            is_confirmed=bool(track.is_confirmed),
            confidence=1.0,
            position=(float(track.x), float(track.y), 0.0),
            velocity=(float(track.vx), float(track.vy), 0.0),
            bbox_size=(float(track.bbox_size[0]), float(track.bbox_size[1]), 0.50),
            bbox_yaw=float(track.bbox_yaw),
            covariance=cov_list[:16],
            total_hits=int(getattr(track, "track_age", 1)),
            consecutive_misses=int(getattr(track, "coast_cycles", 0)),
            time_since_last_update_s=0.0,
        )

    def to_phase13(self) -> TrackedObstacleState:
        cov_arr = np.array(self.covariance, dtype=np.float64)
        if cov_arr.size == 16:
            cov = cov_arr.reshape((4, 4))
        else:
            cov = np.eye(4, dtype=np.float64) * 0.04

        return TrackedObstacleState(
            track_id=int(self.track_id),
            timestamp=float(self.header.stamp.to_sec()),
            x=float(self.position[0]),
            y=float(self.position[1]),
            vx=float(self.velocity[0]),
            vy=float(self.velocity[1]),
            covariance=cov,
            bbox_size=(float(self.bbox_size[0]), float(self.bbox_size[1])),
            bbox_yaw=float(self.bbox_yaw),
            frame_id=self.header.frame_id,
            track_age=int(self.total_hits),
            coast_cycles=int(self.consecutive_misses),
            is_confirmed=bool(self.is_confirmed),
            status=self._parse_track_status(),
        )

    def _parse_track_status(self) -> str:
        st_val = str(self.status)
        if st_val.startswith("TrackStatus."):
            st_val = st_val.split(".", 1)[1]
        return st_val


@dataclass
class TrackedObstacleArrayMsg:
    """In-memory representation of TrackedObstacleArray.msg."""

    header: RosHeader
    tracks: List[TrackedObstacleMsg] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "tracks": [t.to_dict() for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrackedObstacleArrayMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            tracks=[TrackedObstacleMsg.from_dict(t) for t in d.get("tracks", [])],
        )

    @classmethod
    def from_phase13(cls, arr: TrackedObstacleArray, frame_id: str = "odom") -> TrackedObstacleArrayMsg:
        ts = getattr(arr, "header_timestamp", getattr(arr, "timestamp", 0.0))
        f_id = getattr(arr, "frame_id", frame_id)
        header = RosHeader.create(frame_id=f_id, timestamp_s=float(ts))
        tracks = [TrackedObstacleMsg.from_phase13(t, frame_id=f_id) for t in arr.tracks]
        return cls(header=header, tracks=tracks)

    def to_phase13(self) -> TrackedObstacleArray:
        tracks = [t.to_phase13() for t in self.tracks]
        return TrackedObstacleArray(
            header_timestamp=float(self.header.stamp.to_sec()),
            frame_id=self.header.frame_id,
            tracks=tracks,
        )


# ==============================================================================
# 3. PREDICTED TRAJECTORY MESSAGES
# ==============================================================================

@dataclass
class TrajectoryWaypointMsg:
    """In-memory representation of TrajectoryWaypoint.msg."""

    target_time_s: float
    relative_time_s: float
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    covariance: List[float] = field(default_factory=lambda: [0.04, 0.0, 0.0, 0.04])
    uncertainty_radius_m: float = 0.20
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_time_s": round(float(self.target_time_s), 4),
            "relative_time_s": round(float(self.relative_time_s), 4),
            "position": [round(float(v), 4) for v in self.position],
            "velocity": [round(float(v), 4) for v in self.velocity],
            "covariance": [round(float(v), 6) for v in self.covariance],
            "uncertainty_radius_m": round(float(self.uncertainty_radius_m), 4),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrajectoryWaypointMsg:
        return cls(
            target_time_s=float(d["target_time_s"]),
            relative_time_s=float(d["relative_time_s"]),
            position=tuple(float(v) for v in d["position"]),
            velocity=tuple(float(v) for v in d["velocity"]),
            covariance=[float(v) for v in d.get("covariance", [0.04, 0.0, 0.0, 0.04])],
            uncertainty_radius_m=float(d.get("uncertainty_radius_m", 0.20)),
            confidence=float(d.get("confidence", 1.0)),
        )

    @classmethod
    def from_phase13(cls, wp: TrajectoryWaypoint) -> TrajectoryWaypointMsg:
        cov_list = wp.covariance_2d.flatten().tolist() if hasattr(wp, "covariance_2d") and wp.covariance_2d is not None else [0.04, 0.0, 0.0, 0.04]
        return cls(
            target_time_s=float(wp.timestamp),
            relative_time_s=float(getattr(wp, "time_offset", 0.0)),
            position=(float(wp.x), float(wp.y), 0.0),
            velocity=(float(wp.vx), float(wp.vy), 0.0),
            covariance=cov_list[:4],
            uncertainty_radius_m=float(wp.uncertainty_radius),
            confidence=1.0,
        )

    def to_phase13(self) -> TrajectoryWaypoint:
        cov = np.array(self.covariance, dtype=np.float64).reshape((2, 2)) if len(self.covariance) == 4 else np.eye(2) * 0.04
        return TrajectoryWaypoint(
            time_offset=float(self.relative_time_s),
            timestamp=float(self.target_time_s),
            x=float(self.position[0]),
            y=float(self.position[1]),
            vx=float(self.velocity[0]),
            vy=float(self.velocity[1]),
            covariance_2d=cov,
            uncertainty_radius=float(self.uncertainty_radius_m),
        )


@dataclass
class PredictedObstacleTrajectoryMsg:
    """In-memory representation of PredictedObstacleTrajectory.msg."""

    header: RosHeader
    track_id: int
    status: str
    bbox_size: Tuple[float, float, float]
    bbox_yaw: float
    waypoints: List[TrajectoryWaypointMsg] = field(default_factory=list)
    prediction_horizon_s: float = 2.0
    is_unknown: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "track_id": int(self.track_id),
            "status": str(self.status),
            "bbox_size": [round(float(v), 4) for v in self.bbox_size],
            "bbox_yaw": round(float(self.bbox_yaw), 4),
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "prediction_horizon_s": round(float(self.prediction_horizon_s), 4),
            "is_unknown": bool(self.is_unknown),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictedObstacleTrajectoryMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            track_id=int(d["track_id"]),
            status=str(d.get("status", "TRACKING")),
            bbox_size=tuple(float(v) for v in d["bbox_size"]),
            bbox_yaw=float(d.get("bbox_yaw", 0.0)),
            waypoints=[TrajectoryWaypointMsg.from_dict(wp) for wp in d.get("waypoints", [])],
            prediction_horizon_s=float(d.get("prediction_horizon_s", 2.0)),
            is_unknown=bool(d.get("is_unknown", False)),
            confidence=float(d.get("confidence", 1.0)),
        )

    @classmethod
    def from_phase13(cls, traj: PredictedObstacleTrajectory, frame_id: str = "odom") -> PredictedObstacleTrajectoryMsg:
        t0 = traj.waypoints[0].timestamp if traj.waypoints else 0.0
        header = RosHeader.create(frame_id=frame_id, timestamp_s=float(t0))
        wps = [TrajectoryWaypointMsg.from_phase13(wp) for wp in traj.waypoints]
        h_s = float(traj.horizon_s) if hasattr(traj, "horizon_s") else (float(wps[-1].target_time_s - t0) if wps else 2.0)
        c_st = getattr(traj, "current_state", None)
        bbox = (float(c_st.bbox_size[0]), float(c_st.bbox_size[1]), 0.50) if c_st else (0.8, 0.6, 0.5)
        yaw = float(c_st.bbox_yaw) if c_st else 0.0
        st = str(c_st.status) if c_st else "TRACKING"

        return cls(
            header=header,
            track_id=int(traj.track_id),
            status=st,
            bbox_size=bbox,
            bbox_yaw=yaw,
            waypoints=wps,
            prediction_horizon_s=h_s,
            is_unknown=bool(getattr(traj, "is_unknown", False)),
            confidence=1.0,
        )

    def to_phase13(self) -> PredictedObstacleTrajectory:
        from ai.planning.dynamic.data_model import compute_obb_corners
        wps = [wp.to_phase13() for wp in self.waypoints]
        st = TrackedObstacleState(
            track_id=int(self.track_id),
            timestamp=float(self.header.stamp.to_sec()),
            x=float(wps[0].x) if wps else 0.0,
            y=float(wps[0].y) if wps else 0.0,
            vx=float(wps[0].vx) if wps else 0.0,
            vy=float(wps[0].vy) if wps else 0.0,
            covariance=np.eye(4) * 0.04,
            bbox_size=(float(self.bbox_size[0]), float(self.bbox_size[1])),
            bbox_yaw=float(self.bbox_yaw),
            frame_id=self.header.frame_id,
            status=str(self.status),
        )
        polygons = [
            compute_obb_corners(
                (wp.x, wp.y),
                (self.bbox_size[0], self.bbox_size[1]),
                self.bbox_yaw,
                margin_m=wp.uncertainty_radius,
            )
            for wp in wps
        ]
        eff_dt = float(wps[1].time_offset - wps[0].time_offset) if len(wps) > 1 else 0.25
        if eff_dt <= 0.0:
            eff_dt = 0.25
        traj = PredictedObstacleTrajectory(
            track_id=int(self.track_id),
            current_state=st,
            horizon_s=float(self.prediction_horizon_s),
            dt=eff_dt,
            waypoints=wps,
            bounding_polygon_per_step=polygons,
        )
        traj.is_unknown = bool(self.is_unknown)
        traj.bbox_yaw = float(self.bbox_yaw)
        traj.bbox_size = (float(self.bbox_size[0]), float(self.bbox_size[1]))
        return traj



@dataclass
class PredictedTrajectoryArrayMsg:
    """In-memory representation of PredictedTrajectoryArray.msg."""

    header: RosHeader
    trajectories: List[PredictedObstacleTrajectoryMsg] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "trajectories": [t.to_dict() for t in self.trajectories],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PredictedTrajectoryArrayMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            trajectories=[PredictedObstacleTrajectoryMsg.from_dict(t) for t in d.get("trajectories", [])],
        )

    @classmethod
    def from_phase13(cls, arr: PredictedTrajectoryArray, frame_id: str = "odom") -> PredictedTrajectoryArrayMsg:
        ts = getattr(arr, "header_timestamp", getattr(arr, "timestamp", 0.0))
        f_id = getattr(arr, "frame_id", frame_id)
        header = RosHeader.create(frame_id=f_id, timestamp_s=float(ts))
        preds = getattr(arr, "predictions", getattr(arr, "trajectories", []))
        trajs = [PredictedObstacleTrajectoryMsg.from_phase13(t, frame_id=f_id) for t in preds]
        return cls(header=header, trajectories=trajs)

    def to_phase13(self) -> PredictedTrajectoryArray:
        trajs = [t.to_phase13() for t in self.trajectories]
        return PredictedTrajectoryArray(
            header_timestamp=float(self.header.stamp.to_sec()),
            frame_id=self.header.frame_id,
            predictions=trajs,
        )


# ==============================================================================
# 4. DYNAMIC OCCUPANCY MESSAGES
# ==============================================================================

@dataclass
class DynamicOccupancyGridSliceMsg:
    """In-memory representation of DynamicOccupancyGridSlice.msg."""

    header: RosHeader
    target_time_offset_s: float
    target_timestamp_s: float
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    data: List[float] = field(default_factory=list)
    unknown_mask: List[bool] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "target_time_offset_s": round(float(self.target_time_offset_s), 4),
            "target_timestamp_s": round(float(self.target_timestamp_s), 4),
            "resolution": round(float(self.resolution), 4),
            "origin_x": round(float(self.origin_x), 4),
            "origin_y": round(float(self.origin_y), 4),
            "width": int(self.width),
            "height": int(self.height),
            "data": [round(float(v), 4) for v in self.data],
            "unknown_mask": [bool(v) for v in self.unknown_mask],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicOccupancyGridSliceMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            target_time_offset_s=float(d["target_time_offset_s"]),
            target_timestamp_s=float(d.get("target_timestamp_s", 0.0)),
            resolution=float(d["resolution"]),
            origin_x=float(d["origin_x"]),
            origin_y=float(d["origin_y"]),
            width=int(d["width"]),
            height=int(d["height"]),
            data=[float(v) for v in d.get("data", [])],
            unknown_mask=[bool(v) for v in d.get("unknown_mask", [])],
        )


@dataclass
class DynamicOccupancySliceArrayMsg:
    """In-memory representation of DynamicOccupancySliceArray.msg."""

    header: RosHeader
    slices: List[DynamicOccupancyGridSliceMsg] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "slices": [s.to_dict() for s in self.slices],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicOccupancySliceArrayMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            slices=[DynamicOccupancyGridSliceMsg.from_dict(s) for s in d.get("slices", [])],
        )


# ==============================================================================
# 5. BEHAVIORAL DECISION MESSAGE
# ==============================================================================

@dataclass
class BehavioralDecisionMsg:
    """In-memory representation of BehavioralDecision.msg."""

    header: RosHeader
    state: str
    command_v: float
    command_omega: float
    reason: str
    blocking_obstacle_id: int = 0
    dynamic_collision: bool = False
    dynamic_risk: str = "FREE"
    unknown_horizon: bool = False
    wait_elapsed_s: float = 0.0
    wait_timeout: bool = False
    deadlock: bool = False
    deadlock_event: str = "NONE"
    transition: str = "NONE"
    previous_state: str = "PROCEED"
    state_dwell_time_s: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "state": str(self.state),
            "command_v": round(float(self.command_v), 4),
            "command_omega": round(float(self.command_omega), 4),
            "reason": str(self.reason),
            "blocking_obstacle_id": int(self.blocking_obstacle_id),
            "dynamic_collision": bool(self.dynamic_collision),
            "dynamic_risk": str(self.dynamic_risk),
            "unknown_horizon": bool(self.unknown_horizon),
            "wait_elapsed_s": round(float(self.wait_elapsed_s), 4),
            "wait_timeout": bool(self.wait_timeout),
            "deadlock": bool(self.deadlock),
            "deadlock_event": str(self.deadlock_event),
            "transition": str(self.transition),
            "previous_state": str(self.previous_state),
            "state_dwell_time_s": round(float(self.state_dwell_time_s), 4),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BehavioralDecisionMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            state=str(d["state"]),
            command_v=float(d["command_v"]),
            command_omega=float(d["command_omega"]),
            reason=str(d.get("reason", "")),
            blocking_obstacle_id=int(d.get("blocking_obstacle_id", 0)),
            dynamic_collision=bool(d.get("dynamic_collision", False)),
            dynamic_risk=str(d.get("dynamic_risk", "FREE")),
            unknown_horizon=bool(d.get("unknown_horizon", False)),
            wait_elapsed_s=float(d.get("wait_elapsed_s", 0.0)),
            wait_timeout=bool(d.get("wait_timeout", False)),
            deadlock=bool(d.get("deadlock", False)),
            deadlock_event=str(d.get("deadlock_event", "NONE")),
            transition=str(d.get("transition", "NONE")),
            previous_state=str(d.get("previous_state", "PROCEED")),
            state_dwell_time_s=float(d.get("state_dwell_time_s", 0.0)),
            confidence=float(d.get("confidence", 1.0)),
        )

    @classmethod
    def from_phase13(cls, decision: BehavioralDecision, frame_id: str = "odom", timestamp_s: float = 0.0) -> BehavioralDecisionMsg:
        header = RosHeader.create(frame_id=frame_id, timestamp_s=timestamp_s)
        return cls(
            header=header,
            state=decision.state.value,
            command_v=float(decision.command_v),
            command_omega=float(decision.command_omega),
            reason=str(decision.reason),
            blocking_obstacle_id=int(decision.blocking_obstacle_id or 0),
            dynamic_collision=bool(decision.dynamic_collision),
            dynamic_risk=str(decision.dynamic_risk),
            unknown_horizon=bool(decision.unknown_horizon),
            wait_elapsed_s=float(decision.wait_elapsed_s),
            wait_timeout=bool(decision.wait_timeout),
            deadlock=bool(decision.deadlock),
            deadlock_event=str(decision.deadlock_event),
            transition=str(decision.transition),
            previous_state=decision.previous_state.value,
            state_dwell_time_s=float(decision.state_dwell_time_s),
            confidence=float(decision.confidence),
        )

    def to_phase13(self) -> BehavioralDecision:
        return BehavioralDecision(
            state=BehavioralState(self.state),
            command_v=float(self.command_v),
            command_omega=float(self.command_omega),
            reason=str(self.reason),
            blocking_obstacle_id=int(self.blocking_obstacle_id) if self.blocking_obstacle_id > 0 else None,
            dynamic_collision=bool(self.dynamic_collision),
            dynamic_risk=str(self.dynamic_risk),
            unknown_horizon=bool(self.unknown_horizon),
            wait_elapsed_s=float(self.wait_elapsed_s),
            wait_timeout=bool(self.wait_timeout),
            deadlock=bool(self.deadlock),
            deadlock_event=str(self.deadlock_event),
            transition=str(self.transition),
            previous_state=BehavioralState(self.previous_state),
            state_dwell_time_s=float(self.state_dwell_time_s),
            confidence=float(self.confidence),
        )


# ==============================================================================
# 6. DYNAMIC REPLAN EVENT MESSAGE
# ==============================================================================

@dataclass
class DynamicReplanEventMsg:
    """In-memory representation of DynamicReplanEvent.msg."""

    header: RosHeader
    event_id: str
    trigger_type: str
    current_pose: Tuple[float, float, float]
    current_goal: Tuple[float, float]
    blocking_obstacle_ids: List[int] = field(default_factory=list)
    blocked_duration_s: float = 0.0
    deadlock_duration_s: float = 0.0
    current_progress_m: float = 0.0
    reason: str = ""
    confidence: float = 1.0
    success: bool = True
    is_path_stale: bool = False
    old_path_length_m: float = 0.0
    new_path_length_m: float = 0.0
    path_deviation_m: float = 0.0
    planning_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "event_id": str(self.event_id),
            "trigger_type": str(self.trigger_type),
            "current_pose": [round(float(v), 4) for v in self.current_pose],
            "current_goal": [round(float(v), 4) for v in self.current_goal],
            "blocking_obstacle_ids": [int(v) for v in self.blocking_obstacle_ids],
            "blocked_duration_s": round(float(self.blocked_duration_s), 4),
            "deadlock_duration_s": round(float(self.deadlock_duration_s), 4),
            "current_progress_m": round(float(self.current_progress_m), 4),
            "reason": str(self.reason),
            "confidence": round(float(self.confidence), 4),
            "success": bool(self.success),
            "is_path_stale": bool(self.is_path_stale),
            "old_path_length_m": round(float(self.old_path_length_m), 4),
            "new_path_length_m": round(float(self.new_path_length_m), 4),
            "path_deviation_m": round(float(self.path_deviation_m), 4),
            "planning_latency_ms": round(float(self.planning_latency_ms), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DynamicReplanEventMsg:
        return cls(
            header=RosHeader.from_dict(d["header"]),
            event_id=str(d["event_id"]),
            trigger_type=str(d["trigger_type"]),
            current_pose=tuple(float(v) for v in d["current_pose"]),
            current_goal=tuple(float(v) for v in d["current_goal"]),
            blocking_obstacle_ids=[int(v) for v in d.get("blocking_obstacle_ids", [])],
            blocked_duration_s=float(d.get("blocked_duration_s", 0.0)),
            deadlock_duration_s=float(d.get("deadlock_duration_s", 0.0)),
            current_progress_m=float(d.get("current_progress_m", 0.0)),
            reason=str(d.get("reason", "")),
            confidence=float(d.get("confidence", 1.0)),
            success=bool(d.get("success", True)),
            is_path_stale=bool(d.get("is_path_stale", False)),
            old_path_length_m=float(d.get("old_path_length_m", 0.0)),
            new_path_length_m=float(d.get("new_path_length_m", 0.0)),
            path_deviation_m=float(d.get("path_deviation_m", 0.0)),
            planning_latency_ms=float(d.get("planning_latency_ms", 0.0)),
        )

    @classmethod
    def from_phase13(
        cls,
        event: DynamicReplanEvent,
        result: Optional[DynamicReplanResult] = None,
        frame_id: str = "odom",
    ) -> DynamicReplanEventMsg:
        header = RosHeader.create(frame_id=frame_id, timestamp_s=float(event.timestamp))
        succ = result.success if result else True
        stale = result.is_path_stale if result else False
        old_l = result.metrics.old_path_length if (result and result.metrics) else 0.0
        new_l = result.metrics.new_path_length if (result and result.metrics) else 0.0
        dev = result.metrics.path_deviation if (result and result.metrics) else 0.0
        lat = result.planning_latency_ms if result else 0.0

        return cls(
            header=header,
            event_id=str(event.event_id),
            trigger_type=event.trigger_type.value,
            current_pose=event.current_pose,
            current_goal=event.current_goal,
            blocking_obstacle_ids=list(event.blocking_obstacle_ids),
            blocked_duration_s=float(event.blocked_duration),
            deadlock_duration_s=float(event.deadlock_duration),
            current_progress_m=float(event.current_progress),
            reason=str(event.reason),
            confidence=float(event.confidence),
            success=succ,
            is_path_stale=stale,
            old_path_length_m=float(old_l),
            new_path_length_m=float(new_l),
            path_deviation_m=float(dev),
            planning_latency_ms=float(lat),
        )

    def to_phase13(self) -> DynamicReplanEvent:
        return DynamicReplanEvent(
            trigger_type=DynamicReplanTrigger(self.trigger_type),
            timestamp=float(self.header.stamp.to_sec()),
            current_pose=self.current_pose,
            current_goal=self.current_goal,
            current_global_path=None,
            blocking_obstacle_ids=list(self.blocking_obstacle_ids),
            blocked_duration=float(self.blocked_duration_s),
            deadlock_duration=float(self.deadlock_duration_s),
            current_progress=float(self.current_progress_m),
            reason=str(self.reason),
            confidence=float(self.confidence),
            event_id=str(self.event_id),
        )


# Provide backwards-compatible update method on BehavioralStateMachine for node/test evaluation
if not hasattr(BehavioralStateMachine, "update"):
    def _sm_update(
        self: BehavioralStateMachine,
        ugv_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ugv_twist: Tuple[float, float] = (0.0, 0.0),
        timestamp_s: float = 0.0,
        predictions: Optional[Any] = None,
        obstacles: Optional[Any] = None,
        best_trajectory_cost: Optional[float] = None,
        has_safe_trajectory: bool = True,
        **kwargs: Any,
    ) -> BehavioralDecision:
        robot_st = RobotState(
            x=ugv_pose[0],
            y=ugv_pose[1],
            yaw=ugv_pose[2],
            v=ugv_twist[0],
            omega=ugv_twist[1],
            timestamp=timestamp_s,
        )
        has_unknown = False
        if predictions is not None:
            preds_list = getattr(predictions, "predictions", getattr(predictions, "trajectories", predictions))
            if hasattr(preds_list, "__iter__"):
                has_unknown = any(getattr(p, "is_unknown", False) for p in preds_list)

        planner_st = PlannerStatus.SUCCESS if (has_safe_trajectory and not has_unknown) else PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY
        diag = {
            "valid_trajectories": 1 if (has_safe_trajectory and not has_unknown) else 0,
            "dynamic_risk": "UNKNOWN_HORIZON" if has_unknown else "FREE",
            "unknown_horizon": has_unknown,
            "cost": best_trajectory_cost or 0.0,
        }
        return self.evaluate(
            current_state=robot_st,
            best_trajectory=None,
            best_v=ugv_twist[0] if (has_safe_trajectory and not has_unknown) else 0.0,
            best_w=ugv_twist[1] if (has_safe_trajectory and not has_unknown) else 0.0,
            planner_status=planner_st,
            dwa_diagnostics=diag,
            predicted_tracks=predictions or obstacles,
            current_time_s=timestamp_s,
        )
    BehavioralStateMachine.update = _sm_update  # type: ignore[attr-defined]

