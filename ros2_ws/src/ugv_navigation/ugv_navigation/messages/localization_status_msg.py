"""Pure-Python data model and message adapter for LocalizationStatus.msg."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from typing import ClassVar

from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ugv_navigation.utils.time_utils import RosHeader, RosTime


@dataclass
class Point3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Point3D = field(default_factory=Point3D)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class Twist:
    linear: Point3D = field(default_factory=Point3D)
    angular: Point3D = field(default_factory=Point3D)


@dataclass
class PoseWithCovariance:
    pose: Pose = field(default_factory=Pose)
    covariance: list[float] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class TwistWithCovariance:
    twist: Twist = field(default_factory=Twist)
    covariance: list[float] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class LocalizationStatusMsg:
    """In-memory representation of LocalizationStatus.msg."""

    TRACKING_OK: ClassVar[int] = 0
    DEGRADED: ClassVar[int] = 1
    LOST: ClassVar[int] = 2
    RELOCALIZING: ClassVar[int] = 3
    RECOVERED: ClassVar[int] = 4

    header: RosHeader
    child_frame_id: str
    tracking_state: int  # 0=OK, 1=DEGRADED, 2=LOST, 3=RELOCALIZING, 4=RECOVERED
    confidence: float
    is_pose_valid: bool
    allows_spatial_fusion: bool
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)
    tracked_features: int = 0
    inliers: int = 0
    keyframe_id: int = 0
    is_keyframe: bool = False
    relocalized: bool = False

    STATE_MAPPING = {
        TrackingState.TRACKING_OK: 0,
        TrackingState.DEGRADED: 1,
        TrackingState.LOST: 2,
        TrackingState.RELOCALIZING: 3,
        TrackingState.RECOVERED: 4,
    }

    INV_STATE_MAPPING = {
        0: TrackingState.TRACKING_OK,
        1: TrackingState.DEGRADED,
        2: TrackingState.LOST,
        3: TrackingState.RELOCALIZING,
        4: TrackingState.RECOVERED,
    }

    @classmethod
    def from_localization_estimate(
        cls, est: LocalizationEstimate, frame_id: str = "odom", child_frame_id: str = "base_link"
    ) -> LocalizationStatusMsg:
        """Construct from Phase 8 LocalizationEstimate."""
        header = RosHeader.create(frame_id=frame_id, timestamp_s=est.timestamp)
        state_int = cls.STATE_MAPPING.get(est.tracking_state, 2)

        # Convert rotation matrix to quaternion
        R = est.pose[:3, :3]
        t = est.pose[:3, 3]

        # Standard rotation matrix to quaternion
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        pose_msg = Pose(
            position=Point3D(x=float(t[0]), y=float(t[1]), z=float(t[2])),
            orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
        )

        twist_msg = Twist(
            linear=Point3D(x=float(est.linear_velocity[0]), y=float(est.linear_velocity[1]), z=float(est.linear_velocity[2])),
            angular=Point3D(x=float(est.angular_velocity[0]), y=float(est.angular_velocity[1]), z=float(est.angular_velocity[2])),
        )

        cov = est.covariance_proxy.flatten().tolist()

        is_valid = bool(est.is_pose_valid) and (est.tracking_state not in (TrackingState.LOST, TrackingState.RELOCALIZING))
        allows_fusion = bool(est.allows_spatial_fusion) and is_valid

        return cls(
            header=header,
            child_frame_id=child_frame_id,
            tracking_state=state_int,
            confidence=float(est.confidence),
            is_pose_valid=is_valid,
            allows_spatial_fusion=allows_fusion,
            pose=PoseWithCovariance(pose=pose_msg, covariance=cov),
            twist=TwistWithCovariance(twist=twist_msg),
            tracked_features=int(est.tracked_features),
            inliers=int(est.inliers),
            keyframe_id=int(est.keyframe_id),
            is_keyframe=bool(est.is_keyframe),
            relocalized=bool(est.relocalized),
        )

    def to_localization_estimate(self) -> LocalizationEstimate:
        """Convert back to Phase 8 LocalizationEstimate."""
        # Convert quaternion back to rotation matrix
        q = self.pose.pose.orientation
        p = self.pose.pose.position
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

        pose_mat = np.eye(4, dtype=np.float64)
        pose_mat[:3, :3] = R
        pose_mat[:3, 3] = [p.x, p.y, p.z]

        lin_vel = np.array([self.twist.twist.linear.x, self.twist.twist.linear.y, self.twist.twist.linear.z], dtype=np.float64)
        ang_vel = np.array([self.twist.twist.angular.x, self.twist.twist.angular.y, self.twist.twist.angular.z], dtype=np.float64)
        cov_mat = np.array(self.pose.covariance, dtype=np.float64).reshape((6, 6))

        tracking_state = self.INV_STATE_MAPPING.get(self.tracking_state, TrackingState.LOST)

        return LocalizationEstimate(
            pose=pose_mat,
            confidence=self.confidence,
            tracking_state=tracking_state,
            timestamp=self.header.stamp.to_sec(),
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            tracked_features=self.tracked_features,
            inliers=self.inliers,
            keyframe_id=self.keyframe_id,
            is_keyframe=self.is_keyframe,
            relocalized=self.relocalized,
            diagnostics={"covariance": cov_mat},
        )
