"""Visual-Inertial Odometry (VIO) combining stereo visual features with high-rate IMU."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.localization.base import BaseVisualOdometry, CameraIntrinsics, IMUReading, TrackingState
from ai.localization.result import LocalizationEstimate
from ai.localization.stereo_vo import StereoVisualOdometry
from ai.localization.trajectory import Trajectory3D, TrajectoryPoint3D, rotation_matrix_to_quaternion


class VisualInertialOdometry(BaseVisualOdometry):
    """Tightly/loosely-coupled Visual-Inertial Odometry fusing stereo vision with IMU.
    
    Provides high-frequency state updates, orientation stability during fast UGV turns,
    and dead-reckoning bridge over momentary visual tracking dropouts.
    """

    def __init__(
        self,
        camera: CameraIntrinsics,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(camera, config)
        self.vo = StereoVisualOdometry(camera, config)

        # IMU state
        self.velocity = np.zeros(3, dtype=np.float64)  # World frame velocity
        self.gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        self.prev_imu_time: Optional[float] = None
        self.trajectory = Trajectory3D()

    def reset(self) -> None:
        self.vo.reset()
        self.current_pose = np.eye(4, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.state = TrackingState.TRACKING_OK
        self.prev_imu_time = None
        self.trajectory = Trajectory3D()
        self.frame_count = 0

    def integrate_imu(self, imu: IMUReading) -> None:
        """Predict forward step using IMU reading."""
        if self.prev_imu_time is None:
            self.prev_imu_time = imu.timestamp
            return

        dt = imu.timestamp - self.prev_imu_time
        if dt <= 0.0 or dt > 0.5:
            self.prev_imu_time = imu.timestamp
            return

        # 1. Update orientation using angular velocity (Rodrigues formula)
        R_current = self.current_pose[:3, :3]
        d_theta = imu.angular_velocity * dt
        angle = np.linalg.norm(d_theta)
        if angle > 1e-6:
            axis = d_theta / angle
            # Skew symmetric
            K_skew = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0]
            ])
            dR = np.eye(3) + np.sin(angle) * K_skew + (1.0 - np.cos(angle)) * (K_skew @ K_skew)
            R_new = R_current @ dR
            self.current_pose[:3, :3] = R_new

        # 2. Integrate linear acceleration: a_world = R * a_body + gravity
        a_world = R_current @ imu.linear_acceleration + self.gravity
        # Velocity and position forward Euler prediction
        self.velocity += a_world * dt
        self.current_pose[:3, 3] += self.velocity * dt

        self.prev_imu_time = imu.timestamp

    def process_frame(
        self,
        image_left: np.ndarray,
        timestamp: float,
        image_right: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        imu: Optional[IMUReading] = None,
    ) -> LocalizationEstimate:
        self.frame_count += 1

        # Process IMU prediction if available
        if imu is not None:
            self.integrate_imu(imu)

        # Run Visual Odometry update
        vo_estimate = self.vo.process_frame(
            image_left, timestamp, image_right, depth, imu
        )

        # Fuse Visual Measurement with Inertial Prediction
        if vo_estimate.tracking_state in (TrackingState.TRACKING_OK, TrackingState.RECOVERED):
            # Vision is reliable: fuse pose with high weight on visual measurement
            # (Complementary alpha filter: 85% vision, 15% inertial continuity)
            alpha = 0.85
            self.current_pose[:3, 3] = (
                alpha * vo_estimate.pose[:3, 3] + (1.0 - alpha) * self.current_pose[:3, 3]
            )
            # Reset velocity from visual displacement if applicable
            if len(self.trajectory) > 1:
                dt = timestamp - self.trajectory.points[-1].timestamp
                if dt > 1e-3:
                    self.velocity = (self.current_pose[:3, 3] - self.trajectory.points[-1].position) / dt

            self.state = TrackingState.TRACKING_OK
            confidence = vo_estimate.confidence
        elif vo_estimate.tracking_state == TrackingState.DEGRADED:
            # Degraded vision: rely equally on IMU dead-reckoning
            self.state = TrackingState.DEGRADED
            confidence = 0.40
        else:
            # Visual tracking lost: dead-reckon using IMU
            self.state = TrackingState.LOST
            confidence = max(0.10, vo_estimate.confidence * 0.5)

        estimate = LocalizationEstimate(
            timestamp=timestamp,
            pose=self.current_pose.copy(),
            tracking_state=self.state,
            confidence=confidence,
            is_pose_valid=(self.state not in (TrackingState.LOST, TrackingState.RELOCALIZING)),
            linear_velocity=self.velocity.copy(),
            angular_velocity=imu.angular_velocity if imu else np.zeros(3),
            tracked_features=vo_estimate.tracked_features,
            inliers=vo_estimate.inliers,
            keyframe_id=vo_estimate.keyframe_id,
            is_keyframe=vo_estimate.is_keyframe,
            diagnostics={
                "mode": "stereo_inertial_vio",
                "imu_integrated": imu is not None,
                "visual_state": vo_estimate.tracking_state.value
            }
        )
        self.trajectory.append(TrajectoryPoint3D.from_se3(
            timestamp, self.current_pose, self.state, confidence, vo_estimate.inliers
        ))
        return estimate
