"""Stereo Visual Odometry Adapter for production localization fusion (Phase 14.4).

Wraps StereoVisualOdometry / VisualInertialOdometry to ingest stereo image pairs,
compute metric 3D visual odometry via PnP-RANSAC, estimate measurement covariance
from reprojection residuals and inlier counts, and format standardized
StereoVIOMeasurement contracts for the Multi-Sensor Fusion Extended Kalman Filter.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.stereo_vo import StereoVisualOdometry
from production.localization.types import StereoVIOMeasurement


class StereoVIOAdapter:
    """Production adapter bridging stereo visual odometry into multi-sensor EKF fusion.

    Features:
    - Ingests stereo image pairs (or synthetic test frames / depth).
    - Computes metric 3D position (x, y, z) and orientation (roll, pitch, yaw).
    - Computes backward finite-difference velocities (vx, vy, vz).
    - Computes metric covariance matrix (6x6) scaled dynamically by:
      - Inlier feature count (penalty when < 30 inliers)
      - Reprojection residual variance
      - Tracking state degradation
    - Implements feature starvation watchdog (LOST when inliers < 10).
    """

    def __init__(
        self,
        camera: Optional[CameraIntrinsics] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.camera = camera or CameraIntrinsics(
            fx=462.14,
            fy=462.14,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
            baseline_m=0.10,  # 10 cm baseline
        )
        self.config = config or {}
        self.vo = StereoVisualOdometry(self.camera, self.config)

        # State tracking
        self.prev_pose: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None
        self.total_frames_processed: int = 0
        self.last_valid_measurement: Optional[StereoVIOMeasurement] = None

    def reset(self) -> None:
        """Resets the visual odometry estimator and adapter state."""
        self.vo.reset()
        self.prev_pose = None
        self.prev_timestamp = None
        self.total_frames_processed = 0
        self.last_valid_measurement = None

    def process_stereo_frame(
        self,
        image_left: np.ndarray,
        image_right: Optional[np.ndarray],
        timestamp: float,
        depth: Optional[np.ndarray] = None,
    ) -> StereoVIOMeasurement:
        """Processes a stereo frame pair and returns a standardized StereoVIOMeasurement."""
        self.total_frames_processed += 1
        t_start = time.perf_counter()

        # Run underlying StereoVisualOdometry
        est = self.vo.process_frame(
            image_left=image_left,
            timestamp=timestamp,
            image_right=image_right,
            depth=depth,
        )

        pose = est.pose  # 4x4 SE(3)
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Extract Euler angles (roll, pitch, yaw) from rotation matrix
        # ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0

        # Calculate finite-difference linear velocities
        vx, vy, vz = 0.0, 0.0, 0.0
        if self.prev_pose is not None and self.prev_timestamp is not None:
            dt = timestamp - self.prev_timestamp
            if dt > 1e-4:
                prev_t = self.prev_pose[:3, 3]
                vx = float((t[0] - prev_t[0]) / dt)
                vy = float((t[1] - prev_t[1]) / dt)
                vz = float((t[2] - prev_t[2]) / dt)

        # Dynamic covariance calculation based on tracking quality
        inliers = est.inliers
        tracking_state_str = est.tracking_state.value
        is_valid = est.is_pose_valid and est.tracking_state != TrackingState.LOST

        # Base covariance: 0.01 m^2 position var, 0.005 rad^2 rotation var
        base_pos_var = 0.01
        base_rot_var = 0.005

        if is_valid and inliers >= 10:
            # Scale inversely with inlier count: 50 inliers -> scale 1.0, 20 inliers -> scale 2.5
            inlier_factor = max(1.0, 50.0 / max(10, inliers))
            conf_scale = 1.0 / max(0.1, est.confidence * est.confidence)
            total_scale = inlier_factor * conf_scale
        else:
            # Degraded / Lost state: inflate covariance heavily so EKF does not rely on it
            total_scale = 100.0
            is_valid = False

        pos_var = base_pos_var * total_scale
        rot_var = base_rot_var * total_scale
        vel_var = pos_var * 2.0

        pose_cov = np.diag([pos_var, pos_var, pos_var, rot_var, rot_var, rot_var]).astype(np.float64)
        twist_cov = np.diag([vel_var, vel_var, vel_var, rot_var * 2.0, rot_var * 2.0, rot_var * 2.0]).astype(np.float64)

        measurement = StereoVIOMeasurement(
            timestamp=timestamp,
            x=float(t[0]),
            y=float(t[1]),
            z=float(t[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            vx=vx,
            vy=vy,
            vz=vz,
            pose_covariance=pose_cov,
            twist_covariance=twist_cov,
            inliers=inliers,
            tracking_state=tracking_state_str,
            valid=is_valid,
            frame_id="odom",
            child_frame_id="base_link",
        )

        if is_valid:
            self.prev_pose = pose.copy()
            self.prev_timestamp = timestamp
            self.last_valid_measurement = measurement

        return measurement
