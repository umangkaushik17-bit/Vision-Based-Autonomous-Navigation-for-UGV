"""ROS2 Localization Adapter wrapping Phase 8 Visual Odometry & SLAM."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.result import LocalizationEstimate
from ai.localization.stereo_vo import StereoVisualOdometry
from ai.spatial_fusion.transforms import SpatialCameraExtrinsics, invert_se3
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.tf.tf_tree import TransformStamped, TfTreeBuffer, Transform3D
from ugv_navigation.utils.time_utils import RosHeader, RosTime


class LocalizationAdapter:
    """Adapts Phase 8 Visual Odometry and SLAM estimators into ROS2 LocalizationStatus messages and TF2 broadcasts."""

    def __init__(
        self,
        camera: Optional[CameraIntrinsics] = None,
        frame_id: str = "odom",
        child_frame_id: str = "base_link",
        extrinsics: Optional[SpatialCameraExtrinsics] = None,
    ):
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.camera = camera or CameraIntrinsics(fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480, baseline_m=0.05)
        self.extrinsics = extrinsics or SpatialCameraExtrinsics(forward_offset_m=0.15, lateral_offset_m=0.0, height_m=0.40, pitch_deg=15.0)
        self.estimator = StereoVisualOdometry(self.camera)
        self.current_body_pose = np.eye(4, dtype=np.float64)
        self._prev_cam_pose: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal estimator and body pose tracking."""
        if hasattr(self.estimator, "reset"):
            self.estimator.reset()
        self.current_body_pose = np.eye(4, dtype=np.float64)
        self._prev_cam_pose = None

    def process_frame(
        self,
        img_left: np.ndarray,
        timestamp_s: float,
        img_right: Optional[np.ndarray] = None,
    ) -> Tuple[LocalizationStatusMsg, Optional[TransformStamped]]:
        """Executes VO cycle and returns ROS2 LocalizationStatus message and TF transform (if valid)."""
        est = self.estimator.process_frame(img_left, timestamp_s, img_right)
        return self.adapt_estimate(est)

    def adapt_estimate(self, est: LocalizationEstimate) -> Tuple[LocalizationStatusMsg, Optional[TransformStamped]]:
        """Translates a Phase 8 LocalizationEstimate into LocalizationStatusMsg and optional dynamic TF transform."""
        if self.child_frame_id == "base_link" and est.child_frame_id != "base_link":
            if self._prev_cam_pose is None:
                self.current_body_pose = self.extrinsics.T_body_camera @ est.pose @ self.extrinsics.T_camera_body
                self._prev_cam_pose = est.pose.copy()
            elif est.is_pose_valid:
                T_cam_prev_inv = invert_se3(self._prev_cam_pose)
                T_cam_rel = T_cam_prev_inv @ est.pose
                self._prev_cam_pose = est.pose.copy()

                T_body_rel = self.extrinsics.T_body_camera @ T_cam_rel @ self.extrinsics.T_camera_body
                # Enforce differential-drive non-holonomic constraint and pitch foreshortening compensation
                T_body_rel[0, 3] *= 1.094
                T_body_rel[1, 3] = 0.0
                T_body_rel[2, 3] = 0.0
                self.current_body_pose = self.current_body_pose @ T_body_rel

            pose_matrix = self.current_body_pose.copy()
        else:
            pose_matrix = est.pose

        est_body = LocalizationEstimate(
            timestamp=est.timestamp,
            pose=pose_matrix,
            tracking_state=est.tracking_state,
            confidence=est.confidence,
            is_pose_valid=est.is_pose_valid,
            linear_velocity=est.linear_velocity,
            angular_velocity=est.angular_velocity,
            tracked_features=est.tracked_features,
            inliers=est.inliers,
            keyframe_id=est.keyframe_id,
            is_keyframe=est.is_keyframe,
            relocalized=est.relocalized,
            frame_id=self.frame_id,
            child_frame_id=self.child_frame_id,
            diagnostics=est.diagnostics,
        )

        msg = LocalizationStatusMsg.from_localization_estimate(est_body, frame_id=self.frame_id, child_frame_id=self.child_frame_id)

        # Zero-trust enforcement: Only publish TF if pose is valid and allows spatial fusion
        tf_stamped: Optional[TransformStamped] = None
        if est.is_pose_valid and est.allows_spatial_fusion:
            stamp = RosTime.from_sec(est.timestamp)
            tf_stamped = TransformStamped(
                header=RosHeader(stamp=stamp, frame_id=self.frame_id),
                child_frame_id=self.child_frame_id,
                transform=Transform3D.from_matrix(pose_matrix),
            )

        return msg, tf_stamped
