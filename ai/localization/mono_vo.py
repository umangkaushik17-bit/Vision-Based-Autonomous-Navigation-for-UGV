"""Monocular Visual Odometry using ORB feature tracking and 5-point Essential Matrix RANSAC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.localization.base import BaseVisualOdometry, CameraIntrinsics, TrackingState
from ai.localization.preprocessing import CameraPreprocessor
from ai.localization.result import LocalizationEstimate
from ai.localization.trajectory import Trajectory3D, TrajectoryPoint3D


class MonocularVisualOdometry(BaseVisualOdometry):
    """Monocular Visual Odometry using ORB features and 5-point Essential Matrix.
    
    IMPORTANT NOTE ON SCALE AMBIGUITY:
    Monocular camera geometry cannot observe absolute metric scale without external constraints
    (such as known camera height above a flat ground plane or IMU acceleration).
    The recovered translation vector t from recoverPose has unit norm (||t|| = 1.0).
    A baseline translation magnitude assumption is maintained, and raw monocular trajectories
    are explicitly marked as scale-ambiguous.
    """

    def __init__(
        self,
        camera: CameraIntrinsics,
        config: Optional[Dict[str, Any]] = None,
        nominal_speed_mps: float = 0.65
    ):
        super().__init__(camera, config)
        self.preprocessor = CameraPreprocessor(camera)
        self.nominal_speed_mps = nominal_speed_mps

        # ORB Feature Detector
        max_features = self.config.get("max_features", 1500)
        scale_factor = self.config.get("scale_factor", 1.2)
        nlevels = self.config.get("pyramid_levels", 8)
        fast_th = self.config.get("ini_fast_threshold", 20)

        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            fastThreshold=fast_th
        )

        # BFMatcher with Hamming distance for binary ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp: Optional[List[cv2.KeyPoint]] = None
        self.prev_des: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None

        self.trajectory = Trajectory3D()
        self.last_estimate: Optional[LocalizationEstimate] = None

    def reset(self) -> None:
        self.current_pose = np.eye(4, dtype=np.float64)
        self.state = TrackingState.TRACKING_OK
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_timestamp = None
        self.trajectory = Trajectory3D()
        self.frame_count = 0

    def process_frame(
        self,
        image_left: np.ndarray,
        timestamp: float,
        image_right: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        imu: Optional[Any] = None,
    ) -> LocalizationEstimate:
        self.frame_count += 1
        gray = self.preprocessor.process(image_left)

        # Extract features
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 10:
            self.state = TrackingState.LOST
            estimate = LocalizationEstimate(
                timestamp=timestamp,
                pose=self.current_pose.copy(),
                tracking_state=self.state,
                confidence=0.10,
                is_pose_valid=False,
                tracked_features=len(kp) if kp else 0,
                inliers=0,
                keyframe_id=len(self.keyframe_poses),
                diagnostics={"reason": "insufficient_features"}
            )
            self.trajectory.append(TrajectoryPoint3D.from_se3(
                timestamp, self.current_pose, self.state, 0.10, 0
            ))
            return estimate

        # First frame initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            self.prev_timestamp = timestamp
            self.state = TrackingState.TRACKING_OK

            estimate = LocalizationEstimate(
                timestamp=timestamp,
                pose=self.current_pose.copy(),
                tracking_state=self.state,
                confidence=1.0,
                is_pose_valid=True,
                tracked_features=len(kp),
                inliers=len(kp),
                keyframe_id=0,
                is_keyframe=True
            )
            self.keyframe_poses.append(self.current_pose.copy())
            self.trajectory.append(TrajectoryPoint3D.from_se3(
                timestamp, self.current_pose, self.state, 1.0, len(kp)
            ))
            self.last_estimate = estimate
            return estimate

        # Match features between previous frame and current frame
        matches = self.matcher.knnMatch(self.prev_des, des, k=2)

        # Lowe ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                good_matches.append(m_pair[0])

        if len(good_matches) < 8:
            self.state = TrackingState.LOST
            estimate = LocalizationEstimate(
                timestamp=timestamp,
                pose=self.current_pose.copy(),
                tracking_state=self.state,
                confidence=0.15,
                is_pose_valid=False,
                tracked_features=len(kp),
                inliers=len(good_matches),
                keyframe_id=len(self.keyframe_poses),
                diagnostics={"reason": "low_matches"}
            )
            self.trajectory.append(TrajectoryPoint3D.from_se3(
                timestamp, self.current_pose, self.state, 0.15, len(good_matches)
            ))
            return estimate

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])

        # Estimate Essential Matrix with 5-point RANSAC
        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            cameraMatrix=self.camera.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5
        )

        inliers = 0
        if E is not None and mask is not None:
            # Recover relative rotation R and unit translation t
            points_count, R, t, mask_pose = cv2.recoverPose(
                E, pts_curr, pts_prev, cameraMatrix=self.camera.K, mask=mask
            )
            inliers = int(points_count)

            # Estimate scale based on elapsed time and nominal speed
            dt = timestamp - self.prev_timestamp if self.prev_timestamp else 0.033
            scale = max(0.01, self.nominal_speed_mps * dt)

            # Invert relative transform (recoverPose gives prev->curr in camera coordinates)
            # T_relative: transform from previous pose to current pose
            T_rel = np.eye(4, dtype=np.float64)
            T_rel[:3, :3] = R.T
            T_rel[:3, 3] = (-R.T @ t).ravel() * scale

            # Update world pose: T_world_curr = T_world_prev @ T_rel
            self.current_pose = self.current_pose @ T_rel

            # Set tracking state based on inlier count
            if inliers >= 25:
                self.state = TrackingState.TRACKING_OK
                confidence = min(1.0, inliers / 100.0)
            elif inliers >= 12:
                self.state = TrackingState.DEGRADED
                confidence = 0.50
            else:
                self.state = TrackingState.LOST
                confidence = 0.20
        else:
            self.state = TrackingState.LOST
            confidence = 0.10

        # Keyframe decision: if relative motion exceeds threshold or inliers drop
        is_keyframe = False
        if self.state in (TrackingState.TRACKING_OK, TrackingState.DEGRADED):
            if inliers < 30 or (self.frame_count % 10 == 0):
                is_keyframe = True
                self.keyframe_poses.append(self.current_pose.copy())

        # Update previous frame references
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        self.prev_timestamp = timestamp

        estimate = LocalizationEstimate(
            timestamp=timestamp,
            pose=self.current_pose.copy(),
            tracking_state=self.state,
            confidence=confidence,
            is_pose_valid=(self.state not in (TrackingState.LOST, TrackingState.RELOCALIZING)),
            tracked_features=len(kp),
            inliers=inliers,
            keyframe_id=len(self.keyframe_poses),
            is_keyframe=is_keyframe,
            diagnostics={
                "scale_mode": "unobservable_monocular_scaled",
                "good_matches": len(good_matches)
            }
        )
        self.trajectory.append(TrajectoryPoint3D.from_se3(
            timestamp, self.current_pose, self.state, confidence, inliers
        ))
        self.last_estimate = estimate
        return estimate
