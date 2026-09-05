"""Stereo Visual Odometry using ORB features, 3D triangulation, and metric PnP RANSAC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.localization.base import BaseVisualOdometry, CameraIntrinsics, TrackingState
from ai.localization.preprocessing import CameraPreprocessor
from ai.localization.result import LocalizationEstimate
from ai.localization.trajectory import Trajectory3D, TrajectoryPoint3D


class StereoVisualOdometry(BaseVisualOdometry):
    """Stereo Visual Odometry with direct metric scale observability.
    
    Operates by triangulating 3D world landmarks from stereo left/right pairs,
    then solving 3D-to-2D Perspective-n-Point (PnP RANSAC) across consecutive frames.
    Directly produces true metric translation without scale ambiguity.
    """

    def __init__(
        self,
        camera: CameraIntrinsics,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(camera, config)
        self.baseline_m = camera.baseline_m if camera.baseline_m > 0 else 0.05
        self.preprocessor = CameraPreprocessor(camera)

        # ORB Feature Detector
        max_features = self.config.get("max_features", 1500)
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=20
        )

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State storage
        self.prev_left_gray: Optional[np.ndarray] = None
        self.prev_kp: Optional[List[cv2.KeyPoint]] = None
        self.prev_des: Optional[np.ndarray] = None
        self.prev_3d_pts: Optional[np.ndarray] = None # (N, 3) 3D points in previous camera frame
        self.prev_timestamp: Optional[float] = None

        self.trajectory = Trajectory3D()
        self.last_estimate: Optional[LocalizationEstimate] = None

    def reset(self) -> None:
        self.current_pose = np.eye(4, dtype=np.float64)
        self.state = TrackingState.TRACKING_OK
        self.prev_left_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_3d_pts = None
        self.prev_timestamp = None
        self.trajectory = Trajectory3D()
        self.frame_count = 0

    def triangulate_stereo_points(
        self,
        gray_left: np.ndarray,
        gray_right: np.ndarray,
        kp_left: List[cv2.KeyPoint],
        des_left: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match left keypoints to right image and triangulate 3D coordinates.
        
        Returns:
            valid_kp: (M, 2) 2D pixel coordinates in left image
            valid_des: (M, 32) ORB descriptors
            points_3d: (M, 3) 3D metric coordinates [X, Y, Z] in camera optical frame
        """
        kp_right, des_right = self.orb.detectAndCompute(gray_right, None)
        if des_right is None or len(kp_right) < 10:
            return np.empty((0, 2)), np.empty((0, 32)), np.empty((0, 3))

        matches = self.matcher.knnMatch(des_left, des_right, k=2)

        valid_kp_list = []
        valid_des_list = []
        pts_3d_list = []

        fx = self.camera.fx
        fy = self.camera.fy
        cx = self.camera.cx
        cy = self.camera.cy
        b = self.baseline_m

        for m_pair in matches:
            if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                m = m_pair[0]
                pt_l = kp_left[m.queryIdx].pt
                pt_r = kp_right[m.trainIdx].pt

                # Epipolar constraint: vertical disparity must be small (|y_L - y_R| < 3 px)
                # and horizontal disparity must be positive (x_L > x_R)
                disp = pt_l[0] - pt_r[0]
                if abs(pt_l[1] - pt_r[1]) < 3.0 and disp > 1.0:
                    Z = (fx * b) / disp
                    if 0.3 < Z < 30.0:  # Reasonable depth range for UGV
                        X = ((pt_l[0] - cx) * Z) / fx
                        Y = ((pt_l[1] - cy) * Z) / fy

                        valid_kp_list.append(pt_l)
                        valid_des_list.append(des_left[m.queryIdx])
                        pts_3d_list.append([X, Y, Z])

        if not pts_3d_list:
            return np.empty((0, 2)), np.empty((0, 32)), np.empty((0, 3))

        return np.float32(valid_kp_list), np.array(valid_des_list), np.float32(pts_3d_list)

    def process_frame(
        self,
        image_left: np.ndarray,
        timestamp: float,
        image_right: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        imu: Optional[Any] = None,
    ) -> LocalizationEstimate:
        self.frame_count += 1
        gray_l, gray_r = self.preprocessor.rectify_stereo_pair(
            image_left,
            image_right if image_right is not None else image_left
        )

        kp_l, des_l = self.orb.detectAndCompute(gray_l, None)

        if des_l is None or len(kp_l) < 10:
            self.state = TrackingState.LOST
            estimate = LocalizationEstimate(
                timestamp=timestamp,
                pose=self.current_pose.copy(),
                tracking_state=self.state,
                confidence=0.10,
                is_pose_valid=False,
                diagnostics={"reason": "insufficient_features"}
            )
            self.trajectory.append(TrajectoryPoint3D.from_se3(
                timestamp, self.current_pose, self.state, 0.10, 0
            ))
            return estimate

        # Triangulate 3D points for current frame
        pts_2d_curr, des_curr, pts_3d_curr = self.triangulate_stereo_points(
            gray_l, gray_r, kp_l, des_l
        )

        # First frame initialization
        if self.prev_left_gray is None or self.prev_3d_pts is None or len(self.prev_3d_pts) < 10:
            self.prev_left_gray = gray_l
            self.prev_kp = kp_l
            self.prev_des = des_curr
            self.prev_3d_pts = pts_3d_curr
            self.prev_timestamp = timestamp
            self.state = TrackingState.TRACKING_OK

            estimate = LocalizationEstimate(
                timestamp=timestamp,
                pose=self.current_pose.copy(),
                tracking_state=self.state,
                confidence=1.0,
                tracked_features=len(pts_2d_curr),
                inliers=len(pts_2d_curr),
                keyframe_id=0,
                is_keyframe=True,
                child_frame_id="camera_optical",
                diagnostics={"mode": "stereo_initialized", "metric_scale": True}
            )
            self.keyframe_poses.append(self.current_pose.copy())
            self.trajectory.append(TrajectoryPoint3D.from_se3(
                timestamp, self.current_pose, self.state, 1.0, len(pts_2d_curr)
            ))
            self.last_estimate = estimate
            return estimate

        # Match previous frame's 3D landmarks to current frame's 2D features
        matches = self.matcher.knnMatch(self.prev_des, des_l, k=2)
        matched_3d = []
        matched_2d = []

        for m_pair in matches:
            if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                m = m_pair[0]
                matched_3d.append(self.prev_3d_pts[m.queryIdx])
                matched_2d.append(kp_l[m.trainIdx].pt)

        inliers_count = 0
        if len(matched_3d) >= 6:
            p3d = np.ascontiguousarray(np.float32(matched_3d)).reshape(-1, 3)
            p2d = np.ascontiguousarray(np.float32(matched_2d)).reshape(-1, 2)

            # Solve PnP RANSAC to get metric rotation R and translation t
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                p3d,
                p2d,
                self.camera.K,
                distCoeffs=None,
                iterationsCount=200,
                reprojectionError=2.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            trans_norm = float(np.linalg.norm(tvec)) if tvec is not None else float("inf")
            if success and inliers is not None and len(inliers) >= 10 and trans_norm <= 1.5:
                inliers_count = len(inliers)
                R_mat, _ = cv2.Rodrigues(rvec)

                # Relative transform from previous camera frame to current camera frame:
                # P_curr = R * P_prev + t => T_rel_cam = [R, t; 0, 1]
                # Invert to transform previous pose forward:
                T_rel = np.eye(4, dtype=np.float64)
                T_rel[:3, :3] = R_mat.T
                T_rel[:3, 3] = (-R_mat.T @ tvec).ravel()

                # Metric world pose update
                self._last_T_rel = T_rel.copy()
                self.current_pose = self.current_pose @ T_rel

                if inliers_count >= 20:
                    self.state = TrackingState.TRACKING_OK
                    confidence = min(1.0, inliers_count / 80.0)
                else:
                    self.state = TrackingState.DEGRADED
                    confidence = 0.50
            else:
                self.state = TrackingState.LOST
                confidence = 0.15
        else:
            self.state = TrackingState.LOST
            confidence = 0.10

        # Update previous frame state
        self.prev_left_gray = gray_l
        self.prev_kp = kp_l
        self.prev_des = des_curr
        self.prev_3d_pts = pts_3d_curr
        self.prev_timestamp = timestamp

        is_keyframe = (inliers_count < 25 or self.frame_count % 10 == 0)
        if is_keyframe:
            self.keyframe_poses.append(self.current_pose.copy())

        estimate = LocalizationEstimate(
            timestamp=timestamp,
            pose=self.current_pose.copy(),
            tracking_state=self.state,
            confidence=confidence,
            is_pose_valid=(self.state not in (TrackingState.LOST, TrackingState.RELOCALIZING)),
            tracked_features=len(pts_2d_curr),
            inliers=inliers_count,
            keyframe_id=len(self.keyframe_poses),
            is_keyframe=is_keyframe,
            child_frame_id="camera_optical",
            diagnostics={
                "scale_mode": "true_metric_stereo",
                "baseline_m": self.baseline_m,
                "triangulated_landmarks": len(pts_3d_curr),
                "matched_3d": len(matched_3d) if 'matched_3d' in locals() else 0,
                "trans_norm": round(trans_norm, 3) if 'trans_norm' in locals() else None,
            }
        )
        self.trajectory.append(TrajectoryPoint3D.from_se3(
            timestamp, self.current_pose, self.state, confidence, inliers_count
        ))
        self.last_estimate = estimate
        return estimate
