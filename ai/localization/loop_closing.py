"""Place recognition, loop closure detection, and trajectory relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ai.localization.trajectory import Trajectory3D, TrajectoryPoint3D


@dataclass
class KeyframeRecord:
    """Archived keyframe for place recognition and loop closing."""
    keyframe_id: int
    timestamp: float
    pose: np.ndarray        # 4x4 SE(3) matrix
    descriptors: np.ndarray # (N, 32) ORB descriptors
    keypoints: np.ndarray   # (N, 2) 2D pixel coordinates


class LoopClosingEngine:
    """DBoW2-style visual place recognition and pose graph relaxation engine."""

    def __init__(
        self,
        min_inliers_loop: int = 25,
        min_separation_keyframes: int = 20,
        similarity_threshold: float = 0.65
    ):
        self.min_inliers = min_inliers_loop
        self.min_separation = min_separation_keyframes
        self.similarity_th = similarity_threshold
        self.database: List[KeyframeRecord] = []
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.detected_loops: List[Tuple[int, int, float]] = [] # (curr_id, past_id, error_m)

        # Quantitative instrumentation counters
        self.loop_attempts: int = 0
        self.loop_candidates: int = 0
        self.loop_closures_accepted: int = 0
        self.loop_closures_rejected: int = 0
        self.relocalization_attempts: int = 0
        self.relocalization_successes: int = 0
        self.relocalization_failures: int = 0

    def add_keyframe(
        self,
        keyframe_id: int,
        timestamp: float,
        pose: np.ndarray,
        descriptors: np.ndarray,
        keypoints: np.ndarray
    ) -> None:
        """Register a keyframe in the historical visual database."""
        if descriptors is not None and len(descriptors) > 10:
            self.database.append(KeyframeRecord(
                keyframe_id=keyframe_id,
                timestamp=timestamp,
                pose=pose.copy(),
                descriptors=descriptors.copy(),
                keypoints=keypoints.copy()
            ))

    def detect_loop(
        self,
        curr_descriptors: np.ndarray,
        curr_keyframe_id: int
    ) -> Optional[Tuple[int, np.ndarray, float]]:
        """Search keyframe database for place recognition match.
        
        Returns:
            (match_keyframe_id, relative_transform, residual_drift_m) or None
        """
        self.loop_attempts += 1
        if curr_descriptors is None or len(self.database) < self.min_separation:
            return None

        best_match_id = -1
        best_inliers = 0
        candidate_window = self.database[:-self.min_separation]

        for kf in candidate_window:
            matches = self.matcher.knnMatch(curr_descriptors, kf.descriptors, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.70 * m[1].distance]

            if len(good) > 15:
                self.loop_candidates += 1

            if len(good) > best_inliers:
                best_inliers = len(good)
                best_match_id = kf.keyframe_id

        if best_inliers >= self.min_inliers:
            self.loop_closures_accepted += 1
            matched_kf = self.database[best_match_id]
            # Compute loop drift residual
            curr_pos = self.database[-1].pose[:3, 3] if self.database else np.zeros(3)
            loop_pos = matched_kf.pose[:3, 3]
            drift_m = float(np.linalg.norm(curr_pos - loop_pos))

            self.detected_loops.append((curr_keyframe_id, best_match_id, drift_m))
            # Relative transformation
            T_rel = np.linalg.inv(matched_kf.pose) @ self.database[-1].pose
            return best_match_id, T_rel, drift_m

        if best_match_id >= 0:
            self.loop_closures_rejected += 1
        return None

    def relocalize(
        self,
        curr_descriptors: np.ndarray,
        min_inliers_reloc: int = 20
    ) -> Optional[Tuple[int, np.ndarray, int]]:
        """Attempt to relocalize from visual tracking loss against historical keyframes.
        
        Returns:
            (matched_keyframe_id, recovered_pose, inlier_count) or None
        """
        self.relocalization_attempts += 1
        if curr_descriptors is None or not self.database:
            self.relocalization_failures += 1
            return None

        best_kf_id = -1
        best_inliers = 0

        for kf in self.database:
            matches = self.matcher.knnMatch(curr_descriptors, kf.descriptors, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.70 * m[1].distance]
            if len(good) > best_inliers:
                best_inliers = len(good)
                best_kf_id = kf.keyframe_id

        if best_inliers >= min_inliers_reloc:
            self.relocalization_successes += 1
            matched_pose = self.database[best_kf_id].pose.copy()
            return best_kf_id, matched_pose, best_inliers

        self.relocalization_failures += 1
        return None

    def relax_trajectory(
        self,
        trajectory: Trajectory3D,
        loop_match_id: int,
        loop_error: np.ndarray
    ) -> Trajectory3D:
        """Apply linear pose graph relaxation along trajectory to correct accumulated drift."""
        N = len(trajectory)
        if N < 2:
            return trajectory

        relaxed = Trajectory3D(frame_id=trajectory.frame_id, child_frame_id=trajectory.child_frame_id)
        # Distribute loop error linearly from match point to current frame
        for i, pt in enumerate(trajectory.points):
            weight = float(i) / float(N - 1)
            corrected_pos = pt.position - weight * loop_error[:3]
            relaxed.append(TrajectoryPoint3D(
                timestamp=pt.timestamp,
                x=corrected_pos[0],
                y=corrected_pos[1],
                z=corrected_pos[2],
                qx=pt.qx,
                qy=pt.qy,
                qz=pt.qz,
                qw=pt.qw,
                vx=pt.vx,
                vy=pt.vy,
                vz=pt.vz,
                tracking_state=pt.tracking_state,
                confidence=pt.confidence,
                inlier_count=pt.inlier_count
            ))
        return relaxed
