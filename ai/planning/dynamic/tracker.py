"""Multi-frame dynamic obstacle tracker for Phase 13.

Implements persistent multi-object tracking (MOT) without Kalman filtering or
ground-truth leakage:
- Track lifecycle: NEW -> TRACKING -> DEGRADED -> LOST -> EXPIRED
- Persistent, unique track IDs across successive frames
- 1-to-1 Hungarian bipartite matching with spatial distance gating
- Frame transformation from 'base_link' to 'odom'
- Robust coasting across temporary detection dropouts
- Outputs standard TrackedObstacleArray and TrackedObstacleState
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ai.planning.dynamic.association import BaseAssociation, HungarianAssociation
from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    TrackedObstacleArray,
    TrackedObstacleState,
    normalize_angle,
)
from ai.planning.dynamic.kalman_filter import (
    KalmanFilterConfig,
    KalmanTrackFilter,
)


class TrackStatus(str, enum.Enum):
    """Lifecycle state machine for tracked dynamic obstacles."""

    NEW = "NEW"
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


@dataclass
class TrackerConfig:
    """Explicit parameters for multi-frame tracking and lifecycle transitions."""

    max_association_distance_m: float = 1.50
    hits_to_confirm: int = 3
    misses_to_degrade: int = 1
    misses_to_lost: int = 3
    misses_to_expire: int = 5
    max_coast_duration_s: float = 2.0
    min_detection_confidence: float = 0.30
    initial_pos_variance: float = 0.04  # 0.20 m std
    initial_vel_variance: float = 0.25  # 0.50 m/s std
    target_frame: str = "odom"
    use_kalman_filter: bool = False
    filter_config: Optional[KalmanFilterConfig] = None

    def __post_init__(self) -> None:
        if self.max_association_distance_m <= 0.0:
            raise ValueError(f"max_association_distance_m must be positive, got {self.max_association_distance_m}")
        if self.hits_to_confirm < 1:
            raise ValueError(f"hits_to_confirm must be >= 1, got {self.hits_to_confirm}")
        if self.misses_to_degrade < 1:
            raise ValueError(f"misses_to_degrade must be >= 1, got {self.misses_to_degrade}")
        if self.misses_to_lost <= self.misses_to_degrade:
            raise ValueError(
                f"misses_to_lost ({self.misses_to_lost}) must be > misses_to_degrade ({self.misses_to_degrade})"
            )
        if self.misses_to_expire <= self.misses_to_lost:
            raise ValueError(
                f"misses_to_expire ({self.misses_to_expire}) must be > misses_to_lost ({self.misses_to_lost})"
            )
        if self.max_coast_duration_s <= 0.0:
            raise ValueError(f"max_coast_duration_s must be positive, got {self.max_coast_duration_s}")
        if not (0.0 <= self.min_detection_confidence <= 1.0):
            raise ValueError(
                f"min_detection_confidence must be in [0.0, 1.0], got {self.min_detection_confidence}"
            )


class Track:
    """Internal representation of a single tracked dynamic obstacle."""

    def __init__(
        self,
        track_id: int,
        detection: DynamicObstacleDetection,
        config: TrackerConfig,
    ) -> None:
        self.track_id = track_id
        self.config = config
        self.status = TrackStatus.NEW
        self.hits: int = 1
        self.consecutive_hits: int = 1
        self.coast_cycles: int = 0
        self.track_age: int = 1

        self.first_timestamp: float = float(detection.timestamp)
        self.last_timestamp: float = float(detection.timestamp)
        self.last_observation_timestamp: float = float(detection.timestamp)

        self.x: float = float(detection.bbox_center[0])
        self.y: float = float(detection.bbox_center[1])
        # Simple placeholder velocity until updated
        self.vx: float = 0.0
        self.vy: float = 0.0

        self.bbox_size: Tuple[float, float] = detection.bbox_size
        self.bbox_yaw: float = float(detection.bbox_yaw)
        self.confidence: float = float(detection.confidence)
        self.frame_id: str = config.target_frame

        # Kalman Filter state estimator integration
        if self.config.use_kalman_filter:
            kf_cfg = self.config.filter_config or KalmanFilterConfig(
                initial_pos_variance=self.config.initial_pos_variance,
                initial_vel_variance=self.config.initial_vel_variance,
            )
            self.kf: Optional[KalmanTrackFilter] = KalmanTrackFilter(kf_cfg)
            self.kf.initialize(
                position=detection.bbox_center,
                timestamp_s=detection.timestamp,
            )
            self.covariance = self.kf.get_covariance()
        else:
            self.kf = None
            self.covariance = np.diag([
                config.initial_pos_variance,
                config.initial_pos_variance,
                config.initial_vel_variance,
                config.initial_vel_variance,
            ]).astype(np.float64)

        self.history: List[Tuple[float, float, float]] = [(self.last_timestamp, self.x, self.y)]

        # Check if 1 hit is sufficient to confirm
        if self.hits >= self.config.hits_to_confirm:
            self.status = TrackStatus.TRACKING

    @property
    def is_confirmed(self) -> bool:
        return self.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED)

    def predict_position(self, timestamp_s: float) -> Tuple[float, float]:
        """Kinematic coasting for data association."""
        dt = max(0.0, timestamp_s - self.last_timestamp)
        return (self.x + self.vx * dt, self.y + self.vy * dt)

    def update(
        self,
        detection: DynamicObstacleDetection,
        timestamp_s: float,
    ) -> None:
        """Update track with an associated detection observation."""
        dt = timestamp_s - self.last_timestamp
        self.last_timestamp = float(timestamp_s)
        self.last_observation_timestamp = float(timestamp_s)
        self.track_age += 1
        self.hits += 1
        self.consecutive_hits += 1
        self.coast_cycles = 0

        new_x, new_y = detection.bbox_center

        if self.kf is not None:
            # Recursive state-space Kalman update
            accepted, _ = self.kf.update(detection.bbox_center, timestamp_s=timestamp_s)
            state = self.kf.get_state()
            self.x = float(state[0])
            self.y = float(state[1])
            self.vx = float(state[2])
            self.vy = float(state[3])
            self.covariance = self.kf.get_covariance()
        else:
            # Retain simple baseline finite difference if dt > 0
            if dt > 1e-4:
                self.vx = 0.8 * self.vx + 0.2 * ((new_x - self.x) / dt)
                self.vy = 0.8 * self.vy + 0.2 * ((new_y - self.y) / dt)

            self.x = float(new_x)
            self.y = float(new_y)
            # Reset covariance to initial scale on fresh observation
            self.covariance = np.diag([
                self.config.initial_pos_variance,
                self.config.initial_pos_variance,
                self.config.initial_vel_variance,
                self.config.initial_vel_variance,
            ]).astype(np.float64)

        self.bbox_size = detection.bbox_size
        self.bbox_yaw = float(detection.bbox_yaw)
        self.confidence = float(detection.confidence)

        self.history.append((self.last_timestamp, self.x, self.y))
        # Bound history to last 50 entries
        if len(self.history) > 50:
            self.history = self.history[-50:]

        # Lifecycle transition
        if self.status == TrackStatus.NEW:
            if self.hits >= self.config.hits_to_confirm:
                self.status = TrackStatus.TRACKING
        elif self.status in (TrackStatus.DEGRADED, TrackStatus.LOST):
            # Reacquisition
            self.status = TrackStatus.TRACKING

    def mark_missed(self, timestamp_s: float) -> None:
        """Update track state on missed detection."""
        self.coast_cycles += 1
        self.consecutive_hits = 0
        self.track_age += 1

        coast_time = timestamp_s - self.last_observation_timestamp

        if self.kf is not None:
            # Predict forward in state space; covariance expands monotonically
            if timestamp_s > self.last_timestamp:
                self.kf.predict(timestamp_s)
                state = self.kf.get_state()
                self.x = float(state[0])
                self.y = float(state[1])
                self.vx = float(state[2])
                self.vy = float(state[3])
                self.covariance = self.kf.get_covariance()
                self.last_timestamp = float(timestamp_s)
        else:
            # Slightly expand position uncertainty during coasting
            pos_inflation = self.coast_cycles * 0.02
            self.covariance[0, 0] += pos_inflation
            self.covariance[1, 1] += pos_inflation

        # Check max coast duration
        if coast_time > self.config.max_coast_duration_s or self.coast_cycles >= self.config.misses_to_expire:
            self.status = TrackStatus.EXPIRED
            return

        # Status degradation logic
        if self.status == TrackStatus.NEW:
            # Unconfirmed tracks expire fast if missed
            if self.coast_cycles >= 2:
                self.status = TrackStatus.EXPIRED
        elif self.status == TrackStatus.TRACKING:
            if self.coast_cycles >= self.config.misses_to_lost:
                self.status = TrackStatus.LOST
            elif self.coast_cycles >= self.config.misses_to_degrade:
                self.status = TrackStatus.DEGRADED
        elif self.status == TrackStatus.DEGRADED:
            if self.coast_cycles >= self.config.misses_to_lost:
                self.status = TrackStatus.LOST

    def to_state(self) -> TrackedObstacleState:
        """Convert to standard TrackedObstacleState data structure."""
        return TrackedObstacleState(
            track_id=self.track_id,
            timestamp=self.last_timestamp,
            frame_id=self.frame_id,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            covariance=self.covariance.copy(),
            bbox_size=self.bbox_size,
            bbox_yaw=self.bbox_yaw,
            track_age=self.track_age,
            coast_cycles=self.coast_cycles,
            is_confirmed=(self.status == TrackStatus.TRACKING),
            status=self.status.value,
            history=list(self.history),
        )


class MultiFrameTracker:
    """Multi-frame obstacle tracker maintaining persistent tracks in the world ('odom') frame."""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        association: Optional[BaseAssociation] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.association = association or HungarianAssociation()
        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 0
        self._last_timestamp: float = -1.0

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Reset all tracking state and track ID sequence."""
        self._tracks.clear()
        self._next_track_id = 0
        self._last_timestamp = -1.0

    def get_track(self, track_id: int) -> Optional[Track]:
        """Retrieve track by ID."""
        return self._tracks.get(track_id)

    def update(
        self,
        detections: List[DynamicObstacleDetection],
        timestamp_s: float,
        robot_pose: Optional[Tuple[float, float, float]] = None,
    ) -> TrackedObstacleArray:
        """Process incoming detections and update multi-frame tracks.

        Args:
            detections: List of DynamicObstacleDetection objects.
            timestamp_s: Observation timestamp in seconds (must be >= 0).
            robot_pose: (x, y, yaw) in world frame for transforming base_link detections.

        Returns:
            TrackedObstacleArray containing current active tracks.
        """
        if timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")

        self._last_timestamp = timestamp_s

        # 1. Filter detections by confidence & transform to target frame ('odom')
        valid_detections: List[DynamicObstacleDetection] = []
        for det in detections:
            if det.confidence < self.config.min_detection_confidence:
                continue

            if det.frame_id != self.config.target_frame:
                if robot_pose is None:
                    raise ValueError(
                        f"Detection in '{det.frame_id}' cannot be transformed to "
                        f"'{self.config.target_frame}' without robot_pose."
                    )
                det_odom = det.transform_to_frame(self.config.target_frame, robot_pose)
                valid_detections.append(det_odom)
            else:
                valid_detections.append(det)

        # 2. Extract active tracks for association
        active_track_ids = list(self._tracks.keys())
        track_positions = []
        for tid in active_track_ids:
            track = self._tracks[tid]
            pos = track.predict_position(timestamp_s)
            track_positions.append(pos)

        det_positions = [det.bbox_center for det in valid_detections]

        t_pos_arr = np.array(track_positions, dtype=np.float64).reshape(-1, 2)
        d_pos_arr = np.array(det_positions, dtype=np.float64).reshape(-1, 2)

        # 3. Associate detections with active tracks
        matches, unmatched_track_indices, unmatched_det_indices = self.association.associate(
            t_pos_arr,
            d_pos_arr,
            max_distance_m=self.config.max_association_distance_m,
        )

        # 4. Update matched tracks
        for t_idx, d_idx in matches:
            track_id = active_track_ids[t_idx]
            self._tracks[track_id].update(valid_detections[d_idx], timestamp_s)

        # 5. Mark unmatched tracks as missed
        for t_idx in unmatched_track_indices:
            track_id = active_track_ids[t_idx]
            self._tracks[track_id].mark_missed(timestamp_s)

        # 6. Create new tracks for unmatched detections
        for d_idx in unmatched_det_indices:
            new_id = self._next_track_id
            self._next_track_id += 1
            new_track = Track(new_id, valid_detections[d_idx], self.config)
            self._tracks[new_id] = new_track

        # 7. Prune expired tracks
        expired_ids = [
            tid for tid, track in self._tracks.items() if track.status == TrackStatus.EXPIRED
        ]
        for tid in expired_ids:
            del self._tracks[tid]

        # 8. Build TrackedObstacleArray output
        current_states = [track.to_state() for track in self._tracks.values()]
        # Sort by track_id for deterministic output
        current_states.sort(key=lambda s: s.track_id)

        return TrackedObstacleArray(
            header_timestamp=float(timestamp_s),
            frame_id=self.config.target_frame,
            tracks=current_states,
        )
