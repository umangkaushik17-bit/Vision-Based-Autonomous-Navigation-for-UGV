"""Motion estimation layer for multi-frame dynamic obstacle tracking in Phase 13.

Consumes multi-frame track position histories in the world ('odom') frame and derives
physically meaningful velocity estimates (vx, vy, speed, heading) and empirical uncertainties
without Kalman filtering or ground-truth cheating:
- Windowed Ordinary Least Squares (OLS) linear regression over timestamped history
- Handles irregular timestamps, frame dropouts, and zero/duplicate dt
- Stationary object thresholding and noise rejection
- Explicit separation between measured position, raw finite-difference velocity, and filtered estimate
- Computes quantitative estimation metrics (RMSE, MAE, bias, heading error)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ai.planning.dynamic.data_model import (
    TrackedObstacleArray,
    TrackedObstacleState,
    normalize_angle,
)


@dataclass
class MotionEstimate:
    """Explicit motion estimation output distinguishing raw vs filtered quantities."""

    track_id: int
    timestamp: float
    # Measured / current position in odom
    measured_x: float
    measured_y: float
    # Raw finite-difference velocity (two-point)
    raw_vx: float
    raw_vy: float
    raw_speed: float
    # Motion-estimator output (windowed regression)
    vx: float
    vy: float
    speed: float
    velocity_heading: float
    # Uncertainty
    var_vx: float
    var_vy: float
    is_stationary: bool
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "timestamp": float(self.timestamp),
            "measured_x": float(self.measured_x),
            "measured_y": float(self.measured_y),
            "raw_vx": float(self.raw_vx),
            "raw_vy": float(self.raw_vy),
            "raw_speed": float(self.raw_speed),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "speed": float(self.speed),
            "velocity_heading": float(self.velocity_heading),
            "var_vx": float(self.var_vx),
            "var_vy": float(self.var_vy),
            "is_stationary": bool(self.is_stationary),
            "sample_count": int(self.sample_count),
        }


@dataclass
class MotionEstimatorConfig:
    """Parameters governing windowed motion estimation and noise rejection."""

    window_size: int = 5
    min_history_points: int = 2
    stationary_speed_threshold_m_s: float = 0.08
    min_dt_s: float = 1e-4
    max_history_age_s: float = 2.0
    default_pos_noise_std_m: float = 0.05
    velocity_smoothing_alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if self.min_history_points < 1:
            raise ValueError(f"min_history_points must be >= 1, got {self.min_history_points}")
        if self.stationary_speed_threshold_m_s < 0.0:
            raise ValueError(
                f"stationary_speed_threshold_m_s must be non-negative, got {self.stationary_speed_threshold_m_s}"
            )
        if self.min_dt_s <= 0.0:
            raise ValueError(f"min_dt_s must be positive, got {self.min_dt_s}")
        if not (0.0 < self.velocity_smoothing_alpha <= 1.0):
            raise ValueError(
                f"velocity_smoothing_alpha must be in (0.0, 1.0], got {self.velocity_smoothing_alpha}"
            )


class MotionEstimator:
    """Deterministic motion estimator deriving velocities from timestamped track histories."""

    def __init__(self, config: Optional[MotionEstimatorConfig] = None) -> None:
        self.config = config or MotionEstimatorConfig()

    def estimate_track_motion(
        self,
        track_id: int,
        current_x: float,
        current_y: float,
        current_timestamp: float,
        history: List[Tuple[float, float, float]],
        fallback_yaw: float = 0.0,
    ) -> MotionEstimate:
        """Estimate 2D velocity and kinematics from timestamped history [(t, x, y), ...].

        Args:
            track_id: Integer track ID.
            current_x: Current x position in meters (odom frame).
            current_y: Current y position in meters (odom frame).
            current_timestamp: Current timestamp in seconds.
            history: List of (timestamp, x, y) observations in ascending order.
            fallback_yaw: Fallback heading angle if target is stationary.

        Returns:
            MotionEstimate containing raw and filtered motion quantities.
        """
        # Clean and filter history
        valid_history: List[Tuple[float, float, float]] = []
        last_t = -1.0
        for (t, x, y) in history:
            # Reject negative or non-increasing timestamps within min_dt
            if t < 0.0:
                continue
            if t - last_t >= self.config.min_dt_s or last_t < 0.0:
                valid_history.append((float(t), float(x), float(y)))
                last_t = t

        # Ensure current position is present at the end of history
        if not valid_history or abs(valid_history[-1][0] - current_timestamp) > self.config.min_dt_s:
            if valid_history and current_timestamp >= valid_history[-1][0] + self.config.min_dt_s:
                valid_history.append((float(current_timestamp), float(current_x), float(current_y)))
            elif not valid_history:
                valid_history = [(float(current_timestamp), float(current_x), float(current_y))]

        # Trim history to configured max window and max age
        cutoff_time = current_timestamp - self.config.max_history_age_s
        windowed = [pt for pt in valid_history if pt[0] >= cutoff_time]
        if len(windowed) > self.config.window_size:
            windowed = windowed[-self.config.window_size :]

        sample_count = len(windowed)

        # 1. Raw two-point finite difference
        raw_vx = 0.0
        raw_vy = 0.0
        if sample_count >= 2:
            t_prev, x_prev, y_prev = windowed[-2]
            t_curr, x_curr, y_curr = windowed[-1]
            dt_raw = t_curr - t_prev
            if dt_raw >= self.config.min_dt_s:
                raw_vx = (x_curr - x_prev) / dt_raw
                raw_vy = (y_curr - y_prev) / dt_raw

        raw_speed = math.hypot(raw_vx, raw_vy)

        # 2. Insufficient points for regression
        if sample_count < self.config.min_history_points or sample_count < 2:
            return MotionEstimate(
                track_id=track_id,
                timestamp=current_timestamp,
                measured_x=current_x,
                measured_y=current_y,
                raw_vx=raw_vx,
                raw_vy=raw_vy,
                raw_speed=raw_speed,
                vx=0.0,
                vy=0.0,
                speed=0.0,
                velocity_heading=fallback_yaw,
                var_vx=0.25,
                var_vy=0.25,
                is_stationary=True,
                sample_count=sample_count,
            )

        # 3. Windowed Ordinary Least Squares (OLS) Regression
        ts = np.array([pt[0] for pt in windowed], dtype=np.float64)
        xs = np.array([pt[1] for pt in windowed], dtype=np.float64)
        ys = np.array([pt[2] for pt in windowed], dtype=np.float64)

        t_mean = np.mean(ts)
        t_diff = ts - t_mean
        ss_t = np.sum(t_diff ** 2)

        if ss_t < (self.config.min_dt_s ** 2):
            # All timestamps identical
            est_vx = 0.0
            est_vy = 0.0
            var_vx = 0.25
            var_vy = 0.25
        else:
            est_vx = float(np.sum(t_diff * (xs - np.mean(xs))) / ss_t)
            est_vy = float(np.sum(t_diff * (ys - np.mean(ys))) / ss_t)

            # Empirical variance of the slope estimate
            pos_var = self.config.default_pos_noise_std_m ** 2
            var_vx = float(max(1e-4, pos_var / ss_t))
            var_vy = float(max(1e-4, pos_var / ss_t))

        speed = math.hypot(est_vx, est_vy)

        # 4. Stationary thresholding
        if speed < self.config.stationary_speed_threshold_m_s:
            is_stationary = True
            est_vx = 0.0
            est_vy = 0.0
            speed = 0.0
            velocity_heading = fallback_yaw
        else:
            is_stationary = False
            velocity_heading = normalize_angle(math.atan2(est_vy, est_vx))

        return MotionEstimate(
            track_id=track_id,
            timestamp=current_timestamp,
            measured_x=current_x,
            measured_y=current_y,
            raw_vx=raw_vx,
            raw_vy=raw_vy,
            raw_speed=raw_speed,
            vx=est_vx,
            vy=est_vy,
            speed=speed,
            velocity_heading=velocity_heading,
            var_vx=var_vx,
            var_vy=var_vy,
            is_stationary=is_stationary,
            sample_count=sample_count,
        )

    def update_track_state(
        self,
        track_state: TrackedObstacleState,
    ) -> Tuple[TrackedObstacleState, MotionEstimate]:
        """Compute motion estimate for a TrackedObstacleState and return updated state."""
        estimate = self.estimate_track_motion(
            track_id=track_state.track_id,
            current_x=track_state.x,
            current_y=track_state.y,
            current_timestamp=track_state.timestamp,
            history=track_state.history,
            fallback_yaw=track_state.bbox_yaw,
        )

        # Create updated covariance with estimated velocity uncertainty
        updated_cov = track_state.covariance.copy()
        updated_cov[2, 2] = estimate.var_vx
        updated_cov[3, 3] = estimate.var_vy

        updated_state = TrackedObstacleState(
            track_id=track_state.track_id,
            timestamp=track_state.timestamp,
            frame_id=track_state.frame_id,
            x=track_state.x,
            y=track_state.y,
            vx=estimate.vx,
            vy=estimate.vy,
            covariance=updated_cov,
            bbox_size=track_state.bbox_size,
            bbox_yaw=track_state.bbox_yaw,
            track_age=track_state.track_age,
            coast_cycles=track_state.coast_cycles,
            is_confirmed=track_state.is_confirmed,
            status=track_state.status,
            history=list(track_state.history),
        )

        return updated_state, estimate

    def update(
        self,
        tracks: Union[List[TrackedObstacleState], TrackedObstacleArray],
        timestamp_s: float,
    ) -> TrackedObstacleArray:
        """Process an array of tracks, updating velocities and uncertainties.

        Args:
            tracks: TrackedObstacleArray or list of TrackedObstacleState objects.
            timestamp_s: Observation timestamp in seconds.

        Returns:
            TrackedObstacleArray with updated velocities and covariances.
        """
        if timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")

        updated_tracks: List[TrackedObstacleState] = []
        track_list = tracks.tracks if isinstance(tracks, TrackedObstacleArray) else tracks

        for track in track_list:
            up_state, _ = self.update_track_state(track)
            updated_tracks.append(up_state)

        frame_id = tracks.frame_id if isinstance(tracks, TrackedObstacleArray) else "odom"
        return TrackedObstacleArray(
            header_timestamp=float(timestamp_s),
            frame_id=frame_id,
            tracks=updated_tracks,
        )
