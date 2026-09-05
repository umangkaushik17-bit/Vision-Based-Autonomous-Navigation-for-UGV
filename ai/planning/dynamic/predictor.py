"""Multi-step future motion prediction module for tracked dynamic obstacles (Milestone 13.6).

Consumes filtered obstacle state [px, py, vx, vy]^T and 4x4 covariance P from Milestone 13.5
(TrackedObstacleState) and generates spatiotemporal future trajectory predictions:
- Motion model: Constant-Velocity (CV) forward extrapolation
    px(t + dt) = px + vx * dt
    py(t + dt) = py + vy * dt
    vx(t + dt) = vx
    vy(t + dt) = vy
- Covariance propagation: Continuous White-Noise Acceleration (CWNA)
    P_future(dt) = F(dt) * P_current * F(dt)^T + Q_future(dt)
  where F(dt) and Q(dt) are mathematically consistent with the Milestone 13.5 Kalman filter.
- Spatial uncertainty:
    2D position uncertainty radius derived from the maximum eigenvalue of position covariance:
    r_unc(dt) = sigma_scale * sqrt(max(0, lambda_max(P_pos(dt))))
  with expanded oriented bounding boxes per prediction waypoint.
- Lifecycle & Confidence awareness:
    TRACKING: Nominal prediction up to configured horizon.
    DEGRADED: Prediction allowed, with uncertainty inflated by degraded_noise_multiplier.
    LOST: Prediction capped at lost_max_horizon_s, with uncertainty inflated by lost_noise_multiplier.
    EXPIRED: Prediction rejected / excluded (cannot predict terminated tracks).
- Strict State Isolation:
    Consumes strictly filtered track state and covariance. Completely decoupled from external reference data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)


@dataclass
class PredictorConfig:
    """Configuration parameters for the Constant-Velocity Motion Predictor.

    Attributes:
        default_horizon_s: Default maximum future time horizon for predictions (seconds).
        default_dt_s: Default time step between successive prediction waypoints (seconds).
        process_noise_spectral_density: Continuous white-noise acceleration spectral density q (m^2/s^3).
            Consistent with Kalman filter process_noise_accel.
        sigma_scale: Confidence multiplier for position uncertainty radius semi-major axis.
            Default 3.0 corresponds to ~99.7% 2D confidence ellipse.
        degraded_noise_multiplier: Factor multiplying process noise spectral density for DEGRADED tracks.
        lost_noise_multiplier: Factor multiplying process noise spectral density for LOST tracks.
        lost_max_horizon_s: Maximum allowed prediction horizon for LOST tracks (seconds).
            Prevents stale/lost tracks from generating indefinite future predictions.
        min_dt_s: Minimum allowable dt between distinct requested time offsets (seconds).
        max_horizon_s: Upper safety bound on requested prediction horizon (seconds).
    """

    default_horizon_s: float = 2.0
    default_dt_s: float = 0.25
    process_noise_spectral_density: float = 1.0
    sigma_scale: float = 3.0
    degraded_noise_multiplier: float = 2.0
    lost_noise_multiplier: float = 4.0
    lost_max_horizon_s: float = 1.0
    min_dt_s: float = 1e-5
    max_horizon_s: float = 10.0

    def __post_init__(self) -> None:
        if self.default_horizon_s < 0.0:
            raise ValueError(
                f"default_horizon_s must be non-negative, got {self.default_horizon_s}"
            )
        if self.default_dt_s <= 0.0:
            raise ValueError(f"default_dt_s must be positive, got {self.default_dt_s}")
        if self.process_noise_spectral_density < 0.0:
            raise ValueError(
                "process_noise_spectral_density must be non-negative, "
                f"got {self.process_noise_spectral_density}"
            )
        if self.sigma_scale <= 0.0:
            raise ValueError(f"sigma_scale must be strictly positive, got {self.sigma_scale}")
        if self.degraded_noise_multiplier < 1.0:
            raise ValueError(
                "degraded_noise_multiplier must be >= 1.0, "
                f"got {self.degraded_noise_multiplier}"
            )
        if self.lost_noise_multiplier < 1.0:
            raise ValueError(
                f"lost_noise_multiplier must be >= 1.0, got {self.lost_noise_multiplier}"
            )
        if self.lost_max_horizon_s < 0.0:
            raise ValueError(
                f"lost_max_horizon_s must be non-negative, got {self.lost_max_horizon_s}"
            )
        if self.min_dt_s < 0.0:
            raise ValueError(f"min_dt_s must be non-negative, got {self.min_dt_s}")
        if self.max_horizon_s < self.default_horizon_s:
            raise ValueError(
                f"max_horizon_s ({self.max_horizon_s}) must be >= default_horizon_s ({self.default_horizon_s})"
            )


class MotionPredictor:
    """Predicts future spatiotemporal trajectories and uncertainty bounds for dynamic obstacles."""

    def __init__(self, config: Optional[PredictorConfig] = None) -> None:
        self.config = config or PredictorConfig()

    def validate_time_offsets(self, offsets: Sequence[float]) -> List[float]:
        """Validate an explicit sequence of prediction time offsets.

        Rules:
        - Must be non-empty.
        - Must contain only finite numbers (no NaN, no Inf).
        - Must contain only non-negative numbers (dt >= 0.0).
        - Must be strictly monotonically increasing (no duplicate or decreasing values).

        Args:
            offsets: Sequence of future time offsets in seconds.

        Returns:
            Validated list of float time offsets.

        Raises:
            ValueError: If any validation condition is violated.
        """
        if not offsets:
            raise ValueError("time_offsets sequence must not be empty.")

        validated: List[float] = []
        prev_val: Optional[float] = None

        for val in offsets:
            f_val = float(val)
            if not math.isfinite(f_val):
                raise ValueError(f"Non-finite prediction offset rejected: {val}")
            if f_val < 0.0:
                raise ValueError(f"Negative prediction offset rejected: {val}")
            if prev_val is not None:
                if abs(f_val - prev_val) < self.config.min_dt_s:
                    raise ValueError(
                        f"Duplicate or sub-threshold prediction offset rejected: {f_val} "
                        f"(previous={prev_val}, threshold={self.config.min_dt_s})"
                    )
                if f_val < prev_val:
                    raise ValueError(
                        f"Non-monotonic prediction offset sequence rejected: {f_val} < {prev_val}"
                    )
            validated.append(f_val)
            prev_val = f_val

        return validated

    def generate_time_offsets(
        self,
        horizon_s: Optional[float] = None,
        dt_s: Optional[float] = None,
    ) -> List[float]:
        """Generate a uniform sequence of prediction time offsets from 0.0 to horizon_s.

        Args:
            horizon_s: Future time horizon in seconds (default: config.default_horizon_s).
            dt_s: Timestep between waypoints in seconds (default: config.default_dt_s).

        Returns:
            List of monotonically increasing time offsets [0.0, dt, 2*dt, ..., horizon_s].
        """
        horizon = float(self.config.default_horizon_s if horizon_s is None else horizon_s)
        dt = float(self.config.default_dt_s if dt_s is None else dt_s)

        if not math.isfinite(horizon):
            raise ValueError(f"horizon_s must be finite, got {horizon_s}")
        if not math.isfinite(dt):
            raise ValueError(f"dt_s must be finite, got {dt_s}")
        if horizon < 0.0:
            raise ValueError(f"horizon_s must be non-negative, got {horizon}")
        if horizon > self.config.max_horizon_s:
            raise ValueError(
                f"horizon_s ({horizon}s) exceeds max_horizon_s ({self.config.max_horizon_s}s)"
            )
        if dt <= 0.0:
            raise ValueError(f"dt_s must be strictly positive, got {dt}")

        if horizon < 1e-9:
            return [0.0]

        num_steps = int(round(horizon / dt))
        if num_steps <= 0:
            return [0.0, horizon]

        offsets = [round(i * dt, 8) for i in range(num_steps + 1)]
        # Ensure exact horizon endpoint without floating-point drift
        offsets[-1] = horizon
        return self.validate_time_offsets(offsets)

    def _propagate_state_and_cov(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        dt: float,
        q_eff: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute state, covariance, and uncertainty radius at future offset dt.

        Args:
            x0: Initial state vector [px, py, vx, vy]^T.
            P0: Initial 4x4 covariance matrix.
            dt: Future time offset in seconds (>= 0.0).
            q_eff: Effective continuous acceleration process noise spectral density.

        Returns:
            Tuple of:
            - x_pred: Predicted state vector [px, py, vx, vy]^T.
            - P_pos: 2x2 position covariance matrix at dt.
            - r_unc: Positional uncertainty radius in meters.
        """
        if dt < 1e-9:
            # Zero-horizon case: state and covariance are unchanged
            P_pos = P0[:2, :2].copy()
            eigvals = np.linalg.eigvalsh(P_pos)
            max_eigval = max(0.0, float(np.max(eigvals)))
            r_unc = float(self.config.sigma_scale * math.sqrt(max_eigval))
            return x0.copy(), P_pos, r_unc

        # 1. State transition matrix F(dt)
        # [px + vx*dt, py + vy*dt, vx, vy]^T
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 2. State propagation
        x_pred = F @ x0

        # 3. Continuous White-Noise Acceleration (CWNA) process noise Q(dt)
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = q_eff * np.array([
            [dt3 / 3.0, 0.0,       dt2 / 2.0, 0.0      ],
            [0.0,       dt3 / 3.0, 0.0,       dt2 / 2.0],
            [dt2 / 2.0, 0.0,       dt,        0.0      ],
            [0.0,       dt2 / 2.0, 0.0,       dt       ],
        ], dtype=np.float64)

        # 4. Covariance propagation P = F * P0 * F^T + Q
        P_future = F @ P0 @ F.T + Q
        # Enforce numerical symmetry
        P_future = 0.5 * (P_future + P_future.T)

        # 5. Extract 2D position covariance and uncertainty radius
        P_pos = P_future[:2, :2].copy()
        eigvals = np.linalg.eigvalsh(P_pos)
        max_eigval = max(0.0, float(np.max(eigvals)))
        r_unc = float(self.config.sigma_scale * math.sqrt(max_eigval))

        return x_pred, P_pos, r_unc

    def predict_track(
        self,
        track: TrackedObstacleState,
        horizon_s: Optional[float] = None,
        dt_s: Optional[float] = None,
        time_offsets_s: Optional[Sequence[float]] = None,
    ) -> PredictedObstacleTrajectory:
        """Predict future motion trajectory and uncertainty envelope for a single track.

        Args:
            track: TrackedObstacleState to predict from.
            horizon_s: Target future prediction horizon in seconds.
            dt_s: Timestep between waypoints in seconds.
            time_offsets_s: Explicit arbitrary monotonic sequence of future offsets.
                If supplied, overrides horizon_s and dt_s.

        Returns:
            PredictedObstacleTrajectory containing waypoints and spatial bounding polygons.

        Raises:
            ValueError: If track is EXPIRED or offsets are invalid.
        """
        status = track.status.upper()
        if status == "EXPIRED":
            raise ValueError(
                f"Cannot predict trajectory for EXPIRED track (id={track.track_id}). "
                "Expired tracks must not be predicted."
            )

        if horizon_s is not None:
            f_h = float(horizon_s)
            if not math.isfinite(f_h):
                raise ValueError(f"horizon_s must be finite, got {horizon_s}")
            if f_h < 0.0:
                raise ValueError(f"horizon_s must be non-negative, got {horizon_s}")
        if dt_s is not None:
            f_dt = float(dt_s)
            if not math.isfinite(f_dt):
                raise ValueError(f"dt_s must be finite, got {dt_s}")
            if f_dt <= 0.0:
                raise ValueError(f"dt_s must be strictly positive, got {dt_s}")

        # Determine effective process noise spectral density and horizon limits based on status
        q_nominal = float(self.config.process_noise_spectral_density)
        if status == "DEGRADED":
            q_eff = q_nominal * float(self.config.degraded_noise_multiplier)
            max_allowed_horizon = float(self.config.max_horizon_s)
        elif status == "LOST":
            q_eff = q_nominal * float(self.config.lost_noise_multiplier)
            max_allowed_horizon = float(self.config.lost_max_horizon_s)
        else:
            # TRACKING, NEW, or any standard active track
            q_eff = q_nominal
            max_allowed_horizon = float(self.config.max_horizon_s)

        # Resolve time offsets
        if time_offsets_s is not None:
            raw_offsets = self.validate_time_offsets(time_offsets_s)
            # For LOST tracks, truncate offsets exceeding lost_max_horizon_s
            offsets = [t for t in raw_offsets if t <= max_allowed_horizon + 1e-6]
            if not offsets:
                offsets = [0.0]
            effective_horizon = offsets[-1]
            if len(offsets) >= 2:
                effective_dt = offsets[1] - offsets[0]
            else:
                effective_dt = float(self.config.default_dt_s if dt_s is None else dt_s)
        else:
            req_horizon = float(self.config.default_horizon_s if horizon_s is None else horizon_s)
            capped_horizon = min(req_horizon, max_allowed_horizon)
            req_dt = float(self.config.default_dt_s if dt_s is None else dt_s)
            offsets = self.generate_time_offsets(capped_horizon, req_dt)
            effective_horizon = capped_horizon
            effective_dt = req_dt

        # Ensure effective_dt is strictly positive
        if effective_dt <= 0.0:
            effective_dt = float(self.config.default_dt_s)

        # Extract initial state vector and covariance
        x0 = np.array([track.x, track.y, track.vx, track.vy], dtype=np.float64)
        P0 = track.covariance.copy()

        waypoints: List[TrajectoryWaypoint] = []
        bounding_polygons: List[np.ndarray] = []

        for offset in offsets:
            x_pred, P_pos, r_unc = self._propagate_state_and_cov(x0, P0, offset, q_eff)
            t_wp = float(track.timestamp + offset)

            wp = TrajectoryWaypoint(
                time_offset=float(offset),
                timestamp=t_wp,
                x=float(x_pred[0]),
                y=float(x_pred[1]),
                vx=float(x_pred[2]),
                vy=float(x_pred[3]),
                covariance_2d=P_pos,
                uncertainty_radius=float(r_unc),
            )
            waypoints.append(wp)

            # Generate bounding polygon expanded by uncertainty radius
            poly = compute_obb_corners(
                (wp.x, wp.y),
                track.bbox_size,
                track.bbox_yaw,
                margin_m=wp.uncertainty_radius,
            )
            bounding_polygons.append(poly)

        return PredictedObstacleTrajectory(
            track_id=track.track_id,
            current_state=track,
            horizon_s=effective_horizon,
            dt=effective_dt,
            waypoints=waypoints,
            bounding_polygon_per_step=bounding_polygons,
        )

    def predict_tracks(
        self,
        tracks: Union[Sequence[TrackedObstacleState], TrackedObstacleArray],
        horizon_s: Optional[float] = None,
        dt_s: Optional[float] = None,
        time_offsets_s: Optional[Sequence[float]] = None,
    ) -> PredictedTrajectoryArray:
        """Predict future motion trajectories for multiple tracked obstacles.

        Args:
            tracks: Sequence of TrackedObstacleState or TrackedObstacleArray.
            horizon_s: Target future prediction horizon in seconds.
            dt_s: Timestep between waypoints in seconds.
            time_offsets_s: Optional sequence of explicit time offsets.

        Returns:
            PredictedTrajectoryArray containing predicted trajectories for all active tracks.
            EXPIRED tracks are cleanly omitted.
        """
        if isinstance(tracks, TrackedObstacleArray):
            track_list = tracks.tracks
            header_timestamp = float(tracks.header_timestamp)
            frame_id = str(tracks.frame_id)
        else:
            track_list = list(tracks)
            header_timestamp = float(track_list[0].timestamp) if track_list else 0.0
            frame_id = str(track_list[0].frame_id) if track_list else "odom"

        predictions: List[PredictedObstacleTrajectory] = []

        for track in track_list:
            if track.status.upper() == "EXPIRED":
                # Do not predict expired tracks
                continue
            pred = self.predict_track(
                track=track,
                horizon_s=horizon_s,
                dt_s=dt_s,
                time_offsets_s=time_offsets_s,
            )
            predictions.append(pred)

        return PredictedTrajectoryArray(
            header_timestamp=header_timestamp,
            frame_id=frame_id,
            predictions=predictions,
        )
