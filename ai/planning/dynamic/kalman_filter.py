"""Discrete-time Constant-Velocity (CV) Kalman Filter for dynamic obstacle tracking in Phase 13.

Implements a recursive, probabilistic linear state estimator in the canonical world ('odom') frame:
- State vector: x = [px, py, vx, vy]^T
- Covariance: P (4x4 positive semi-definite)
- Motion model: Constant-Velocity (CV) with dynamic time delta dt:
    px_{k+1} = px_k + vx_k * dt
    py_{k+1} = py_k + vy_k * dt
    vx_{k+1} = vx_k
    vy_{k+1} = vy_k
- Process noise: Continuous White-Noise Acceleration (CWNA) discretized over actual dt:
    Q(dt) = q * [[dt^3/3, 0, dt^2/2, 0],
                 [0, dt^3/3, 0, dt^2/2],
                 [dt^2/2, 0, dt, 0],
                 [0, dt^2/2, 0, dt]]
- Measurement model: Position observations z = [px, py]^T, H = [[1,0,0,0],[0,1,0,0]]
- Measurement noise: R = diag(sigma_p^2, sigma_p^2)
- Outlier gating: Mahalanobis innovation distance squared d_M^2 = y^T S^{-1} y <= chi^2_2 threshold
- Numerical stability: Joseph stabilized covariance update:
    P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
  with explicit symmetrization and positive-definiteness enforcement.
- Dropouts & coasting: Predict-only cycles during observation loss with monotonically expanding covariance.
- Reacquisition: Smooth recovery without state divergence or ID disruption.
- Irregular timestamps: Explicit handling of dt variations, duplicate timestamps, and monotonicity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ai.planning.dynamic.data_model import normalize_angle


@dataclass
class KalmanFilterConfig:
    """Configuration parameters for the Constant-Velocity Kalman Track Filter.

    Attributes:
        process_noise_accel: Continuous acceleration power spectral density q (m^2/s^3).
            Represents expected magnitude of unmodeled target accelerations.
        measurement_noise_pos: Standard deviation of position sensor noise sigma_p (m).
        initial_pos_variance: Initial diagonal variance for px, py in P_0 (m^2).
        initial_vel_variance: Initial diagonal variance for vx, vy in P_0 ((m/s)^2).
        gate_chi2_threshold: Chi-square innovation threshold for 2 degrees of freedom (DOF).
            Default 9.21 corresponds to 99% confidence interval. Outliers exceeding this are rejected.
        enable_gating: If True, reject measurements exceeding gate_chi2_threshold.
        max_predict_dt_s: Maximum allowable dt for a single prediction step (s).
        min_predict_dt_s: Minimum dt threshold below which prediction is treated as identity (s).
    """

    process_noise_accel: float = 1.0
    measurement_noise_pos: float = 0.05
    initial_pos_variance: float = 0.04  # 0.20 m std
    initial_vel_variance: float = 1.0   # 1.00 m/s std
    gate_chi2_threshold: float = 9.21   # 2-DOF 99% chi-square threshold
    enable_gating: bool = True
    max_predict_dt_s: float = 5.0
    min_predict_dt_s: float = 1e-5

    def __post_init__(self) -> None:
        if self.process_noise_accel <= 0.0:
            raise ValueError(
                f"process_noise_accel must be strictly positive, got {self.process_noise_accel}"
            )
        if self.measurement_noise_pos <= 0.0:
            raise ValueError(
                f"measurement_noise_pos must be strictly positive, got {self.measurement_noise_pos}"
            )
        if self.initial_pos_variance <= 0.0:
            raise ValueError(
                f"initial_pos_variance must be strictly positive, got {self.initial_pos_variance}"
            )
        if self.initial_vel_variance <= 0.0:
            raise ValueError(
                f"initial_vel_variance must be strictly positive, got {self.initial_vel_variance}"
            )
        if self.gate_chi2_threshold <= 0.0:
            raise ValueError(
                f"gate_chi2_threshold must be strictly positive, got {self.gate_chi2_threshold}"
            )
        if self.max_predict_dt_s <= 0.0:
            raise ValueError(
                f"max_predict_dt_s must be strictly positive, got {self.max_predict_dt_s}"
            )
        if self.min_predict_dt_s < 0.0:
            raise ValueError(
                f"min_predict_dt_s must be non-negative, got {self.min_predict_dt_s}"
            )


class KalmanTrackFilter:
    """Recursive 2D Constant-Velocity state estimator for dynamic obstacle tracking."""

    # Measurement matrix H observing 2D position [px, py] from state [px, py, vx, vy]
    _H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)

    def __init__(self, config: Optional[KalmanFilterConfig] = None) -> None:
        self.config = config or KalmanFilterConfig()

        # State vector [px, py, vx, vy]^T
        self._x: Optional[np.ndarray] = None
        # Covariance matrix P (4x4)
        self._P: Optional[np.ndarray] = None

        self._last_timestamp: float = -1.0
        self._initialized: bool = False

        # Diagnostic tracking
        self._predict_count: int = 0
        self._update_count: int = 0
        self._rejected_count: int = 0

        # Measurement noise covariance R (2x2)
        r_var = float(self.config.measurement_noise_pos ** 2)
        self._R = np.array([
            [r_var, 0.0],
            [0.0, r_var],
        ], dtype=np.float64)

    @property
    def is_initialized(self) -> bool:
        """Whether the filter has been initialized with a valid observation."""
        return self._initialized

    @property
    def last_timestamp(self) -> float:
        """Timestamp of the most recent state estimate in seconds."""
        return self._last_timestamp

    @property
    def position(self) -> Tuple[float, float]:
        """Current estimated (px, py) in world frame."""
        if not self._initialized or self._x is None:
            raise RuntimeError("Filter is not initialized.")
        return (float(self._x[0]), float(self._x[1]))

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated (vx, vy) in world frame."""
        if not self._initialized or self._x is None:
            raise RuntimeError("Filter is not initialized.")
        return (float(self._x[2]), float(self._x[3]))

    @property
    def speed(self) -> float:
        """Estimated linear speed magnitude in m/s."""
        if not self._initialized or self._x is None:
            raise RuntimeError("Filter is not initialized.")
        return float(math.hypot(self._x[2], self._x[3]))

    @property
    def heading(self) -> float:
        """Estimated course heading angle in radians [-pi, pi]."""
        if not self._initialized or self._x is None:
            raise RuntimeError("Filter is not initialized.")
        return float(math.atan2(self._x[3], self._x[2]))

    @property
    def position_uncertainty(self) -> float:
        """Position standard deviation (Euclidean trace metric) in meters."""
        if not self._initialized or self._P is None:
            raise RuntimeError("Filter is not initialized.")
        return float(math.sqrt(max(0.0, self._P[0, 0] + self._P[1, 1])))

    @property
    def velocity_uncertainty(self) -> float:
        """Velocity standard deviation (Euclidean trace metric) in m/s."""
        if not self._initialized or self._P is None:
            raise RuntimeError("Filter is not initialized.")
        return float(math.sqrt(max(0.0, self._P[2, 2] + self._P[3, 3])))

    @property
    def rejection_stats(self) -> Tuple[int, int]:
        """Returns (rejected_measurements_count, total_update_attempts)."""
        return (self._rejected_count, self._update_count)

    def initialize(
        self,
        position: Tuple[float, float],
        timestamp_s: float,
        initial_velocity: Tuple[float, float] = (0.0, 0.0),
        initial_covariance: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize filter state and covariance at a given observation.

        Args:
            position: Initial (px, py) in canonical world frame.
            timestamp_s: Observation timestamp in seconds (must be >= 0).
            initial_velocity: Prior (vx, vy) in canonical world frame (default: (0, 0)).
            initial_covariance: Optional custom 4x4 covariance matrix.
        """
        if timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")

        px, py = float(position[0]), float(position[1])
        vx, vy = float(initial_velocity[0]), float(initial_velocity[1])

        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(vx) and math.isfinite(vy)):
            raise ValueError(f"Initial state values must be finite, got pos={position}, vel={initial_velocity}")

        self._x = np.array([px, py, vx, vy], dtype=np.float64)

        if initial_covariance is not None:
            init_P = np.asarray(initial_covariance, dtype=np.float64)
            if init_P.shape != (4, 4):
                raise ValueError(f"initial_covariance must have shape (4, 4), got {init_P.shape}")
            self._P = self._validate_and_symmetrize_cov(init_P)
        else:
            self._P = np.diag([
                float(self.config.initial_pos_variance),
                float(self.config.initial_pos_variance),
                float(self.config.initial_vel_variance),
                float(self.config.initial_vel_variance),
            ]).astype(np.float64)

        self._last_timestamp = float(timestamp_s)
        self._initialized = True
        self._predict_count = 0
        self._update_count = 0
        self._rejected_count = 0

    def predict(self, timestamp_s: float) -> np.ndarray:
        """Propagate state and covariance forward to timestamp_s using CV motion model.

        Args:
            timestamp_s: Target timestamp in seconds (must be >= current last_timestamp).

        Returns:
            Predicted state vector [px, py, vx, vy]^T.
        """
        if not self._initialized or self._x is None or self._P is None:
            raise RuntimeError("Filter must be initialized before predict.")

        if timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {timestamp_s}")

        dt = float(timestamp_s) - self._last_timestamp

        # Policy for non-monotonic timestamps
        if dt < -self.config.min_predict_dt_s:
            raise ValueError(
                f"Non-monotonic timestamp: target {timestamp_s:.6f}s is prior to last "
                f"estimate {self._last_timestamp:.6f}s (dt={dt:.6f}s)"
            )

        # Policy for duplicate / near-zero timestamps
        if dt < self.config.min_predict_dt_s:
            # Zero-time elapsed: state and covariance remain identical
            self._last_timestamp = float(timestamp_s)
            return self._x.copy()

        # Handle large dt gracefully (cap effective dt for noise to avoid numerical overflow)
        effective_dt = min(dt, self.config.max_predict_dt_s)

        # 1. State Transition Matrix F(dt)
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 2. Continuous White-Noise Acceleration (CWNA) Process Noise Matrix Q(dt)
        q = float(self.config.process_noise_accel)
        dt2 = effective_dt * effective_dt
        dt3 = dt2 * effective_dt

        Q = q * np.array([
            [dt3 / 3.0, 0.0,       dt2 / 2.0, 0.0      ],
            [0.0,       dt3 / 3.0, 0.0,       dt2 / 2.0],
            [dt2 / 2.0, 0.0,       effective_dt, 0.0   ],
            [0.0,       dt2 / 2.0, 0.0,       effective_dt],
        ], dtype=np.float64)

        # 3. Predict state: x_k|k-1 = F * x_k-1|k-1
        new_x = F @ self._x

        # 4. Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        new_P = F @ self._P @ F.T + Q

        # Enforce symmetry and validate positive definiteness
        self._P = self._validate_and_symmetrize_cov(new_P)
        self._x = new_x
        self._last_timestamp = float(timestamp_s)
        self._predict_count += 1

        return self._x.copy()

    def update(
        self,
        measurement: Tuple[float, float],
        timestamp_s: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Update filter state with a 2D position observation [px, py].

        Optionally advances time to timestamp_s via predict() if timestamp_s > last_timestamp.

        Args:
            measurement: (px, py) observation in canonical world frame.
            timestamp_s: Optional observation timestamp in seconds.

        Returns:
            Tuple of:
                accepted (bool): True if measurement was within gating threshold and incorporated;
                                 False if rejected as an outlier.
                mahalanobis_dist (float): Innovation Mahalanobis distance d_M.
        """
        if not self._initialized or self._x is None or self._P is None:
            raise RuntimeError("Filter must be initialized before update.")

        # If timestamp is provided, advance state to observation time
        if timestamp_s is not None:
            dt = float(timestamp_s) - self._last_timestamp
            if dt < -self.config.min_predict_dt_s:
                raise ValueError(
                    f"Non-monotonic timestamp in update: {timestamp_s:.6f}s < {self._last_timestamp:.6f}s"
                )
            if dt >= self.config.min_predict_dt_s:
                self.predict(timestamp_s)

        mx, my = float(measurement[0]), float(measurement[1])
        if not (math.isfinite(mx) and math.isfinite(my)):
            raise ValueError(f"Measurement values must be finite, got {measurement}")

        z = np.array([mx, my], dtype=np.float64)
        self._update_count += 1

        # 1. Innovation residual: y = z - H * x_pred
        y = z - (self._H @ self._x)

        # 2. Innovation covariance: S = H * P_pred * H^T + R
        S = self._H @ self._P @ self._H.T + self._R

        # Invert S stably (2x2 matrix)
        det_S = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if det_S < 1e-12:
            S_inv = np.linalg.pinv(S)
        else:
            inv_det = 1.0 / det_S
            S_inv = np.array([
                [ S[1, 1] * inv_det, -S[0, 1] * inv_det],
                [-S[1, 0] * inv_det,  S[0, 0] * inv_det],
            ], dtype=np.float64)

        # 3. Mahalanobis distance squared: d_M^2 = y^T * S^{-1} * y
        d_M_sq = float(y.T @ S_inv @ y)
        d_M = float(math.sqrt(max(0.0, d_M_sq)))

        # 4. Outlier Gating
        if self.config.enable_gating and d_M_sq > self.config.gate_chi2_threshold:
            # Reject observation: preserve predicted state and covariance
            self._rejected_count += 1
            return (False, d_M)

        # 5. Kalman Gain: K = P_pred * H^T * S^{-1}  (4x2)
        K = self._P @ self._H.T @ S_inv

        # 6. State update: x_k|k = x_k|k-1 + K * y
        self._x = self._x + K @ y

        # 7. Numerically stable Joseph form covariance update:
        # P_k|k = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(4, dtype=np.float64) - K @ self._H
        updated_P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        # Symmetrize and validate positive semi-definiteness
        self._P = self._validate_and_symmetrize_cov(updated_P)

        return (True, d_M)

    def predict_and_update(
        self,
        measurement: Tuple[float, float],
        timestamp_s: float,
    ) -> Tuple[bool, float]:
        """Convenience atomic predict + update cycle."""
        return self.update(measurement, timestamp_s=timestamp_s)

    def get_state(self) -> np.ndarray:
        """Return a copy of the current 4D state vector [px, py, vx, vy]^T."""
        if not self._initialized or self._x is None:
            raise RuntimeError("Filter is not initialized.")
        return self._x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the current 4x4 state covariance matrix P."""
        if not self._initialized or self._P is None:
            raise RuntimeError("Filter is not initialized.")
        return self._P.copy()

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._x = None
        self._P = None
        self._last_timestamp = -1.0
        self._initialized = False
        self._predict_count = 0
        self._update_count = 0
        self._rejected_count = 0

    def _validate_and_symmetrize_cov(self, P: np.ndarray) -> np.ndarray:
        """Ensure numerical symmetry, finiteness, and positive semi-definiteness."""
        if not np.all(np.isfinite(P)):
            raise FloatingPointError(f"Covariance matrix contains NaN or Inf values:\n{P}")

        # Enforce exact symmetry: P = 0.5 * (P + P^T)
        sym_P = 0.5 * (P + P.T)

        # Check eigenvalues
        eigvals, eigvecs = np.linalg.eigh(sym_P)
        min_eig = np.min(eigvals)

        # If slightly negative due to floating-point truncation, project onto positive semi-definite cone
        if min_eig < 0.0:
            if min_eig < -1e-5:
                raise FloatingPointError(
                    f"Covariance matrix lost positive definiteness (min eigenvalue: {min_eig:.6e}):\n{sym_P}"
                )
            # Clip negative eigenvalues to zero and reconstruct
            clipped_eigvals = np.maximum(eigvals, 0.0)
            sym_P = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T
            sym_P = 0.5 * (sym_P + sym_P.T)

        return sym_P
