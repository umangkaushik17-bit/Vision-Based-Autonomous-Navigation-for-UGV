"""Data types, contracts, and measurement representations for GPS-denied localization fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PoseWithCovariance:
    """3D position and orientation with 6x6 spatial covariance."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 1e-3)


@dataclass
class TwistWithCovariance:
    """Linear and angular velocities with 6x6 covariance."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vroll: float = 0.0
    vpitch: float = 0.0
    vyaw: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 1e-3)


@dataclass
class WheelOdometryMeasurement:
    """Wheel encoder odometry measurement in robot base_link and odom frames."""

    timestamp: float
    vx: float                            # Forward linear velocity in base_link (m/s)
    vyaw: float                          # Angular velocity in base_link (rad/s)
    pose_x: float = 0.0                  # Dead-reckoned integrated X (m)
    pose_y: float = 0.0                  # Dead-reckoned integrated Y (m)
    pose_yaw: float = 0.0                # Dead-reckoned integrated Yaw (rad)
    twist_covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 0.02)
    pose_covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 0.10)
    frame_id: str = "odom"
    child_frame_id: str = "base_link"
    valid: bool = True


@dataclass
class IMUMeasurement:
    """6-Axis Inertial Measurement Unit reading (angular velocity + linear acceleration)."""

    timestamp: float
    angular_velocity: np.ndarray          # [wx, wy, wz] (rad/s)
    linear_acceleration: np.ndarray       # [ax, ay, az] (m/s^2)
    orientation_rpy: Optional[np.ndarray] = None  # [roll, pitch, yaw] if onboard AHRS available
    angular_velocity_covariance: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64) * 1e-4)
    linear_acceleration_covariance: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64) * 1e-3)
    orientation_covariance: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64) * 1e-4)
    frame_id: str = "imu_link"
    valid: bool = True


@dataclass
class StereoVIOMeasurement:
    """Stereo Visual Odometry measurement with metric scale observability and covariance."""

    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    pose_covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 0.05)
    twist_covariance: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64) * 0.05)
    inliers: int = 100
    tracking_state: str = "TRACKING_OK"
    valid: bool = True
    frame_id: str = "odom"
    child_frame_id: str = "base_link"


@dataclass
class FusedLocalizationState:
    """15-dimensional state estimate from the Multi-Sensor Fusion Extended Kalman Filter.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az]^T
    """

    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float
    vroll: float
    vpitch: float
    vyaw: float
    ax: float
    ay: float
    az: float
    covariance: np.ndarray                # (15, 15) float64 state error covariance
    tracking_state: str = "FUSED_TRACKING_OK"  # FUSED_TRACKING_OK, COASTING_IMU_WHEEL, DEGRADED, LOST
    active_sensors: List[str] = field(default_factory=list)
    confidence: float = 1.0               # In [0.0, 1.0]

    # Metrics
    distance_traveled_m: float = 0.0
    cumulative_drift_m: float = 0.0
    drift_per_100m: float = 0.0

    @property
    def pose_xy_yaw(self) -> Tuple[float, float, float]:
        """Convenient (x, y, yaw) tuple for 2D ground navigation."""
        return (self.x, self.y, self.yaw)

    @property
    def position_uncertainty_radius_m(self) -> float:
        """2D position standard deviation sqrt(sigma_x^2 + sigma_y^2)."""
        var_x = float(self.covariance[0, 0])
        var_y = float(self.covariance[1, 1])
        return math.sqrt(max(0.0, var_x + var_y))
