"""Base classes, data contracts, and tracking states for visual localization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TrackingState(Enum):
    """Explicit tracking states required for safe autonomous navigation."""
    TRACKING_OK = "TRACKING_OK"        # High-confidence feature tracking (> 20 inliers)
    DEGRADED = "DEGRADED"              # Marginal tracking (10-20 inliers, scale or parallax weak)
    LOST = "LOST"                      # Tracking failed (< 8 inliers or severe occlusion)
    RELOCALIZING = "RELOCALIZING"      # Searching keyframe database / DBoW2 place recognition
    RECOVERED = "RECOVERED"            # Re-acquired pose after tracking failure


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters and optional stereo baseline."""
    fx: float = 462.14
    fy: float = 462.14
    cx: float = 320.00
    cy: float = 240.00
    width: int = 640
    height: int = 480
    baseline_m: float = 0.05  # Stereo baseline in meters (0.0 for monocular)

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        """Return inverse of intrinsic matrix."""
        return np.linalg.inv(self.K)


@dataclass
class CameraExtrinsics:
    """Extrinsic mount position and orientation relative to robot base_link."""
    # base_link: +X forward, +Y left, +Z up
    x_forward_m: float = 0.15
    y_left_m: float = 0.00
    z_up_m: float = 0.40
    pitch_deg: float = 15.0  # Downward tilt
    roll_deg: float = 0.0
    yaw_deg: float = 0.0

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return 3x3 rotation from base_link to camera optical frame."""
        # Optical frame: +X right, +Y down, +Z forward
        R_base = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)

        pitch_rad = np.radians(self.pitch_deg)
        cp = np.cos(pitch_rad)
        sp = np.sin(pitch_rad)
        R_pitch = np.array([
            [cp, 0.0, -sp],
            [0.0, 1.0, 0.0],
            [sp, 0.0, cp]
        ], dtype=np.float64)

        return R_base @ R_pitch

    @property
    def translation_vector(self) -> np.ndarray:
        """Return 3x1 translation in camera frame: t = -R * t_robot."""
        t_robot = np.array([self.x_forward_m, self.y_left_m, self.z_up_m], dtype=np.float64)
        return -self.rotation_matrix @ t_robot


@dataclass
class IMUReading:
    """Synchronized IMU measurement at a specific timestamp."""
    timestamp: float
    angular_velocity: np.ndarray    # [wx, wy, wz] in rad/s
    linear_acceleration: np.ndarray # [ax, ay, az] in m/s^2


class BaseVisualOdometry(ABC):
    """Abstract base class for visual odometry and SLAM estimators."""

    def __init__(self, camera: CameraIntrinsics, config: Optional[Dict[str, Any]] = None):
        self.camera = camera
        self.config = config or {}
        self.state: TrackingState = TrackingState.TRACKING_OK
        self.current_pose: np.ndarray = np.eye(4, dtype=np.float64) # 4x4 SE(3) world-to-body pose
        self.keyframe_poses: List[np.ndarray] = []
        self.frame_count: int = 0

    @abstractmethod
    def process_frame(
        self,
        image_left: np.ndarray,
        timestamp: float,
        image_right: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        imu: Optional[IMUReading] = None,
    ) -> Any:
        """Process one synchronized sensor frame and update pose estimate."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset estimator to initial origin pose."""
        pass
