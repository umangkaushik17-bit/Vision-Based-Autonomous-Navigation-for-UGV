"""Standardized localization result contract for downstream fusion (Phase 9) and ROS2 handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

from ai.localization.base import TrackingState
from ai.localization.trajectory import rotation_matrix_to_quaternion


@dataclass
class LocalizationEstimate:
    """Canonical 6-DoF localization state estimate produced by visual localization.
    
    This contract serves as the official input to:
      1. Phase 9 (Perception + Localization Spatial Fusion)
      2. Costmap temporal motion compensation
      3. Global / Local path tracking (A* / DWA)
      4. Future ROS2 /odom and /tf publishers
    """
    timestamp: float
    pose: np.ndarray                            # 4x4 SE(3) transformation matrix (world to body)
    tracking_state: TrackingState = TrackingState.TRACKING_OK
    confidence: float = 1.0                     # Normalized localization confidence [0.0, 1.0]
    is_pose_valid: bool = True                  # Strictly FALSE if LOST or RELOCALIZING
    linear_velocity: np.ndarray = field(        # [vx, vy, vz] in m/s (body frame)
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    angular_velocity: np.ndarray = field(       # [wx, wy, wz] in rad/s (body frame)
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    tracked_features: int = 0
    inliers: int = 0
    keyframe_id: int = 0
    is_keyframe: bool = False
    relocalized: bool = False
    frame_id: str = "odom"
    child_frame_id: str = "base_link"
    sequence_id: str = "default"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        """Extract (3,) position [x, y, z] in world coordinate frame (meters)."""
        return self.pose[:3, 3]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Extract (3, 3) rotation matrix."""
        return self.pose[:3, :3]

    @property
    def quaternion(self) -> np.ndarray:
        """Extract unit quaternion [qx, qy, qz, qw] (Hamilton convention)."""
        return rotation_matrix_to_quaternion(self.rotation_matrix)

    @property
    def trusted_pose(self) -> Optional[np.ndarray]:
        """Return 4x4 SE(3) pose ONLY if valid and trusted.
        
        Zero-trust contract: Returns None when LOST or RELOCALIZING so downstream
        modules cannot silently consume a stale or unanchored pose.
        """
        if not self.is_pose_valid or self.tracking_state in (TrackingState.LOST, TrackingState.RELOCALIZING):
            return None
        return self.pose.copy()

    @property
    def is_reliable(self) -> bool:
        """Check if localization is fully reliable for nominal autonomous navigation."""
        return (
            self.is_pose_valid
            and self.tracking_state in (TrackingState.TRACKING_OK, TrackingState.RECOVERED)
            and self.confidence >= 0.70
        )

    @property
    def allows_spatial_fusion(self) -> bool:
        """Determine if Phase 9 spatial fusion is permitted to integrate local costmaps.
        
        TRACKING_OK / RECOVERED: Allowed with normal weighting.
        DEGRADED: Allowed with conservative confidence weighting (v <= 0.35 m/s).
        LOST / RELOCALIZING: Strictly forbidden (returns False).
        """
        if not self.is_pose_valid or self.tracking_state in (TrackingState.LOST, TrackingState.RELOCALIZING):
            return False
        return self.confidence >= 0.30

    @property
    def covariance_proxy(self) -> np.ndarray:
        """6x6 diagonal covariance proxy [x, y, z, roll, pitch, yaw] based on tracking confidence.
        
        Scaling formula: scale = 1.0 / (c * c) where c = max(0.05, confidence).
        - TRACKING_OK (c in [0.70, 1.00]): scale in [1.0x, 2.04x].
        - DEGRADED (c in [0.30, 0.69]): scale in [2.10x, 11.11x].
        - Absolute floor (c = 0.05): maximum scale of 400.0x.
        """
        base_var_pos = 0.01  # (10 cm standard deviation squared)
        base_var_rot = 0.005 # (~4 degrees squared)
        # Scale variance inversely with squared confidence
        c = max(0.05, self.confidence)
        scale = 1.0 / (c * c)
        cov = np.diag([
            base_var_pos * scale, base_var_pos * scale, base_var_pos * scale,
            base_var_rot * scale, base_var_rot * scale, base_var_rot * scale
        ])
        return cov

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization and diagnostic logging."""
        pos = self.position
        quat = self.quaternion
        return {
            "timestamp": self.timestamp,
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "linear_velocity": [float(v) for v in self.linear_velocity],
            "angular_velocity": [float(w) for w in self.angular_velocity],
            "tracking_state": self.tracking_state.value,
            "confidence": float(self.confidence),
            "tracked_features": self.tracked_features,
            "inliers": self.inliers,
            "keyframe_id": self.keyframe_id,
            "is_keyframe": self.is_keyframe,
            "relocalized": self.relocalized,
        }
