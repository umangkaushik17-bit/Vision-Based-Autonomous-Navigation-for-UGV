"""Rigid SE(3) coordinate transformation engine for spatial fusion.

Canonical Frame Definitions:
  - 'world' / 'odom': Fixed inertial navigation frame (+X East/Forward, +Y North/Left, +Z Up).
  - 'base_link': Robot body frame on ground contact plane (+X Forward, +Y Left, +Z Up).
  - 'camera_optical': Camera sensor frame (+X Right, +Y Down, +Z Optical Forward).

Transform Notation:
  - T_A_B transforms homogeneous points expressed in frame B into coordinates in frame A:
      p_A = T_A_B @ p_B
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


def euler_to_rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from Euler angles (ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ], dtype=np.float64)
    return R


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) in radians from 3x3 rotation matrix (ZYX convention)."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def create_se3_matrix(
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None
) -> np.ndarray:
    """Construct 4x4 SE(3) transformation matrix from 3x3 rotation and (3,) translation."""
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()[:3]
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 SE(3) transformation matrix: T^{-1} = [R^T, -R^T @ t; 0, 1]."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    R_inv = R.T
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -R_inv @ t
    return T_inv


def transform_points_3d(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Vectorized transformation of (N, 3) 3D points by 4x4 SE(3) matrix T.
    
    Formula: P_A = P_B @ R^T + t^T
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {pts.shape}")

    R = T[:3, :3]
    t = T[:3, 3]
    return pts @ R.T + t


@dataclass
class SpatialCameraExtrinsics:
    """Defines physical mounting parameters of camera relative to robot base_link."""

    forward_offset_m: float = 0.15      # X: distance ahead of center (meters)
    lateral_offset_m: float = 0.00      # Y: distance left of center (meters)
    height_m: float = 0.40              # Z: height above ground contact (meters)
    pitch_deg: float = 15.0             # Downward tilt angle (degrees)
    yaw_deg: float = 0.0                # Heading angle (degrees)
    roll_deg: float = 0.0               # Bank angle (degrees)

    @property
    def translation(self) -> np.ndarray:
        """Translation vector [x, y, z] from base_link to camera mount."""
        return np.array([self.forward_offset_m, self.lateral_offset_m, self.height_m], dtype=np.float64)

    @property
    def T_body_camera(self) -> np.ndarray:
        """4x4 SE(3) matrix transforming points from camera_optical frame to base_link frame.
        
        Account for:
          1. Optical to ROS body frame alignment:
             X_opt (right) -> -Y_body
             Y_opt (down)  -> -Z_body
             Z_opt (forward) -> +X_body
          2. Extrinsic Euler mounting rotation (pitch, roll, yaw)
          3. Lever-arm translation offset
        """
        # Standard optical to vehicle frame rotation:
        # p_vehicle_optical_origin = R_opt2body @ p_optical
        R_opt2body = np.array([
            [0.0,  0.0,  1.0],  # +X_body = +Z_optical (forward)
            [-1.0, 0.0,  0.0],  # +Y_body = -X_optical (left)
            [0.0, -1.0,  0.0]   # +Z_body = -Y_optical (up)
        ], dtype=np.float64)

        # Vehicle mount orientation (pitch downward tilt rotates about vehicle Y axis)
        pitch_rad = np.radians(self.pitch_deg)
        yaw_rad = np.radians(self.yaw_deg)
        roll_rad = np.radians(self.roll_deg)
        R_mount = euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)

        R_total = R_mount @ R_opt2body
        return create_se3_matrix(R_total, self.translation)

    @property
    def T_camera_body(self) -> np.ndarray:
        """4x4 SE(3) transform from base_link into camera_optical frame."""
        return invert_se3(self.T_body_camera)
