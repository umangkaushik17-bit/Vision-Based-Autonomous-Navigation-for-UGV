"""Camera geometry models, intrinsics, extrinsics, and coordinate frame transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic matrix parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3x3 Intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    @property
    def K_inv(self) -> np.ndarray:
        """3x3 Inverse intrinsic matrix."""
        return np.linalg.inv(self.K)

    @classmethod
    def default_640x480(cls, fov_deg: float = 69.4) -> CameraIntrinsics:
        """Standard wide-angle 640x480 camera (e.g. Intel RealSense D435 / Oak-D)."""
        width = 640
        height = 480
        # Calculate focal length from horizontal FOV
        fx = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class CameraExtrinsics:
    """Rigid body transformation from vehicle footprint ground frame to camera optical frame."""

    pitch_deg: float  # Downward tilt from horizontal (> 0 is tilted toward ground)
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    height_m: float = 0.40  # Optical center height above ground contact surface (meters)
    forward_offset_m: float = 0.15  # Distance from vehicle base_link center forward to camera

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 Rotation matrix transforming vectors from Robot frame (X-fwd, Y-left, Z-up) to Camera optical frame (X-right, Y-down, Z-fwd)."""
        pitch = np.radians(self.pitch_deg)
        roll = np.radians(self.roll_deg)
        yaw = np.radians(self.yaw_deg)

        # Standard ROS camera optical frame convention:
        # Robot frame: X forward, Y left, Z up
        # Optical frame: X right, Y down, Z forward
        R_robot_to_optical = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Pitch rotation around robot Y axis (tilt down = negative pitch angle in standard Euler)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        R_pitch = np.array([
            [cp, 0.0, -sp],
            [0.0, 1.0, 0.0],
            [sp, 0.0, cp]
        ], dtype=np.float32)

        return R_robot_to_optical @ R_pitch

    @property
    def translation_vector(self) -> np.ndarray:
        """3x1 Translation vector in camera optical frame."""
        # Camera position relative to base_link ground contact
        t_robot = np.array([self.forward_offset_m, 0.0, self.height_m], dtype=np.float32)
        return -self.rotation_matrix @ t_robot
