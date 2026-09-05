"""Dataset loaders and sequence adapters for outdoor visual odometry and SLAM benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np

from ai.localization.base import IMUReading


@dataclass
class SequenceFrame:
    """A single synchronized sensor frame in a localization sequence."""
    index: int
    timestamp: float
    image_left: np.ndarray
    image_right: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    imu: Optional[IMUReading] = None
    ground_truth_pose: Optional[np.ndarray] = None  # 4x4 SE(3) matrix


class RUGDSequenceLoader:
    """Loads outdoor RUGD video frame sequences with timestamps."""

    def __init__(
        self,
        sequence_dir: Union[str, Path],
        fps: float = 10.0,
        target_size: Optional[Tuple[int, int]] = (640, 480),
        max_frames: Optional[int] = None
    ):
        self.sequence_dir = Path(sequence_dir)
        self.fps = fps
        self.dt = 1.0 / fps
        self.target_size = target_size
        self.max_frames = max_frames

        # Search for images
        rgb_dir = self.sequence_dir / "rgb" if (self.sequence_dir / "rgb").exists() else self.sequence_dir
        self.image_paths = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
        if self.max_frames:
            self.image_paths = self.image_paths[:self.max_frames]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[SequenceFrame]:
        for idx, p in enumerate(self.image_paths):
            img = cv2.imread(str(p))
            if img is None:
                continue
            if self.target_size and (img.shape[1] != self.target_size[0] or img.shape[0] != self.target_size[1]):
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

            timestamp = idx * self.dt
            yield SequenceFrame(
                index=idx,
                timestamp=timestamp,
                image_left=img,
                image_right=None,
                depth=None,
                imu=None,
                ground_truth_pose=None,
            )


class SyntheticStereoSequenceGenerator:
    """Generates synthetic stereo image sequences with known 6-DoF ground truth poses and IMU data.
    
    Used for quantitative ATE/RPE evaluation and sensor mode benchmarking where physical
    outdoor ground-truth poses are not provided by raw video datasets.
    """

    def __init__(
        self,
        num_frames: int = 100,
        fps: float = 30.0,
        baseline_m: float = 0.05,
        width: int = 640,
        height: int = 480,
        motion_type: str = "curved"  # "straight", "curved", "turn", "loop"
    ):
        self.num_frames = num_frames
        self.fps = fps
        self.dt = 1.0 / fps
        self.baseline_m = baseline_m
        self.width = width
        self.height = height
        self.motion_type = motion_type

        # Generate ground truth 3D landmarks along the forward motion path (+X forward)
        rng = np.random.RandomState(42)
        self.num_landmarks = 400
        # Forward X: -2m to 35m, Lateral Y: -4m to 4m, Ground elevation Z: -0.1m to 0.4m
        self.landmarks = rng.uniform(
            low=[-2.0, -4.0, -0.1],
            high=[35.0, 4.0, 0.4],
            size=(self.num_landmarks, 3)
        )

    def __len__(self) -> int:
        return self.num_frames

    def get_ground_truth_pose(self, t: float) -> np.ndarray:
        """Compute true 4x4 SE(3) robot pose in world frame at time t."""
        T = np.eye(4, dtype=np.float64)
        if self.motion_type == "straight":
            x = 0.5 * t
            y = 0.0
            theta = 0.0
        elif self.motion_type == "curved":
            x = 0.5 * t
            y = 0.4 * np.sin(0.3 * t)
            theta = 0.12 * np.cos(0.3 * t)
        elif self.motion_type == "loop":
            radius = 2.5
            omega = 2.0 * np.pi / max(1.0, self.num_frames * self.dt)
            x = radius * np.sin(omega * t)
            y = radius * (1.0 - np.cos(omega * t))
            theta = omega * t
        else:
            x = 0.5 * t
            y = 0.0
            theta = 0.0

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0]
        ], dtype=np.float64)

        T[:3, :3] = R
        T[:3, 3] = [x, y, 0.0]
        return T

    def render_frame_views(self, T_world_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize left and right camera images by projecting 3D landmarks."""
        img_left = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 110
        img_right = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 110

        fx, fy = 462.14, 462.14
        cx, cy = self.width / 2.0, self.height / 2.0

        R_wb = T_world_body[:3, :3]
        t_wb = T_world_body[:3, 3]

        # Camera extrinsics relative to robot body:
        # Optical frame: +X right, +Y down, +Z forward
        pitch = np.radians(15.0)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        R_pitch = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], dtype=np.float64)
        R_bc = R_base @ R_pitch
        t_bc = np.array([0.15, 0.0, 0.40], dtype=np.float64)

        R_wc = R_wb @ R_bc
        t_wc = t_wb + R_wb @ t_bc

        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        # Right camera baseline offset along camera +X
        t_cw_right = t_cw - np.array([self.baseline_m, 0.0, 0.0])

        for lm in self.landmarks:
            p_cam_l = R_cw @ lm + t_cw
            p_cam_r = R_cw @ lm + t_cw_right

            if p_cam_l[2] > 0.4 and p_cam_r[2] > 0.4:
                ul = int(fx * p_cam_l[0] / p_cam_l[2] + cx)
                vl = int(fy * p_cam_l[1] / p_cam_l[2] + cy)

                ur = int(fx * p_cam_r[0] / p_cam_r[2] + cx)
                vr = int(fy * p_cam_r[1] / p_cam_r[2] + cy)

                # Draw high-contrast corner markers for ORB
                if 10 <= ul < self.width - 10 and 10 <= vl < self.height - 10:
                    cv2.drawMarker(img_left, (ul, vl), (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2)
                    cv2.circle(img_left, (ul, vl), 2, (0, 0, 0), -1)

                if 10 <= ur < self.width - 10 and 10 <= vr < self.height - 10:
                    cv2.drawMarker(img_right, (ur, vr), (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2)
                    cv2.circle(img_right, (ur, vr), 2, (0, 0, 0), -1)

        # Add subtle noise
        noise = np.random.normal(0, 2, img_left.shape).astype(np.int16)
        img_left = np.clip(img_left.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img_right = np.clip(img_right.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img_left, img_right

    def __iter__(self) -> Iterator[SequenceFrame]:
        for idx in range(self.num_frames):
            t = idx * self.dt
            T_gt = self.get_ground_truth_pose(t)
            img_l, img_r = self.render_frame_views(T_gt)

            # Synthesize IMU reading (finite difference of pose + noise)
            dt = self.dt
            if idx > 0:
                T_prev = self.get_ground_truth_pose((idx - 1) * dt)
                v = (T_gt[:3, 3] - T_prev[:3, 3]) / dt
                R_diff = T_gt[:3, :3] @ T_prev[:3, :3].T
                # Small angle approximation for angular velocity
                w = np.array([R_diff[2, 1], R_diff[0, 2], R_diff[1, 0]]) / dt
            else:
                v = np.zeros(3)
                w = np.zeros(3)

            # Add gravity vector to accelerometer
            accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0, 0.04, 3)
            gyro = w + np.random.normal(0, 0.004, 3)

            imu = IMUReading(
                timestamp=t,
                angular_velocity=gyro,
                linear_acceleration=accel,
            )

            yield SequenceFrame(
                index=idx,
                timestamp=t,
                image_left=img_l,
                image_right=img_r,
                depth=None,
                imu=imu,
                ground_truth_pose=T_gt,
            )
