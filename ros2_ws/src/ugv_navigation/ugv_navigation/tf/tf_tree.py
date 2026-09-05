"""Canonical TF2 frame tree and high-precision transform buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple

from ugv_navigation.messages.localization_status_msg import Point3D, Quaternion
from ugv_navigation.utils.time_utils import RosHeader, RosTime
from ai.spatial_fusion.transforms import create_se3_matrix, invert_se3, euler_to_rotation_matrix


@dataclass
class Transform3D:
    translation: Point3D = field(default_factory=Point3D)
    rotation: Quaternion = field(default_factory=Quaternion)

    def to_matrix(self) -> np.ndarray:
        """Convert translation and quaternion to 4x4 SE(3) matrix."""
        tx, ty, tz = self.translation.x, self.translation.y, self.translation.z
        qx, qy, qz, qw = self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w

        R = np.array([
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx**2 + qy**2)],
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> Transform3D:
        """Construct from 4x4 SE(3) matrix."""
        R = T[:3, :3]
        t = T[:3, 3]

        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return cls(
            translation=Point3D(x=float(t[0]), y=float(t[1]), z=float(t[2])),
            rotation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
        )


@dataclass
class TransformStamped:
    header: RosHeader
    child_frame_id: str
    transform: Transform3D


class TfLookupError(Exception):
    pass


class TfExtrapolationError(Exception):
    pass


class TfTreeBuffer:
    """Pure-Python implementation of a TF2 Transform Buffer conforming to ROS2 standards."""

    FRAME_ODOM = "odom"
    FRAME_BASE_LINK = "base_link"
    FRAME_CAMERA_LINK = "camera_link"
    FRAME_CAMERA_OPTICAL = "camera_optical"

    def __init__(self, cache_time_s: float = 10.0, max_extrapolation_s: float = 0.10):
        self.cache_time_s = cache_time_s
        self.max_extrapolation_s = max_extrapolation_s
        # Dynamic transforms: (parent, child) -> list of (timestamp_s, TransformStamped)
        self._dynamic_transforms: Dict[Tuple[str, str], List[Tuple[float, TransformStamped]]] = {}
        # Static transforms: (parent, child) -> TransformStamped
        self._static_transforms: Dict[Tuple[str, str], TransformStamped] = {}

    def set_static_transform(self, transform: TransformStamped) -> None:
        key = (transform.header.frame_id, transform.child_frame_id)
        self._static_transforms[key] = transform

    def set_transform(self, transform: TransformStamped) -> None:
        key = (transform.header.frame_id, transform.child_frame_id)
        t_sec = transform.header.stamp.to_sec()
        if key not in self._dynamic_transforms:
            self._dynamic_transforms[key] = []
        buf = self._dynamic_transforms[key]
        buf.append((t_sec, transform))
        # Keep buffer sorted by timestamp
        buf.sort(key=lambda x: x[0])
        # Purge entries older than cache_time_s
        cutoff = t_sec - self.cache_time_s
        self._dynamic_transforms[key] = [item for item in buf if item[0] >= cutoff]

    def _get_direct_transform_mat(self, parent: str, child: str, time_s: Optional[float]) -> np.ndarray:
        # Check static first
        key = (parent, child)
        if key in self._static_transforms:
            return self._static_transforms[key].transform.to_matrix()

        # Check inverse static
        inv_key = (child, parent)
        if inv_key in self._static_transforms:
            return invert_se3(self._static_transforms[inv_key].transform.to_matrix())

        # Check dynamic
        if key in self._dynamic_transforms:
            buf = self._dynamic_transforms[key]
            if not buf:
                raise TfLookupError(f"No transforms recorded for {parent} -> {child}")
            if time_s is None:
                # Latest transform
                return buf[-1][1].transform.to_matrix()
            # Find nearest timestamp
            best_diff = float("inf")
            best_tf = None
            for stamp, tf in buf:
                diff = abs(stamp - time_s)
                if diff < best_diff:
                    best_diff = diff
                    best_tf = tf
            if best_diff > self.max_extrapolation_s:
                raise TfExtrapolationError(
                    f"Lookup {parent} -> {child} at t={time_s:.3f}s exceeds max extrapolation ({best_diff:.3f}s > {self.max_extrapolation_s:.3f}s)"
                )
            return best_tf.transform.to_matrix()

        # Check inverse dynamic
        if inv_key in self._dynamic_transforms:
            T_child_parent = self._get_direct_transform_mat(child, parent, time_s)
            return invert_se3(T_child_parent)

        raise TfLookupError(f"No direct transform available between '{parent}' and '{child}'")

    def lookup_transform_matrix(self, target_frame: str, source_frame: str, time_s: Optional[float] = None) -> np.ndarray:
        """Computes SE(3) matrix that transforms points from source_frame into target_frame."""
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        # Canonical chain lookup
        # Supported graph: odom <-> base_link <-> camera_link <-> camera_optical
        frames = [self.FRAME_ODOM, self.FRAME_BASE_LINK, self.FRAME_CAMERA_LINK, self.FRAME_CAMERA_OPTICAL]
        
        # Route through base_link if traversing across odom and camera frames
        if (target_frame, source_frame) in self._static_transforms or (target_frame, source_frame) in self._dynamic_transforms:
            return self._get_direct_transform_mat(target_frame, source_frame, time_s)
        if (source_frame, target_frame) in self._static_transforms or (source_frame, target_frame) in self._dynamic_transforms:
            return self._get_direct_transform_mat(target_frame, source_frame, time_s)

        # Composite lookup:
        # Example 1: odom -> camera_optical = T(odom->base_link) * T(base_link->camera_link) * T(camera_link->camera_optical)
        if target_frame == self.FRAME_ODOM and source_frame == self.FRAME_CAMERA_OPTICAL:
            T_ob = self._get_direct_transform_mat(self.FRAME_ODOM, self.FRAME_BASE_LINK, time_s)
            T_bc = self._get_direct_transform_mat(self.FRAME_BASE_LINK, self.FRAME_CAMERA_LINK, time_s)
            T_co = self._get_direct_transform_mat(self.FRAME_CAMERA_LINK, self.FRAME_CAMERA_OPTICAL, time_s)
            return T_ob @ T_bc @ T_co

        if target_frame == self.FRAME_ODOM and source_frame == self.FRAME_CAMERA_LINK:
            T_ob = self._get_direct_transform_mat(self.FRAME_ODOM, self.FRAME_BASE_LINK, time_s)
            T_bc = self._get_direct_transform_mat(self.FRAME_BASE_LINK, self.FRAME_CAMERA_LINK, time_s)
            return T_ob @ T_bc

        if target_frame == self.FRAME_BASE_LINK and source_frame == self.FRAME_CAMERA_OPTICAL:
            T_bc = self._get_direct_transform_mat(self.FRAME_BASE_LINK, self.FRAME_CAMERA_LINK, time_s)
            T_co = self._get_direct_transform_mat(self.FRAME_CAMERA_LINK, self.FRAME_CAMERA_OPTICAL, time_s)
            return T_bc @ T_co

        if target_frame == self.FRAME_CAMERA_OPTICAL and source_frame == self.FRAME_ODOM:
            T_target_source = self.lookup_transform_matrix(self.FRAME_ODOM, self.FRAME_CAMERA_OPTICAL, time_s)
            return invert_se3(T_target_source)

        if target_frame == self.FRAME_CAMERA_OPTICAL and source_frame == self.FRAME_BASE_LINK:
            T_target_source = self.lookup_transform_matrix(self.FRAME_BASE_LINK, self.FRAME_CAMERA_OPTICAL, time_s)
            return invert_se3(T_target_source)

        raise TfLookupError(f"No composite transform path found between '{target_frame}' and '{source_frame}'")

    def lookup_transform(self, target_frame: str, source_frame: str, time_s: Optional[float] = None) -> TransformStamped:
        """Returns standard TransformStamped object."""
        T = self.lookup_transform_matrix(target_frame, source_frame, time_s)
        stamp = RosTime.from_sec(time_s if time_s is not None else 0.0)
        return TransformStamped(
            header=RosHeader(stamp=stamp, frame_id=target_frame),
            child_frame_id=source_frame,
            transform=Transform3D.from_matrix(T),
        )
