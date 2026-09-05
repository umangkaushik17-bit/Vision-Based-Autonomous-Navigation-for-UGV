"""3D Trajectory representation, SE(3) pose transformations, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import scipy.io as sio

from ai.localization.base import TrackingState


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [qx, qy, qz, qw] (Hamilton convention)."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    norm = np.linalg.norm(q)
    if norm > 0:
        qx, qy, qz, qw = q / norm

    return np.array([
        [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx**2 + qy**2)]
    ], dtype=np.float64)


@dataclass
class TrajectoryPoint3D:
    """A single timestamped 6-DoF pose along the robot trajectory."""
    timestamp: float
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    tracking_state: TrackingState = TrackingState.TRACKING_OK
    confidence: float = 1.0
    inlier_count: int = 0

    def __post_init__(self):
        # Enforce unit quaternion normalization
        q = np.array([self.qx, self.qy, self.qz, self.qw], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm > 0 and abs(norm - 1.0) > 1e-6:
            q = q / norm
            self.qx, self.qy, self.qz, self.qw = q[0], q[1], q[2], q[3]

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def quaternion(self) -> np.ndarray:
        return np.array([self.qx, self.qy, self.qz, self.qw], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quaternion_to_rotation_matrix(self.quaternion)

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Return 4x4 SE(3) transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T

    @classmethod
    def from_se3(
        cls,
        timestamp: float,
        T: np.ndarray,
        tracking_state: TrackingState = TrackingState.TRACKING_OK,
        confidence: float = 1.0,
        inliers: int = 0
    ) -> TrajectoryPoint3D:
        """Construct from 4x4 SE(3) transformation matrix."""
        pos = T[:3, 3]
        q = rotation_matrix_to_quaternion(T[:3, :3])
        return cls(
            timestamp=timestamp,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            qx=float(q[0]),
            qy=float(q[1]),
            qz=float(q[2]),
            qw=float(q[3]),
            tracking_state=tracking_state,
            confidence=confidence,
            inlier_count=inliers
        )


@dataclass
class Trajectory3D:
    """Full 6-DoF trajectory conforming strictly to Phase 7 SLAM interface."""
    points: List[TrajectoryPoint3D] = field(default_factory=list)
    frame_id: str = "map"
    child_frame_id: str = "base_link"

    def __len__(self) -> int:
        return len(self.points)

    def append(self, pt: TrajectoryPoint3D) -> None:
        self.points.append(pt)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([p.timestamp for p in self.points], dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        return np.array([[p.x, p.y, p.z] for p in self.points], dtype=np.float64)

    @property
    def quaternions(self) -> np.ndarray:
        return np.array([[p.qx, p.qy, p.qz, p.qw] for p in self.points], dtype=np.float64)

    @property
    def duration_s(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    @property
    def path_length_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        pos = self.positions
        diffs = np.diff(pos, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def to_numpy_canonical(self) -> np.ndarray:
        """Return (N, 8) matrix: [timestamp, x, y, z, qx, qy, qz, qw]."""
        if not self.points:
            return np.empty((0, 8), dtype=np.float64)
        return np.column_stack([self.timestamps, self.positions, self.quaternions])

    def to_csv(self, filepath: Union[str, Path]) -> Path:
        """Export trajectory to CSV conforming strictly to Phase 7 interface."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_numpy_canonical()
        header = "timestamp,x,y,z,qx,qy,qz,qw"
        np.savetxt(str(filepath), data, delimiter=",", header=header, comments="", fmt="%.6f")
        return filepath

    def to_mat(self, filepath: Union[str, Path], var_name: str = "trajectory") -> Path:
        """Export trajectory to native MATLAB .mat dictionary."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_numpy_canonical()
        sio.savemat(str(filepath), {
            var_name: data,
            "duration_s": self.duration_s,
            "length_m": self.path_length_m,
            "num_points": len(self.points),
            "frame_id": self.frame_id,
            "child_frame_id": self.child_frame_id,
        })
        return filepath

    @classmethod
    def from_csv(cls, filepath: Union[str, Path]) -> Trajectory3D:
        """Load canonical trajectory from CSV file."""
        data = np.genfromtxt(str(filepath), delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        traj = cls()
        for row in data:
            traj.append(TrajectoryPoint3D(
                timestamp=float(row[0]),
                x=float(row[1]),
                y=float(row[2]),
                z=float(row[3]),
                qx=float(row[4]),
                qy=float(row[5]),
                qz=float(row[6]),
                qw=float(row[7]),
            ))
        return traj
