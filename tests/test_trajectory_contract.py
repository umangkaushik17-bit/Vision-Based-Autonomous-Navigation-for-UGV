"""Unit tests for TrajectoryPoint3D, Trajectory3D, and canonical I/O contracts."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from ai.localization.base import TrackingState
from ai.localization.trajectory import (
    Trajectory3D,
    TrajectoryPoint3D,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def test_quaternion_matrix_round_trip():
    """Verify conversion between rotation matrix and unit quaternion."""
    # Identity rotation
    R_eye = np.eye(3, dtype=np.float64)
    q_eye = rotation_matrix_to_quaternion(R_eye)
    assert np.allclose(q_eye, [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(quaternion_to_rotation_matrix(q_eye), R_eye)

    # 45 deg rotation about Z axis
    theta = np.radians(45.0)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    q_z = rotation_matrix_to_quaternion(R_z)
    assert abs(np.linalg.norm(q_z) - 1.0) < 1e-6
    R_rec = quaternion_to_rotation_matrix(q_z)
    assert np.allclose(R_z, R_rec, atol=1e-6)


def test_trajectory_point_se3_construction():
    """Verify TrajectoryPoint3D creation from SE(3) matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [2.0, 3.5, -0.2]
    pt = TrajectoryPoint3D.from_se3(timestamp=5.0, T=T)

    assert pt.timestamp == 5.0
    assert np.allclose(pt.position, [2.0, 3.5, -0.2])
    assert np.allclose(pt.transformation_matrix, T, atol=1e-6)


def test_trajectory_csv_and_mat_io():
    """Verify CSV and MAT export and import round-trip."""
    traj = Trajectory3D()
    for i in range(10):
        t = i * 0.1
        traj.append(TrajectoryPoint3D(
            timestamp=t,
            x=0.5 * t,
            y=0.1 * t,
            z=0.0,
            qx=0.0, qy=0.0, qz=0.0, qw=1.0
        ))

    assert len(traj) == 10
    assert abs(traj.duration_s - 0.9) < 1e-4
    assert traj.path_length_m > 0.0

    canon_mat = traj.to_numpy_canonical()
    assert canon_mat.shape == (10, 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_traj.csv"
        mat_path = Path(tmpdir) / "test_traj.mat"

        traj.to_csv(csv_path)
        traj.to_mat(mat_path)

        assert csv_path.exists()
        assert mat_path.exists()

        # Reload CSV
        reloaded = Trajectory3D.from_csv(csv_path)
        assert len(reloaded) == 10
        assert np.allclose(reloaded.positions, traj.positions, atol=1e-4)
