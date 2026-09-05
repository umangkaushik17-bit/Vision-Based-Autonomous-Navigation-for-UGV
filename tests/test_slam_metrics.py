"""Unit tests for Phase 8 ATE and RPE trajectory evaluation."""

import numpy as np
import pytest

from ai.localization.evaluation import (
    align_trajectories_umeyama,
    compute_ate,
    compute_rpe,
    evaluate_trajectory_pair,
)
from ai.localization.trajectory import Trajectory3D, TrajectoryPoint3D


def test_umeyama_alignment_exact_match():
    """Verify Umeyama alignment removes rigid translation and rotation without scale distortion."""
    t = np.linspace(0, 10, 50)
    x = 2.0 * np.sin(t)
    y = 1.5 * np.cos(t)
    z = 0.1 * t
    gt = np.column_stack([x, y, z])

    # Offset by [0.5, -0.2, 0.1] and rotate by 30 deg
    theta = np.radians(30.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    trans = np.array([0.5, -0.2, 0.1])
    est = (R @ gt.T + trans[:, None]).T

    aligned, R_est, t_est = align_trajectories_umeyama(gt, est)

    # Post-alignment error should be zero
    err = np.linalg.norm(gt - aligned, axis=1)
    assert np.max(err) < 1e-10


def test_compute_ate_zero_and_noisy():
    """Verify compute_ate returns 0.0 on identical trajectories and expected RMSE on noise."""
    gt = np.random.RandomState(42).randn(100, 3)

    # Identical
    ate_zero = compute_ate(gt, gt, do_align=False)
    assert ate_zero.rmse == 0.0
    assert ate_zero.max == 0.0

    # Perturbed with 0.05m Gaussian noise
    noise = np.random.RandomState(123).normal(0.0, 0.05, gt.shape)
    ate_noisy = compute_ate(gt, gt + noise, do_align=False)
    assert abs(ate_noisy.rmse - 0.05 * np.sqrt(3)) < 0.02


def test_compute_rpe():
    """Verify compute_rpe evaluates drift over delta interval."""
    gt = np.column_stack([np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)])
    est = np.column_stack([np.linspace(0, 9.5, 50), np.zeros(50), np.zeros(50)])

    rpe = compute_rpe(gt, est, delta_step=5)
    assert rpe.num_pairs == 45
    assert rpe.rmse > 0.0


def test_evaluate_trajectory_pair():
    """Verify end-to-end evaluation dictionary production."""
    gt_traj = Trajectory3D()
    est_traj = Trajectory3D()

    for i in range(20):
        t = i * 0.1
        gt_traj.append(TrajectoryPoint3D(timestamp=t, x=float(i), y=0.0, z=0.0))
        est_traj.append(TrajectoryPoint3D(timestamp=t, x=float(i) + 0.02, y=0.0, z=0.0))

    metrics = evaluate_trajectory_pair(gt_traj, est_traj)
    assert metrics["num_evaluated_poses"] == 20
    assert metrics["ate_raw_rmse_m"] > 0.0
    assert metrics["ate_aligned_rmse_m"] < 1e-6
