"""Quantitative SLAM trajectory evaluation using Phase 7 ATE and RPE formulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import scipy.io as sio

from ai.localization.trajectory import Trajectory3D


@dataclass
class ATEMetrics:
    rmse: float
    mean: float
    median: float
    std: float
    max: float
    num_points: int
    errors: np.ndarray


@dataclass
class RPEMetrics:
    delta_step: int
    rmse: float
    mean: float
    median: float
    std: float
    max: float
    num_pairs: int
    errors: np.ndarray


def align_trajectories_umeyama(
    traj_gt: np.ndarray,
    traj_est: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid SE(3) Umeyama alignment preserving metric scale (scale s = 1.0).
    
    Inputs:
        traj_gt: (N, 3) reference positions
        traj_est: (N, 3) estimated positions
    Returns:
        traj_est_aligned: (N, 3) aligned estimated positions
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    N, dim = traj_gt.shape
    if traj_est.shape != (N, dim):
        raise ValueError("Trajectories must have identical shapes for alignment")

    mu_gt = np.mean(traj_gt, axis=0)
    mu_est = np.mean(traj_est, axis=0)

    X = (traj_gt - mu_gt).T   # (dim, N)
    Y = (traj_est - mu_est).T # (dim, N)

    H = Y @ X.T / N
    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(dim)
    if d < 0:
        S[-1, -1] = -1.0

    R = Vt.T @ S @ U.T
    t = mu_gt - R @ mu_est

    aligned = (R @ traj_est.T + t[:, None]).T
    return aligned, R, t


def compute_ate(
    traj_gt: np.ndarray,
    traj_est: np.ndarray,
    do_align: bool = True
) -> ATEMetrics:
    """Calculate Absolute Trajectory Error (ATE) matching Phase 7 MATLAB engine."""
    if do_align:
        eval_traj, _, _ = align_trajectories_umeyama(traj_gt, traj_est)
    else:
        eval_traj = traj_est

    diffs = traj_gt - eval_traj
    errors = np.sqrt(np.sum(diffs**2, axis=1))

    return ATEMetrics(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        max=float(np.max(errors)),
        num_points=len(errors),
        errors=errors
    )


def compute_rpe(
    traj_gt: np.ndarray,
    traj_est: np.ndarray,
    delta_step: int = 10
) -> RPEMetrics:
    """Calculate Relative Pose Error (RPE) matching Phase 7 MATLAB engine."""
    N = len(traj_gt)
    if N <= delta_step:
        raise ValueError(f"Trajectory length {N} must be > delta_step {delta_step}")

    num_pairs = N - delta_step
    delta_gt = traj_gt[delta_step:] - traj_gt[:-delta_step]
    delta_est = traj_est[delta_step:] - traj_est[:-delta_step]

    rpe_diffs = delta_gt - delta_est
    errors = np.sqrt(np.sum(rpe_diffs**2, axis=1))

    return RPEMetrics(
        delta_step=delta_step,
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        max=float(np.max(errors)),
        num_pairs=num_pairs,
        errors=errors
    )


def evaluate_trajectory_pair(
    gt_trajectory: Trajectory3D,
    est_trajectory: Trajectory3D,
    delta_step: int = 10
) -> Dict[str, Any]:
    """Evaluate paired trajectories using Phase 7 standards."""
    gt_pos = gt_trajectory.positions
    est_pos = est_trajectory.positions

    # Align lengths if needed
    N = min(len(gt_pos), len(est_pos))
    if N < 5:
        return {"status": "insufficient_points"}

    gt_pos = gt_pos[:N]
    est_pos = est_pos[:N]

    ate_raw = compute_ate(gt_pos, est_pos, do_align=False)
    ate_aligned = compute_ate(gt_pos, est_pos, do_align=True)
    rpe = compute_rpe(gt_pos, est_pos, delta_step=min(delta_step, N // 2))

    return {
        "num_evaluated_poses": N,
        "ate_raw_rmse_m": ate_raw.rmse,
        "ate_raw_max_m": ate_raw.max,
        "ate_aligned_rmse_m": ate_aligned.rmse,
        "ate_aligned_mean_m": ate_aligned.mean,
        "ate_aligned_median_m": ate_aligned.median,
        "ate_aligned_std_m": ate_aligned.std,
        "ate_aligned_max_m": ate_aligned.max,
        "rpe_rmse_m": rpe.rmse,
        "rpe_mean_m": rpe.mean,
        "rpe_max_m": rpe.max,
        "duration_s": est_trajectory.duration_s,
        "path_length_m": est_trajectory.path_length_m,
    }


def export_for_matlab_evaluation(
    gt_trajectory: Trajectory3D,
    est_trajectory: Trajectory3D,
    filepath: Union[str, Path],
    sequence_name: str = "evaluated_sequence"
) -> Path:
    """Export paired trajectories to MATLAB v7 .mat file for verification with matlab/slamming/."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    gt_canon = gt_trajectory.to_numpy_canonical()
    est_canon = est_trajectory.to_numpy_canonical()

    sio.savemat(str(filepath), {
        "sequence_name": sequence_name,
        "ground_truth": gt_canon,
        "estimated": est_canon,
        "timestamps": gt_canon[:, 0] if len(gt_canon) else np.array([]),
    })
    return filepath
