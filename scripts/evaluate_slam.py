"""CLI tool to evaluate estimated SLAM trajectories against ground truth using Phase 7 ATE/RPE metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from ai.localization.evaluation import evaluate_trajectory_pair, export_for_matlab_evaluation
from ai.localization.trajectory import Trajectory3D
from ai.localization.datasets import SyntheticStereoSequenceGenerator


def evaluate_slam(
    est_trajectory_path: str,
    sequence_type: str = "synthetic",
    num_frames: int = 50,
    output_json: str = "outputs/slam/comparisons/evaluation_report.json"
):
    print("\n" + "=" * 78)
    print("       SIH26126 PHASE 8: QUANTITATIVE SLAM TRAJECTORY EVALUATION")
    print(f"       Estimated Trajectory: {est_trajectory_path}")
    print("=" * 78)

    est_path = Path(est_trajectory_path)
    if not est_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {est_path}")

    est_traj = Trajectory3D.from_csv(est_path)

    # Build reference ground-truth trajectory
    gt_traj = Trajectory3D()
    if sequence_type == "synthetic":
        gen = SyntheticStereoSequenceGenerator(num_frames=len(est_traj))
        for frame in gen:
            T_gt = frame.ground_truth_pose
            pos = T_gt[:3, 3]
            from ai.localization.trajectory import rotation_matrix_to_quaternion, TrajectoryPoint3D
            q = rotation_matrix_to_quaternion(T_gt[:3, :3])
            gt_traj.append(TrajectoryPoint3D(
                timestamp=frame.timestamp,
                x=pos[0], y=pos[1], z=pos[2],
                qx=q[0], qy=q[1], qz=q[2], qw=q[3]
            ))
    else:
        print("  [NOTE] Real outdoor RUGD dataset does not provide RTK-GPS ground truth.")
        print("         Evaluating internal tracking smoothness and relative drift.")
        return

    # Evaluate metrics
    metrics = evaluate_trajectory_pair(gt_traj, est_traj)

    print("------------------------------------------------------------------------------")
    print(f"  Evaluated Poses       : {metrics.get('num_evaluated_poses', 0)}")
    print(f"  Trajectory Duration   : {metrics.get('duration_s', 0.0):.2f} s")
    print(f"  Estimated Path Length : {metrics.get('path_length_m', 0.0):.2f} m")
    print(f"  ATE RMSE (Raw)        : {metrics.get('ate_raw_rmse_m', 0.0):.4f} m")
    print(f"  ATE RMSE (Aligned)    : {metrics.get('ate_aligned_rmse_m', 0.0):.4f} m")
    print(f"  ATE Mean Error        : {metrics.get('ate_aligned_mean_m', 0.0):.4f} m")
    print(f"  ATE Max Error         : {metrics.get('ate_aligned_max_m', 0.0):.4f} m")
    print(f"  RPE RMSE (Delta=10)   : {metrics.get('rpe_rmse_m', 0.0):.4f} m")
    print("------------------------------------------------------------------------------")

    # Save to JSON
    out_json_path = Path(output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Evaluation metrics saved to: {out_json_path}")

    # Export paired trajectory for MATLAB slamming validation
    mat_out = Path("matlab/data") / "phase_8_evaluated_slam.mat"
    export_for_matlab_evaluation(gt_traj, est_traj, mat_out, sequence_name=est_path.stem)
    print(f"  [OK] Paired trajectory exported for MATLAB: {mat_out}")
    print("=" * 78 + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate estimated SLAM trajectory against ground truth.")
    parser.add_argument("--trajectory", default="outputs/slam/trajectories/stereo_synthetic_trajectory.csv")
    parser.add_argument("--sequence-type", choices=["synthetic", "rugd"], default="synthetic")
    args = parser.parse_args()

    evaluate_slam(args.trajectory, sequence_type=args.sequence_type)


if __name__ == "__main__":
    main()
