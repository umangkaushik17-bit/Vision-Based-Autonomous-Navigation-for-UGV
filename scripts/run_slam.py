"""CLI to execute visual localization / odometry on image sequences or synthetic scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import yaml

from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.datasets import RUGDSequenceLoader, SyntheticStereoSequenceGenerator
from ai.localization.diagnostics import TrackingDiagnostics
from ai.localization.mono_vo import MonocularVisualOdometry
from ai.localization.stereo_vo import StereoVisualOdometry
from ai.localization.visual_inertial import VisualInertialOdometry


def run_slam(
    mode: str = "stereo",
    sequence_type: str = "synthetic",
    sequence_path: str = "data/processed/sequences/rugd/creek",
    num_frames: int = 50,
    output_dir: str = "outputs/slam/trajectories"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    camera = CameraIntrinsics(
        fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480, baseline_m=0.05
    )

    print("\n" + "=" * 78)
    print(f"       SIH26126 PHASE 8: VISUAL LOCALIZATION PIPELINE")
    print(f"       Mode: {mode.upper()} | Sequence: {sequence_type} ({num_frames} frames)")
    print("=" * 78)

    # Initialize SLAM / VO Estimator
    if mode == "monocular":
        estimator = MonocularVisualOdometry(camera)
    elif mode == "stereo":
        estimator = StereoVisualOdometry(camera)
    elif mode == "stereo_imu":
        estimator = VisualInertialOdometry(camera)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Initialize dataset loader
    if sequence_type == "rugd":
        loader = RUGDSequenceLoader(sequence_path, max_frames=num_frames)
    else:
        loader = SyntheticStereoSequenceGenerator(num_frames=num_frames, baseline_m=camera.baseline_m)

    diagnostics = TrackingDiagnostics()

    t_start = time.perf_counter()
    for frame in loader:
        est = estimator.process_frame(
            image_left=frame.image_left,
            timestamp=frame.timestamp,
            image_right=frame.image_right,
            depth=frame.depth,
            imu=frame.imu
        )
        diagnostics.record(est)
        if frame.index % 10 == 0 or frame.index == len(loader) - 1:
            print(f"  Frame {frame.index:03d}/{len(loader):03d} | State: {est.tracking_state.value:<12} | "
                  f"Inliers: {est.inliers:03d} | Pos: ({est.position[0]:.2f}, {est.position[1]:.2f}, {est.position[2]:.2f})m")

    elapsed = time.perf_counter() - t_start
    fps = len(loader) / elapsed if elapsed > 0 else 0.0

    print("-" * 78)
    print(f"  Processing Complete: {len(loader)} frames in {elapsed:.2f}s ({fps:.1f} FPS)")

    # Save trajectory to CSV and MAT
    csv_file = out_path / f"{mode}_{sequence_type}_trajectory.csv"
    mat_file = out_path / f"{mode}_{sequence_type}_trajectory.mat"
    estimator.trajectory.to_csv(csv_file)
    estimator.trajectory.to_mat(mat_file)
    print(f"  [OK] Saved Trajectory CSV: {csv_file}")
    print(f"  [OK] Saved Trajectory MAT: {mat_file}")

    # Export diagnostics
    diag_file = Path("outputs/slam/diagnostics") / f"{mode}_{sequence_type}_diagnostics.csv"
    diagnostics.export_csv(diag_file)
    summary = diagnostics.compute_summary()
    print(f"  Diagnostics: Tracking OK = {summary.get('tracking_ok_pct', 0):.1f}%, Mean Inliers = {summary.get('mean_inliers', 0):.1f}")
    print("=" * 78 + "\n")

    return estimator.trajectory, summary


def main():
    parser = argparse.ArgumentParser(description="Execute visual localization / SLAM on a sequence.")
    parser.add_argument("--mode", choices=["monocular", "stereo", "stereo_imu"], default="stereo")
    parser.add_argument("--sequence-type", choices=["synthetic", "rugd"], default="synthetic")
    parser.add_argument("--sequence-path", default="data/processed/sequences/rugd/creek")
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()

    run_slam(
        mode=args.mode,
        sequence_type=args.sequence_type,
        sequence_path=args.sequence_path,
        num_frames=args.frames
    )


if __name__ == "__main__":
    main()
