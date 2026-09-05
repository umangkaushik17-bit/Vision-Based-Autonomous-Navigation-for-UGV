"""Comprehensive benchmark script evaluating Monocular vs Stereo vs Stereo+IMU."""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import psutil

from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.datasets import SyntheticStereoSequenceGenerator
from ai.localization.diagnostics import TrackingDiagnostics
from ai.localization.evaluation import evaluate_trajectory_pair
from ai.localization.mono_vo import MonocularVisualOdometry
from ai.localization.stereo_vo import StereoVisualOdometry
from ai.localization.visual_inertial import VisualInertialOdometry
from ai.localization.trajectory import Trajectory3D, TrajectoryPoint3D, rotation_matrix_to_quaternion


def benchmark_mode(mode: str, num_frames: int = 50) -> dict:
    camera = CameraIntrinsics(
        fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480, baseline_m=0.05
    )

    if mode == "monocular":
        estimator = MonocularVisualOdometry(camera)
    elif mode == "stereo":
        estimator = StereoVisualOdometry(camera)
    elif mode == "stereo_imu":
        estimator = VisualInertialOdometry(camera)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    gen = SyntheticStereoSequenceGenerator(num_frames=num_frames, baseline_m=camera.baseline_m)
    gt_traj = Trajectory3D()
    diagnostics = TrackingDiagnostics()

    latencies_ms = []

    process = psutil.Process()
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    for frame in gen:
        # Build reference ground-truth
        T_gt = frame.ground_truth_pose
        pos = T_gt[:3, 3]
        q = rotation_matrix_to_quaternion(T_gt[:3, :3])
        gt_traj.append(TrajectoryPoint3D(
            timestamp=frame.timestamp,
            x=pos[0], y=pos[1], z=pos[2],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3]
        ))

        t0 = time.perf_counter()
        est = estimator.process_frame(
            image_left=frame.image_left,
            timestamp=frame.timestamp,
            image_right=frame.image_right,
            depth=frame.depth,
            imu=frame.imu
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        diagnostics.record(est)

    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    ram_used_mb = max(0.0, ram_end_mb - ram_start_mb)

    eval_res = evaluate_trajectory_pair(gt_traj, estimator.trajectory)
    diag_summary = diagnostics.compute_summary()

    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mode": mode,
        "num_frames": num_frames,
        "fps": fps,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "ram_used_mb": ram_used_mb,
        "tracking_ok_pct": diag_summary.get("tracking_ok_pct", 0.0),
        "tracking_loss_events": diag_summary.get("tracking_loss_events", 0),
        "ate_raw_rmse_m": eval_res.get("ate_raw_rmse_m", 0.0),
        "ate_aligned_rmse_m": eval_res.get("ate_aligned_rmse_m", 0.0),
        "rpe_rmse_m": eval_res.get("rpe_rmse_m", 0.0),
        "tracking_ok_pct": diag_summary.get("tracking_ok_pct", 0.0),
        "tracking_availability_pct": diag_summary.get("tracking_availability_pct", 0.0),
        "degraded_pct": diag_summary.get("degraded_pct", 0.0),
        "lost_pct": diag_summary.get("lost_pct", 0.0),
        "loss_events": diag_summary.get("loss_events", 0),
        "longest_continuous_ok": diag_summary.get("longest_continuous_ok_frames", 0),
        "longest_continuous_avail": diag_summary.get("longest_continuous_available_frames", 0),
        "scale_observable": (mode != "monocular"),
    }


def benchmark_rugd_sequences(sequences: list[str], max_frames: int = 50) -> list[dict]:
    """Audit feature tracking availability on real outdoor RUGD sequences."""
    from ai.localization.datasets import RUGDSequenceLoader

    camera = CameraIntrinsics(
        fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480
    )
    rugd_results = []
    for seq in sequences:
        seq_path = f"data/processed/sequences/rugd/{seq}"
        if not Path(seq_path).exists():
            continue
        loader = RUGDSequenceLoader(seq_path, max_frames=max_frames)
        vo = MonocularVisualOdometry(camera)
        diag = TrackingDiagnostics()
        t0 = time.perf_counter()
        for frame in loader:
            res = vo.process_frame(frame.image_left, frame.timestamp)
            diag.record(res)
        elapsed = time.perf_counter() - t0
        summary = diag.compute_summary()
        rugd_results.append({
            "sequence_id": seq,
            "total_frames": len(loader),
            "fps": len(loader) / elapsed if elapsed > 0 else 0.0,
            "track_ok_pct": summary.get("tracking_ok_pct", 0.0),
            "degraded_pct": summary.get("degraded_pct", 0.0),
            "lost_pct": summary.get("lost_pct", 0.0),
            "track_availability_pct": summary.get("tracking_availability_pct", 0.0),
            "loss_events": summary.get("loss_events", 0),
            "longest_continuous_ok": summary.get("longest_continuous_ok_frames", 0),
            "mean_inliers": summary.get("mean_inliers", 0.0),
        })
    return rugd_results


def main():
    print("\n" + "=" * 105)
    print("      SIH26126 PHASE 8: COMPREHENSIVE SENSOR MODE & TRACKING BENCHMARK")
    print("=" * 105)

    # 1. Synthetic Sequence Sensor Mode Comparison
    print("\n--- PART 1: SYNTHETIC STEREO & VIO BENCHMARK (Exact Ground Truth) ---")
    modes = ["monocular", "stereo", "stereo_imu"]
    results = []

    for mode in modes:
        print(f"\nRunning Synthetic Benchmark: {mode.upper()} (50 frames)")
        res = benchmark_mode(mode, num_frames=50)
        results.append(res)
        print(f"  FPS: {res['fps']:.1f} | Mean Latency: {res['mean_latency_ms']:.2f}ms | P95: {res['p95_latency_ms']:.2f}ms")
        print(f"  ATE Raw: {res['ate_raw_rmse_m']:.4f}m | ATE Aligned: {res['ate_aligned_rmse_m']:.4f}m | RPE: {res['rpe_rmse_m']:.4f}m")
        print(f"  Track OK%: {res['tracking_ok_pct']:.1f}% | Availability%: {res['tracking_availability_pct']:.1f}% | Loss Events: {res['loss_events']}")

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'Configuration':<14} | {'ATE Raw(m)':<11} | {'ATE Align(m)':<12} | {'RPE (m)':<9} | {'FPS':<7} | {'Track OK%':<10} | {'Avail %':<8} | {'Scale?':<10}")
    print("-" * 105)
    for r in results:
        scale_str = "Metric" if r["scale_observable"] else "Unscaled"
        print(f"{r['mode']:<14} | {r['ate_raw_rmse_m']:<11.4f} | {r['ate_aligned_rmse_m']:<12.4f} | {r['rpe_rmse_m']:<9.4f} | {r['fps']:<7.1f} | {r['tracking_ok_pct']:<10.1f} | {r['tracking_availability_pct']:<8.1f} | {scale_str:<10}")
    print("=" * 105)

    # 2. Real Outdoor RUGD Tracking Audit
    print("\n--- PART 2: REAL OUTDOOR RUGD SEQUENCE TRACKING AUDIT ---")
    rugd_seqs = ["creek", "park-1", "trail", "village"]
    rugd_results = benchmark_rugd_sequences(rugd_seqs, max_frames=50)

    print("\n" + "=" * 105)
    print(f"{'Sequence ID':<14} | {'Frames':<8} | {'Mean Inliers':<13} | {'Track OK%':<10} | {'Degraded%':<10} | {'Lost%':<8} | {'Avail%':<8} | {'Loss Events':<11}")
    print("-" * 105)
    for r in rugd_results:
        print(f"{r['sequence_id']:<14} | {r['total_frames']:<8} | {r['mean_inliers']:<13.1f} | {r['track_ok_pct']:<10.1f} | {r['degraded_pct']:<10.1f} | {r['lost_pct']:<8.1f} | {r['track_availability_pct']:<8.1f} | {r['loss_events']:<11}")
    print("=" * 105)

    # Save to JSON
    out_dir = Path("outputs/slam/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "rugd_tracking_summary.json", "w") as f:
        json.dump(rugd_results, f, indent=2)

    print(f"\n[OK] Saved Synthetic Benchmark: {out_dir / 'benchmark_summary.json'}")
    print(f"[OK] Saved RUGD Tracking Audit:  {out_dir / 'rugd_tracking_summary.json'}\n")


if __name__ == "__main__":
    main()
