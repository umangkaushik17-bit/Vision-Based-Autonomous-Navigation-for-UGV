"""Video-rate benchmark evaluating latency percentiles (P50, P95), throughput (FPS), and temporal stability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

from ai.traversability.postprocess import TemporalTraversabilityFilter
from ai.traversability.sam_tp import SAMTPTraversabilityPredictor
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("benchmark_video_sam_tp")


def run_video_benchmark(
    num_frames: int = 15,
    sequence_path: str = "data/processed/sequences/own_data/own_seq_01_dirt/rgb",
    output_dir: Path = Path("outputs/traversability/baseline")
) -> Dict:
    """Run sequential video benchmark measuring true latency and temporal stability."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_dir = Path(sequence_path)
    frame_paths = sorted(list(seq_dir.glob("*.png")) + list(seq_dir.glob("*.jpg")))
    if not frame_paths:
        # Fallback to any sequence
        frame_paths = sorted(list(Path("data/processed/sequences").rglob("*_rgb.png")))
    if not frame_paths:
        frame_paths = sorted(list(Path("data/processed/sequences").rglob("*.png")))

    frame_paths = frame_paths[:num_frames]
    logger.info(f"Loaded {len(frame_paths)} sequential frames for video benchmark.")

    predictor = SAMTPTraversabilityPredictor(contrast_refine=True)
    predictor.warmup()

    filter_ema = TemporalTraversabilityFilter(alpha=0.7)

    total_latencies: List[float] = []
    inference_latencies: List[float] = []
    prep_latencies: List[float] = []
    post_latencies: List[float] = []

    raw_diffs: List[float] = []
    smoothed_diffs: List[float] = []

    prev_raw: np.ndarray | None = None
    prev_smoothed: np.ndarray | None = None

    for i, fp in enumerate(frame_paths):
        t0 = time.perf_counter()

        # 1. Preprocessing / Read
        t_prep_start = time.perf_counter()
        bgr = cv2.imread(str(fp))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prep_s = time.perf_counter() - t_prep_start

        # 2. Inference
        res = predictor.predict(rgb)
        inf_s = res.inference_s

        # 3. Postprocessing & EMA smoothing
        t_post_start = time.perf_counter()
        raw_mask = res.mask
        smoothed_mask = filter_ema.update(raw_mask)
        post_s = time.perf_counter() - t_post_start

        total_s = time.perf_counter() - t0

        total_latencies.append(total_s)
        inference_latencies.append(inf_s)
        prep_latencies.append(prep_s)
        post_latencies.append(post_s)

        # Measure inter-frame jitter (pixel MAE)
        if prev_raw is not None:
            raw_diff = float(np.mean(np.abs(raw_mask - prev_raw)))
            smoothed_diff = float(np.mean(np.abs(smoothed_mask - prev_smoothed)))
            raw_diffs.append(raw_diff)
            smoothed_diffs.append(smoothed_diff)

        prev_raw = raw_mask.copy()
        prev_smoothed = smoothed_mask.copy()

    # Calculate statistics
    total_arr = np.array(total_latencies)
    inf_arr = np.array(inference_latencies)
    prep_arr = np.array(prep_latencies)
    post_arr = np.array(post_latencies)

    measured_fps = float(len(total_latencies) / np.sum(total_arr))
    p50_latency = float(np.percentile(total_arr, 50))
    p95_latency = float(np.percentile(total_arr, 95))
    mean_latency = float(np.mean(total_arr))

    mean_raw_jitter = float(np.mean(raw_diffs)) if raw_diffs else 0.0
    mean_smoothed_jitter = float(np.mean(smoothed_diffs)) if smoothed_diffs else 0.0
    jitter_reduction_pct = float((1.0 - (mean_smoothed_jitter / max(mean_raw_jitter, 1e-6))) * 100.0)

    video_results = {
        "model_name": predictor.model_name,
        "device": predictor.device,
        "num_frames_evaluated": len(frame_paths),
        "fps": {
            "measured_fps": measured_fps,
            "theoretical_pure_inf_fps": float(1.0 / np.mean(inf_arr))
        },
        "latency_seconds": {
            "mean_total": mean_latency,
            "p50_median": p50_latency,
            "p95_high": p95_latency,
            "mean_preprocessing": float(np.mean(prep_arr)),
            "mean_inference": float(np.mean(inf_arr)),
            "mean_postprocessing": float(np.mean(post_arr))
        },
        "temporal_stability": {
            "raw_interframe_jitter_mae": mean_raw_jitter,
            "smoothed_interframe_jitter_mae": mean_smoothed_jitter,
            "jitter_reduction_percent": jitter_reduction_pct
        }
    }

    results_file = output_dir / "video_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(video_results, f, indent=2)

    logger.info("Video Benchmark Complete:")
    logger.info(f"  Device: {predictor.device}")
    logger.info(f"  Measured FPS: {measured_fps:.2f} Hz")
    logger.info(f"  Mean Latency: {mean_latency*1000:.1f} ms (P50: {p50_latency*1000:.1f} ms, P95: {p95_latency*1000:.1f} ms)")
    logger.info(f"  Temporal Jitter Reduction: {jitter_reduction_pct:.1f}%")

    return video_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-TP Video Rate Benchmark")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of frames to evaluate")
    args = parser.parse_args()

    run_video_benchmark(num_frames=args.num_frames)
