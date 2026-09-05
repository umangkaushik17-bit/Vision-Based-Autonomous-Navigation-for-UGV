"""End-to-End Perception Latency and Throughput Benchmark (Phase 4).

Measures wall-clock latency, throughput (FPS), P50, and P95 percentiles across:
1. Model A (SAM-TP) alone
2. Model B (RUGD) alone
3. Obstacle & Hazard Detector alone
4. Safety Perception Fusion Engine (End-to-End)

Accurately reports CPU measured numbers and marks GPU benchmark as PENDING.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np

from ai.obstacles import UnifiedHazardDetector
from ai.perception_fusion import SafetyPerceptionFusion
from ai.traversability.rugd_model import RUGDTraversabilityPredictor
from ai.traversability.sam_tp import SAMTPTraversabilityPredictor

OUTPUT_DIR = Path("outputs/perception/benchmarks")


def benchmark_component(name: str, fn, iterations: int = 15, warmup: int = 3) -> Dict[str, Any]:
    """Run warm-up and timed iterations for a callable perception component."""
    print(f"\n[BENCHMARK] Testing {name} ({warmup} warmup + {iterations} timed iterations)...")

    # Warmup
    for _ in range(warmup):
        _ = fn()

    latencies_ms: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = fn()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

    latencies = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    p95_ms = float(np.percentile(latencies, 95))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"  -> Mean Latency: {mean_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    print(f"  -> P50 (Median): {p50_ms:.2f} ms | P95: {p95_ms:.2f} ms")
    print(f"  -> Throughput  : {fps:.2f} FPS")

    return {
        "component": name,
        "iterations": iterations,
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "min_latency_ms": round(min_ms, 2),
        "max_latency_ms": round(max_ms, 2),
        "fps": round(fps, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark perception components and fusion pipeline")
    parser.add_argument("--image", type=str, default="frame.jpg", help="Input test frame")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    if not img_path.exists():
        # Fallback synthetic dummy frame (480, 640, 3)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("\n" + "=" * 70)
    print("      SIH26126 PHASE 4: PERCEPTION LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 70)
    print("Hardware Environment: Host AMD Ryzen CPU (GPU Extrapolation Policy: NOT CLAIMED)")

    # 1. Initialize models
    pred_a = SAMTPTraversabilityPredictor()
    pred_b = RUGDTraversabilityPredictor()
    hazard_detector = UnifiedHazardDetector()
    fusion_engine = SafetyPerceptionFusion()

    # Benchmark 1: Model A (SAM-TP) alone
    bench_a = benchmark_component(
        "Model A (SAM-TP Mini+)",
        lambda: pred_a.predict(rgb),
        iterations=args.iterations,
        warmup=args.warmup
    )

    # Benchmark 2: Model B (RUGD ONNX) alone
    bench_b = benchmark_component(
        "Model B (RUGD Traversability Net)",
        lambda: pred_b.predict(rgb),
        iterations=args.iterations,
        warmup=args.warmup
    )

    # Benchmark 3: Obstacle & Hazard Detector alone
    bench_hazards = benchmark_component(
        "Unified Hazard Detector",
        lambda: hazard_detector.detect(rgb),
        iterations=args.iterations,
        warmup=args.warmup
    )

    # Benchmark 4: End-to-End Safety Perception Fusion Pipeline
    def run_end_to_end():
        res_a = pred_a.predict(rgb)
        res_b = pred_b.predict(rgb)
        dets = hazard_detector.detect(rgb)
        return fusion_engine.fuse(rgb, res_a, res_b, dets)

    bench_e2e = benchmark_component(
        "End-to-End Safety Perception Fusion",
        run_end_to_end,
        iterations=args.iterations,
        warmup=args.warmup
    )

    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": "cpu",
            "gpu_benchmark_status": "PENDING (Host is CPU-only, GPU benchmark not run)"
        },
        "frame_dimensions": list(rgb.shape[:2]),
        "components": {
            "sam_tp_model_a": bench_a,
            "rugd_model_b": bench_b,
            "unified_hazard_detector": bench_hazards,
            "end_to_end_fusion": bench_e2e
        }
    }

    report_path = OUTPUT_DIR / "perception_latency_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 70)
    print("                     BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Model A (SAM-TP)       : {bench_a['mean_latency_ms']} ms ({bench_a['fps']} FPS)")
    print(f"  Model B (RUGD)         : {bench_b['mean_latency_ms']} ms ({bench_b['fps']} FPS)")
    print(f"  Hazard Detector        : {bench_hazards['mean_latency_ms']} ms ({bench_hazards['fps']} FPS)")
    print(f"  End-to-End Fusion Total: {bench_e2e['mean_latency_ms']} ms ({bench_e2e['fps']} FPS)")
    print(f"\n[OK] Latency report saved to: {report_path}")


if __name__ == "__main__":
    main()
