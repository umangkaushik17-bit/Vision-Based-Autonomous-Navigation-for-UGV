"""High-precision inference benchmark tool for traversability models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.models.traversability_net import create_traversability_model

def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    input_resolution=(480, 640),
    num_warmup: int = 10,
    num_iterations: int = 40
) -> dict:
    """Benchmark end-to-end tensor inference latencies."""
    h, w = input_resolution
    dummy_tensor = torch.randn(1, 3, h, w, device=device)

    # 1. Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 2. Benchmark loop
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model(dummy_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms

    latencies_arr = np.array(latencies)
    avg_latency = float(np.mean(latencies_arr))
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = (param_count * 4) / (1024 * 1024)

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "Host CPU",
        "input_resolution": [h, w],
        "parameter_count": param_count,
        "model_size_mb": round(param_mb, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "fps": round(fps, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Traversability Model latency and FPS")
    parser.add_argument("--output-report", type=str, default="outputs/reports/benchmark_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_traversability_model().to(device)

    print(f"\n--- [TRAVERSABILITY BENCHMARK] Device: {device} ---")
    results = benchmark_model(model, device)

    print("\n" + "=" * 55)
    print(f"Device              : {results['device_name']}")
    print(f"Resolution          : {results['input_resolution']}")
    print(f"Parameters          : {results['parameter_count']:,} (~{results['model_size_mb']} MB)")
    print(f"Avg Latency         : {results['avg_latency_ms']} ms")
    print(f"P50 Median Latency  : {results['p50_latency_ms']} ms")
    print(f"P95 Tail Latency    : {results['p95_latency_ms']} ms")
    print(f"Throughput          : {results['fps']} FPS")
    print("=" * 55 + "\n")

    out_p = Path(args.output_report).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[REPORT] Benchmark metrics saved to: {out_p}")

if __name__ == "__main__":
    main()
