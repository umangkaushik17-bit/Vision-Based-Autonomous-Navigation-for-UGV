"""Phase 14.2 Perception Latency and Performance Benchmark Tests.

Validates:
- >20 FPS end-to-end perception throughput on host hardware
- Sub-50ms average total pipeline latency
- Full breakdown logging (preprocessing_ms, inference_ms, postprocessing_ms)
- Phase 13 AdaptiveTraversabilityPredictor backward compatibility
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from production.perception import (
    AdaptiveTraversabilityPredictor,
    RealtimeTraversabilityPipeline,
    SyntheticCameraSource,
)


def test_end_to_end_perception_throughput_exceeds_20_fps():
    """Verify that end-to-end perception achieves >20 FPS (>20 Hz) on host hardware."""
    pipe = RealtimeTraversabilityPipeline(backend="AUTO")
    cam = SyntheticCameraSource()
    pipe.warmup(3)

    latencies = []
    for _ in range(30):
        frame = cam.read()
        res = pipe.predict(frame)
        assert res.valid is True
        latencies.append(res.total_ms)

    mean_lat_ms = float(np.mean(latencies))
    fps = 1000.0 / mean_lat_ms

    print(f"\n[PHASE 14.2 BENCHMARK] Backend: {pipe.active_backend_name}")
    print(f"[PHASE 14.2 BENCHMARK] Mean Latency: {mean_lat_ms:.2f} ms")
    print(f"[PHASE 14.2 BENCHMARK] P95 Latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"[PHASE 14.2 BENCHMARK] Throughput:   {fps:.1f} FPS")

    assert fps >= 20.0, f"Perception throughput fell below 20 FPS: {fps:.1f} FPS (mean {mean_lat_ms:.2f} ms)"


def test_latency_breakdown_consistency():
    """Verify that preprocessing, inference, and postprocessing sum consistently to total_ms."""
    pipe = RealtimeTraversabilityPipeline(backend="AUTO")
    dummy = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    res = pipe.predict(dummy)

    assert res.preprocessing_ms > 0.0
    assert res.inference_s > 0.0
    assert res.postprocessing_ms >= 0.0

    sum_stages = res.preprocessing_ms + (res.inference_s * 1000.0) + res.postprocessing_ms
    # Total ms should be greater than or equal to sum of stages (allowing small timing overhead)
    assert res.total_ms >= sum_stages - 1.0


def test_phase13_perception_contract_backward_compatibility():
    """Verify that Phase 13 AdaptiveTraversabilityPredictor executes without divergence."""
    legacy_pred = AdaptiveTraversabilityPredictor(backend="AUTO")
    dummy = np.random.randint(50, 200, (400, 640, 3), dtype=np.uint8)
    res = legacy_pred.predict(dummy)

    assert res.mask.shape == (400, 640)
    assert not np.isnan(res.mask).any()
    assert (res.mask >= 0.0).all() and (res.mask <= 1.0).all()
    assert res.inference_s > 0.0
    assert res.valid is True
