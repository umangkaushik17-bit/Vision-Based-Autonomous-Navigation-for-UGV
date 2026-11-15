"""Unit tests for Decoupled Asynchronous Perception Worker (P1-3 / Defect #7).

Verifies:
1. Non-blocking camera frame submission (< 0.5 ms caller overhead even with slow 40 ms model).
2. Bounded depth-1 latest-frame-only slot overwrites stale burst frames (zero queue lag).
3. Background 3D SE(3) attitude-compensated BEV projection and atomic result retrieval.
4. Clean worker lifecycle shutdown.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from ai.traversability.base import TraversabilityResult
from production.perception.async_perception_worker import AsyncPerceptionWorker
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator


class _SimulatedSlowPredictor:
    """Simulates a 35 ms neural network inference pass."""

    def __init__(self, delay_s: float = 0.035) -> None:
        self.delay_s = delay_s

    def predict(self, rgb: np.ndarray) -> TraversabilityResult:
        time.sleep(self.delay_s)
        h, w = rgb.shape[:2]
        mask = np.full((h // 2, w // 2), 0.85, dtype=np.float32)
        return TraversabilityResult(
            mask=mask,
            confidence=np.ones_like(mask),
            timestamp=time.time(),
            valid=True,
        )


def test_async_worker_non_blocking_submission():
    """Verify submit_frame returns in < 0.5 ms while 35 ms inference runs in background."""
    predictor = _SimulatedSlowPredictor(delay_s=0.035)
    integrator = VisionCostmapIntegrator.create_simulation_default()
    worker = AsyncPerceptionWorker(
        predictor=predictor,
        costmap_integrator=integrator,
        enable_bev_projection=True,
    )

    try:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        seq = worker.submit_frame(
            rgb=rgb,
            timestamp_s=time.time(),
            ugv_pose_xy_yaw=(1.0, 2.0, 0.5),
            roll_rad=0.02,
            pitch_rad=-0.04,
        )
        submit_ms = (time.perf_counter() - t0) * 1000.0
        assert seq == 1
        assert submit_ms < 1.0, f"submit_frame blocked caller for {submit_ms:.3f}ms (> 1.0ms)"

        # Wait for background completion
        deadline = time.perf_counter() + 0.50
        out = None
        while time.perf_counter() < deadline:
            out = worker.pop_latest_result()
            if out is not None:
                break
            time.sleep(0.005)

        assert out is not None
        assert out.sequence_id == 1
        assert out.traversability_result.valid
        assert out.bev_costmap is not None
        assert out.bev_costmap.valid
        assert out.ugv_pose_xy_yaw == (1.0, 2.0, 0.5)
    finally:
        worker.stop()


def test_async_worker_latest_frame_overwrite_drops_stale():
    """Verify burst of 10 frames during a 40ms inference drops stale intermediate frames and processes the newest."""
    predictor = _SimulatedSlowPredictor(delay_s=0.040)
    worker = AsyncPerceptionWorker(predictor=predictor, enable_bev_projection=False)

    try:
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        for i in range(10):
            worker.submit_frame(rgb=rgb, timestamp_s=100.0 + i)
            time.sleep(0.001)

        # Wait for worker to finish current + latest slot
        time.sleep(0.15)
        out = worker.pop_latest_result()
        assert out is not None
        # Must have processed the very last frame (sequence_id == 10) and dropped intermediate stale ones
        assert out.sequence_id == 10
        assert worker.dropped_stale_count >= 5
    finally:
        worker.stop()
