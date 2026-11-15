"""Unit tests for Workstream WS-3A: Dynamic ONNX Export & Mixed-Precision INT8 PTQ Quantization."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from production.perception.base import PerceptionFrameInput
from production.perception.realtime_pipeline import RealtimeTraversabilityPipeline


def test_quantized_onnx_models_exist_and_meet_safety_parity():
    """Verify both dynamic FP32 ONNX and INT8 QDQ ONNX models exist and satisfy <1% diff and 0 false-safe leakage."""
    fp32_path = Path("outputs/exports/traversability_model_dyn.onnx")
    int8_path = Path("outputs/exports/traversability_model_int8.onnx")
    report_path = Path("reports/onnx_int8_quantization_benchmark.json")

    assert fp32_path.exists(), f"Missing {fp32_path}"
    assert int8_path.exists(), f"Missing {int8_path}"
    assert report_path.exists(), f"Missing {report_path}"

    with open(report_path, "r") as f:
        metrics = json.load(f)

    assert metrics["mean_drivability_diff"] < 0.01, f"Drivability diff {metrics['mean_drivability_diff']} >= 0.01"
    assert metrics["false_safe_pixel_increases"] == 0, f"False-safe pixel increases: {metrics['false_safe_pixel_increases']}"


def test_realtime_pipeline_onnx_fp32_and_int8_execution():
    """Verify RealtimeTraversabilityPipeline executes both PERCEPTION_ONNX_CPU and PERCEPTION_ONNX_INT8_CPU."""
    pipe_fp32 = RealtimeTraversabilityPipeline(backend="PERCEPTION_ONNX_CPU", generate_overlay=False)
    pipe_int8 = RealtimeTraversabilityPipeline(backend="PERCEPTION_ONNX_INT8_CPU", generate_overlay=False)

    assert pipe_fp32.active_backend_name == "PERCEPTION_ONNX_CPU"
    assert pipe_int8.active_backend_name == "PERCEPTION_ONNX_INT8_CPU"

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:200, :, :] = [180, 140, 100]
    img[200:, :, :] = [60, 120, 70]
    img[250:320, 280:360, :] = [20, 20, 20]

    res_fp32 = pipe_fp32.predict(PerceptionFrameInput(image=img, timestamp=1.0))
    res_int8 = pipe_int8.predict(PerceptionFrameInput(image=img, timestamp=1.0))

    assert res_fp32.valid and res_int8.valid
    assert res_fp32.mask.shape == (480, 640)
    assert res_int8.mask.shape == (480, 640)
    mean_diff = float(np.mean(np.abs(res_fp32.mask - res_int8.mask)))
    assert mean_diff < 0.02, f"End-to-end mask difference {mean_diff:.4f} >= 0.02"
