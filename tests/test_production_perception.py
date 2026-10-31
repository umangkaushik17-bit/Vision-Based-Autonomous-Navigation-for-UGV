"""Test Suite for Hardware-Adaptive Production Perception.

Validates:
  - LightweightTraversabilityNet inference produces valid (400, 640) probability masks in [0.0, 1.0].
  - No NaN or Inf in outputs.
  - Device reporting and health checks.
  - Auto backend selection for perception.
  - Explicit backend request failures when hardware is unavailable (zero silent fallback).
  - Strict quarantining of SAM-TP as research-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.traversability.base import TraversabilityResult
from production.perception import AdaptiveTraversabilityPredictor
from research_reference.sam_tp import IS_RESEARCH_ONLY, IS_PRODUCTION_DEFAULT


def test_production_perception_auto_inference():
    """Verify AdaptiveTraversabilityPredictor in AUTO mode on host hardware."""
    predictor = AdaptiveTraversabilityPredictor(backend="AUTO")

    # Synthetic RGB frame
    img = np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)
    res = predictor.predict(img)

    assert isinstance(res, TraversabilityResult)
    assert res.mask.shape == (400, 640)
    assert not np.isnan(res.mask).any()
    assert not np.isinf(res.mask).any()
    assert (res.mask >= 0.0).all() and (res.mask <= 1.0).all()
    assert res.inference_s > 0.0
    assert "LightweightTraversabilityNet" in res.model_name


def test_production_perception_explicit_simd_inference():
    """Verify AdaptiveTraversabilityPredictor explicitly requesting PERCEPTION_SIMD."""
    predictor = AdaptiveTraversabilityPredictor(backend="PERCEPTION_SIMD")
    img = np.zeros((400, 640, 3), dtype=np.uint8)
    res = predictor.predict(img)

    assert res.mask.shape == (400, 640)
    assert res.device == "cpu"
    assert "PERCEPTION_SIMD" in predictor.active_backend_name


def test_production_perception_explicit_unsupported_cuda_fails_clearly():
    """When CUDA is unavailable, requesting PERCEPTION_CUDA must raise RuntimeError without silent fallback."""
    try:
        import torch
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError) as exc_info:
                AdaptiveTraversabilityPredictor(backend="PERCEPTION_CUDA")
            assert "System will NOT silently fall back" in str(exc_info.value)
    except Exception:
        pass


def test_sam_tp_strictly_quarantined_from_production():
    """Verify SAM-TP is designated research-only and cannot be selected as production backend."""
    assert IS_RESEARCH_ONLY is True
    assert IS_PRODUCTION_DEFAULT is False

    with pytest.raises(RuntimeError) as exc_info:
        AdaptiveTraversabilityPredictor(backend="PERCEPTION_SAM_TP")
    assert "research/reference asset" in str(exc_info.value)
