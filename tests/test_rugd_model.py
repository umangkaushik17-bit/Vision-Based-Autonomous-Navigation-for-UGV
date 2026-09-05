"""Unit tests for RUGD Traversability Model Adapter (Model B)."""

import pytest
import numpy as np
from pathlib import Path

from ai.traversability.rugd_model import RUGDTraversabilityPredictor
from ai.traversability.base import TraversabilityResult


@pytest.fixture
def rugd_predictor():
    return RUGDTraversabilityPredictor()


def test_rugd_predictor_properties(rugd_predictor):
    """Verify predictor properties and model naming."""
    assert rugd_predictor.model_name == "RUGD-Traversability-Net (Model B)"
    assert rugd_predictor.device in ["cpu", "cuda"]
    info = rugd_predictor.get_model_info()
    assert info["model_id"] == "rugd_traversability_v1"
    assert "checkpoint" in info


def test_rugd_prediction_output_contract(rugd_predictor):
    """Verify output contract adhering to TraversabilityResult standard."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[200:, :] = 120  # Gray ground

    res = rugd_predictor.predict(dummy_img)

    assert isinstance(res, TraversabilityResult)
    assert res.mask.shape == (480, 640)
    assert res.mask.dtype == np.float32
    assert res.mask.min() >= 0.0
    assert res.mask.max() <= 1.0
    assert res.logits.shape == (480, 640)
    assert res.overlay.shape == (480, 640, 3)
    assert res.input_shape == (480, 640)
    assert res.inference_s > 0.0
    assert res.confidence is not None
    assert res.confidence.shape == (480, 640)


def test_rugd_prediction_from_filepath(rugd_predictor):
    """Verify prediction works directly from image filepath."""
    test_path = Path("frame.jpg")
    if test_path.exists():
        res = rugd_predictor.predict(test_path)
        assert res.mask.shape == (480, 640)
        assert 0.0 <= res.mean_drivability <= 1.0


def test_rugd_warmup(rugd_predictor):
    """Verify warmup pass executes and returns execution time."""
    warmup_s = rugd_predictor.warmup()
    assert isinstance(warmup_s, float)
    assert warmup_s > 0.0
