"""Unit tests for SAM-TP traversability predictor adapter."""

from pathlib import Path
import numpy as np
import pytest

from ai.traversability.base import TraversabilityPredictorBase, TraversabilityResult
from ai.traversability.sam_tp import SAMTPTraversabilityPredictor


@pytest.fixture(scope="module")
def predictor():
    """Shared fixture loading SAM-TP once for the test session."""
    p = SAMTPTraversabilityPredictor(device="cpu", contrast_refine=True)
    return p


def test_predictor_inheritance(predictor):
    """Verify SAMTPTraversabilityPredictor complies with TraversabilityPredictorBase contract."""
    assert isinstance(predictor, TraversabilityPredictorBase)
    assert predictor.device == "cpu"
    assert "SAM-TP" in predictor.model_name


def test_predict_array_input(predictor):
    """Verify inference on synthetic RGB array returns valid TraversabilityResult."""
    # Synthetic 240x320 RGB frame
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    # Add light road in bottom center
    frame[120:, 80:240] = [180, 180, 160]

    res = predictor.predict(frame)

    assert isinstance(res, TraversabilityResult)
    assert res.mask.shape == (240, 320)
    assert res.mask.dtype == np.float32
    assert res.mask.min() >= 0.0
    assert res.mask.max() <= 1.0
    assert res.overlay.shape == (240, 320, 3)
    assert res.overlay.dtype == np.uint8
    assert res.inference_s > 0.0
    assert res.input_shape == (240, 320)


def test_predict_deterministic(predictor):
    """Verify that SAM-TP inference is strictly deterministic (zero random seeds/dropout at test time)."""
    frame = np.full((180, 240, 3), 100, dtype=np.uint8)

    res1 = predictor.predict(frame)
    res2 = predictor.predict(frame)

    np.testing.assert_allclose(res1.mask, res2.mask, atol=1e-5)


def test_predict_file_path(predictor):
    """Verify inference accepts a filepath argument."""
    test_img = Path("frame.jpg")
    if test_img.exists():
        res = predictor.predict(test_img)
        assert res.mask.shape == (480, 640)
        assert 0.0 <= res.mean_drivability <= 1.0
