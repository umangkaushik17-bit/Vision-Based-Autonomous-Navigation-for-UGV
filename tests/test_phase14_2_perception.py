"""Phase 14.2 Realtime Perception Closure Test Suite.

Validates:
- Explicit TraversabilityResult contract fields and types
- UNKNOWN != SAFE safety invariant
- Input frame validation (blackout, corrupted, NaN, shape)
- Conservative fail-safe fallback (zero free space, 100% unknown)
- Hardware-adaptive backend selection and zero silent fallback
- Asymmetric temporal stability filtering
- Camera source abstraction (Synthetic, Sequence)
- Segmentation quality metrics on synthetic outdoor sequences
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from ai.traversability.base import PerceptionFrameInput, TraversabilityResult
from production.perception import (
    AdaptiveTraversabilityPredictor,
    CameraConfig,
    RealtimeTraversabilityPipeline,
    SequenceCameraSource,
    SyntheticCameraSource,
    TemporalStabilityFilter,
)


@pytest.fixture(scope="module")
def pipeline():
    pipe = RealtimeTraversabilityPipeline(backend="AUTO")
    pipe.warmup(2)
    return pipe


def test_perception_contract_fields_and_types(pipeline):
    """Verify TraversabilityResult contains all explicit Phase 14.2 contract fields."""
    dummy = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    res = pipeline.predict(dummy)

    assert isinstance(res, TraversabilityResult)
    assert res.valid is True
    assert res.status_message == "OK"
    assert res.mask.shape == (480, 640)
    assert res.mask.dtype == np.float32
    assert (res.mask >= 0.0).all() and (res.mask <= 1.0).all()

    assert res.confidence is not None
    assert res.confidence.shape == (480, 640)
    assert (res.confidence >= 0.0).all() and (res.confidence <= 1.0).all()

    assert res.unknown_mask is not None
    assert res.unknown_mask.shape == (480, 640)
    assert res.unknown_mask.dtype == bool

    assert res.input_shape == (480, 640)
    assert res.timestamp > 0.0
    assert res.frame_id == "camera_optical_frame"
    assert res.preprocessing_ms >= 0.0
    assert res.inference_s > 0.0
    assert res.postprocessing_ms >= 0.0
    assert res.total_ms > 0.0
    assert pipeline.active_backend_name in res.model_name


def test_unknown_not_equal_safe_invariant(pipeline):
    """Verify UNKNOWN != SAFE: unknown regions are strictly excluded from safe paths."""
    # Create synthetic frame with high unknown/risky region
    dummy = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    res = pipeline.predict(dummy)

    safe_mask = res.get_safe_mask(threshold=0.5)
    assert safe_mask.dtype == bool

    # Where unknown_mask is True, safe_mask MUST be False
    if res.unknown_mask is not None and res.unknown_mask.any():
        assert not (safe_mask & res.unknown_mask).any(), "Safe mask leaked into unknown region!"

    # Binary mask on an invalid result must be all False
    invalid_res = TraversabilityResult(
        mask=np.ones((10, 10), dtype=np.float32),
        logits=np.zeros((10, 10), dtype=np.float32),
        overlay=np.zeros((10, 10, 3), dtype=np.uint8),
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        input_shape=(10, 10),
        inference_s=0.0,
        device="cpu",
        model_name="test",
        valid=False,
    )
    assert not invalid_res.get_safe_mask().any()
    assert not invalid_res.get_binary_mask().any()
    assert invalid_res.get_obstacle_mask().all()
    assert invalid_res.mean_drivability == 0.0


def test_input_validation_blackout_safe_fallback(pipeline):
    """Camera blackout (black frame) must trigger safe conservative fallback."""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.predict(black_img)

    assert res.valid is False
    assert "BLACKOUT" in res.status_message
    assert (res.mask == 0.0).all(), "Drivability must be 0.0 on blackout failure!"
    assert res.unknown_mask.all(), "Unknown mask must be True everywhere on blackout!"
    assert (res.confidence == 0.0).all()
    assert not res.get_safe_mask().any()


def test_input_validation_corrupted_nan_safe_fallback(pipeline):
    """Corrupted frame with NaN or wrong shape must trigger safe fallback."""
    nan_img = np.full((480, 640, 3), np.nan, dtype=np.float32)
    res = pipeline.predict(nan_img)

    assert res.valid is False
    assert "NAN_OR_INF" in res.status_message
    assert (res.mask == 0.0).all()
    assert res.unknown_mask.all()

    # Invalid dimension
    bad_img = np.zeros((480, 640), dtype=np.uint8)
    res_bad = pipeline.predict(bad_img)
    assert res_bad.valid is False
    assert (res_bad.mask == 0.0).all()


def test_explicit_backend_selection_and_no_silent_fallback():
    """Explicit backend requests must not silently fall back to other devices."""
    # Explicit CPU
    p_cpu = RealtimeTraversabilityPipeline(backend="PERCEPTION_CPU")
    assert p_cpu.active_backend_name == "PERCEPTION_CPU"
    assert p_cpu.device == "cpu"

    # Explicit CUDA on host without CUDA must raise clear RuntimeError
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            RealtimeTraversabilityPipeline(backend="PERCEPTION_CUDA")
        assert "System will NOT silently fall back" in str(exc_info.value)


def test_temporal_stability_filter_asymmetric_latching():
    """Temporal filter must latch obstacles with zero lag, and smooth clearing noise."""
    flt = TemporalStabilityFilter(alpha_hazard=1.0, alpha_recovery=0.70)

    # Frame 1: completely safe (1.0)
    m1 = np.ones((10, 10), dtype=np.float32)
    out1, _ = flt.apply(m1, timestamp=1.0)
    assert np.allclose(out1, 1.0)

    # Frame 2: obstacle appears in center (drops from 1.0 to 0.0)
    m2 = m1.copy()
    m2[4:6, 4:6] = 0.0
    out2, _ = flt.apply(m2, timestamp=1.033)

    # Zero-lag obstacle check: where drivability dropped, out2 must drop instantly to 0.0!
    assert np.allclose(out2[4:6, 4:6], 0.0), "Obstacle appearance must be instantaneous!"

    # Frame 3: momentary noisy clearing (rises from 0.0 to 1.0)
    m3 = m1.copy()
    out3, _ = flt.apply(m3, timestamp=1.066)
    # With alpha_recovery = 0.70, value should rise from 0.0 to 0.70 (not instantly to 1.0)
    assert out3[4, 4] < 0.90, "Clearing should be smoothed to prevent hazard flutter!"

    # Frame 4: Timestamp gap > 0.5s (frame drop) resets filter
    m4 = np.zeros((10, 10), dtype=np.float32)
    out4, _ = flt.apply(m4, timestamp=2.50)  # >0.5s jump
    assert np.allclose(out4, 0.0)


def test_camera_sources_synthetic_and_sequence():
    """Verify camera interfaces stream valid PerceptionFrameInput frames."""
    # Synthetic camera
    synth = SyntheticCameraSource(base_pattern="TRAIL")
    assert synth.is_opened()
    frame = synth.read()
    assert isinstance(frame, PerceptionFrameInput)
    assert frame.image.shape == (480, 640, 3)
    assert frame.timestamp > 0.0
    synth.close()
    assert not synth.is_opened()

    # Sequence camera
    seq_path = Path("data/processed/sequences/synthetic/syn_seq_01_trail_sun")
    if seq_path.exists():
        seq_cam = SequenceCameraSource(sequence_dir=seq_path, loop=False)
        assert seq_cam.is_opened()
        f1 = seq_cam.read()
        assert f1 is not None
        assert f1.image.shape == (480, 640, 3)
        seq_cam.close()


def test_perception_quality_metrics_on_synthetic_sequence(pipeline):
    """Evaluate perception accuracy against ground truth annotations on syn_seq_01."""
    seq_path = Path("data/processed/sequences/synthetic/syn_seq_01_trail_sun")
    if not seq_path.exists():
        pytest.skip("Synthetic sequence syn_seq_01_trail_sun not present")

    import cv2
    rgb_files = sorted(list((seq_path / "rgb").glob("*.png")))[:5]
    mask_files = sorted(list((seq_path / "masks").glob("*.png")))[:5]

    ious, recalls = [], []
    for rgb_p, mask_p in zip(rgb_files, mask_files):
        bgr = cv2.imread(str(rgb_p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)

        res = pipeline.predict(rgb)
        assert res.valid is True

        # GT: 1 = Trail (Safe), 0 = Vegetation/Obstacle, 2 = Rock, 255 = Sky
        gt_safe = (gt == 1)
        pred_safe = res.get_safe_mask(threshold=0.5)

        inter = np.logical_and(pred_safe, gt_safe).sum()
        union = np.logical_or(pred_safe, gt_safe).sum()
        iou = inter / union if union > 0 else 0.0
        rec = inter / gt_safe.sum() if gt_safe.sum() > 0 else 0.0

        ious.append(iou)
        recalls.append(rec)

    mean_iou = float(np.mean(ious))
    mean_rec = float(np.mean(recalls))

    assert mean_iou >= 0.80, f"Safe Path IoU too low: {mean_iou:.4f} < 0.80"
    assert mean_rec >= 0.85, f"Safe Path Recall too low: {mean_rec:.4f} < 0.85"


def test_perception_quality_across_diverse_terrains(pipeline):
    """Verify stability across rocks, grass, shadows, and ditch sequences."""
    seq_names = [
        "syn_seq_03_rocks_sun",
        "syn_seq_04_grass_overcast",
        "syn_seq_05_rocks_shadows",
        "syn_seq_07_ditch_unseen_02",
    ]
    import cv2

    for name in seq_names:
        seq_dir = Path(f"data/processed/sequences/synthetic/{name}/rgb")
        if not seq_dir.exists():
            continue

        imgs = sorted(list(seq_dir.glob("*.png")))[:3]
        for img_p in imgs:
            bgr = cv2.imread(str(img_p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = pipeline.predict(rgb)

            assert res.valid is True
            assert not np.isnan(res.mask).any()
            assert not np.isinf(res.mask).any()
            assert res.total_ms > 0.0
            assert res.confidence is not None
