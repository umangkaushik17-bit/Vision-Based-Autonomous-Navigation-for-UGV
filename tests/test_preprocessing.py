"""Automated tests for configurable preprocessing pipeline and data preservation."""

import numpy as np
import pytest

from ai.core.config import PreprocessingConfig, VisionConfig, load_vision_config
from ai.core.frame_source import VisionFrame
from ai.core.preprocessing import Preprocessor, ProcessedFrame
from ai.core.processor import CannyEdgeProcessor, GrayscaleProcessor

def test_preprocessing_pipeline_and_preservation():
    cfg = PreprocessingConfig()
    cfg.resize.target_width = 320
    cfg.resize.target_height = 240
    cfg.normalization.enabled = True
    cfg.normalization.dtype = "float32"

    raw_input = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    raw_clone = raw_input.copy()

    vision_frame = VisionFrame(
        raw_frame=raw_input,
        frame_index=0,
        timestamp=0.0,
        source_type="image"
    )

    preprocessor = Preprocessor(cfg)
    processed = preprocessor.process(vision_frame)

    # 1. Output dimensions and types
    assert processed.preprocessed_resolution == (320, 240)
    assert processed.preprocessed_frame.dtype == np.float32
    assert np.min(processed.preprocessed_frame) >= 0.0
    assert np.max(processed.preprocessed_frame) <= 1.0

    # 2. Strict preservation of original frame
    assert processed.original_resolution == (640, 480)
    assert processed.original_frame.dtype == np.uint8
    assert np.array_equal(processed.original_frame, raw_clone), "Original frame must not be mutated!"
    assert np.array_equal(raw_input, raw_clone), "Raw input array must remain completely untouched!"

def test_preprocessing_color_and_clahe():
    cfg = PreprocessingConfig()
    cfg.color_conversion.target_format = "RGB"
    cfg.clahe.enabled = True
    cfg.denoising.enabled = True
    cfg.denoising.method = "bilateral"
    cfg.normalization.enabled = False

    raw = np.full((100, 100, 3), 120, dtype=np.uint8)
    vision_frame = VisionFrame(raw_frame=raw, frame_index=0, timestamp=0.0, source_type="image")

    preprocessor = Preprocessor(cfg)
    processed = preprocessor.process(vision_frame)
    assert processed.preprocessed_frame.shape == (cfg.resize.target_height, cfg.resize.target_width, 3)

def test_baseline_processors():
    raw = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    vision_frame = VisionFrame(raw_frame=raw, frame_index=0, timestamp=0.0, source_type="image")
    preprocessor = Preprocessor()
    processed = preprocessor.process(vision_frame)

    canny = CannyEdgeProcessor()
    out_canny = canny.process(processed)
    assert out_canny.shape == (480, 640, 3)
    assert out_canny.dtype == np.uint8

    gray_proc = GrayscaleProcessor()
    out_gray = gray_proc.process(processed)
    assert out_gray.shape == (480, 640, 3)
    assert out_gray.dtype == np.uint8
