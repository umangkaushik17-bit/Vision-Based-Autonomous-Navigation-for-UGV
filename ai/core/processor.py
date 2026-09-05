"""Modular vision processors for feature extraction and pipeline validation.

Designed as an easily swappable interface so future deep learning models
(TraversabilityModel, ObstacleDetector, DepthModel, VisualOdometry) can drop in seamlessly.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from ai.core.config import ProcessingConfig, load_vision_config
from ai.core.preprocessing import ProcessedFrame

class BaseVisionProcessor(abc.ABC):
    """Abstract interface for all downstream vision analysis modules."""

    @abc.abstractmethod
    def process(self, processed_frame: ProcessedFrame) -> np.ndarray:
        """Process preprocessed frame and return visual output (H, W, 3) uint8 BGR for display."""
        pass

class CannyEdgeProcessor(BaseVisionProcessor):
    """Baseline outdoor edge feature detector for terrain boundary and obstacle outline analysis."""

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, processed_frame: ProcessedFrame) -> np.ndarray:
        img = processed_frame.preprocessed_frame

        # Ensure uint8 in range [0, 255]
        if img.dtype in (np.float32, np.float64):
            gray_input = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray_input = img

        # Convert to single-channel grayscale if needed
        if gray_input.ndim == 3:
            gray = cv2.cvtColor(gray_input, cv2.COLOR_RGB2GRAY)
        else:
            gray = gray_input

        # Smooth slightly before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)

        # Convert single-channel edge mask to 3-channel green/cyan overlay on original for rich display
        edge_bgr = np.zeros((*edges.shape, 3), dtype=np.uint8)
        edge_bgr[edges > 0] = (0, 255, 128) # Bright emerald green edges

        # Alpha-blend with original raw frame (resized to match)
        raw_resized = cv2.resize(processed_frame.original_frame, (edges.shape[1], edges.shape[0]))
        blended = cv2.addWeighted(raw_resized, 0.65, edge_bgr, 0.85, 0)
        return blended

class GrayscaleProcessor(BaseVisionProcessor):
    """Simple baseline converting frame to 3-channel grayscale."""

    def process(self, processed_frame: ProcessedFrame) -> np.ndarray:
        img = processed_frame.preprocessed_frame
        if img.dtype in (np.float32, np.float64):
            img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = img

        if img_uint8.ndim == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def create_processor(config: Optional[ProcessingConfig] = None) -> BaseVisionProcessor:
    """Factory function instantiating the configured vision processor."""
    if config is None:
        config = load_vision_config().processing

    algo = config.baseline_processor.lower()
    if algo == "canny_edge":
        return CannyEdgeProcessor(
            low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold
        )
    elif algo in ("grayscale", "gray"):
        return GrayscaleProcessor()
    else:
        return CannyEdgeProcessor()
