"""Configurable preprocessing pipeline for outdoor vision data.

Features:
- Strict preservation of the raw input frame.
- Contrast Limited Adaptive Histogram Equalization (CLAHE) for sunlight/shadows.
- Denoising (Gaussian / Bilateral filter).
- Aspect-preserving or exact resizing.
- Color space conversion (RGB, HSV, LAB, GRAY).
- Value range normalization ([0.0, 1.0]) and dtype conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from ai.core.config import PreprocessingConfig, VisionConfig, load_vision_config
from ai.core.frame_source import VisionFrame
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.preprocessing")

@dataclass
class ProcessedFrame:
    """Encapsulates both the immutable original frame and the transformed preprocessed frame."""
    original_frame: np.ndarray                # Untouched raw frame (H, W, 3) uint8 BGR
    preprocessed_frame: np.ndarray            # Transformed frame (H', W', C)
    frame_index: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def original_resolution(self) -> Tuple[int, int]:
        """(width, height) of original frame."""
        return self.original_frame.shape[1], self.original_frame.shape[0]

    @property
    def preprocessed_resolution(self) -> Tuple[int, int]:
        """(width, height) of preprocessed frame."""
        return self.preprocessed_frame.shape[1], self.preprocessed_frame.shape[0]

class Preprocessor:
    """High-performance frame preprocessor driven by configuration."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        if config is None:
            config = load_vision_config().preprocessing
        self.config = config
        self._clahe: Optional[cv2.CLAHE] = None

        if self.config.clahe.enabled:
            grid = tuple(self.config.clahe.tile_grid_size)
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe.clip_limit,
                tileGridSize=grid
            )

    def process(self, frame: VisionFrame) -> ProcessedFrame:
        """Execute the full preprocessing pipeline on a VisionFrame.
        
        Guarantees that `frame.raw_frame` is never modified in-place.
        """
        # 1. Preserve original frame reference/copy
        raw = frame.raw_frame
        # Work on a copy for transformations
        img = raw.copy()
        meta: Dict[str, Any] = {
            "source_type": frame.source_type,
            "source_info": frame.source_info,
            "original_shape": raw.shape
        }

        # 2. Optional Crop
        if self.config.crop.enabled:
            h, w = img.shape[:2]
            top = max(0, self.config.crop.top_pixels)
            bottom = min(h, h - self.config.crop.bottom_pixels)
            left = max(0, self.config.crop.left_pixels)
            right = min(w, w - self.config.crop.right_pixels)
            if bottom > top and right > left:
                img = img[top:bottom, left:right]
                meta["cropped_shape"] = img.shape

        # 3. CLAHE Adaptive Contrast Enhancement (Handles harsh outdoor shadows)
        if self.config.clahe.enabled and self._clahe is not None:
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert to LAB, apply CLAHE on L (luminance) channel only, and convert back
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = self._clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            elif img.ndim == 2:
                img = self._clahe.apply(img)

        # 4. Optional Denoising
        if self.config.denoising.enabled:
            method = self.config.denoising.method.lower()
            k = self.config.denoising.kernel_size
            if k % 2 == 0:
                k += 1 # Kernel size must be odd
            sigma = self.config.denoising.sigma

            if method == "gaussian":
                img = cv2.GaussianBlur(img, (k, k), sigma)
            elif method == "bilateral":
                # Bilateral preserves sharp obstacle edges
                img = cv2.bilateralFilter(img, d=k, sigmaColor=75, sigmaSpace=sigma * 25)
            elif method == "median":
                img = cv2.medianBlur(img, k)

        # 5. Color Conversion
        target_fmt = self.config.color_conversion.target_format.upper()
        if target_fmt == "RGB" and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif target_fmt == "GRAY":
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif target_fmt == "HSV" and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif target_fmt == "LAB" and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 6. Spatial Resize
        if self.config.resize.enabled:
            tgt_w = self.config.resize.target_width
            tgt_h = self.config.resize.target_height
            if (img.shape[1], img.shape[0]) != (tgt_w, tgt_h):
                interp_map = {
                    "linear": cv2.INTER_LINEAR,
                    "cubic": cv2.INTER_CUBIC,
                    "nearest": cv2.INTER_NEAREST,
                    "area": cv2.INTER_AREA
                }
                interp = interp_map.get(self.config.resize.interpolation.lower(), cv2.INTER_LINEAR)
                img = cv2.resize(img, (tgt_w, tgt_h), interpolation=interp)

        # 7. Normalization & Dtype Conversion
        if self.config.normalization.enabled:
            rmin = self.config.normalization.range_min
            rmax = self.config.normalization.range_max
            img = img.astype(np.float32) / 255.0
            if (rmin, rmax) != (0.0, 1.0):
                img = img * (rmax - rmin) + rmin

            if self.config.normalization.dtype.lower() == "float32":
                img = img.astype(np.float32)
            else:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        meta["final_shape"] = img.shape
        meta["final_dtype"] = str(img.dtype)

        return ProcessedFrame(
            original_frame=raw,
            preprocessed_frame=img,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            metadata=meta
        )
