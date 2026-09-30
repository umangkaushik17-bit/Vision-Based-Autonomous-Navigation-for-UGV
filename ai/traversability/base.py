"""Abstract interfaces and unified data contracts for Traversability Prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class PerceptionFrameInput:
    """Unified input contract for perception frames."""
    image: np.ndarray
    timestamp: float = 0.0
    frame_id: str = "camera_optical_frame"
    camera_info: Optional[Dict[str, Any]] = None


@dataclass
class TraversabilityResult:
    """Unified output structure for image-space traversability prediction.

    Attributes:
        mask: Continuous traversability probability map (H, W) in [0.0, 1.0], where 1.0 = SAFE/DRIVABLE.
        logits: Raw unnormalized network logits (H, W), float32.
        confidence: Per-pixel model confidence (H, W) in [0.0, 1.0] if available.
        unknown_mask: Boolean mask (H, W) indicating unobservable/sky/void pixels.
        overlay: Debug visualization (H, W, 3) blending frame with traversability overlay (green=safe, red=blocked).
        image: Original input frame in RGB format (H, W, 3) uint8.
        input_shape: Spatial dimensions (height, width) of input image.
        inference_s: Wall-clock inference execution duration in seconds.
        device: Active compute hardware (e.g. 'cpu', 'cuda', 'cuda:0').
        model_name: Unique identifier string for model architecture & checkpoint.
        timestamp: Perception timestamp in seconds.
        frame_id: Coordinate frame identifier of camera optical sensor.
        preprocessing_ms: Duration of preprocessing in milliseconds.
        postprocessing_ms: Duration of postprocessing in milliseconds.
        total_ms: Complete perception stage latency in milliseconds.
        valid: Whether perception output is healthy and trustworthy.
        status_message: Informational or diagnostic status string.
    """

    mask: np.ndarray
    logits: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    input_shape: Tuple[int, int] = (0, 0)
    inference_s: float = 0.0
    device: str = "cpu"
    model_name: str = "lightweight_traversability"
    confidence: Optional[np.ndarray] = None
    unknown_mask: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: str = "camera_optical_frame"
    preprocessing_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_ms: float = 0.0
    valid: bool = True
    status_message: str = "OK"


    def get_binary_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Threshold continuous traversability score into a boolean binary navigable mask."""
        if not self.valid:
            return np.zeros(self.mask.shape, dtype=bool)
        return (self.mask >= threshold).astype(bool)

    def get_safe_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Return boolean mask of verified safe traversable regions (UNKNOWN != SAFE)."""
        if not self.valid:
            return np.zeros(self.mask.shape, dtype=bool)
        safe = self.mask >= threshold
        if self.unknown_mask is not None:
            safe = safe & (~self.unknown_mask)
        return safe

    def get_obstacle_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Return boolean mask of confirmed obstacles/hazards."""
        if not self.valid:
            return np.ones(self.mask.shape, dtype=bool)
        obs = self.mask < threshold
        if self.unknown_mask is not None:
            obs = obs & (~self.unknown_mask)
        return obs

    @property
    def is_valid(self) -> bool:
        """Health check flag for downstream navigation integration."""
        return self.valid and not np.isnan(self.mask).any() and not np.isinf(self.mask).any()

    @property
    def mean_drivability(self) -> float:
        """Average drivability score across all valid non-unknown pixels."""
        if not self.valid:
            return 0.0
        if self.unknown_mask is not None:
            valid = ~self.unknown_mask
            if valid.any():
                return float(np.mean(self.mask[valid]))
        return float(np.mean(self.mask))


class TraversabilityPredictorBase(ABC):
    """Abstract base class establishing the standard project interface for traversability predictors."""

    @abstractmethod
    def predict(self, image: Union[np.ndarray, str, Path]) -> TraversabilityResult:
        """Run traversability inference on an input RGB frame or image filepath.

        Args:
            image: RGB numpy array (H, W, 3) or filepath string/Path.

        Returns:
            TraversabilityResult adhering to project standard data contract.
        """
        pass

    @abstractmethod
    def warmup(self) -> float:
        """Execute a synthetic forward pass to compile kernels and allocate buffers.

        Returns:
            Warmup duration in seconds.
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return active compute device identifier string."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return human-readable model identifier."""
        pass
