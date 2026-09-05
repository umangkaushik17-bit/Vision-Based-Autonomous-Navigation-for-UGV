"""Abstract interfaces and unified data contracts for Traversability Prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


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
    """

    mask: np.ndarray
    logits: np.ndarray
    overlay: np.ndarray
    image: np.ndarray
    input_shape: Tuple[int, int]
    inference_s: float
    device: str
    model_name: str
    confidence: Optional[np.ndarray] = None
    unknown_mask: Optional[np.ndarray] = None

    def get_binary_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Threshold continuous traversability score into a boolean binary navigable mask."""
        return (self.mask >= threshold).astype(bool)

    @property
    def mean_drivability(self) -> float:
        """Average drivability score across all valid non-unknown pixels."""
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
