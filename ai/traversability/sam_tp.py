"""Production adapter integrating SAM-TP (SAM 2.1 Hiera-Tiny) into the SIH26126 pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

from ai.traversability.base import TraversabilityPredictorBase, TraversabilityResult
from ai.traversability.postprocess import refine_traversability_by_contrast
from ai.traversability.visualization import create_traversability_overlay
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.traversability.sam_tp")


class SAMTPTraversabilityPredictor(TraversabilityPredictorBase):
    """Production wrapper for SAM-TP (SAM 2.1 Hiera-Tiny learned traversability model).

    This class provides a stable, zero-leakage interface isolating the project from third-party
    package internals, supporting both CPU and GPU execution with safety contrast refinement.
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        config: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        contrast_refine: bool = True,
        overlay_alpha: float = 0.45
    ) -> None:
        """Initialize the SAM-TP predictor.

        Args:
            checkpoint: Optional explicit path to `checkpoint_finetuned_v2.pt`.
                        If None, auto-resolved from cache or Hugging Face Hub.
            config: Optional path to `sam2.1_custom2.yaml`.
            device: Compute device ('cpu', 'cuda', etc.). Defaults to best available.
            contrast_refine: Whether to apply per-frame luminance contrast refinement.
            overlay_alpha: Blending transparency for debug visualization overlays.
        """
        try:
            from rover_traversability import TraversabilityPredictor
        except ImportError as exc:
            raise ImportError(
                "rover_traversability package is required. Install via:\n"
                "  pip install -e references/sana-earth-rover-policy/traversability"
            ) from exc

        logger.info(f"Initializing SAM-TP Predictor (device requested: {device})...")
        self._predictor = TraversabilityPredictor(
            checkpoint=checkpoint,
            config=config,
            device=device,
            contrast_refine=contrast_refine,
            overlay_alpha=overlay_alpha
        )
        self._device = self._predictor.device
        self._contrast_refine = contrast_refine
        self._overlay_alpha = overlay_alpha
        self._model_name = "SAM-TP_Mini+_v2_HieraTiny"
        logger.info(f"SAM-TP ready on compute device: {self._device}")

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, image: Union[np.ndarray, str, Path]) -> TraversabilityResult:
        """Execute single-frame traversability prediction.

        Args:
            image: Either an RGB numpy array (H, W, 3) uint8 or filepath.

        Returns:
            TraversabilityResult adhering to standard project specification.
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image).resolve()
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            # Load with OpenCV and convert BGR -> RGB
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Failed to read image at: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            rgb = image
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Expected RGB image of shape (H, W, 3), got {rgb.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Run underlying SAM-TP predictor
        raw_res = self._predictor.predict(rgb)

        mask = np.asarray(raw_res.mask, dtype=np.float32)
        logits = np.asarray(raw_res.logits, dtype=np.float32)

        # Confidence metric derived from distance to decision boundary |p - 0.5| * 2
        confidence = np.abs(mask - 0.5) * 2.0

        # Construct result object
        return TraversabilityResult(
            mask=mask,
            logits=logits,
            confidence=confidence,
            unknown_mask=None,
            overlay=raw_res.overlay,
            image=rgb,
            input_shape=(rgb.shape[0], rgb.shape[1]),
            inference_s=float(raw_res.inference_s),
            device=self._device,
            model_name=self._model_name
        )

    def warmup(self) -> float:
        """Run dummy inference pass to prime PyTorch computation graphs."""
        return float(self._predictor.warmup())
