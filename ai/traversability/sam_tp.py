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
        overlay_alpha: float = 0.45,
        use_dense: bool = True
    ) -> None:
        """Initialize the traversability predictor.

        By default, routes to the high-throughput DenseTraversabilityPredictor (MobileNetV3-Small RELLIS-3D).

        Args:
            checkpoint: Optional explicit path to model checkpoint.
            config: Optional path to config file.
            device: Compute device ('cpu', 'cuda', etc.). Defaults to best available.
            contrast_refine: Whether to apply per-frame luminance contrast refinement.
            overlay_alpha: Blending transparency for debug visualization overlays.
            use_dense: If True, uses the fast RELLIS-3D model instead of legacy SAM-TP.
        """
        self._dense_fallback = None
        self._predictor = None

        force_dense = use_dense or (os.environ.get("USE_DENSE_TRAVERSABILITY", "1") == "1")
        if force_dense:
            from ai.traversability.dense_predictor import DenseTraversabilityPredictor
            logger.info("Using production DenseTraversabilityPredictor (LightweightTraversabilityNet RELLIS-3D).")
            self._dense_fallback = DenseTraversabilityPredictor(
                checkpoint=checkpoint if checkpoint and Path(checkpoint).exists() else None,
                device=device,
                overlay_alpha=overlay_alpha
            )
            self._device = self._dense_fallback.device
            self._model_name = self._dense_fallback.model_name
        else:
            try:
                from rover_traversability import TraversabilityPredictor
                logger.info(f"Initializing legacy SAM-TP Predictor (device requested: {device})...")
                self._predictor = TraversabilityPredictor(
                    checkpoint=checkpoint,
                    config=config,
                    device=device,
                    contrast_refine=contrast_refine,
                    overlay_alpha=overlay_alpha
                )
                self._device = self._predictor.device
                self._model_name = "SAM-TP_Mini+_v2_HieraTiny"
                logger.info(f"SAM-TP ready on compute device: {self._device}")
            except Exception as exc:
                logger.warning(
                    f"rover_traversability package not found or failed ({exc}). "
                    "Seamlessly routing to production DenseTraversabilityPredictor (MobileNetV3-Small RELLIS-3D)."
                )
                from ai.traversability.dense_predictor import DenseTraversabilityPredictor
                self._dense_fallback = DenseTraversabilityPredictor(
                    checkpoint=checkpoint if checkpoint and Path(checkpoint).exists() else None,
                    device=device,
                    overlay_alpha=overlay_alpha
                )
                self._device = self._dense_fallback.device
                self._model_name = self._dense_fallback.model_name

        self._contrast_refine = contrast_refine
        self._overlay_alpha = overlay_alpha

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

        # If running dense model, delegate directly
        if self._dense_fallback is not None:
            return self._dense_fallback.predict(rgb)

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
        if self._dense_fallback is not None:
            return float(self._dense_fallback.warmup())
        if self._predictor is not None:
            return float(self._predictor.warmup())
        return 0.0

