"""Hardware-Adaptive Production Traversability Predictor.

Uses LightweightTraversabilityNet as the authoritative production model.
Automatically qualifies and selects the fastest validated inference backend (CPU, SIMD, CUDA, Intel GPU)
via the AutoBackendSelector.
Strictly quarantines SAM-TP to research/reference usage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from ai.traversability.base import TraversabilityPredictorBase, TraversabilityResult
from ai.traversability.dense_predictor import DenseTraversabilityPredictor
from production.runtime.auto_selector import AutoBackendSelector, SelectionResult

logger = logging.getLogger("production.perception.adaptive_predictor")


class AdaptiveTraversabilityPredictor(TraversabilityPredictorBase):
    """Production Hardware-Adaptive Dense Traversability Predictor."""

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        backend: str = "AUTO",
        target_wh: Tuple[int, int] = (640, 400),
        selector: Optional[AutoBackendSelector] = None,
    ) -> None:
        self.requested_backend = backend.upper().strip()
        self.selector = selector or AutoBackendSelector()
        self.target_wh = target_wh
        self.checkpoint = checkpoint

        # Execute Auto selection or explicit qualification
        self.selection_result: SelectionResult = self.selector.select_perception_backend(
            requested=self.requested_backend
        )
        self.active_backend_name = self.selection_result.selected_name
        self.benchmark_latency_ms = self.selection_result.benchmark_latency_ms

        # Map to device
        device_map = {
            "PERCEPTION_CPU": "cpu",
            "PERCEPTION_SIMD": "cpu",
            "PERCEPTION_CUDA": "cuda:0",
            "PERCEPTION_INTEL_GPU": "xpu:0",
        }
        target_device = device_map.get(self.active_backend_name, "cpu")

        logger.info(
            f"Initializing AdaptiveTraversabilityPredictor with validated backend '{self.active_backend_name}' "
            f"on device '{target_device}' (benchmark: {self.benchmark_latency_ms:.2f} ms)"
        )
        self._predictor = DenseTraversabilityPredictor(
            checkpoint=self.checkpoint,
            device=target_device,
            target_wh=self.target_wh,
        )

    def predict(self, image: Union[np.ndarray, str, Path]) -> TraversabilityResult:
        """Run traversability inference on the calibrated hardware backend."""
        return self._predictor.predict(image)

    def warmup(self) -> float:
        """Warm up the active backend."""
        return self._predictor.warmup()

    @property
    def device(self) -> str:
        return self._predictor.device

    @property
    def model_name(self) -> str:
        return f"{self._predictor.model_name} [{self.active_backend_name}]"

    def get_selection_info(self) -> Dict[str, Any]:
        return {
            "active_backend": self.active_backend_name,
            "device": self.device,
            "benchmark_latency_ms": self.benchmark_latency_ms,
            "selection_reason": self.selection_result.selection_reason,
        }
