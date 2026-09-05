"""Temporal stabilization filter for consecutive video frames to prevent prediction flickering."""

from __future__ import annotations

from typing import Optional
import numpy as np

class TemporalStabilizer:
    """Exponential Moving Average (EMA) smoother for consecutive frame probability distributions."""

    def __init__(self, ema_alpha: float = 0.65):
        self.ema_alpha = ema_alpha
        self.running_probs: Optional[np.ndarray] = None

    def reset(self):
        """Reset internal history at sequence boundaries."""
        self.running_probs = None

    def update(self, current_probs: np.ndarray) -> np.ndarray:
        """Smooth class probabilities across time and return stabilized class predictions.
        
        Args:
            current_probs: (C, H, W) or (H, W, C) float array of class probabilities
            
        Returns:
            stabilized_preds: (H, W) stabilized uint8 class indices
        """
        # Ensure format is (H, W, C)
        if current_probs.shape[0] in (3, 4) and current_probs.ndim == 3:
            current_probs = np.transpose(current_probs, (1, 2, 0))

        if self.running_probs is None or self.running_probs.shape != current_probs.shape:
            self.running_probs = current_probs.copy()
        else:
            self.running_probs = self.ema_alpha * self.running_probs + (1.0 - self.ema_alpha) * current_probs

        stabilized_preds = np.argmax(self.running_probs, axis=-1).astype(np.uint8)
        return stabilized_preds
