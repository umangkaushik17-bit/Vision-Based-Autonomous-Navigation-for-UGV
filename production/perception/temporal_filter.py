"""Temporal Stability Filter for Realtime Traversability Perception.

Provides asymmetric temporal filtering:
- Instant hazard latching (alpha = 1.0 when drivability drops or obstacles appear)
- Smooth recovery on noise clearing (alpha = 0.7 when drivability rises)
- Resets on timestamp discontinuities or frame drop (> 0.5s)
- Tracks filter execution overhead (< 0.5 ms)
"""

from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np


class TemporalStabilityFilter:
    """Asymmetric temporal filter for image-space drivability probability maps."""

    def __init__(
        self,
        alpha_hazard: float = 1.0,  # Zero lag for new obstacles
        alpha_recovery: float = 0.70,  # Smoothing for noisy clearing
        max_dt_s: float = 0.50,  # Max allowable frame gap before state reset
        enabled: bool = True,
    ) -> None:
        self.alpha_hazard = alpha_hazard
        self.alpha_recovery = alpha_recovery
        self.max_dt_s = max_dt_s
        self.enabled = enabled

        self._prev_mask: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._last_filter_ms: float = 0.0

    @property
    def last_filter_ms(self) -> float:
        return self._last_filter_ms

    def reset(self) -> None:
        """Reset internal filter state."""
        self._prev_mask = None
        self._prev_timestamp = None
        self._last_filter_ms = 0.0

    def apply(
        self,
        mask: np.ndarray,
        timestamp: float,
        valid: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Apply asymmetric temporal filter to drivability mask.

        Args:
            mask: Continuous drivability array (H, W) in [0.0, 1.0].
            timestamp: Frame timestamp in seconds.
            valid: Whether frame is valid and uncorrupted.

        Returns:
            Tuple of (filtered_mask, filter_duration_ms).
        """
        if not self.enabled or not valid:
            self.reset()
            return mask.copy(), 0.0

        t0 = time.perf_counter()

        # Check for timestamp discontinuity or first frame
        if (
            self._prev_mask is None
            or self._prev_mask.shape != mask.shape
            or self._prev_timestamp is None
            or (timestamp - self._prev_timestamp) > self.max_dt_s
            or (timestamp < self._prev_timestamp)
        ):
            self._prev_mask = mask.copy()
            self._prev_timestamp = timestamp
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._last_filter_ms = dt_ms
            return mask.copy(), dt_ms

        # Asymmetric alpha selection:
        # Where current mask is lower than previous (becoming more hazardous/unsafe):
        # use alpha_hazard (1.0 = instant obstacle detection).
        # Where current mask is higher (clearing):
        # use alpha_recovery (smoothing).
        hazard_mask = mask < self._prev_mask
        alpha = np.where(hazard_mask, self.alpha_hazard, self.alpha_recovery).astype(np.float32)

        # M_t = alpha * M_curr + (1 - alpha) * M_prev
        filtered = alpha * mask + (1.0 - alpha) * self._prev_mask
        filtered = np.clip(filtered, 0.0, 1.0)

        self._prev_mask = filtered.copy()
        self._prev_timestamp = timestamp

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._last_filter_ms = dt_ms
        return filtered, dt_ms
