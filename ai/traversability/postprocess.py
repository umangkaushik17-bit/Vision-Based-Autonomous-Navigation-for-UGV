"""Post-processing and safety refinement filters for traversability predictions."""

from __future__ import annotations

from typing import Optional
import numpy as np


def refine_traversability_by_contrast(
    trav: np.ndarray,
    rgb: np.ndarray,
    drivable_thresh: float = 0.5,
    darkness_ratio: float = 0.65,
    min_reference_pixels: int = 500,
) -> np.ndarray:
    """Safety refinement: mark pixels significantly darker than reference ground as non-traversable.

    Why this is required:
    SAM-TP (and most drivable-ground segmenters) are trained on ground-vs-sky/background.
    A dark object lying flat on sunlit terrain (e.g. rocks, ditches, low debris, or other rovers)
    can be mistakenly segmented as continuous ground. This filter computes per-frame ground
    luminance from high-confidence drivable pixels and downgrades anomalous dark pockets.

    Args:
        trav: Continuous traversability mask (H, W) in [0.0, 1.0].
        rgb: Input RGB frame (H, W, 3) uint8.
        drivable_thresh: Threshold identifying candidate ground reference pixels.
        darkness_ratio: Threshold fraction below ground median luminance triggering obstacle downgrade.
        min_reference_pixels: Minimum number of reference ground pixels required to activate filter.

    Returns:
        Refined traversability mask (H, W) float32 in [0.0, 1.0].
    """
    t = np.asarray(trav, dtype=np.float32)
    if rgb is None or rgb.shape[:2] != t.shape:
        return t

    # Standard ITU-R BT.601 perceived luminance
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    ground_mask = t > float(drivable_thresh)
    if int(ground_mask.sum()) < int(min_reference_pixels):
        return t

    median_ground_lum = float(np.median(lum[ground_mask]))
    # Safeguard against uniform night / dusk conditions
    if median_ground_lum < 30.0:
        return t

    obstacle_mask = ground_mask & (lum < (median_ground_lum * float(darkness_ratio)))
    if not obstacle_mask.any():
        return t

    out = t.copy()
    out[obstacle_mask] = 0.0
    return out


class TemporalTraversabilityFilter:
    """Exponential moving average (EMA) temporal smoother across sequential video frames."""

    def __init__(self, alpha: float = 0.7) -> None:
        """
        Args:
            alpha: Weight of current frame [0.0, 1.0].
                   1.0 = no smoothing (raw instantaneous prediction).
                   0.7 = balanced smoothing with responsive dynamics.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self.alpha = float(alpha)
        self.prev_mask: Optional[np.ndarray] = None

    def update(self, mask: np.ndarray) -> np.ndarray:
        """Filter the incoming traversability mask using temporal EMA."""
        current = np.asarray(mask, dtype=np.float32)
        if self.prev_mask is None or self.prev_mask.shape != current.shape:
            self.prev_mask = current.copy()
            return current

        smoothed = self.alpha * current + (1.0 - self.alpha) * self.prev_mask
        self.prev_mask = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        """Clear temporal history (e.g. upon sequence switch or discontinuous teleport)."""
        self.prev_mask = None
