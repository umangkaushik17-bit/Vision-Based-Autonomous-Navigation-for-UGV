"""Classical non-deep-learning computer vision baseline for outdoor traversability."""

from __future__ import annotations

import cv2
import numpy as np

from ai.datasets.base import TraversabilityTargetClass

class BaselineTerrainClassifier:
    """Classical heuristic classifier combining HSV color spaces, local texture variance, and edge density.
    
    Serves as the quantifiable baseline against which deep learning models are evaluated.
    """

    def __init__(
        self,
        edge_threshold_unsafe: float = 0.18,
        texture_var_threshold: float = 250.0
    ):
        self.edge_threshold_unsafe = edge_threshold_unsafe
        self.texture_var_threshold = texture_var_threshold

    def predict(self, bgr_image: np.ndarray) -> np.ndarray:
        """Segment an outdoor image into SAFE (0), UNKNOWN (1), UNSAFE (2), and VOID (255)."""
        h, w = bgr_image.shape[:2]
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        output_mask = np.full((h, w), TraversabilityTargetClass.UNKNOWN, dtype=np.uint8)

        # 1. Sky / Horizon Detection (Top section with high brightness and low saturation)
        horizon_y = int(h * 0.35)
        sky_mask = (hsv[:horizon_y, :, 2] > 180) & (hsv[:horizon_y, :, 1] < 60)
        output_mask[:horizon_y, :][sky_mask] = TraversabilityTargetClass.VOID

        # 2. Ground Region (Below horizon)
        ground_gray = gray[horizon_y:, :]
        ground_hsv = hsv[horizon_y:, :]

        # 3. Edge Feature Density (Obstacles, boulders, sharp rocks have dense high-frequency edges)
        edges = cv2.Canny(ground_gray, 60, 140)
        # Compute local edge density via box blur
        edge_density = cv2.boxFilter(edges.astype(np.float32) / 255.0, -1, (15, 15))

        # 4. Color heuristics for traversable terrain:
        # - Brown/tan dirt trails: H in [10, 30], S in [30, 180], V in [40, 200]
        # - Green grass: H in [35, 85], S in [40, 220], V in [40, 220]
        # - Gray asphalt/gravel: S < 30, V in [50, 180]
        dirt_mask = (ground_hsv[:, :, 0] >= 10) & (ground_hsv[:, :, 0] <= 30) & (ground_hsv[:, :, 1] >= 30) & (ground_hsv[:, :, 2] >= 40)
        grass_mask = (ground_hsv[:, :, 0] >= 35) & (ground_hsv[:, :, 0] <= 85) & (ground_hsv[:, :, 1] >= 40) & (ground_hsv[:, :, 2] >= 40)
        gravel_mask = (ground_hsv[:, :, 1] < 35) & (ground_hsv[:, :, 2] >= 50) & (ground_hsv[:, :, 2] <= 180)
        traversable_surface = (dirt_mask | grass_mask | gravel_mask)

        # 5. Classify Ground Pixels
        ground_labels = np.full((h - horizon_y, w), TraversabilityTargetClass.UNKNOWN, dtype=np.uint8)

        # Flat smooth surface with low edge density -> SAFE
        safe_mask = traversable_surface & (edge_density < self.edge_threshold_unsafe)
        ground_labels[safe_mask] = TraversabilityTargetClass.SAFE

        # High edge density or deep dark shadows -> UNSAFE (Obstacles, boulders, logs, deep pits)
        unsafe_mask = (edge_density >= self.edge_threshold_unsafe) | (ground_hsv[:, :, 2] < 30)
        ground_labels[unsafe_mask] = TraversabilityTargetClass.UNSAFE

        output_mask[horizon_y:, :] = ground_labels
        return output_mask
