"""Image preprocessing, stereo rectification, and adaptive contrast enhancement."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from ai.localization.base import CameraIntrinsics


class CameraPreprocessor:
    """Preprocesses raw camera images for visual feature tracking and stereo matching."""

    def __init__(
        self,
        camera: CameraIntrinsics,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.camera = camera
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR or RGB image to 8-bit single-channel grayscale."""
        if image.ndim == 2:
            return image
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def enhance_contrast(self, image_gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast in outdoor shadows and harsh sun glare."""
        if not self.enable_clahe:
            return image_gray
        return self.clahe.apply(image_gray)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run standard preprocessing pipeline: grayscale + optional CLAHE."""
        gray = self.to_grayscale(image)
        return self.enhance_contrast(gray)

    def rectify_stereo_pair(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Verify and process synchronized stereo image pair."""
        proc_left = self.process(img_left)
        proc_right = self.process(img_right)
        return proc_left, proc_right
