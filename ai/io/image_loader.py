"""Dedicated image loading and validation functions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.io.image_loader")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def load_image(image_path: str, color_format: str = "BGR") -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load, validate, and convert color space for an image file.
    
    Args:
        image_path: Path to the image file.
        color_format: Requested output format ('BGR', 'RGB', or 'GRAY').
        
    Returns:
        Tuple of (image_array, (width, height))
        
    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If unsupported format or corrupt image.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to decode image file (corrupted or unreadable): {image_path}")

    h, w = img.shape[:2]

    # Convert color format if requested
    fmt_upper = color_format.upper()
    if fmt_upper == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif fmt_upper == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif fmt_upper == "BGR":
        pass
    else:
        raise ValueError(f"Unknown target color format: {color_format}")

    logger.debug(f"Loaded image: {path.name} ({w}x{h}) in {fmt_upper} format.")
    return img, (w, h)
