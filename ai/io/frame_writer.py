"""Frame and video export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.io.frame_writer")

def save_image(image: np.ndarray, output_path: str) -> str:
    """Save an image to disk, automatically ensuring parent directories exist.
    
    Args:
        image: np.ndarray image to save (uint8 BGR).
        output_path: Target destination filepath.
        
    Returns:
        The resolved output file path.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # If image is float32 normalized in [0, 1], scale to uint8 [0, 255]
    if image.dtype in (np.float32, np.float64):
        image_to_save = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        image_to_save = image

    success = cv2.imwrite(str(path), image_to_save)
    if not success:
        raise IOError(f"Failed to write image to destination: {path}")

    logger.info(f"Saved processed image to: {path}")
    return str(path)

class VideoWriter:
    """Encoder streaming processed frames to an MP4 video file."""

    def __init__(self, output_path: str, fps: float = 30.0, width: int = 640, height: int = 480):
        self.output_path = str(Path(output_path).resolve())
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written: int = 0

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use mp4v or avc1 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        if not self._writer.isOpened():
            # Fallback to AVI / XVID if MP4 fails
            logger.warning("mp4v codec failed; attempting fallback XVID...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._writer = cv2.VideoWriter(
                str(path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

        if not self._writer.isOpened():
            raise IOError(f"Could not open VideoWriter at: {self.output_path}")

        logger.info(f"Opened VideoWriter: {path.name} ({self.width}x{self.height} @ {self.fps:.1f} FPS)")

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to the video stream."""
        if self._writer is None:
            self.open()

        # Convert float to uint8 if needed
        if frame.dtype in (np.float32, np.float64):
            frame_to_write = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame_to_write = frame

        # Ensure 3-channel BGR
        if frame_to_write.ndim == 2:
            frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_GRAY2BGR)

        # Ensure correct resolution
        if (frame_to_write.shape[1], frame_to_write.shape[0]) != (self.width, self.height):
            frame_to_write = cv2.resize(frame_to_write, (self.width, self.height))

        self._writer.write(frame_to_write)
        self._frames_written += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(f"Closed VideoWriter: {self.output_path} (Wrote {self._frames_written} frames)")
