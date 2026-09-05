"""Streaming video reader and metadata extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.io.video_loader")

class VideoReader:
    """High-level generator-based streaming video reader."""

    def __init__(self, video_path: str):
        self.video_path = str(Path(video_path).resolve())
        self._cap: Optional[cv2.VideoCapture] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._opened: bool = False

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        path = Path(self.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found at path: {self.video_path}")

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Unable to decode video (corrupted or unsupported codec): {self.video_path}")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(raw_fps) if raw_fps > 0 else 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._opened = True

        logger.debug(
            f"Initialized VideoReader: {path.name} | {self._width}x{self._height} @ {self._fps:.1f} FPS"
        )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_sec(self) -> float:
        return (self._total_frames / self._fps) if (self._fps > 0 and self._total_frames > 0) else 0.0

    def stream_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator streaming (frame_index, frame_array) sequentially."""
        if not self._opened or self._cap is None:
            self.open()

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
