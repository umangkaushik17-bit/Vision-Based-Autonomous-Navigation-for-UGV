"""Universal FrameSource abstraction unifying images, videos, and live camera streams."""

from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, Optional, Tuple
import cv2
import numpy as np

from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.frame_source")

@dataclass
class VisionFrame:
    """Standardized multi-modal frame container passing through the vision pipeline."""
    raw_frame: np.ndarray                        # Original unmodified frame (H, W, C)
    frame_index: int                             # Sequential frame count (0-indexed)
    timestamp: float                             # Monotonic capture timestamp in seconds
    source_type: str                             # 'image', 'video', or 'camera'
    source_info: Dict[str, Any] = field(default_factory=dict) # Width, height, source FPS, etc.

    @property
    def width(self) -> int:
        return self.raw_frame.shape[1]

    @property
    def height(self) -> int:
        return self.raw_frame.shape[0]

    @property
    def channels(self) -> int:
        return self.raw_frame.shape[2] if self.raw_frame.ndim == 3 else 1

class FrameSource(abc.ABC):
    """Abstract Base Class for all frame-producing sources."""

    def __init__(self):
        self._is_opened: bool = False
        self._frame_count_read: int = 0

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """Native frame width."""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """Native frame height."""
        pass

    @property
    @abc.abstractmethod
    def fps(self) -> Optional[float]:
        """Nominal or reported source FPS."""
        pass

    @property
    def total_frames(self) -> Optional[int]:
        """Total number of frames available, or None if live stream."""
        return None

    @property
    def duration_sec(self) -> Optional[float]:
        """Total stream duration in seconds, or None if unknown."""
        return None

    @abc.abstractmethod
    def open(self) -> bool:
        """Open the frame source."""
        pass

    @abc.abstractmethod
    def read(self) -> Optional[VisionFrame]:
        """Read and return the next VisionFrame, or None if stream ended."""
        pass

    @abc.abstractmethod
    def release(self):
        """Release any hardware, network, or file system handles."""
        pass

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __iter__(self) -> Iterator[VisionFrame]:
        """Yield frames sequentially until end of stream."""
        if not self._is_opened:
            self.open()
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

class ImageSource(FrameSource):
    """Single static image treated as a 1-frame streaming source."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = str(Path(image_path).resolve())
        self._cached_frame: Optional[np.ndarray] = None
        self._has_yielded: bool = False

    def open(self) -> bool:
        path = Path(self.image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"Corrupted or unreadable image file: {self.image_path}")

        self._cached_frame = img
        self._is_opened = True
        self._has_yielded = False
        logger.info(f"Loaded image: {path.name} ({img.shape[1]}x{img.shape[0]})")
        return True

    @property
    def width(self) -> int:
        return self._cached_frame.shape[1] if self._cached_frame is not None else 0

    @property
    def height(self) -> int:
        return self._cached_frame.shape[0] if self._cached_frame is not None else 0

    @property
    def fps(self) -> Optional[float]:
        return 1.0

    @property
    def total_frames(self) -> int:
        return 1

    @property
    def duration_sec(self) -> float:
        return 1.0

    def read(self) -> Optional[VisionFrame]:
        if not self._is_opened or self._has_yielded:
            return None
        self._has_yielded = True
        self._frame_count_read = 1
        return VisionFrame(
            raw_frame=self._cached_frame.copy(),
            frame_index=0,
            timestamp=time.perf_counter(),
            source_type="image",
            source_info={
                "file_path": self.image_path,
                "width": self.width,
                "height": self.height
            }
        )

    def release(self):
        self._cached_frame = None
        self._is_opened = False

class VideoSource(FrameSource):
    """Streaming frame source for recorded video files (MP4, AVI, MOV, MKV)."""

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = str(Path(video_path).resolve())
        self._cap: Optional[cv2.VideoCapture] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 30.0
        self._total_frames: int = 0

    def open(self) -> bool:
        path = Path(self.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file (unsupported codec or corrupt): {self.video_path}")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps_val) if fps_val > 0.0 else 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._is_opened = True

        logger.info(
            f"Opened video stream: {path.name} | Resolution: {self._width}x{self._height} | "
            f"FPS: {self._fps:.1f} | Total Frames: {self._total_frames}"
        )
        return True

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

    def read(self) -> Optional[VisionFrame]:
        if not self._is_opened or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.info(f"Video stream completed. Total frames streamed: {self._frame_count_read}")
            return None

        current_idx = self._frame_count_read
        self._frame_count_read += 1

        return VisionFrame(
            raw_frame=frame,
            frame_index=current_idx,
            timestamp=time.perf_counter(),
            source_type="video",
            source_info={
                "video_path": self.video_path,
                "width": self._width,
                "height": self._height,
                "source_fps": self._fps
            }
        )

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

class CameraSource(FrameSource):
    """Live webcam or USB camera feed with graceful fallback if unattached."""

    def __init__(self, camera_index: int = 0, target_width: int = 640, target_height: int = 480, target_fps: int = 30):
        super().__init__()
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = float(target_fps)

    def open(self) -> bool:
        logger.info(f"Attempting connection to live camera device index {self.camera_index}...")
        # Use DirectShow on Windows for fast device capture discovery
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)

        if not self._cap.isOpened():
            error_msg = (
                f"\n[CameraSource Error] No physical camera found at device index {self.camera_index}.\n"
                f"-> Solution: Use recorded video testing instead:\n"
                f"   python scripts/process_video.py --input data/test/sample.mp4\n"
            )
            logger.warning(error_msg)
            raise RuntimeError(error_msg)

        # Set requested properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._actual_fps = float(reported_fps) if reported_fps > 0 else float(self.target_fps)
        self._is_opened = True

        logger.info(
            f"Camera {self.camera_index} connected successfully! Resolution: {self._actual_width}x{self._actual_height} "
            f"@ {self._actual_fps:.1f} FPS"
        )
        return True

    @property
    def width(self) -> int:
        return self._actual_width

    @property
    def height(self) -> int:
        return self._actual_height

    @property
    def fps(self) -> float:
        return self._actual_fps

    def read(self) -> Optional[VisionFrame]:
        if not self._is_opened or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera device.")
            return None

        current_idx = self._frame_count_read
        self._frame_count_read += 1

        return VisionFrame(
            raw_frame=frame,
            frame_index=current_idx,
            timestamp=time.perf_counter(),
            source_type="camera",
            source_info={
                "camera_index": self.camera_index,
                "width": self._actual_width,
                "height": self._actual_height,
                "reported_fps": self._actual_fps
            }
        )

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False
        logger.info(f"Camera index {self.camera_index} released.")
