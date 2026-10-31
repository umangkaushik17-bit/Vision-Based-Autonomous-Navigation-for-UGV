"""Unified Camera Interface Abstraction for UGV Navigation.

Supports:
- ROS2 camera topic subscription (/camera/image_raw, /camera/camera_info)
- Dataset sequence replay (RELLIS-3D / RUGD / synthetic sequences)
- Local OpenCV webcam / USB camera
- Synthetic frame generator for deterministic unit testing
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ai.traversability.base import PerceptionFrameInput

logger = logging.getLogger("production.perception.camera_interface")


@dataclass
class CameraConfig:
    """Configuration parameters for camera input sources."""
    source_type: str = "SYNTHETIC"  # "ROS2", "SEQUENCE", "OPENCV", "SYNTHETIC"
    topic_name: str = "/camera/image_raw"
    info_topic_name: str = "/camera/camera_info"
    device_id: int = 0
    sequence_path: Optional[str] = None
    width: int = 640
    height: int = 480
    fps: float = 30.0
    frame_id: str = "camera_optical_frame"
    timeout_s: float = 1.0


class BaseCameraSource(ABC):
    """Abstract base class for all camera input sources."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._frame_count: int = 0
        self._start_time: float = time.time()

    @abstractmethod
    def read(self) -> Optional[PerceptionFrameInput]:
        """Fetch the next available camera frame."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera source is valid and streaming."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying camera resources."""
        pass

    @property
    def frame_id(self) -> str:
        return self.config.frame_id

    @property
    def fps(self) -> float:
        return self.config.fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.config.width, self.config.height)

    def get_camera_info(self) -> Optional[Dict[str, Any]]:
        """Return camera intrinsic parameters if available."""
        # Default pinhole camera model based on standard 80 deg FOV (1.396 rad)
        fx = (self.config.width / 2.0) / np.tan(1.3962634 / 2.0)
        fy = fx
        cx = self.config.width / 2.0
        cy = self.config.height / 2.0
        return {
            "width": self.config.width,
            "height": self.config.height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frame_id": self.config.frame_id,
        }


class SyntheticCameraSource(BaseCameraSource):
    """Deterministic synthetic camera generator for headless validation."""

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        base_pattern: str = "TRAIL",
    ) -> None:
        super().__init__(config or CameraConfig(source_type="SYNTHETIC"))
        self.base_pattern = base_pattern.upper()
        self._opened = True
        self._current_time: float = time.time()

    def is_opened(self) -> bool:
        return self._opened

    def close(self) -> None:
        self._opened = False

    def read(self) -> Optional[PerceptionFrameInput]:
        if not self._opened:
            return None

        w, h = self.config.width, self.config.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        if self.base_pattern == "TRAIL":
            # Sky top half (blue)
            frame[: h // 2, :] = [180, 130, 70]
            # Ground bottom half (green grass)
            frame[h // 2 :, :] = [34, 139, 34]
            # Center dirt trail (brownish trapezoid)
            pts = np.array(
                [
                    [w // 2 - 30, h // 2],
                    [w // 2 + 30, h // 2],
                    [w // 2 + 160, h],
                    [w // 2 - 160, h],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [pts], (100, 140, 180))  # BGR dirt in RGB
        elif self.base_pattern == "OBSTACLE":
            frame[: h // 2, :] = [180, 130, 70]
            frame[h // 2 :, :] = [34, 139, 34]
            pts = np.array(
                [
                    [w // 2 - 30, h // 2],
                    [w // 2 + 30, h // 2],
                    [w // 2 + 160, h],
                    [w // 2 - 160, h],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [pts], (100, 140, 180))
            # Place rock/boulder in center of trail
            cv2.circle(frame, (w // 2, int(h * 0.75)), 35, (60, 60, 60), -1)
        elif self.base_pattern == "BLACK":
            # Completely black frame for blackout testing
            pass

        self._frame_count += 1
        dt = 1.0 / max(1.0, self.config.fps)
        self._current_time += dt

        return PerceptionFrameInput(
            image=frame,
            timestamp=self._current_time,
            frame_id=self.config.frame_id,
            camera_info=self.get_camera_info(),
        )


class SequenceCameraSource(BaseCameraSource):
    """Replays image sequences from dataset directories."""

    def __init__(
        self,
        sequence_dir: Union[str, Path],
        config: Optional[CameraConfig] = None,
        loop: bool = True,
    ) -> None:
        cfg = config or CameraConfig(source_type="SEQUENCE")
        super().__init__(cfg)
        self.sequence_dir = Path(sequence_dir)
        self.loop = loop

        # Discover RGB image files
        rgb_dir = self.sequence_dir / "rgb" if (self.sequence_dir / "rgb").exists() else self.sequence_dir
        self.image_files: List[Path] = sorted(
            [p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in sequence directory: {self.sequence_dir}")

        self._index: int = 0
        self._opened: bool = True
        self._sim_time: float = 0.0

    def is_opened(self) -> bool:
        return self._opened and len(self.image_files) > 0

    def close(self) -> None:
        self._opened = False

    def read(self) -> Optional[PerceptionFrameInput]:
        if not self._opened or not self.image_files:
            return None

        if self._index >= len(self.image_files):
            if self.loop:
                self._index = 0
            else:
                return None

        img_path = self.image_files[self._index]
        self._index += 1
        self._frame_count += 1

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning(f"Could not read image: {img_path}")
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (self.config.width, self.config.height):
            rgb = cv2.resize(rgb, (self.config.width, self.config.height), interpolation=cv2.INTER_LINEAR)

        dt = 1.0 / max(1.0, self.config.fps)
        self._sim_time += dt

        return PerceptionFrameInput(
            image=rgb,
            timestamp=self._sim_time,
            frame_id=self.config.frame_id,
            camera_info=self.get_camera_info(),
        )


class OpenCVCameraSource(BaseCameraSource):
    """Local USB webcam or video file reader via OpenCV."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        cfg = config or CameraConfig(source_type="OPENCV")
        super().__init__(cfg)
        self._cap = cv2.VideoCapture(self.config.device_id)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[PerceptionFrameInput]:
        if not self.is_opened():
            return None

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._frame_count += 1

        return PerceptionFrameInput(
            image=rgb,
            timestamp=time.time(),
            frame_id=self.config.frame_id,
            camera_info=self.get_camera_info(),
        )


class ROS2CameraSource(BaseCameraSource):
    """ROS2 Image and CameraInfo subscriber adapter."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        cfg = config or CameraConfig(source_type="ROS2")
        super().__init__(cfg)
        self._latest_input: Optional[PerceptionFrameInput] = None
        self._latest_camera_info: Optional[Dict[str, Any]] = None
        self._opened: bool = True

    def is_opened(self) -> bool:
        return self._opened

    def close(self) -> None:
        self._opened = False

    def update_from_ros_image(
        self,
        image_np: np.ndarray,
        timestamp_sec: float,
        frame_id: Optional[str] = None,
    ) -> None:
        """Callback interface invoked by ROS2 image subscriber."""
        self._latest_input = PerceptionFrameInput(
            image=image_np,
            timestamp=timestamp_sec,
            frame_id=frame_id or self.config.frame_id,
            camera_info=self._latest_camera_info or self.get_camera_info(),
        )

    def update_from_ros_info(self, camera_info_dict: Dict[str, Any]) -> None:
        """Callback interface invoked by ROS2 CameraInfo subscriber."""
        self._latest_camera_info = camera_info_dict

    def read(self) -> Optional[PerceptionFrameInput]:
        """Fetch latest arrived ROS frame or None if stale/missing."""
        if not self._opened or self._latest_input is None:
            return None

        now = time.time()
        # If timestamp is wall-clock and older than timeout_s, flag stale
        if abs(now - self._latest_input.timestamp) > self.config.timeout_s and self._latest_input.timestamp > 1e8:
            return None

        return self._latest_input
