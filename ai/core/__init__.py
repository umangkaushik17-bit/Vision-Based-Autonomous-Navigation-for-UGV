"""Core vision abstractions, preprocessors, processors, and timing utilities."""

from ai.core.frame_source import (
    VisionFrame,
    FrameSource,
    ImageSource,
    VideoSource,
    CameraSource
)
from ai.core.config import VisionConfig, load_vision_config
from ai.core.preprocessing import ProcessedFrame, Preprocessor
from ai.core.processor import (
    BaseVisionProcessor,
    CannyEdgeProcessor,
    GrayscaleProcessor,
    create_processor
)
from ai.core.timing import FrameTimer, FPSMeter, FrameTimingSnapshot, PerformanceReport

__all__ = [
    "VisionFrame",
    "FrameSource",
    "ImageSource",
    "VideoSource",
    "CameraSource",
    "VisionConfig",
    "load_vision_config",
    "ProcessedFrame",
    "Preprocessor",
    "BaseVisionProcessor",
    "CannyEdgeProcessor",
    "GrayscaleProcessor",
    "create_processor",
    "FrameTimer",
    "FPSMeter",
    "FrameTimingSnapshot",
    "PerformanceReport"
]
