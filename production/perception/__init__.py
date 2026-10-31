"""Production Perception Package.

Exposes AdaptiveTraversabilityPredictor (using LightweightTraversabilityNet) and unified contracts.
"""

from __future__ import annotations

from production.perception.adaptive_predictor import AdaptiveTraversabilityPredictor
from production.perception.base import (
    PerceptionFrameInput,
    TraversabilityPredictorBase,
    TraversabilityResult,
)
from production.perception.camera_interface import (
    BaseCameraSource,
    CameraConfig,
    OpenCVCameraSource,
    ROS2CameraSource,
    SequenceCameraSource,
    SyntheticCameraSource,
)
from production.perception.realtime_pipeline import RealtimeTraversabilityPipeline
from production.perception.temporal_filter import TemporalStabilityFilter
from production.perception.vision_costmap_integrator import (
    LocalBEVCostmap,
    VisionCostmapConfig,
    VisionCostmapIntegrator,
)

__all__ = [
    "AdaptiveTraversabilityPredictor",
    "RealtimeTraversabilityPipeline",
    "TemporalStabilityFilter",
    "CameraConfig",
    "BaseCameraSource",
    "SyntheticCameraSource",
    "SequenceCameraSource",
    "OpenCVCameraSource",
    "ROS2CameraSource",
    "PerceptionFrameInput",
    "TraversabilityPredictorBase",
    "TraversabilityResult",
    "LocalBEVCostmap",
    "VisionCostmapConfig",
    "VisionCostmapIntegrator",
]

