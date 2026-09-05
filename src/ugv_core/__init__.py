"""Core library for SIH26126 Autonomous UGV project.

Contains data models, device abstraction, configuration loader, and logging.
"""

from ugv_core.types import (
    TraversabilityClass,
    RiskLevel,
    MissionState,
    Pose2D,
    Pose3D,
    ObstacleBBox,
    VelocityCommand,
    CostmapGrid,
    FramePacket
)
from ugv_core.device import get_device, get_device_info, configure_cpu_threading, DeviceInfo
from ugv_core.logger import get_logger
from ugv_core.config import SystemConfig, load_config

__all__ = [
    "TraversabilityClass",
    "RiskLevel",
    "MissionState",
    "Pose2D",
    "Pose3D",
    "ObstacleBBox",
    "VelocityCommand",
    "CostmapGrid",
    "FramePacket",
    "get_device",
    "get_device_info",
    "configure_cpu_threading",
    "DeviceInfo",
    "get_logger",
    "SystemConfig",
    "load_config"
]
