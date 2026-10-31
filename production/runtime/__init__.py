"""Production Runtime Package.

Exposes HardwareProfiler, BackendRegistry, AutoBackendSelector, and RuntimeAdaptationManager.
"""

from __future__ import annotations

from production.runtime.adaptation_manager import (
    AdaptationConfig,
    AdaptationDecision,
    OperationalState,
    RuntimeAdaptationManager,
    RuntimeMetrics,
)
from production.runtime.auto_selector import (
    AutoBackendSelector,
    SelectionResult,
)
from production.runtime.backend_registry import (
    BackendCategory,
    BackendDescriptor,
    BackendRegistry,
)
from production.runtime.hardware_profiler import (
    AcceleratorInfo,
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    HardwareProfiler,
    RuntimeEnvironmentInfo,
    Status,
)

__all__ = [
    "Status",
    "CPUInfo",
    "GPUInfo",
    "AcceleratorInfo",
    "RuntimeEnvironmentInfo",
    "HardwareProfile",
    "HardwareProfiler",
    "BackendCategory",
    "BackendDescriptor",
    "BackendRegistry",
    "AutoBackendSelector",
    "SelectionResult",
    "OperationalState",
    "AdaptationConfig",
    "RuntimeMetrics",
    "AdaptationDecision",
    "RuntimeAdaptationManager",
]
