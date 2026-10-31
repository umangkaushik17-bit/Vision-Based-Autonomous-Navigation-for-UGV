"""Hardware-Agnostic Production Runtime Architecture Package.

Provides a clean logical separation between production and research/reference code:
  production.perception -> LightweightTraversabilityNet (hardware-adaptive)
  production.planning   -> Native C++ DWA (hardware-adaptive: SIMD/Scalar/CUDA/Intel GPU)
  production.safety     -> Authoritative Safety Governor with compute adaptation watchdog
  production.runtime    -> HardwareProfiler, BackendRegistry, AutoBackendSelector, AdaptationManager
"""

from __future__ import annotations

from production.perception import AdaptiveTraversabilityPredictor
from production.planning import AdaptiveDWAPlanner
from production.runtime import (
    AutoBackendSelector,
    BackendCategory,
    BackendDescriptor,
    BackendRegistry,
    HardwareProfile,
    HardwareProfiler,
    OperationalState,
    RuntimeAdaptationManager,
    RuntimeMetrics,
)
from production.safety import (
    AuthoritativeSafetyGovernor,
    LatchedHazard,
    SafetyGovernor,
    SafetyGovernorConfig,
)

__all__ = [
    "AdaptiveTraversabilityPredictor",
    "AdaptiveDWAPlanner",
    "HardwareProfiler",
    "HardwareProfile",
    "BackendRegistry",
    "BackendDescriptor",
    "BackendCategory",
    "AutoBackendSelector",
    "RuntimeAdaptationManager",
    "RuntimeMetrics",
    "OperationalState",
    "AuthoritativeSafetyGovernor",
    "SafetyGovernor",
    "SafetyGovernorConfig",
    "LatchedHazard",
]
