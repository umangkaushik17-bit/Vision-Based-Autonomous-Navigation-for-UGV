"""Hardware-independent DWA acceleration backends package."""

from ai.planning.backends.backend_interface import (
    BackendHealth,
    DWABackend,
    DWABackendType,
)
from ai.planning.backends.cpu_backend import CPUDWABackend
from ai.planning.backends.cpp_backend import (
    CPPDWABackend,
    CPPSimdDWABackend,
    CPPScalarDWABackend,
)
from ai.planning.backends.cuda_backend import CUDADWABackend
from ai.planning.backends.intel_gpu_backend import IntelGPUDWABackend
from ai.planning.backends.npu_backend import NPUDWABackend
from ai.planning.backends.vectorized_cpu_backend import VectorizedCPUDWABackend
from ai.planning.backends.unified_planner import UnifiedDWAPlanner

__all__ = [
    "DWABackendType",
    "BackendHealth",
    "DWABackend",
    "CPUDWABackend",
    "CPPDWABackend",
    "CPPSimdDWABackend",
    "CPPScalarDWABackend",
    "CUDADWABackend",
    "IntelGPUDWABackend",
    "NPUDWABackend",
    "VectorizedCPUDWABackend",
    "UnifiedDWAPlanner",
]
