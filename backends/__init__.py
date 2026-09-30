"""Hardware Backends Package.

Modular hardware execution backends categorized by compute architecture:
  backends.cpu
  backends.simd
  backends.cuda
  backends.intel_gpu
  backends.other_supported
"""

from __future__ import annotations

from backends.cpu import CPPScalarDWABackend, CPUDWABackend
from backends.cuda import CUDADWABackend
from backends.intel_gpu import IntelGPUDWABackend
from backends.simd import CPPSimdDWABackend

__all__ = [
    "CPPScalarDWABackend",
    "CPUDWABackend",
    "CPPSimdDWABackend",
    "CUDADWABackend",
    "IntelGPUDWABackend",
]
