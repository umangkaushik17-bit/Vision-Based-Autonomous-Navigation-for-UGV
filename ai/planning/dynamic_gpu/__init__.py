"""Hardware-accelerated planning package for UGV dynamic obstacle navigation.

Exports:
    GPUBackend: Device manager for CUDA and CPU tensor operations.
    GPURollout: Batched unicycle trajectory generator.
    GPUCollisionChecker: Vectorized static and dynamic collision checking.
    GPUScorer: Batched multi-term trajectory scoring.
    AcceleratedDWAPlanner: High-performance drop-in replacement for DWAPlanner.
"""

from ai.planning.dynamic_gpu.gpu_backend import GPUBackend
from ai.planning.dynamic_gpu.gpu_collision import GPUCollisionChecker
from ai.planning.dynamic_gpu.gpu_dwa import AcceleratedDWAPlanner
from ai.planning.dynamic_gpu.gpu_rollout import GPURollout
from ai.planning.dynamic_gpu.gpu_scoring import GPUScorer

__all__ = [
    "GPUBackend",
    "GPURollout",
    "GPUCollisionChecker",
    "GPUScorer",
    "AcceleratedDWAPlanner",
]
