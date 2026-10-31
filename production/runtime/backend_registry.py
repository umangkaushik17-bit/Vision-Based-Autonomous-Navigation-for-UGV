"""Unified Backend Registry for DWA Planning and Traversability Perception.

Maintains hardware requirements, runtime dependencies, validation status, and benchmark
metrics for all candidate execution backends.
Enforces that research/reference components (SAM-TP, Python DWA) are strictly quarantined
from production AUTO selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from production.runtime.hardware_profiler import HardwareProfile, Status


class BackendCategory(str, Enum):
    DWA_PLANNER = "DWA_PLANNER"
    TRAVERSABILITY_PERCEPTION = "TRAVERSABILITY_PERCEPTION"


@dataclass
class BackendDescriptor:
    name: str
    category: BackendCategory
    hardware_requirements: List[str]
    runtime_requirements: List[str]
    available: bool = False
    supported: bool = True
    validated: bool = False
    safety_validated: bool = False
    benchmark_latency_ms: Optional[float] = None
    benchmark_throughput_fps: Optional[float] = None
    memory_requirement_mb: Optional[float] = None
    reason_if_rejected: Optional[str] = None
    factory: Optional[Callable[[], Any]] = None
    is_reference_only: bool = False  # If True, never eligible for AUTO production selection

    @property
    def auto_eligible(self) -> bool:
        """A backend is AUTO-eligible only if available, supported, validated, safety_validated, and not reference-only."""
        return (
            self.available
            and self.supported
            and self.validated
            and self.safety_validated
            and not self.is_reference_only
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "hardware_requirements": self.hardware_requirements,
            "runtime_requirements": self.runtime_requirements,
            "available": self.available,
            "supported": self.supported,
            "validated": self.validated,
            "safety_validated": self.safety_validated,
            "benchmark_latency_ms": self.benchmark_latency_ms,
            "benchmark_throughput_fps": self.benchmark_throughput_fps,
            "memory_requirement_mb": self.memory_requirement_mb,
            "reason_if_rejected": self.reason_if_rejected,
            "auto_eligible": self.auto_eligible,
            "is_reference_only": self.is_reference_only,
        }


class BackendRegistry:
    """Registry managing candidates for both Perception and DWA execution."""

    def __init__(self) -> None:
        self._backends: Dict[str, BackendDescriptor] = {}

    def register(self, desc: BackendDescriptor) -> None:
        self._backends[desc.name] = desc

    def get(self, name: str) -> Optional[BackendDescriptor]:
        return self._backends.get(name)

    def get_by_category(self, category: BackendCategory) -> List[BackendDescriptor]:
        return [b for b in self._backends.values() if b.category == category]

    def get_auto_candidates(self, category: BackendCategory) -> List[BackendDescriptor]:
        """Return backends eligible for AUTO calibration and selection."""
        return [
            b for b in self.get_by_category(category)
            if b.available and b.supported and not b.is_reference_only
        ]

    def all_descriptors(self) -> List[BackendDescriptor]:
        return list(self._backends.values())

    @classmethod
    def create_default(cls, profile: HardwareProfile) -> BackendRegistry:
        """Create and populate registry based on detected hardware profile."""
        registry = cls()
        cls._populate_dwa_backends(registry, profile)
        cls._populate_perception_backends(registry, profile)
        return registry

    @staticmethod
    def _populate_dwa_backends(registry: BackendRegistry, profile: HardwareProfile) -> None:
        # 1. C++ Scalar Backend: Ultimate portable CPU fallback (no SIMD/AVX2 required)
        has_scalar_bin = profile.runtime.has_cpp_scalar_binary
        registry.register(
            BackendDescriptor(
                name="CPP_SCALAR",
                category=BackendCategory.DWA_PLANNER,
                hardware_requirements=["Generic x86 / ARM CPU"],
                runtime_requirements=["Compiled dwa_cpp_scalar shared library"],
                available=has_scalar_bin,
                supported=True,
                validated=has_scalar_bin,
                safety_validated=has_scalar_bin,
                reason_if_rejected=None if has_scalar_bin else "dwa_cpp_scalar binary missing",
                memory_requirement_mb=10.0,
            )
        )

        # 2. C++ SIMD Backend: High-performance vectorised CPU (AVX2/AVX-512/NEON)
        has_simd_hw = profile.cpu.has_avx2 or profile.cpu.has_avx512 or profile.cpu.has_neon
        has_simd_bin = profile.runtime.has_cpp_simd_binary
        simd_avail = has_simd_hw and has_simd_bin
        simd_reject = None
        if not has_simd_hw:
            simd_reject = "CPU lacks AVX2/AVX-512/NEON vector instructions"
        elif not has_simd_bin:
            simd_reject = "dwa_cpp binary missing"

        registry.register(
            BackendDescriptor(
                name="CPP_SIMD",
                category=BackendCategory.DWA_PLANNER,
                hardware_requirements=["x86 AVX2/AVX-512 or ARM NEON CPU"],
                runtime_requirements=["Compiled dwa_cpp shared library (OpenMP + AVX2)"],
                available=simd_avail,
                supported=True,
                validated=simd_avail,
                safety_validated=simd_avail,
                reason_if_rejected=simd_reject,
                memory_requirement_mb=12.0,
            )
        )

        # 3. NVIDIA CUDA Backend
        has_cuda = profile.runtime.cuda_available
        registry.register(
            BackendDescriptor(
                name="CUDA",
                category=BackendCategory.DWA_PLANNER,
                hardware_requirements=["NVIDIA GPU with Compute Capability >= 6.0"],
                runtime_requirements=["CUDA runtime", "PyTorch with CUDA support"],
                available=has_cuda,
                supported=True,
                validated=has_cuda,
                safety_validated=has_cuda,
                reason_if_rejected=None if has_cuda else "NVIDIA GPU or PyTorch CUDA runtime unavailable",
                memory_requirement_mb=256.0,
            )
        )

        # 4. Intel GPU Backend (Intel XPU / DirectML)
        has_intel_gpu = profile.runtime.xpu_available or profile.runtime.directml_available
        registry.register(
            BackendDescriptor(
                name="INTEL_GPU",
                category=BackendCategory.DWA_PLANNER,
                hardware_requirements=["Intel integrated GPU (Core Ultra / Arc / Iris Xe)"],
                runtime_requirements=["PyTorch XPU or torch-directml runtime"],
                available=has_intel_gpu,
                supported=True,
                validated=has_intel_gpu,
                safety_validated=has_intel_gpu,
                reason_if_rejected=None if has_intel_gpu else "Intel GPU runtime (XPU or DirectML) unavailable",
                memory_requirement_mb=256.0,
            )
        )

        # 5. NPU Backend: Strictly UNSUPPORTED for DWA
        registry.register(
            BackendDescriptor(
                name="NPU",
                category=BackendCategory.DWA_PLANNER,
                hardware_requirements=["Neural Processing Unit"],
                runtime_requirements=["NPU Runtime"],
                available=False,
                supported=False,
                validated=False,
                safety_validated=False,
                reason_if_rejected="NPU is UNSUPPORTED for DWA planning (branch-heavy polygon SAT geometry incompatible with fixed neural engines)",
            )
        )

        # 6. Python DWA: Golden Reference / Regression only (NEVER auto-eligible in production)
        registry.register(
            BackendDescriptor(
                name="PYTHON_REFERENCE",
                category=BackendCategory.DWA_PLANNER,
                hardware_requirements=["Host CPU"],
                runtime_requirements=["Python runtime", "NumPy"],
                available=True,
                supported=True,
                validated=True,
                safety_validated=True,
                is_reference_only=True,
                reason_if_rejected="Reference only: Quarantined from production navigation execution",
            )
        )

    @staticmethod
    def _populate_perception_backends(registry: BackendRegistry, profile: HardwareProfile) -> None:
        # Production model: LightweightTraversabilityNet
        # 1. Perception CPU
        torch_ok = profile.runtime.torch_version is not None
        registry.register(
            BackendDescriptor(
                name="PERCEPTION_CPU",
                category=BackendCategory.TRAVERSABILITY_PERCEPTION,
                hardware_requirements=["Generic x86 / ARM CPU"],
                runtime_requirements=["PyTorch CPU runtime"],
                available=torch_ok,
                supported=True,
                validated=torch_ok,
                safety_validated=torch_ok,
                reason_if_rejected=None if torch_ok else "PyTorch not installed",
                memory_requirement_mb=150.0,
            )
        )

        # 2. Perception SIMD (MKL / OpenMP accelerated CPU inference)
        simd_avail = torch_ok and (profile.cpu.has_avx2 or profile.cpu.has_neon)
        registry.register(
            BackendDescriptor(
                name="PERCEPTION_SIMD",
                category=BackendCategory.TRAVERSABILITY_PERCEPTION,
                hardware_requirements=["AVX2 / AVX-512 / NEON CPU"],
                runtime_requirements=["PyTorch with oneDNN / MKL optimization"],
                available=simd_avail,
                supported=True,
                validated=simd_avail,
                safety_validated=simd_avail,
                reason_if_rejected=None if simd_avail else "CPU lacks SIMD or PyTorch missing MKL/oneDNN",
                memory_requirement_mb=150.0,
            )
        )

        # 3. Perception CUDA
        has_cuda = profile.runtime.cuda_available
        registry.register(
            BackendDescriptor(
                name="PERCEPTION_CUDA",
                category=BackendCategory.TRAVERSABILITY_PERCEPTION,
                hardware_requirements=["NVIDIA GPU"],
                runtime_requirements=["PyTorch CUDA runtime"],
                available=has_cuda,
                supported=True,
                validated=has_cuda,
                safety_validated=has_cuda,
                reason_if_rejected=None if has_cuda else "CUDA runtime unavailable",
                memory_requirement_mb=500.0,
            )
        )

        # 4. Perception TensorRT (Marked EXPERIMENTAL/UNSUPPORTED unless verified)
        registry.register(
            BackendDescriptor(
                name="PERCEPTION_TENSORRT",
                category=BackendCategory.TRAVERSABILITY_PERCEPTION,
                hardware_requirements=["NVIDIA GPU with Tensor Cores"],
                runtime_requirements=["TensorRT runtime", "torch_tensorrt"],
                available=False,
                supported=False,
                validated=False,
                safety_validated=False,
                reason_if_rejected="TensorRT engine conversion not currently compiled or validated for LightweightTraversabilityNet",
            )
        )

        # 5. Perception Intel GPU (XPU / DirectML)
        has_intel_gpu = profile.runtime.xpu_available or profile.runtime.directml_available
        registry.register(
            BackendDescriptor(
                name="PERCEPTION_INTEL_GPU",
                category=BackendCategory.TRAVERSABILITY_PERCEPTION,
                hardware_requirements=["Intel Core Ultra / Arc / Iris Xe GPU"],
                runtime_requirements=["PyTorch XPU or DirectML"],
                available=has_intel_gpu,
                supported=True,
                validated=has_intel_gpu,
                safety_validated=has_intel_gpu,
                reason_if_rejected=None if has_intel_gpu else "Intel GPU runtime unavailable",
                memory_requirement_mb=500.0,
            )
        )

        # 6. SAM-TP: Research / Reference Only (NEVER auto-eligible in production)
        registry.register(
            BackendDescriptor(
                name="PERCEPTION_SAM_TP",
                category=BackendCategory.TRAVERSABILITY_PERCEPTION,
                hardware_requirements=["High-end GPU / Heavy CPU"],
                runtime_requirements=["Segment Anything, GroundingDINO, heavy dependencies"],
                available=False,  # Excluded from production deployment
                supported=True,
                validated=False,
                safety_validated=False,
                is_reference_only=True,
                reason_if_rejected="Reference only: Quarantined from production runtime. Deployment package excludes SAM-TP.",
            )
        )
