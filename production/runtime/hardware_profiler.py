"""Hardware Profiler for Autonomous UGV Navigation.

Inspects CPU instruction sets (AVX-512, AVX2, AVX, SSE, NEON), GPU compute runtimes
(NVIDIA CUDA, Intel XPU, DirectML), and specialized accelerators (NPU).
Operates safely across x86_64, aarch64, Windows, and Linux without crashing on missing metrics.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class CPUInfo:
    architecture: str
    vendor: str
    brand: str
    physical_cores: int
    logical_threads: int
    has_avx512: bool = False
    has_avx2: bool = False
    has_avx: bool = False
    has_sse4_2: bool = False
    has_sse2: bool = False
    has_neon: bool = False

    @property
    def highest_simd(self) -> str:
        if self.has_avx512:
            return "AVX-512"
        if self.has_avx2:
            return "AVX2"
        if self.has_avx:
            return "AVX"
        if self.has_sse4_2:
            return "SSE4.2"
        if self.has_sse2:
            return "SSE2"
        if self.has_neon:
            return "NEON"
        return "SCALAR"


@dataclass
class GPUInfo:
    vendor: str
    device_name: str
    total_memory_mb: Optional[float] = None
    driver: Optional[str] = None
    runtime: str = "none"  # "cuda", "xpu", "directml", "none"
    compute_capability: Optional[str] = None
    status: Status = Status.UNAVAILABLE


@dataclass
class AcceleratorInfo:
    accelerator_type: str
    status: Status
    notes: str


@dataclass
class RuntimeEnvironmentInfo:
    os_name: str
    python_version: str
    torch_version: Optional[str] = None
    cuda_available: bool = False
    xpu_available: bool = False
    directml_available: bool = False
    has_cpp_simd_binary: bool = False
    has_cpp_scalar_binary: bool = False


@dataclass
class HardwareProfile:
    cpu: CPUInfo
    gpus: List[GPUInfo] = field(default_factory=list)
    accelerators: List[AcceleratorInfo] = field(default_factory=list)
    runtime: RuntimeEnvironmentInfo = field(default_factory=lambda: RuntimeEnvironmentInfo(
        os_name=platform.system(), python_version=platform.python_version()
    ))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        gpu_summary = ", ".join(
            f"{g.device_name} ({g.runtime.upper()}: {g.status.value})" for g in self.gpus
        ) if self.gpus else "None detected"
        acc_summary = ", ".join(
            f"{a.accelerator_type} ({a.status.value})" for a in self.accelerators
        ) if self.accelerators else "None"
        return (
            f"CPU: {self.cpu.brand or self.cpu.architecture} [{self.cpu.logical_threads} threads, "
            f"SIMD: {self.cpu.highest_simd}] | GPUs: [{gpu_summary}] | Accelerators: [{acc_summary}]"
        )


class HardwareProfiler:
    """Discovers and profiles host compute hardware without side effects or fatal exceptions."""

    _cached_profile: Optional[HardwareProfile] = None

    @classmethod
    def profile(cls, force_refresh: bool = False) -> HardwareProfile:
        if cls._cached_profile is not None and not force_refresh:
            return cls._cached_profile

        cpu_info = cls._detect_cpu()
        gpus = cls._detect_gpus()
        accelerators = cls._detect_accelerators()
        runtime_info = cls._detect_runtime()

        profile = HardwareProfile(
            cpu=cpu_info,
            gpus=gpus,
            accelerators=accelerators,
            runtime=runtime_info,
        )
        cls._cached_profile = profile
        return profile

    @classmethod
    def _detect_cpu(cls) -> CPUInfo:
        arch = platform.machine().lower()
        system = platform.system()
        logical_threads = os.cpu_count() or 1
        physical_cores = logical_threads // 2 if logical_threads > 1 else 1

        try:
            import psutil
            p_cores = psutil.cpu_count(logical=False)
            if p_cores:
                physical_cores = p_cores
        except Exception:
            pass

        vendor = platform.processor() or "Unknown"
        brand = vendor

        has_avx512 = False
        has_avx2 = False
        has_avx = False
        has_sse4_2 = False
        has_sse2 = False
        has_neon = False

        if arch in ("x86_64", "amd64", "x86"):
            has_sse2 = True
            if system == "Windows":
                try:
                    # Windows kernel32 IsProcessorFeaturePresent
                    # 40 = PF_AVX2_INSTRUCTIONS_AVAILABLE
                    # 30 = PF_AVX_INSTRUCTIONS_AVAILABLE
                    # 13 = PF_SSE3_INSTRUCTIONS_AVAILABLE
                    k32 = ctypes.windll.kernel32
                    has_avx2 = bool(k32.IsProcessorFeaturePresent(40))
                    has_avx = bool(k32.IsProcessorFeaturePresent(30)) or has_avx2
                    has_sse4_2 = bool(k32.IsProcessorFeaturePresent(13))
                    # 50 = PF_AVX512F_INSTRUCTIONS_AVAILABLE (Windows 10/11 newer builds)
                    try:
                        has_avx512 = bool(k32.IsProcessorFeaturePresent(50))
                    except Exception:
                        has_avx512 = False
                except Exception:
                    pass
            elif system == "Linux":
                try:
                    with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        has_avx512 = "avx512f" in content
                        has_avx2 = "avx2" in content
                        has_avx = "avx" in content
                        has_sse4_2 = "sse4_2" in content
                        has_sse2 = "sse2" in content
                except Exception:
                    pass
        elif "arm" in arch or "aarch64" in arch:
            has_neon = True
            if system == "Linux":
                try:
                    with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        has_neon = "neon" in content or "asimd" in content
                except Exception:
                    pass

        return CPUInfo(
            architecture=arch,
            vendor=vendor,
            brand=brand,
            physical_cores=physical_cores,
            logical_threads=logical_threads,
            has_avx512=has_avx512,
            has_avx2=has_avx2,
            has_avx=has_avx,
            has_sse4_2=has_sse4_2,
            has_sse2=has_sse2,
            has_neon=has_neon,
        )

    @classmethod
    def _detect_gpus(cls) -> List[GPUInfo]:
        gpus: List[GPUInfo] = []

        # 1. PyTorch CUDA inspection
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    dev_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    mem_mb = props.total_memory / (1024 * 1024)
                    cap = f"{props.major}.{props.minor}"
                    gpus.append(
                        GPUInfo(
                            vendor="NVIDIA",
                            device_name=dev_name,
                            total_memory_mb=round(mem_mb, 1),
                            runtime="cuda",
                            compute_capability=cap,
                            status=Status.AVAILABLE,
                        )
                    )
        except Exception:
            pass

        # 2. PyTorch XPU inspection (Intel GPU / Arc / iGPU)
        try:
            import torch
            if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
                for i in range(torch.xpu.device_count()):
                    name = torch.xpu.get_device_name(i)
                    gpus.append(
                        GPUInfo(
                            vendor="Intel",
                            device_name=name,
                            runtime="xpu",
                            status=Status.AVAILABLE,
                        )
                    )
        except Exception:
            pass

        # 3. DirectML inspection
        try:
            import torch_directml
            if torch_directml.is_available():
                dev_count = torch_directml.device_count()
                for i in range(dev_count):
                    name = torch_directml.device_name(i)
                    vendor = "Intel" if "intel" in name.lower() else ("NVIDIA" if "nvidia" in name.lower() else "AMD")
                    gpus.append(
                        GPUInfo(
                            vendor=vendor,
                            device_name=name,
                            runtime="directml",
                            status=Status.AVAILABLE,
                        )
                    )
        except Exception:
            pass

        # 4. Fallback WMI / lspci check for physical GPU presence when runtimes not loaded
        if not gpus:
            gpus.extend(cls._detect_system_gpus_fallback())

        return gpus

    @classmethod
    def _detect_system_gpus_fallback(cls) -> List[GPUInfo]:
        """Detect physical display adapters when CUDA/XPU PyTorch packages are not installed."""
        fallback: List[GPUInfo] = []
        if platform.system() == "Windows":
            try:
                cmd = "powershell -Command \"Get-CimInstance Win32_VideoController | Select-Object -Property Caption, AdapterRAM\""
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    for line in res.stdout.strip().splitlines():
                        line_clean = line.strip()
                        if not line_clean or "Caption" in line_clean or "---" in line_clean:
                            continue
                        vendor = "Intel" if "intel" in line_clean.lower() else (
                            "NVIDIA" if "nvidia" in line_clean.lower() else "AMD"
                        )
                        fallback.append(
                            GPUInfo(
                                vendor=vendor,
                                device_name=line_clean,
                                runtime="none",
                                status=Status.UNAVAILABLE,  # Present in hardware, but no runtime installed
                            )
                        )
            except Exception:
                pass
        elif platform.system() == "Linux":
            try:
                res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if "VGA" in line or "3D" in line:
                            vendor = "NVIDIA" if "nvidia" in line.lower() else (
                                "Intel" if "intel" in line.lower() else "AMD"
                            )
                            fallback.append(
                                GPUInfo(
                                    vendor=vendor,
                                    device_name=line.strip(),
                                    runtime="none",
                                    status=Status.UNAVAILABLE,
                                )
                            )
            except Exception:
                pass
        return fallback

    @classmethod
    def _detect_accelerators(cls) -> List[AcceleratorInfo]:
        accelerators: List[AcceleratorInfo] = []
        # NPU detection / architectural classification:
        # NPU is strictly UNSUPPORTED for branch-heavy DWA planning
        # It may be evaluated in future phases for neural perception only.
        npu_detected = False
        if platform.system() == "Windows":
            try:
                cmd = "powershell -Command \"Get-PnpDevice -FriendlyName '*NPU*' -ErrorAction SilentlyContinue\""
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
                if res.returncode == 0 and "NPU" in res.stdout:
                    npu_detected = True
            except Exception:
                pass

        accelerators.append(
            AcceleratorInfo(
                accelerator_type="NPU",
                status=Status.UNSUPPORTED,
                notes=(
                    "NPU detected on host" if npu_detected else "NPU runtime unconfigured"
                ) + " — Strictly UNSUPPORTED for DWA planning (branch-heavy polygon SAT collision geometry is incompatible with fixed tensor NPU architectures).",
            )
        )
        return accelerators

    @classmethod
    def _detect_runtime(cls) -> RuntimeEnvironmentInfo:
        torch_ver = None
        cuda_avail = False
        xpu_avail = False
        dml_avail = False

        try:
            import torch
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            xpu_avail = hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)()
        except Exception:
            pass

        try:
            import torch_directml
            dml_avail = torch_directml.is_available()
        except Exception:
            pass

        # Check compiled native C++ binaries
        project_root = Path(__file__).resolve().parents[2]
        native_dir = project_root / "ai" / "planning" / "native_cpp"
        has_simd = (
            (native_dir / "dwa_cpp.dll").exists() or
            (native_dir / "libdwa_cpp.so").exists() or
            (native_dir / "libdwa_cpp.dylib").exists()
        )
        has_scalar = (
            (native_dir / "dwa_cpp_scalar.dll").exists() or
            (native_dir / "libdwa_cpp_scalar.so").exists() or
            (native_dir / "libdwa_cpp_scalar.dylib").exists()
        )

        return RuntimeEnvironmentInfo(
            os_name=platform.system(),
            python_version=platform.python_version(),
            torch_version=torch_ver,
            cuda_available=cuda_avail,
            xpu_available=xpu_avail,
            directml_available=dml_avail,
            has_cpp_simd_binary=has_simd,
            has_cpp_scalar_binary=has_scalar,
        )
