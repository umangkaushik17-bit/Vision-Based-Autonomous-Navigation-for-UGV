"""Comprehensive Test Suite for Hardware-Aware AUTO Backend Selection (Scenarios 1-7).

Validates:
  Scenario 1: Generic CPU only (no SIMD) -> Scalar C++ selected
  Scenario 2: CPU + SIMD (AVX2/NEON) -> Fast SIMD C++ selected
  Scenario 3: CPU + Intel GPU -> Empirical benchmark decides
  Scenario 4: CPU + NVIDIA CUDA -> Empirical benchmark decides
  Scenario 5: GPU present but initialization/calibration fails -> AUTO smoothly selects validated fallback
  Scenario 6: Explicit backend (CUDA) requested but unavailable -> Hard RuntimeError, zero silent fallback
  Scenario 7: No accelerator present -> Robust scalar/SIMD CPU selection

Clearly differentiates REAL HOST HARDWARE VALIDATION from CONTROLLED FIXTURE/MOCK TESTING.
"""

from __future__ import annotations

import pytest

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
    CPUInfo,
    HardwareProfile,
    HardwareProfiler,
    RuntimeEnvironmentInfo,
    Status,
)


# ==============================================================================
# REAL HOST HARDWARE TESTS
# ==============================================================================

def test_real_hardware_auto_dwa_selection():
    """REAL HARDWARE: Test that AUTO mode selects a validated C++ backend on current machine."""
    selector = AutoBackendSelector()
    res = selector.select_dwa_backend("AUTO", benchmark_iterations=3)

    assert isinstance(res, SelectionResult)
    assert res.category == BackendCategory.DWA_PLANNER
    assert res.selected_name in ("CPP_SIMD", "CPP_SCALAR")
    assert res.benchmark_latency_ms > 0.0
    assert res.selected_descriptor.safety_validated
    assert not res.selected_descriptor.is_reference_only


def test_real_hardware_auto_perception_selection():
    """REAL HARDWARE: Test that AUTO mode selects validated LightweightTraversabilityNet on current machine."""
    selector = AutoBackendSelector()
    res = selector.select_perception_backend("AUTO", benchmark_iterations=2)

    assert isinstance(res, SelectionResult)
    assert res.category == BackendCategory.TRAVERSABILITY_PERCEPTION
    assert res.selected_name in ("PERCEPTION_SIMD", "PERCEPTION_CPU")
    assert res.benchmark_latency_ms > 0.0
    assert res.selected_descriptor.safety_validated
    assert not res.selected_descriptor.is_reference_only


def test_real_hardware_explicit_unsupported_cuda_fails_clearly():
    """REAL HARDWARE: If CUDA is absent on host, explicit request must raise RuntimeError with zero silent fallback."""
    profile = HardwareProfiler.profile()
    if not profile.runtime.cuda_available:
        selector = AutoBackendSelector()
        with pytest.raises(RuntimeError) as exc_info:
            selector.select_dwa_backend("CUDA")
        assert "System will NOT silently fall back" in str(exc_info.value)


# ==============================================================================
# LOGIC & CAPABILITY FIXTURE TESTS (SCENARIOS 1-7)
# ==============================================================================

def test_scenario_1_cpu_only_scalar_selected():
    """Scenario 1: Generic CPU without SIMD (e.g. legacy x86 or scalar ARM).
    Expected: CPP_SCALAR selected.
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(
            architecture="x86_64", vendor="Generic", brand="Generic",
            physical_cores=4, logical_threads=4,
            has_avx512=False, has_avx2=False, has_avx=False,
            has_sse4_2=False, has_sse2=True, has_neon=False
        ),
        runtime=RuntimeEnvironmentInfo(
            os_name="Linux", python_version="3.10.0",
            cuda_available=False, xpu_available=False, directml_available=False,
            has_cpp_simd_binary=False, has_cpp_scalar_binary=True
        )
    )
    reg = BackendRegistry.create_default(mock_profile)
    selector = AutoBackendSelector(registry=reg, profile=mock_profile)

    # Patch calibration to simulate scalar latency
    def mock_calib(backend_name: str, iterations: int = 5):
        if backend_name == "CPP_SCALAR":
            return True, 6.2, "Verified"
        return False, 0.0, "Unavailable"

    selector._benchmark_dwa_candidate = mock_calib

    res = selector.select_dwa_backend("AUTO")
    assert res.selected_name == "CPP_SCALAR"
    assert res.benchmark_latency_ms == 6.2


def test_scenario_2_cpu_with_simd_selected():
    """Scenario 2: Modern CPU with AVX2 or NEON.
    Expected: Best validated CPU backend (CPP_SIMD) selected.
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(
            architecture="x86_64", vendor="Intel", brand="Core Ultra",
            physical_cores=8, logical_threads=16,
            has_avx2=True, has_avx=True
        ),
        runtime=RuntimeEnvironmentInfo(
            os_name="Windows", python_version="3.10.0",
            cuda_available=False, xpu_available=False, directml_available=False,
            has_cpp_simd_binary=True, has_cpp_scalar_binary=True
        )
    )
    reg = BackendRegistry.create_default(mock_profile)
    selector = AutoBackendSelector(registry=reg, profile=mock_profile)

    def mock_calib(backend_name: str, iterations: int = 5):
        if backend_name == "CPP_SIMD":
            return True, 1.1, "Verified"
        elif backend_name == "CPP_SCALAR":
            return True, 5.8, "Verified"
        return False, 0.0, "Unavailable"

    selector._benchmark_dwa_candidate = mock_calib

    res = selector.select_dwa_backend("AUTO")
    assert res.selected_name == "CPP_SIMD"
    assert res.benchmark_latency_ms == 1.1


def test_scenario_3_cpu_and_intel_gpu_benchmark_decides():
    """Scenario 3: System with CPU SIMD and Intel GPU.
    Demonstrates that empirical benchmark decides (does NOT blindly choose GPU).
    Case A: Intel GPU is slower than SIMD (e.g. Intel Core Ultra validation) -> SIMD wins.
    Case B: Intel GPU is faster -> Intel GPU wins.
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(
            architecture="x86_64", vendor="Intel", brand="Core Ultra 7",
            physical_cores=16, logical_threads=22, has_avx2=True
        ),
        runtime=RuntimeEnvironmentInfo(
            os_name="Windows", python_version="3.10.0",
            xpu_available=True, has_cpp_simd_binary=True, has_cpp_scalar_binary=True
        )
    )

    # Case A: SIMD (1.2 ms) outperforms Intel GPU (8.5 ms)
    reg_a = BackendRegistry.create_default(mock_profile)
    selector_a = AutoBackendSelector(registry=reg_a, profile=mock_profile)
    selector_a._benchmark_dwa_candidate = lambda name, iterations=5, **kwargs: (
        (True, 1.2, "OK") if name == "CPP_SIMD" else
        ((True, 8.5, "OK") if name == "INTEL_GPU" else (True, 5.5, "OK"))
    )
    res_a = selector_a.select_dwa_backend("AUTO")
    assert res_a.selected_name == "CPP_SIMD"

    # Case B: Intel GPU (0.8 ms) outperforms SIMD (1.2 ms)
    reg_b = BackendRegistry.create_default(mock_profile)
    selector_b = AutoBackendSelector(registry=reg_b, profile=mock_profile)
    selector_b._benchmark_dwa_candidate = lambda name, iterations=5, **kwargs: (
        (True, 0.8, "OK") if name == "INTEL_GPU" else
        ((True, 1.2, "OK") if name == "CPP_SIMD" else (True, 5.5, "OK"))
    )
    res_b = selector_b.select_dwa_backend("AUTO")
    assert res_b.selected_name == "INTEL_GPU"


def test_scenario_4_cpu_and_cuda_benchmark_decides():
    """Scenario 4: System with CPU and NVIDIA CUDA.
    Benchmark decides based on validated latency.
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(architecture="x86_64", vendor="AMD", brand="Ryzen", physical_cores=8, logical_threads=16, has_avx2=True),
        runtime=RuntimeEnvironmentInfo(
            os_name="Linux", python_version="3.10.0",
            cuda_available=True, has_cpp_simd_binary=True, has_cpp_scalar_binary=True
        )
    )
    reg = BackendRegistry.create_default(mock_profile)
    selector = AutoBackendSelector(registry=reg, profile=mock_profile)

    selector._benchmark_dwa_candidate = lambda name, iterations=5, **kwargs: (
        (True, 0.6, "Verified CUDA") if name == "CUDA" else
        ((True, 1.5, "Verified SIMD") if name == "CPP_SIMD" else (True, 6.0, "Verified Scalar"))
    )
    res = selector.select_dwa_backend("AUTO")
    assert res.selected_name == "CUDA"
    assert res.benchmark_latency_ms == 0.6


def test_scenario_5_gpu_initialization_fails_auto_selects_fallback():
    """Scenario 5: GPU is reported by hardware, but warmup/calibration fails (e.g. CUDA OOM or runtime mismatch).
    Expected: AUTO mode rejects failed GPU and selects best remaining validated backend (CPP_SIMD).
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(architecture="x86_64", vendor="Intel", brand="i7", physical_cores=8, logical_threads=16, has_avx2=True),
        runtime=RuntimeEnvironmentInfo(
            os_name="Windows", python_version="3.10.0",
            cuda_available=True, has_cpp_simd_binary=True, has_cpp_scalar_binary=True
        )
    )
    reg = BackendRegistry.create_default(mock_profile)
    selector = AutoBackendSelector(registry=reg, profile=mock_profile)

    # CUDA fails during benchmark with OOM or driver crash
    selector._benchmark_dwa_candidate = lambda name, iterations=5, **kwargs: (
        (False, 0.0, "CUDA Driver initialization failed: Out of memory") if name == "CUDA" else
        ((True, 1.3, "Verified SIMD") if name == "CPP_SIMD" else (True, 5.9, "Verified Scalar"))
    )

    res = selector.select_dwa_backend("AUTO")
    assert res.selected_name == "CPP_SIMD"
    assert res.benchmark_latency_ms == 1.3
    assert res.all_diagnostics["CUDA"]["passed"] is False


def test_scenario_6_explicit_requested_backend_fails_without_silent_fallback():
    """Scenario 6: User explicitly requests CUDA, but CUDA is unavailable or broken.
    Expected: Clear, immediate RuntimeError. Zero silent fallback to CPU.
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(architecture="x86_64", vendor="Intel", brand="Core", physical_cores=4, logical_threads=8),
        runtime=RuntimeEnvironmentInfo(
            os_name="Linux", python_version="3.10.0",
            cuda_available=False, has_cpp_simd_binary=True, has_cpp_scalar_binary=True
        )
    )
    reg = BackendRegistry.create_default(mock_profile)
    selector = AutoBackendSelector(registry=reg, profile=mock_profile)

    with pytest.raises(RuntimeError) as exc_info:
        selector.select_dwa_backend("CUDA")
    assert "System will NOT silently fall back" in str(exc_info.value)


def test_scenario_7_no_accelerator_fallback_to_scalar_or_simd():
    """Scenario 7: Embedded/mobile system without any GPU or accelerator.
    Expected: Safe selection of scalar or SIMD C++.
    """
    mock_profile = HardwareProfile(
        cpu=CPUInfo(
            architecture="arm64", vendor="ARM", brand="Cortex-A53",
            physical_cores=4, logical_threads=4, has_neon=True
        ),
        runtime=RuntimeEnvironmentInfo(
            os_name="Linux", python_version="3.10.0",
            cuda_available=False, xpu_available=False, directml_available=False,
            has_cpp_simd_binary=True, has_cpp_scalar_binary=True
        )
    )
    reg = BackendRegistry.create_default(mock_profile)
    selector = AutoBackendSelector(registry=reg, profile=mock_profile)

    selector._benchmark_dwa_candidate = lambda name, iterations=5, **kwargs: (
        (True, 3.2, "NEON SIMD Verified") if name == "CPP_SIMD" else
        ((True, 9.8, "Scalar Verified") if name == "CPP_SCALAR" else (False, 0.0, "N/A"))
    )

    res = selector.select_dwa_backend("AUTO")
    assert res.selected_name == "CPP_SIMD"
    assert res.benchmark_latency_ms == 3.2
