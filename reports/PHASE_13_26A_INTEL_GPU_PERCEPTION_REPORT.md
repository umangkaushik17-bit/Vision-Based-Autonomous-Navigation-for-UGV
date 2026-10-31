# PHASE 13.26A — INTEL GPU/XPU PERCEPTION ACCELERATION REPORT
**Safe, Additive, Backward-Compatible Implementation & Validation**

**Project**: Vision-Based Autonomous Navigation for UGV (SIH2026)  
**Date**: 2026-09-30T17:30:00+05:30  
**Status**: VALIDATED & AUDITED — INTEL GPU PERCEPTION ACCELERATION VERIFIED  
**Branch**: `demo-01-interactive-ugv`  
**Starting Commit**: `3fe36cb`  

---

## 1. Executive Summary & Objective

The objective of Phase 13.26A was to accelerate the existing frozen perception model (`LightweightTraversabilityNet`) on the native Intel integrated GPU (Intel XPU, Meteor Lake Arc Xe cores) available on the host machine (Intel Core Ultra 7 155H), determining whether Intel GPU execution reduces inference latency compared to the CPU baseline while preserving 100% backward compatibility, strict numerical parity, safety semantics, and zero modifications to downstream C++ DWA planning.

### Key Measured Outcomes:
1. **Model Inference Latency**:
   - **Host CPU**: **62.61 ms** mean (P95: 75.03 ms)
   - **Intel GPU (XPU:0)**: **29.19 ms** mean (P95: 33.62 ms)
   - **Inference Speedup**: **2.14x** (reduces pure neural compute by >33 ms per frame)
2. **Total Perception Stage Latency** (includes cv2 preprocessing, inference, hardware sync, softmax, postprocessing, overlay):
   - **Host CPU**: **74.48 ms** mean (P95: 86.84 ms, 13.4 FPS)
   - **Intel GPU (XPU:0)**: **45.25 ms** mean (P95: 50.96 ms, 22.1 FPS)
   - **Total Perception Speedup**: **1.65x**
3. **End-to-End Navigation Loop (Synchronous Perception + C++ DWA)**:
   - **CPU Perception + C++ DWA**: **88.87 ms** (11.3 Hz)
   - **Intel XPU Perception + C++ DWA**: **65.25 ms** (15.3 Hz)
   - **Overall Loop Speedup**: **1.36x**
4. **Parity & Safety Validation**:
   - Evaluated across **10 representative real and synthetic scenarios**.
   - **Classification Agreement**: **100.0%**.
   - **Max Absolute Mask Discrepancy**: **0.0000** (below $10^{-4}$).
   - **False-Safe Divergence**: **0 pixels** (zero unsafe transitions).
   - **NaN / Inf**: Zero across all runs.
5. **Architectural Safety**:
   - Additive, zero-breakage design: CPU path remains the default fallback.
   - Explicit `device="xpu"` on a machine without XPU fails cleanly with an actionable `RuntimeError` (no silent fallback).
   - Downstream `TraversabilityResult` structure and C++ DWA local planning remain 100% invariant.

---

## 2. Hardware Environment

- **Host Processor**: Intel(R) Core(TM) Ultra 7 155H (16 physical cores: 6 Performance, 8 Efficient, 2 Low-Power Efficient; 22 threads)
- **Host GPU**: Intel(R) Graphics (Meteor Lake integrated Arc GPU with 8 Xe-cores)
- **Shared Video Memory**: ~2.0 GB dynamic allocation
- **Display Driver**: Intel Graphics Driver 32.0.101.8831
- **Host NPU**: Intel(R) AI Boost (NPU 3720) — *reserved for future matrix compute, excluded from DWA/perception branching*

---

## 3. Software Environment & XPU Runtime

- **Host Python Environment**: `.venv_xpu` (Dedicated virtual environment with PyTorch XPU runtime)
- **PyTorch Version**: `2.6.0+xpu`
- **XPU Availability**: `torch.xpu.is_available() == True`
- **Device Count**: 1 active device (`xpu:0`)
- **Device Identifier**: `Intel(R) Graphics`
- **SYCL Backend**: Direct SYCL/OpenCL runtime provided by Intel oneAPI deep learning wheels

---

## 4. Exact Model & Weights Configuration

- **Model Class**: `ai.models.traversability_net.LightweightTraversabilityNet`
- **Backbone Architecture**: `mobilenet_v3_small` with multi-scale skip connections and lightweight decoder
- **Parameter Count**: ~3.2 million parameters (lightweight edge budget)
- **Semantic Classes**: 3 classes:
  - Class 0: SAFE (traversable soil, grass, packed dirt)
  - Class 1: UNKNOWN / RISKY (tall vegetation, shadow boundaries)
  - Class 2: UNSAFE (rocks, boulders, water, obstacles)
- **Frozen Weights Checkpoint**: `outputs/rellis3d/checkpoints/best.pt`
- **Predictor Wrapper**: `ai.traversability.dense_predictor.DenseTraversabilityPredictor`
- **Target Resolution**: Resized to `(640, 400)` during preprocessing; output continuous drivability mask and overlay returned at original frame resolution.

---

## 5. Device Selection Mechanism & Fail-Fast Guarantees

In `ai/traversability/dense_predictor.py`, device resolution is implemented beneath `DenseTraversabilityPredictor.__init__()`:

```python
# Safe, additive, backward-compatible device selection
if device is None or str(device).lower().strip() == "auto":
    if torch.cuda.is_available():
        self._device = "cuda:0"
    elif hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        self._device = "xpu:0"
    else:
        self._device = "cpu"
else:
    dev_str = str(device).lower().strip()
    if dev_str in ("xpu", "xpu:0"):
        has_xpu = hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)()
        if not has_xpu:
            raise RuntimeError(
                f"XPU device '{device}' explicitly requested, but PyTorch XPU runtime "
                "(torch.xpu.is_available()) is unavailable on this environment. "
                "System will NOT silently fall back to CPU."
            )
        self._device = "xpu:0" if dev_str == "xpu" else dev_str
    elif dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' explicitly requested, but NVIDIA CUDA is unavailable "
                "(torch.cuda.is_available() returned False). System will NOT silently fall back to CPU."
            )
        self._device = dev_str
    elif dev_str == "cpu":
        self._device = "cpu"
    else:
        self._device = dev_str
```

### Invariant Rules:
1. **Explicit Request (`device="xpu"`)**: Requires valid XPU runtime. If missing, raises `RuntimeError` immediately. Silent fallback is prohibited.
2. **Automatic Request (`device="auto"` or `device=None`)**: Selects CUDA if present, then XPU if verified available, otherwise gracefully operates on CPU.
3. **Explicit CPU Request (`device="cpu"`)**: Unconditionally runs on host CPU.

---

## 6. Hardware Synchronization for Accurate Benchmarking

Because PyTorch XPU execution on Intel GPUs uses asynchronous SYCL command queues, accurate timing requires proper hardware barriers before reading timers:

```python
t0 = time.perf_counter()
with torch.no_grad():
    logits_tensor = self.model(tensor_in)
    probs_tensor = F.softmax(logits_tensor, dim=1).squeeze(0)

    if self._torch_device.type == "cuda":
        torch.cuda.synchronize(self._torch_device)
    elif self._torch_device.type == "xpu":
        torch.xpu.synchronize(self._torch_device)
dt = time.perf_counter() - t0
```
This guarantees that reported latencies reflect complete physical GPU execution.

---

## 7. Numerical Parity Audit (Phase 8)

Evaluated across 10 diverse scenarios comparing CPU vs Intel XPU:

| Scenario Name | Type | Dimensions | Max Abs Logits Diff | Max Abs Mask Diff | Class Agreement | False-Safe Divergence | Status |
|---|---|---|---|---|---|---|---|
| **Dirt Trail 0000** | Real | (480, 640, 3) | 0.00002 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Dirt Trail 0001** | Real | (480, 640, 3) | 0.00002 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Dirt Trail 0002** | Real | (480, 640, 3) | 0.00002 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Dirt Trail 0003** | Real | (480, 640, 3) | 0.00001 | 0.00000 | **100.0%** | 0 | **PASS** |
| **Test Sample** | Real | (480, 640, 3) | 0.00001 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Processed Sample** | Real | (480, 640, 3) | 0.00001 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Open Traversable Trail** | Synthetic | (400, 640, 3) | 0.00003 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Central Obstacle** | Synthetic | (400, 640, 3) | 0.00003 | 0.00001 | **100.0%** | 0 | **PASS** |
| **Non-Traversable Mud** | Synthetic | (400, 640, 3) | 0.00005 | 0.00004 | **100.0%** | 0 | **PASS** |
| **Low-Contrast Twilight** | Synthetic | (400, 640, 3) | 0.00002 | 0.00001 | **100.0%** | 0 | **PASS** |

**Parity Summary**: Maximum absolute discrepancy on traversability probabilities across all pixels in all 10 scenarios is $\le 0.00004$. Output classes agree 100.0% with zero safety inversions.

---

## 8. Safety Validation (Phase 9)

- **Obstacle Detection Consistency**: Every pixel identified as lethal obstacle by CPU is identically classified by Intel XPU.
- **Corridor Safety**: Safe path traversability probability distributions match with negligible floating-point differences ($< 10^{-4}$).
- **Zero Divergence Hazards**: False-safe divergence count is **0** across all evaluated frames.
- **Stability**: Zero NaNs and zero Infs observed during initialization, warmup, and 200+ continuous benchmark cycles.

---

## 9. Granular Latency Benchmark (Phase 10)

Benchmarked on real outdoor test frame (`own_seq_01_dirt_0000.png`, $480 \times 640 \times 3$) over **100 measured iterations** per device following 15 warmup passes:

| Pipeline Stage | CPU Mean (ms) | CPU P50 (ms) | CPU P95 (ms) | XPU Mean (ms) | XPU P50 (ms) | XPU P95 (ms) | Measured Speedup |
|---|---|---|---|---|---|---|---|
| **Preprocessing** (Resize, Norm, Tensor) | 8.80 | 8.54 | 10.98 | 11.42 | 11.16 | 13.40 | 0.77x (H2D copy) |
| **Model Inference** (Forward Pass + Sync) | **62.61** | **60.06** | **75.03** | **29.19** | **28.63** | **33.62** | **2.14x Faster** |
| **Postprocessing** (Drivability, Overlay) | 3.06 | 3.03 | 3.97 | 4.64 | 4.61 | 5.48 | 0.66x (D2H copy) |
| **Total Perception Stage** | **74.48** | **72.01** | **86.84** | **45.25** | **44.57** | **50.96** | **1.65x Faster** |

### Latency Distribution Breakdown:
- **Pure Model Inference Throughput**:
  - CPU: **16.0 FPS**
  - Intel XPU: **34.3 FPS** (**+18.3 FPS increase**)
- **Full Perception Stage Throughput**:
  - CPU: **13.4 FPS**
  - Intel XPU: **22.1 FPS** (**+8.7 FPS increase**)

---

## 10. End-to-End Navigation Cycle Comparison (Phase 13)

Synchronous benchmark coupling perception with the native C++ DWA local planner (`libdwa_cpp.dll`):

| Configuration | Perception Latency (ms) | C++ DWA Latency (ms) | Total Planning Cycle (ms) | Effective Frequency (Hz) |
|---|---|---|---|---|
| **CPU Perception + C++ DWA** | 87.14 | 1.73 | 88.87 | 11.3 Hz |
| **Intel XPU Perception + C++ DWA** | 63.83 | 1.42 | **65.25** | **15.3 Hz** |

- **Cycle Latency Reduction**: Reduces full navigation loop time from **88.87 ms to 65.25 ms**.
- **End-to-End Speedup**: **1.36x**.
- **Planner Invariance**: Native C++ DWA execution latency remained constant at **~1.4 – 1.7 ms**, confirming complete separation between perception acceleration and planning layers.

---

## 11. Regression Testing & Invariants (Phase 15 & 16)

1. **Unit & Integration Suite**:
   - `tests/test_intel_gpu_perception.py`:
     - `test_cpu_predictor_lifecycle`: PASS
     - `test_xpu_predictor_lifecycle`: PASS
     - `test_auto_device_selection`: PASS
     - `test_cpu_xpu_parity`: PASS
   - `tests/test_traversability_model.py`: PASS (Architecture, parameters, gradients verified).
2. **Backward Compatibility**:
   - Standard CPU execution continues to function with zero degradation.
   - On CPU-only environments (e.g., standard WSL Linux without XPU wheel), `DenseTraversabilityPredictor` defaults to `cpu` automatically without throwing errors.

---

## 12. Files Modified & Added

1. **Modified Production Files**:
   - `ai/traversability/dense_predictor.py`: Added additive XPU device selection, fail-fast checking, and XPU hardware synchronization.
   - `simulation/nodes/interactive_ugv_navigation_node.py`: Added explicit device logging (`PERCEPTION DEVICE: ...`) at startup.
2. **Added Benchmark & Test Files**:
   - `scripts/benchmark_perception_intel_gpu.py`: Automated parity, granular benchmarking, and end-to-end evaluation harness.
   - `tests/test_intel_gpu_perception.py`: Unit and regression tests covering CPU, XPU, and parity.
3. **Generated Reports & Artifacts**:
   - `reports/PHASE_13_26A_INTEL_GPU_PERCEPTION_STATE.md`: Initial audit and state specification.
   - `reports/perception_intel_gpu_validation.json`: Structured machine-readable benchmark telemetry.
   - `reports/PHASE_13_26A_INTEL_GPU_PERCEPTION_REPORT.md`: This comprehensive engineering report.

---

## 13. Conclusion & Validation Status

**INTEL XPU PERCEPTION ACCELERATION IS OFFICIALLY VALIDATED.**

- **Execution Verified**: Real SYCL tensor execution confirmed on Meteor Lake Intel Arc integrated GPU (`Intel(R) Graphics`).
- **Speedup Confirmed**: Pure model inference accelerated by **2.14x** (62.61 ms $\rightarrow$ 29.19 ms); total perception accelerated by **1.65x** (74.48 ms $\rightarrow$ 45.25 ms).
- **Parity Confirmed**: 100.0% classification agreement with zero false-safe leakage.
- **Safety Intact**: C++ DWA, tracking, costmaps, and safety governors remain entirely unchanged.
