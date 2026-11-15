# SIH26126 — Master Optimization, C++17 SIMD & Hardware Acceleration Reference

**Project**: SIH26126 — Vision-Based Autonomous Navigation for UGV in Outdoor Environments  
**Document**: `docs/OPTIMIZATION_AND_ACCELERATION_MASTER.md`  
**Workstream Ledger**: [`docs/OPTIMIZATION_WORKSTREAMS_MASTER.md`](OPTIMIZATION_WORKSTREAMS_MASTER.md)  
**Detailed P0/P1 Report**: [`reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md`](../reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md)  
**Archived Optimization Reports**: [`docs/optimization/`](optimization/) and [`reports/optimization/`](../reports/optimization/)  
**Status**: **ALL 4 OPTIMIZATION TIERS (`WS-1A` THROUGH `WS-4A`) + `P0/P1` HARDENING COMPLETE & VERIFIED**  

---

## 1. Executive Summary of Computational Speedups

Across the development of SIH26126, every computational bottleneck in global planning, local trajectory rollout, 2.5D terrain geometry, spatial fusion, neural perception, and ROS 2 IPC was systematically accelerated:

| Subsystem | Original Pure-Python Baseline | Optimized Production Implementation | Measured Latency | Speedup Factor |
| :--- | :---: | :--- | :---: | :---: |
| **Local DWA Planner (375 rollouts + 8-axis SAT)** | `5,673.7 ms` (Python CPU) | **Native C++17 AVX2 + OpenMP (`dwa_cpp.dll` / `libdwa_cpp.so`)** | **`1.32 ms`** | **$4,298\times$** |
| **Local DWA Planner (GPU Tensor Backend)** | `5,673.7 ms` (Python CPU) | **PyTorch CUDA Tensor Kernel on NVIDIA Tesla T4** | **`58.31 ms`** | **$97.3\times$** |
| **Global $A^*$ Planner ($120\times 120$ Grid + Smoothing)** | `669.3 ms` (Python CPU) | **Native C++17 AVX2 + JPS (`astar_cpp.dll` / `libastar_cpp.so`)** | **`1.53–2.65 ms`** | **$252\times\text{--}437\times$** |
| **Global $A^*$ Planner ($50\times 50$ Local Grid)** | `168.0 ms` (Python CPU) | **Native C++17 AVX2 (`astar_cpp.dll` / `libastar_cpp.so`)** | **`0.45 ms`** | **$371\times$** |
| **Semantic Traversability Inference** | `355.5 ms` (PyTorch FP32 CPU) | **Mixed-Precision QDQ INT8 ONNX (`traversability_model_int8.onnx`)** | **`12.52 ms`** | **$28.4\times$** |
| **ROS 2 Camera Callback Blocking Time** | `21.5 ms` (Synchronous) | **`AsyncPerceptionWorker` (Depth-1 Slot Background Daemon)** | **`< 0.10 ms`** | **$215\times$** |
| **SE(3) Attitude-Compensated BEV Projection** | `18.4 ms` (`np.add.at` loop) | **Pre-Cached Ray LUT + `np.bincount` + O(1) Min-Pooling** | **`1.61 ms`** | **$11.4\times$** |
| **2.5D Elevation Binning & Outlier Rejection** | `14.2 ms` (Per-cell Python) | **`ElevationMapper` Bounded-Span `np.bincount` + Kalman EMA** | **`0.99 ms`** | **$14.3\times$** |
| **2.5D Normals, Slope, Step, Ditch & Belly Crown** | `24.0 ms` (Scipy filters) | **`VectorizedTerrainGeometryEstimator` (`cv2` SIMD Stencils)** | **`1.86 ms`** | **$12.9\times$** |
| **WorldTerrainMap Spatial Fusion ($200\times 200$)** | `28.2 ms` (Phase 9 baseline) | **1D Flat-Indexed Vectorized SE(3) Fusion + C++ Radial Dilation** | **`< 1.50 ms`** | **$18.8\times$** |
| **Photometric CIE-LAB CLAHE + Glare/Shadow Filter** | `8.58 ms` (Full-res float32) | **`PhotometricResilienceProcessor` Compact Grid + `cv2.LUT`** | **`0.92 ms`** | **$9.3\times$** |
| **ROS 2 Map IPC ($200\times 200$ 10-Layer Grid)** | `35.6 ms` (ROSIDL copy) | **Intra-Process `LoanedMessageHandle` / POD Shared Memory** | **`0.04 ms / 0.75 ms`** | **$47\times\text{--}890\times$** |

---

## 2. Optimization Tiers 1–4 Architecture Breakdown

### Tier 1: Core Execution & Real-Time 2.5D Geometry Optimization (`WS-1A`, `WS-1B`, `WS-1C`)
1. **`WS-1A` — Native C++17 Global $A^*$ & Terrain-Aware Path Smoothing (`ai/planning/native_cpp/astar_core.cpp`):**
   - Uses a thread-local generation-indexed visited array (`visited_gen_`) for $O(1)$ zero-allocation resets across planning calls.
   - Implements Jump Point Search (JPS) directional neighbor pruning on uniform-cost regions and terrain-aware line-of-sight shortcutting (`smooth_path`) that preserves low-slope switchback detours.
   - Compiled for both Windows (`astar_cpp.dll`, `astar_cpp_scalar.dll`) and Linux (`libastar_cpp.so`, `libastar_cpp_scalar.so`).
2. **`WS-1B` — Dynamic SE(3) Attitude-Compensated IPM (`production/perception/vision_costmap_integrator.py`):**
   - Compensates camera rays in real time for EKF chassis roll $\phi$ and pitch $\theta$ via $\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$ in $1.61\,\text{ms}$, keeping stationary ground obstacle drift $\le 0.10\,\text{m}$ under $\pm 12^\circ$ vehicle tilting.
3. **`WS-1C` — Vectorized 2.5D Terrain Geometry & Undercarriage Crown Guard (`production/perception/terrain_geometry.py`):**
   - Computes boundary-aware finite-difference surface normals $[n_x, n_y, n_z]$, slope $\alpha = \arccos(n_z)$, positive steps $\Delta Z^+ \ge 0.12\,\text{m}$, negative ditches $\Delta Z^- \le -0.12\,\text{m}$, and $5\times 5 / 7\times 7$ wheel-ring vs. $3\times 3$ belly high-centering crown height $h_{\text{crown}} \ge 0.12\,\text{m}$.

---

### Tier 2: Spatial Fusion & Dynamic State Hardening (`WS-2A`, `WS-2B`)
1. **`WS-2A` — Native C++ Radial Costmap Inflation & In-Place Dynamic Occupancy (`ai/spatial_fusion/fusion_engine.py`, `ai/planning/dynamic/occupancy_layer.py`):**
   - Precomputes circular structuring kernels for single-pass C++ `cv2.dilate` obstacle inflation and reuses spatiotemporal slice buffers in-place inside `DynamicOccupancyLayer._reset_slices()`.
2. **`WS-2B` — Semantic-Weighted Hazard Memory & CTRV Turn-Rate Predictor (`ai/planning/safety_governor.py`, `ai/planning/dynamic/predictor.py`):**
   - Applies semantic class TTL and decay multipliers (extending boulder/vehicle/pedestrian latches while rapidly decaying shadow/glare/grass false positives) and Coordinated Turn-Rate & Velocity (CTRV) trajectory extrapolation.

---

### Tier 3: Perception Model Quantization & Asynchronous Execution (`WS-3A`, `P1-3`)
1. **`WS-3A` — Dynamic ONNX Export & Mixed-Precision INT8 PTQ (`scripts/quantize_traversability_int8.py`):**
   - Exports `outputs/exports/traversability_model_dyn.onnx` and quantizes backbone convolutions to INT8 (`outputs/exports/traversability_model_int8.onnx`) with $0.517\%$ mean drivability diff and **zero false-safe pixel increases**.
2. **`P1-3` — Decoupled Asynchronous Perception Worker (`production/perception/async_perception_worker.py`):**
   - Uses a depth-1 latest-frame slot so the ROS 2 camera callback deposits frames in $< 0.1\,\text{ms}$ while neural inference and SE(3) BEV projection run on a background daemon thread.

---

### Tier 4: Zero-Copy ROS 2 IPC & Multi-Backend Hardware Abstraction (`WS-4A`, `UnifiedDWAPlanner`)
1. **`WS-4A` — Intra-Process Loaned Handles & Cross-Process POD Shared Memory (`ros2_ws/src/ugv_navigation/ugv_navigation/ipc/zero_copy_ipc.py`):**
   - `IntraProcessLoanedChannel`: Transfers read-only frozen NumPy buffers across same-process nodes in $< 0.05\,\text{ms}$.
   - `SharedMemoryWorldMapChannel` & `SharedMemoryCameraFrameChannel`: Lock-free sequence-numbered `multiprocessing.shared_memory` ring buffers transferring $200\times 200$ 10-layer maps in $< 0.8\,\text{ms}$.
2. **Multi-Backend Hardware Arbitration (`ai/planning/backends/unified_planner.py`):**
   - Supports `AUTO`, `CPP_SIMD` (AVX2/OpenMP), `CPP_SCALAR` (portable ARM/x86), `CUDA` (NVIDIA Tesla T4 / Jetson), `INTEL_GPU` (Intel Arc / XPU), `VECTORIZED_CPU`, and `CPU` reference with zero-copy `uint8` views and preallocated diagnostic buffers.
