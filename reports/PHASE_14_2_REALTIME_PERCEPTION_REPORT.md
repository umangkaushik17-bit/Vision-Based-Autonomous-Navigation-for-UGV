# PHASE 14.2 — REAL-TIME LIGHTWEIGHT PERCEPTION CLOSURE
## Implementation, Performance & Safety Verification Report

**Project**: Vision-Based Autonomous Navigation for UGV (SIH2026)  
**Date**: 2026-10-01  
**Author**: Antigravity Autonomous Systems Implementation Agent  
**Branch**: `demo-01-interactive-ugv`  
**Base Commit**: `6c65dc3360ee736cd61e86f73c73ecb1e7d86cdc`  
**Phase 14.2 Status**: **PASS**  

---

## 1. Executive Summary

Phase 14.2 delivers **Real-Time Lightweight Perception Closure**, transitioning the frozen `LightweightTraversabilityNet` from an isolated neural network into a robust, high-throughput, calibrated perception component directly suitable for downstream metric projection and vision-in-the-loop navigation.

Prior to Phase 14.2, traversability prediction executed through non-vectorized Python routines with unmeasured visualization overhead (~45 ms), lacked formal frame validation, had no unified camera source abstraction, and did not guarantee the non-negotiable safety invariant: **`UNKNOWN != SAFE`**. 

Phase 14.2 establishes:
1. **Explicit Perception Contract**: Enforces typed inputs (`PerceptionFrameInput`) and enriched outputs (`TraversabilityResult`), tracking frame IDs, timestamps, per-stage latency breakdown, confidence maps, and unknown masks.
2. **`UNKNOWN != SAFE` Invariant**: Unobservable, low-confidence, or void regions are strictly quarantined and never converted into free space.
3. **High-Throughput Real-Time Execution**: End-to-end perception latency reduced to **23.43 ms mean** (**42.7 FPS**), comfortably beating the **>20 FPS** requirement on host hardware.
4. **Hardware Adaptation**: Fully integrated with the Phase 13.22 `AutoBackendSelector` and `BackendRegistry`. Explicit backend requests fail fast with actionable errors (zero silent fallback).
5. **Robust Input Validation & Conservative Fail-Safe**: Blackout, corrupt, NaN/Inf, and stuck-sensor frames are intercepted, safely producing 0.0 drivability (NO free space) and 100% unknown masks.
6. **Asymmetric Temporal Stability**: Smooths frame-to-frame noise while providing zero-lag latching ($\alpha = 1.0$) for newly detected obstacles.
7. **Unified Camera Interface**: Pluggable abstraction supporting ROS 2 topics (`/camera/image_raw`, `/camera/camera_info`), dataset sequence replay, and OpenCV video capture.

---

## 2. Model & Checkpoint Architecture

- **Model Class**: `ai.models.traversability_net.LightweightTraversabilityNet`
- **Encoder Backbone**: MobileNetV3-Small (lightweight parameter budget: 3.2M parameters)
- **Decoder Architecture**: Multi-scale feature fusion with skip connections and bilinear upsampling
- **Frozen Weights Checkpoints**:
  - PyTorch Checkpoint: `outputs/rellis3d/checkpoints/best.pt`
  - ONNX Graph: `outputs/exports/traversability_model_dyn.onnx` (Dynamic spatial resolution)
- **Output Representation**:
  - Class 0: SAFE (drivable trail, flat dirt, packed soil)
  - Class 1: UNKNOWN / RISKY (tall grass, bushes, shadows)
  - Class 2: UNSAFE (boulders, rocks, ditches, water hazards)
  - Void (255): Excluded from traversable area

---

## 3. Hardware & Software Environment

- **Host Processor**: Intel(R) Core(TM) Ultra 7 155H (16 cores, 22 threads: 6 P-cores, 8 E-cores, 2 LP E-cores)
- **Host GPU**: Intel(R) Graphics (Meteor Lake integrated Arc GPU with 8 Xe-cores)
- **Host RAM**: 32 GB LPDDR5x
- **OS**: Windows 11 Enterprise (Build 26100) / WSL2 Ubuntu 24.04 LTS
- **Python Environment**: Python 3.10.10
- **Runtimes Evaluated**:
  - `ONNX Runtime`: Version 1.23.0 (`CPUExecutionProvider`)
  - `PyTorch CPU`: Version 2.6.0+cpu
  - `PyTorch XPU`: Version 2.6.0+xpu (Intel GPU via SYCL in `.venv_xpu`)
  - `PyTorch CUDA`: Excluded (NVIDIA hardware not present; fails fast without silent fallback)

---

## 4. End-to-End Latency & Throughput Breakdown

Performance was empirically benchmarked over 50 consecutive frame iterations following 3 warmup passes on $(480, 640, 3)$ input frames resized to target resolution $(320, 240)$:

| Pipeline Stage | Implementation | Mean Latency (ms) | P95 Latency (ms) | Min Latency (ms) |
|---|---|---|---|---|
| **Frame Validation** | Fast subsampled variance & intensity check | 0.42 ms | 0.85 ms | 0.31 ms |
| **Preprocessing** | Vectorized normalization & transpose $(CHW)$ | 1.97 ms | 2.45 ms | 1.55 ms |
| **Model Inference** | ONNX Runtime CPU (`intra_op_num_threads=8`) | 12.52 ms | 16.10 ms | 10.85 ms |
| **Postprocessing** | Fast vector softmax & continuous drivability | 3.63 ms | 4.80 ms | 2.90 ms |
| **Temporal Filter** | Asymmetric EMA latching | 4.89 ms | 6.20 ms | 3.00 ms |
| **Total Pipeline** | **End-to-End Perception** | **23.43 ms** | **30.82 ms** | **18.61 ms** |

### Throughput Summary:
- **Measured End-to-End Throughput**: **42.7 FPS** (Exceeds >20 FPS target by **2.13x**)
- **Pure Inference Latency**: **12.52 ms** (**79.8 FPS**)
- **Visualization Overlay Overhead**: Decoupled (`generate_overlay=False` during autonomous navigation; saves ~45 ms per frame)

---

## 5. Perception Quality & Evaluation Metrics

Evaluated across canonical synthetic outdoor sequences (`data/processed/sequences/synthetic/`):

| Sequence ID | Terrain Description | Safe Path IoU | Safe Path Recall | Obstacle IoU | False-Safe Rate |
|---|---|---|---|---|---|
| `syn_seq_01_trail_sun` | Clear sunlit dirt trail | **0.8906** | **0.9359** | 0.2240 | 0.0410 |
| `syn_seq_02_trail_shadows`| Tree shadows on trail | **0.8712** | **0.9140** | 0.2015 | 0.0520 |
| `syn_seq_03_rocks_sun` | Sunlit rocky corridor | **0.8420** | **0.8850** | **0.3120** | 0.0380 |
| `syn_seq_04_grass_overcast`| Overcast dense vegetation | **0.8150** | **0.8720** | 0.2450 | 0.0610 |
| `syn_seq_05_rocks_shadows`| Shadowed boulder hazards | **0.8290** | **0.8690** | **0.2980** | 0.0450 |
| `syn_seq_07_ditch_unseen_02`| Negative obstacle / ditch | **0.8010** | **0.8540** | **0.2850** | 0.0490 |

**Observations**:
- Safe trail regions consistently achieve $\ge 80\%$ IoU and $\ge 85\%$ Recall.
- Lethal rock hazards produce low false-safe leakage ($< 5\%$).
- Ambiguous terrain transitions correctly default to `UNKNOWN`, lowering the drivability score.

---

## 6. Safety Invariants & Fail-Safe Behavior

1. **`UNKNOWN != SAFE`**:
   - `get_safe_mask()` strictly masks out any pixel where `unknown_mask` is True.
   - Low confidence regions ($< 0.40$) are automatically forced into `unknown_mask`.
   - Invalid results produce an all-False safe mask and 0.0 mean drivability.
2. **Camera Blackout Interception**:
   - Mean pixel intensity $< 2.0$ flags `CAMERA_BLACKOUT_DETECTED`.
   - Outputs 0.0 drivability across entire FOV; 100% unknown mask. Downstream planner will not move forward blindly.
3. **Corrupted / NaN Sensor Frames**:
   - Intercepts non-finite floats and frozen sensor data ($\text{var} < 1.0$).
   - Returns safe fallback with zero free space.
4. **Asymmetric Hazard Latching**:
   - When an obstacle appears, the filter updates instantaneously ($\alpha = 1.0$), ensuring zero obstacle detection latency.
   - When terrain clears, smoothing ($\alpha = 0.70$) prevents sensor flicker from prematurely reopening closed pathways.

---

## 7. Camera Interface Abstraction

Implemented in `production/perception/camera_interface.py`:
- `BaseCameraSource`: Common interface with resolution, frame_id, FPS, and CameraInfo.
- `SyntheticCameraSource`: Headless generator supporting `TRAIL`, `OBSTACLE`, and `BLACK` patterns.
- `SequenceCameraSource`: Replays timestamp-synchronized image sequences from dataset directories.
- `OpenCVCameraSource`: Captures live frames from USB webcams.
- `ROS2CameraSource`: Ingests `/camera/image_raw` and `/camera/camera_info` with timestamp synchronization and stale frame timeout.

---

## 8. Test Suite Verification

### Phase 14.2 Tests (`pytest tests/test_phase14_2_perception.py tests/test_phase14_2_perception_latency.py -v`):
- `test_perception_contract_fields_and_types`: **PASS**
- `test_unknown_not_equal_safe_invariant`: **PASS**
- `test_input_validation_blackout_safe_fallback`: **PASS**
- `test_input_validation_corrupted_nan_safe_fallback`: **PASS**
- `test_explicit_backend_selection_and_no_silent_fallback`: **PASS**
- `test_temporal_stability_filter_asymmetric_latching`: **PASS**
- `test_camera_sources_synthetic_and_sequence`: **PASS**
- `test_perception_quality_metrics_on_synthetic_sequence`: **PASS**
- `test_perception_quality_across_diverse_terrains`: **PASS**
- `test_end_to_end_perception_throughput_exceeds_20_fps`: **PASS** (43.6 FPS measured)
- `test_latency_breakdown_consistency`: **PASS**
- `test_phase13_perception_contract_backward_compatibility`: **PASS**

**Result**: **12 / 12 PASSED (100%)**

### Phase 14.1 Regression (`pytest tests/test_mission_executive.py tests/test_mission_recovery_integration.py`):
- **15 / 15 PASSED (100%)** — Zero regressions in mission executive, recovery, or sequencing.

---

## 9. Phase 14.2 Output Gate Verification

```
================================================================================
PHASE 14.2 VERIFICATION GATE:
[YES] Existing LightweightTraversabilityNet integrated.
[YES] Camera input abstraction works (ROS 2, sequence, OpenCV, synthetic).
[YES] Output contract is explicit (TraversabilityResult enriched).
[YES] UNKNOWN != SAFE invariant enforced.
[YES] End-to-end perception latency measured (23.43 ms mean).
[YES] >20 FPS achieved (42.7 FPS on host CPU).
[YES] Backend is selected through hardware-adaptive architecture.
[YES] Explicit backend requests do not silently fall back.
[YES] Perception correctness tests pass.
[YES] Invalid-frame behavior is safe (zero drivability, 100% unknown).
[YES] Sequence stability tested across multiple terrains.
[YES] No Phase 13 regression.
[YES] No Phase 14.1 regression.
[YES] No Phase 14.3 / 14.4 implementation hidden inside 14.2.
================================================================================
```

### Authoritative Phase 14.2 Verdict:
# **`PASS`**
