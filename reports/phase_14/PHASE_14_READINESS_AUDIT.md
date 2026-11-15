# PHASE 14 READINESS & ARCHITECTURE AUDIT
## GPS-Denied Outdoor Navigation & Mission Executive

**Date**: 2026-10-01  
**Auditor**: Antigravity Assistant (Advanced Agentic Architecture & Readiness Pass)  
**Authoritative Branch**: `demo-01-interactive-ugv`  
**Current HEAD Commit**: `c748fb8` (`fix(runtime): verify and correct hardware-adaptive runtime architecture and AVX2 gating`)  
**Repository Working Tree**: Clean  
**Inspection Type**: Strict READ-ONLY Architectural & Empirical Audit  

---

## 1. Executive Summary

This audit evaluates the repository's concrete readiness to transition from the frozen Phase 13 / Phase 13.22 navigation baseline into **Phase 14 — GPS-Denied Outdoor Navigation & Mission Executive**.

### Core Verdict
The repository is **READY_TO_START_PHASE_14_1_ONLY**.

The proposed full Phase 14 roadmap (14.1 through 14.6) cannot be started as an undifferentiated monolith because significant hardware, algorithmic, and calibration gaps exist in later milestones (14.2–14.6). However, the planning, recovery, safety governance, and actuation infrastructure required for **Milestone 14.1 (Mission Executive & Recovery Behavior Trees)** is completely implemented, verified, and frozen. Milestone 14.1 can begin immediately with strict, bounded interfaces and zero changes to the underlying Phase 13 DWA planner.

### Key Empirical Findings:
1. **DWA Core Frozen & Decoupled**: The C++ DWA planner (`ai/planning/native_cpp/dwa_core.cpp`, `libdwa_cpp.so` / `dwa_cpp.dll`) operates as a stateless local trajectory evaluator consuming `(RobotState, Path, Costmap, DynamicOccupancyLayer, Predictions)`. It does not manage missions, goals, or multi-waypoint progressions. A Mission Executive sits strictly *above* global replanning and DWA.
2. **Recovery & Deadlock Infrastructure Already Exists**: Multi-signal deadlock detection, rear static/dynamic clearance verification, safe reverse maneuvers (-0.15 m/s, 0.45 m), and replan triggering were already designed, implemented, and benchmark-validated in Phase 13.24 (`ai/planning/recovery_controller.py`). Milestone 14.1 does not need to invent recovery mechanics from scratch; it only needs to wrap them in a formal mission state machine.
3. **Perception Model Exists, But Costmap Loop Is Disconnected**: `LightweightTraversabilityNet` (`ai/models/traversability_net.py`) is already trained on RELLIS-3D/RUGD and achieves ~94.8 ms on CPU (~10.5 FPS) and ~76 ms on Intel GPU. However, in the live Gazebo runtime (`interactive_ugv_navigation_node.py`), perception inference is only published as an RViz HUD overlay; the navigation costmap relies on a static hardcoded boulder. Furthermore, `PerceptionAdapter` projects perspective images to BEV using an uncalibrated 2D image resize (`cv2.resize`) rather than true ground-plane Inverse Perspective Mapping (IPM).
4. **Dynamic Obstacles Are Currently Telemetry-Driven**: Dynamic obstacle tracks in the Gazebo demonstration originate from `/dynamic_obstacle/odom` passed through a synthetic sensor adapter with noise and FOV filtering (`_generate_sensor_observation`), not from real-time computer vision detection.
5. **Localization Fusion Is Reference-Only**: Stereo Visual Odometry (`StereoVisualOdometry`) and Visual-Inertial Odometry (`VisualInertialOdometry`) exist in `ai/localization/`, but they are evaluated offline on datasets (`scripts/run_slam.py`). There is no active `robot_localization` EKF node running in the ROS 2 launch graph; active navigation relies directly on Gazebo `/odom`.
6. **Hardware Reality Constraints**: The development environment is a single Windows 11 host (Intel Core Ultra 7 155H, Intel Arc integrated GPU, 32 GB RAM, WSL2 Ubuntu 24.04). No NVIDIA Jetson Orin, physical stereo camera, physical IMU, or outdoor test course is physically attached. Milestone 14.6 (Embedded Edge Deployment) is strictly an architectural target and cannot be hardware-validated on this machine.

---

## 2. Current Phase 13 Frozen Baseline

### 2.1 Git Identification
- **Branch**: `demo-01-interactive-ugv`
- **HEAD Commit**: `c748fb8`
- **Working Tree**: Clean (`git status` reports nothing to commit)

### 2.2 Verified Production Components
The system has established a strictly verified, hardware-adaptive runtime pipeline:

```
[HardwareProfiler]
       ↓
[BackendRegistry]
       ↓
[AutoBackendSelector]
       ↓
[Validated Planner Backend: C++ AVX2 (libdwa_cpp.so / dwa_cpp.dll)]
       ↓
[Runtime Adaptation & Latency Gating]
       ↓
[Authoritative Safety Governor & Hazard Latching]
       ↓
[Safe Command Buffer @ 20 Hz Dispatch]
       ↓
[/cmd_vel]
```

Specific verified assets:
- **`HardwareProfiler`** (`ai/hardware/profiler.py`): Detects CPU features (AVX2, AVX512, SSE4.2), physical/logical cores, RAM, GPU vendors (Intel, NVIDIA, AMD), and OpenCL/DirectML environments.
- **`BackendRegistry`** (`ai/planning/backends/backend_registry.py`): Formally registers backends (`CPP_CPU`, `SCALAR_CPP_CPU`, `INTEL_GPU_PTH`, `INTEL_GPU_OPENVINO`, `CUDA_GPU`, `PYTHON_REFERENCE`) and validates binary integrity before selection.
- **`CPPDWABackend`** (`ai/planning/backends/cpp_backend.py`): FFI wrapper invoking native SIMD AVX2 C++ DWA. Executes in **0.65 – 1.9 ms** per step (eliminating the legacy Python 3,211 ms bottleneck).
- **`ScalarCPPDWABackend`** (`ai/planning/backends/scalar_cpp_backend.py`): Clean scalar fallback for environments lacking AVX2.
- **`UnifiedDWAPlanner`** (`ai/planning/backends/unified_planner.py`): Central planner orchestrator with deterministic `AUTO` selection, behavioral arbitration, and candidate trajectory logging.
- **`LightweightTraversabilityNet`** (`ai/models/traversability_net.py`): MobileNetV3-Small encoder with lightweight decoder (~3.2M parameters) trained on RELLIS-3D/RUGD (`outputs/rellis3d/checkpoints/best.pt`). Validated on CPU (~94.8 ms) and Intel integrated GPU (~76 ms).
- **`SafetyGovernor` & `RecentHazardMemory`** (`ai/planning/safety_governor.py`): Latches dropped or occluded obstacle tracks with dynamic TTL (3.5s – 5.0s), strictly enforcing the core safety principle: *"Not detected does NOT mean safe"*.
- **`RecoveryController`** (`ai/planning/recovery_controller.py`): Multi-signal deadlock detector (DWA failure, spatial lack of progress, chattering), rear clearance validator, and deterministic reverse maneuver (-0.15 m/s, 0.45 m).
- **`SafeCommandBuffer`** (`ai/control/command_buffer.py`): Continuous 20 Hz command dispatcher with command freshness watchdog (0.35s / 0.50s timeout) and smooth deceleration.

### 2.3 Known Limitations of Frozen Baseline
1. **Single Goal Focus**: The entire navigation stack only accepts a single `(goal_x, goal_y)` via `/goal_pose`. It has no concept of sequential waypoints, route progression, or mission abort criteria.
2. **Costmap Decoupled from Live Perception**: `interactive_ugv_navigation_node.py` runs `LightweightTraversabilityNet` on camera frames, but discards the cost tensor for navigation and only publishes an RViz image overlay. The active costmap is a static synthetic boulder at `(3.5, 0.15)`.
3. **Naive Perspective-to-BEV Projection**: Where perception is adapted (`ugv_navigation/adapters/perception_adapter.py`), it resizes the 2D image directly to the grid dimensions via `cv2.resize`. It lacks camera matrix intrinsics $K$, extrinsic tilt $\theta$, and ground-plane ray-casting / IPM homography.
4. **Synthetic Obstacle Feeds**: Dynamic obstacle tracks in simulation are derived from Gazebo ground-truth odometry (`/dynamic_obstacle/odom`) with added Gaussian noise, rather than real vision detection.
5. **No Active State Estimation EKF**: The navigation stack uses raw `/odom` from Gazebo. Multi-sensor fusion (VIO + IMU + Wheel Odom) is absent from the active runtime graph.

---

## 3. Phase 14 Proposed Architecture

The proposed Phase 14 architecture aims to achieve autonomous outdoor navigation across long courses without GPS. The following target architecture diagram highlights what currently exists (solid) versus what is missing (dashed):

```mermaid
flowchart TD
    subgraph MISSING_MISSION["Milestone 14.1 — Mission Executive (To Build)"]
        WPM[Waypoint Mission Manager<br/>W1 → W2 → ... → Wn]
        MSM[Mission State Machine / Behavior Tree<br/>NAVIGATING / PAUSED / RECOVERY / ABORTED]
    end

    subgraph PHASE13_PLANNING["Phase 13 Frozen Planning Core (Existing & Verified)"]
        A_STAR[EventDrivenGlobalReplanner<br/>A* Global Path]
        DWA[UnifiedDWAPlanner<br/>Native C++ AVX2 @ 1ms]
        REC[RecoveryController<br/>Deadlock & Safe Reverse]
        GOV[AuthoritativeSafetyGovernor<br/>Corridor & Hazard Watchdog]
        BUF[SafeCommandBuffer<br/>20 Hz Continuous Dispatch]
    end

    subgraph PERCEPTION_GAP["Milestones 14.2 & 14.3 — Perception (Existing vs Gap)"]
        LTN[LightweightTraversabilityNet<br/>RELLIS-3D Model (EXISTING)]
        IPM[Calibrated Inverse Perspective Mapping<br/>Camera → Metric BEV (MISSING)]
        FUSION[SpatialFusionEngine<br/>Local BEV → WorldTerrainMap (EXISTING)]
        YOLO[Vision Dynamic Obstacle Detector<br/>RGB → 3D Bounding Box (MISSING)]
    end

    subgraph LOCALIZATION_GAP["Milestone 14.4 — GPS-Denied State Estimation (Research Only)"]
        GZ_ODOM[Gazebo /odom Ground Truth (EXISTING)]
        VO_REF[StereoVisualOdometry / VIO<br/>Offline Reference Only (EXISTING)]
        EKF[robot_localization EKF Node<br/>Multi-Sensor Fusion (MISSING)]
    end

    %% Connections
    WPM -->|Current Target Waypoint| A_STAR
    MSM -->|Mode / Abort / Recovery Commands| DWA
    MSM -->|Deadlock Triggers| REC
    
    A_STAR -->|Global Path| DWA
    DWA -->|Candidate Velocities| REC
    REC -->|Safe Velocities| GOV
    GOV -->|Governed Velocities| BUF
    BUF -->|/cmd_vel @ 20 Hz| DIFF_DRIVE[UGV 4WD Actuation]

    LTN -->|Traversability Mask| IPM
    IPM -.->|Local Metric BEV Costmap| FUSION
    FUSION -.->|Fused Costmap| DWA
    FUSION -.->|Base Map| A_STAR

    GZ_ODOM -->|Active Pose| DWA
    GZ_ODOM -->|Active Pose| REC
    GZ_ODOM -->|Active Pose| MSM
    EKF -.->|Future Fused Odom| DWA
```

---

## 4. Milestone Readiness Matrix

| Milestone | Existing Code | Maturity | Validation | Dependencies | Blockers | Current-Hardware Feasible | Recommendation |
|---|---|---|---|---|---|---|---|
| **14.1 — Mission Executive & Recovery Behavior Trees** | `src/ugv_mission/` (skeleton), `ai/planning/recovery_controller.py`, `ai/planning/dynamic/behavioral_state.py` | Partially implemented (Recovery: Production-ready; Executive: Stub) | Unit / Benchmark tested for recovery; 0 mission tests | Phase 13 `RecoveryController`, `UnifiedDWAPlanner`, `AStarPlanner`, `SafeCommandBuffer` | None | **YES** | **RECOMMENDED: Start immediately under strictly bounded scope.** |
| **14.2 — Lightweight Real-Time Perception Closure** | `ai/models/traversability_net.py`, `ai/traversability/dense_predictor.py`, `outputs/rellis3d/` | Production-ready model; Missing IPM projection | Offline dataset validation (RELLIS-3D); CPU/GPU inference verified | PyTorch, torchvision, camera intrinsics | Uncalibrated perspective-to-BEV projection (`cv2.resize` hack) | **YES** | **RE-SCOPE: Do NOT design a new model. Add calibrated ground projection (IPM) to existing net.** |
| **14.3 — End-to-End Vision-in-the-Loop Integration** | `interactive_ugv_navigation_node.py`, `perception_node.py`, `spatial_fusion_node.py` | Partially implemented (Overlay works; costmap feed disconnected) | Simulation tested in Gazebo with synthetic telemetry obstacles | Milestone 14.2 (IPM), `SpatialFusionEngine`, vision-based obstacle detector | Live costmap disconnected from camera; dynamic obstacles telemetry-driven | **YES** (in Gazebo) | **DEFER until 14.1 & 14.2 are complete. Connect camera costmap to navigation.** |
| **14.4 — Robust GPS-Denied Localization Fusion** | `ai/localization/stereo_vo.py`, `visual_inertial.py`, `ugv_navigation/nodes/localization_node.py` | Research / Reference only | Offline dataset runs (`scripts/run_slam.py`); data contract tests | Stereo camera feed, IMU, `robot_localization` EKF node | No EKF node in launch graph; VO not closed-loop validated | **PARTIAL** (Simulation in Gazebo only; no hardware sensors) | **TREAT AS PARALLEL RESEARCH STREAM. Keep decoupled from 14.1.** |
| **14.5 — Long-Horizon Multi-Waypoint Benchmark** | `simulation/worlds/demo_outdoor.sdf`, `scripts/test_all_missions.py` | Missing for 100m course; short 10m scenarios exist | Short single-goal runs tested in Gazebo and SIL | Milestones 14.1, 14.3, 14.4, large-scale world SDF | Gated on 14.1, 14.3, 14.4; no 100m outdoor simulation world | **YES** (in Gazebo once 100m SDF world is created) | **DEFER until core milestones 14.1–14.4 are verified.** |
| **14.6 — Embedded Edge Deployment Validation** | `docker/Dockerfile`, deployment scaffolding | Architectural target only | None on embedded hardware | Physical NVIDIA Jetson Orin, TensorRT, 4GB VRAM envelope | No embedded hardware physically available | **NO** (Hardware not present) | **DEFER. Do not attempt hardware sign-off on host Windows machine.** |

---

## 5. Repository Component Inventory

| Component / Subsystem | Primary Path | Implementation Class / Symbol | Existing Status | Test Coverage |
|---|---|---|---|---|
| **Mission Subsystem** | `src/ugv_mission/` | `__version__ = "0.1.0"` (empty skeleton) | Stub / Interface only | 0 tests |
| **Recovery Controller** | `ai/planning/recovery_controller.py` | `RecoveryController`, `RecoveryState`, `RecoveryConfig` | Production-ready / Reusable | Tested in `test_safety_workstream_benchmarks.py` & `test_recovery_replanning.py` |
| **Behavioral State Machine** | `ai/planning/dynamic/behavioral_state.py` | `BehavioralStateMachine`, `BehavioralDecision` | Production-ready / Reusable | 26 unit tests (`test_behavioral_state.py`) |
| **Global Replanner** | `ai/planning/dynamic/global_replanner.py` | `EventDrivenGlobalReplanner`, `DynamicReplanResult` | Production-ready / Reusable | 29 unit tests (`test_global_replanning.py`) |
| **Native C++ DWA** | `ai/planning/native_cpp/` | `dwa_plan_c()`, `libdwa_cpp.so`, `dwa_cpp.dll` | Production-ready / Reusable | 8 parity tests (`test_dwa_cpp_parity.py`) |
| **Unified DWA Planner** | `ai/planning/backends/unified_planner.py` | `UnifiedDWAPlanner`, `DWABackendType` | Production-ready / Reusable | 7 backend tests (`test_dwa_backends.py`) |
| **Safety Governor** | `ai/planning/safety_governor.py` | `SafetyGovernor`, `RecentHazardMemory` | Production-ready / Reusable | Tested across 10 benchmarks (B01–B10) |
| **Command Buffer** | `ai/control/command_buffer.py` | `SafeCommandBuffer`, `SafeCommandBufferConfig` | Production-ready / Reusable | 8 tests (`test_continuous_command_buffer.py`) |
| **Traversability Model** | `ai/models/traversability_net.py` | `LightweightTraversabilityNet` | Production-ready / Reusable | Offline validation (`outputs/rellis3d/`) |
| **Dense Predictor** | `ai/traversability/dense_predictor.py` | `DenseTraversabilityPredictor` | Production-ready / Reusable | 4 tests (`test_intel_gpu_perception.py`) |
| **Perception Adapter** | `ros2_ws/src/.../adapters/perception_adapter.py` | `PerceptionAdapter.process_image()` | Partially implemented (Naive `cv2.resize`) | 3 tests (`test_ros2_adapters.py`) |
| **Spatial Fusion Engine** | `ai/spatial_fusion/fusion_engine.py` | `SpatialFusionEngine.fuse_local_costmap()` | Production-ready / Reusable | 5 tests (`test_spatial_fusion.py`) |
| **Stereo Visual Odometry** | `ai/localization/stereo_vo.py` | `StereoVisualOdometry` | Research / Reference only | Contract tests only (`test_localization.py`) |
| **Visual-Inertial Odometry** | `ai/localization/visual_inertial.py` | `VisualInertialOdometry` | Research / Reference only | Contract tests only (`test_localization.py`) |
| **ROS2 Dynamic Node Graph** | `ros2_ws/src/ugv_navigation/ugv_navigation/nodes/` | 13 nodes (`planner_node`, `behavioral_state_node`, etc.) | Production-ready in ROS2 | 29 tests (`test_ros2_dynamic_integration.py`) |

---

## 6. Existing Interface Map

To ensure Phase 14 does not break Phase 13 contracts, all existing interfaces were audited:

### 6.1 DWA Planner Contract (`ai/planning/backends/backend_interface.py`)
```python
def compute_trajectory(
    current_state: RobotState,
    global_path: Optional[Path],
    costmap: WorldTerrainMap,
    dynamic_layer: Optional[DynamicOccupancyLayer] = None,
    predicted_tracks: Optional[List[PredictedObstacleTrajectory]] = None,
) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]
```
*Verdict*: Fully intact. Milestone 14.1 feeds `current_state`, `global_path` (from current waypoint), `costmap`, and dynamic context, receiving velocities without touching DWA math.

### 6.2 Recovery Controller Contract (`ai/planning/recovery_controller.py`)
```python
def step(
    current_time_s: float,
    robot_pose: Tuple[float, float, float],
    is_deadlocked: bool,
    hazards: List[LatchedHazard],
    costmap: Optional[WorldTerrainMap] = None,
) -> Tuple[Optional[float], Optional[float], RecoveryState, str]
```
*Verdict*: Fully intact. `is_deadlocked` is evaluated from planner status and spatial displacement. Safe reverse velocities `(rec_v, rec_w)` and state transitions (`REVERSING`, `REPLAN_READY`, `ABORTED_SAFE_STOP`) are directly usable by the mission executive.

### 6.3 Safety Governor Contract (`ai/planning/safety_governor.py`)
```python
def evaluate(
    requested_v: float,
    requested_w: float,
    robot_pose: Tuple[float, float, float],
    hazards: List[LatchedHazard],
    current_time_s: float,
) -> Tuple[float, float, str, Dict[str, Any]]
```
*Verdict*: Fully authoritative. The safety governor clamps or stops any command issued by DWA or Recovery if forward corridor or rear clearance is violated.

### 6.4 Actuation Contract (`ai/control/command_buffer.py`)
```python
def set_command(
    linear_v: float,
    angular_w: float,
    timestamp_s: float,
    is_safe: bool = True,
    source: str = "DWA",
    reason: str = "OK",
) -> None
```
*Verdict*: High-frequency continuous dispatch @ 20 Hz with watchdog timeout.

### 6.5 Mission Executive Required Interface (Phase 14.1 Boundary)
Milestone 14.1 must sit cleanly on top of this stack:
```
[MissionExecutive]
  ├── Inputs: WaypointList, RobotState, PlannerStatus, RecoveryState
  ├── Outputs: ActiveGoalPose (to A* Replanner), MissionState, Telemetry
  └── Behavior: Coordinates W1 → W2 → ... → Wn, triggers Recovery on stall, aborts on failure.
```
**Zero Phase 13 interfaces need modification.**

---

## 7. Synthetic vs Real Data Path Audit

| Pipeline Stage | Current Implementation | Source of Data | Classification | Action for Phase 14 |
|---|---|---|---|---|
| **Camera Feed** | Gazebo Camera Plugin → `ros_gz_bridge` → `/camera/image_raw` | Simulated Gazebo 3D world | Real topic / Simulated sensor | Keep for 14.1–14.5; physical camera only for 14.6 |
| **Traversability Inference** | `LightweightTraversabilityNet.predict()` | Live image frame via PyTorch | **REAL INFERENCE** (Model weights active) | Retain existing model |
| **BEV Projection** | `cv2.resize` in `PerceptionAdapter` | Naive 2D image resize | **SYNTHETIC / UNPHYSICAL HACK** | **MUST REPLACE** with Calibrated IPM in 14.2 |
| **Static Costmap** | Hardcoded rock boulder at `(3.5, 0.15)` in `WorldTerrainMap` | Hardcoded geometric formula in node init | **SYNTHETIC FIXTURE** | **MUST REPLACE** with fused perception in 14.3 |
| **Dynamic Obstacles** | `/dynamic_obstacle/odom` + Gaussian noise filter | Gazebo model state topic | **SYNTHETIC TELEMETRY ADAPTER** | Replace with vision-based detector in 14.3 |
| **Dynamic Tracking** | `MultiFrameTracker` with Kalman filtering | Detection messages | **REAL ALGORITHMIC CODE** | Fully validated; retain |
| **Local Planning** | `CPPDWABackend` (`libdwa_cpp.so`) | Grid costmap + dynamic occupancy | **REAL NATIVE CODE** | Fully validated; retain frozen |
| **Localization** | Gazebo `/odom` topic broadcast as `odom -> base_link` | Gazebo ground truth odometry | **SIMULATED IDEAL SENSOR** | Add noise/drift injection for 14.1; EKF for 14.4 |
| **Command Output** | `/cmd_vel` @ 20 Hz via `ros_gz_bridge` | `SafeCommandBuffer` | **REAL ACTUATION MESSAGE** | Drives Gazebo diff-drive plugin |

---

## 8. Localization Readiness Audit

1. **Existence**:
   - `ai/localization/stereo_vo.py` (274 lines): ORB-based stereo feature matching, 3D triangulation, PnP RANSAC.
   - `ai/localization/visual_inertial.py` (145 lines): Loosely/tightly coupled VIO integrating angular velocity and linear acceleration.
   - `ugv_navigation/nodes/localization_node.py` (92 lines): ROS 2 wrapper for visual odometry.
2. **Maturity**: **Research / Reference only**.
3. **Empirical Validation**:
   - Tested only on offline image sequences (`data/processed/sequences/rugd/creek`, synthetic stereo) via `scripts/run_slam.py`.
   - Unit tests (`tests/test_localization.py`) only test dataclass contracts and coordinate conversions, not estimator accuracy or convergence.
   - **Never tested in closed-loop navigation**. In Gazebo, the node graph uses `/odom` from Gazebo diff-drive.
4. **Sufficiency for GPS-Denied Multi-Waypoint (100m class)**:
   - **NO**. Raw dead reckoning / wheel odometry drifts significantly over 100 meters due to wheel slip on rough terrain. Without fused visual odometry and IMU (via an EKF like `robot_localization`), long-horizon waypoint navigation will experience large drift errors.
   - However, for Milestone 14.1 testing in simulation, Gazebo `/odom` is completely sufficient.

---

## 9. Perception Readiness Audit

1. **Model Existence & Architecture**:
   - `ai/models/traversability_net.py` implements `LightweightTraversabilityNet`.
   - Encoder: `mobilenet_v3_small`. Decoder: lightweight feature aggregation (~3.2M parameters).
   - Checkpoint: `outputs/rellis3d/checkpoints/best.pt` (13.6 MB).
2. **Maturity**: **Production-ready / Reusable**.
3. **Speed & Latency Reality**:
   - Verified latency on dev CPU: **94.8 ms** (~10.5 FPS).
   - Verified latency on dev Intel GPU: **76.1 ms** (~13.1 FPS).
   - Latency on remote Kaggle T4 GPU: **14.2 ms** (~70 FPS).
   - Note on the Proposed Roadmap's ">20 FPS" requirement: On the current host CPU, the model runs at ~10.5 FPS. This is completely adequate for an outdoor UGV operating at 0.5 – 1.2 m/s (spatial displacement between frames is < 0.1 m). Replacing this working model with Fast-SCNN or MobileNetV3-UNet would cause massive retraining churn without solving the real bottleneck.
4. **Ground-Plane Projection Defect**:
   - `PerceptionAdapter.process_image()` currently converts perspective masks to BEV using:
     ```python
     pred_bev = cv2.resize(res.mask, (W, H), interpolation=cv2.INTER_LINEAR)
     ```
   - This is unphysical. Perspective foreshortening means pixels near the top of the image represent ground 10 meters away, while pixels near the bottom represent ground 0.5 meters away. A linear resize severely distorts obstacle geometry.
   - **Milestone 14.2 must focus on Calibrated IPM (Inverse Perspective Mapping), NOT a new neural network.**

---

## 10. Mission Executive Readiness Audit

1. **Existence**:
   - `src/ugv_mission/` exists as a package directory containing only an empty `__init__.py`.
   - There are no waypoint queue managers, mission status representations, or multi-goal sequencing logic anywhere in `ai/` or `ros2_ws/`.
2. **Maturity**: **Stub / Interface only**.
3. **Current Goal Handling**:
   - Navigation nodes accept a single `/goal_pose`. When the UGV reaches within 0.35m of `(goal_x, goal_y)`, it commands an emergency stop and sets `mission_completed = True`.
4. **Readiness for Milestone 14.1**:
   - The underlying planners (`UnifiedDWAPlanner`, `AStarPlanner`), recovery controllers (`RecoveryController`), and safety watchdogs (`SafetyGovernor`, `SafeCommandBuffer`) are 100% ready to receive goals from a sequential mission executive.
   - Building the mission executive in `src/ugv_mission/` is pure software engineering requiring zero modifications to the lower navigation layers.

---

## 11. Recovery / Safety Readiness Audit

1. **Existence & Maturity**:
   - `ai/planning/recovery_controller.py`: **Production-ready / Reusable**.
   - Verified across benchmarks B01–B10 in Phase 13.24.
2. **Safety Guarantees Implemented**:
   - **Multi-Signal Deadlock Detection**: Triggers when DWA status is `NO_SAFE_TRAJECTORY`, decision is `STOP`/`EMERGENCY_STOP`, or spatial net displacement is $< 0.15\text{ m}$ over a 2.5-second window.
   - **Rear Static & Dynamic Clearance Verification**: Before any reverse motion is commanded, `RecoveryController.check_rear_clearance()` tests the rear footprint corridor against both static obstacles in `WorldTerrainMap` and dynamic/latched hazards in `RecentHazardMemory`.
   - **Deterministic Reverse**: Reverses at $-0.15\text{ m/s}$ for exactly $0.45\text{ m}$, then triggers an A* global replan around the blocking obstruction.
   - **Safe Termination**: If reverse clearance is blocked or replanning fails after recovery attempts, the controller enters `ABORTED_SAFE_STOP` and latches safe zero velocity.
3. **Phase 14 Safety Adherence**:
   - The mission executive in Phase 14.1 will interface directly with this state machine. At no point will recovery motion be initiated without positive rear clearance.

---

## 12. Hardware & Deployment Readiness Audit

1. **Development Host Reality**:
   - OS: Windows 11 Enterprise (Build 26100)
   - CPU: Intel Core Ultra 7 155H (16 cores / 22 threads)
   - GPU: Intel Arc Graphics (Integrated GPU, Level Zero / DirectML support)
   - Linux Subsystem: WSL2 Ubuntu 24.04 LTS (ROS 2 Jazzy, Gazebo Sim Harmonic 8.7.0)
2. **Missing Physical Assets**:
   - NO physical UGV chassis or microcontroller (STM32/CAN/serial) connected.
   - NO physical stereo camera (e.g. Intel RealSense D435i / ZED2) connected.
   - NO physical 6-axis IMU connected.
   - NO physical outdoor test environment.
   - NO NVIDIA Jetson Orin developer kit present on this host.
3. **Validation Implications**:
   - Milestones 14.1, 14.2, 14.3, 14.4 (simulation), and 14.5 (simulation) **CAN** be thoroughly tested using Gazebo Sim Harmonic and Python SIL harnesses on this machine.
   - Milestone 14.6 (Embedded Edge Hardware Validation) **CANNOT** be executed or signed off on this host.

---

## 13. Phase Ordering Recommendation

The roadmap proposed sequence: `14.1 → 14.2 → 14.3 → 14.4 → 14.5 → 14.6`.

### Critical Critique & Optimized Sequencing:

```
[Phase 14.1: Mission Executive & Sequential Waypoints]
       │ (Uses existing frozen DWA + Recovery on Gazebo /odom)
       ▼
[Phase 14.2: Calibrated Ground-Plane Projection (IPM) & Perception Loop Closure]
       │ (Calibrate camera-to-BEV; DO NOT replace LightweightTraversabilityNet)
       ▼
[Phase 14.3: End-to-End Vision-in-the-Loop Costmap Integration]
       │ (Fuse camera traversability into WorldTerrainMap; remove static rock hack)
       ▼
┌──────────────────────────────────────────────┐
│ Milestone 14.4: GPS-Denied Localization Fusion│ (Parallelizable research stream;
│ (Stereo VO + IMU + EKF in simulation)         │  can proceed alongside 14.2/14.3)
└──────────────────────────────────────────────┘
       │
       ▼
[Phase 14.5: Long-Horizon Multi-Waypoint Benchmark (~100m)]
       │ (Simulate large outdoor course with terrain, boulders, and waypoints)
       ▼
[Phase 14.6: Embedded Edge Deployment (Jetson / TensorRT)]
       (BLOCKED until physical target hardware is provisioned)
```

### Rationale:
1. **14.1 does not depend on 14.2, 14.3, or 14.4**: The mission executive manages topological waypoints and high-level states. It operates entirely above the local planner and can be validated immediately using Gazebo odometry and existing DWA/Recovery.
2. **14.2 must be re-scoped**: Developing a new neural network (Fast-SCNN) is an anti-pattern when `LightweightTraversabilityNet` already achieves ~94.8 ms on CPU and ~14 ms on GPU. The real flaw is the uncalibrated `cv2.resize` projection.
3. **14.4 is an independent estimator stream**: Localization drift does not prevent building the mission executive or closing the camera costmap loop in simulation.

---

## 14. Phase 14 Blockers

The following items are **HARD BLOCKERS** for specific subsequent milestones, but **NONE** block Phase 14.1:

| Blocker ID | Description | Impacted Milestone | Solution Required |
|---|---|---|---|
| **BLK-01** | Missing Calibrated IPM (Perspective-to-BEV) transformation | Milestone 14.2, 14.3 | Implement camera ray-casting or homography projection in `PerceptionAdapter`. |
| **BLK-02** | Active costmap decoupled from camera inference in live ROS 2 node | Milestone 14.3 | Connect `PerceptionNode` output to `SpatialFusionEngine.fuse_local_costmap()`. |
| **BLK-03** | Dynamic obstacles telemetry-dependent (no vision detector) | Milestone 14.3 | Integrate a lightweight 2D/3D detector (e.g. YOLO/MobileNet-SSD) on camera frames. |
| **BLK-04** | No active EKF node fusing VIO + IMU + Wheel Odom in ROS 2 graph | Milestone 14.4 | Configure `robot_localization` EKF node or custom state estimator. |
| **BLK-05** | Absence of 100m outdoor simulation world SDF | Milestone 14.5 | Construct procedural or mesh-based 100m trail SDF world in Gazebo. |
| **BLK-06** | Absence of physical NVIDIA Jetson Orin edge hardware | Milestone 14.6 | Provision physical Jetson hardware or restrict 14.6 to containerized mock validation. |

**There are ZERO blockers for Phase 14.1.**

---

## 15. Phase 14 Risks

1. **Wheel Slip & Odometry Drift (Medium)**: In outdoor terrain, differential drive wheel slip causes significant dead-reckoning drift. If 14.4 is delayed, multi-waypoint navigation on long courses in simulation must inject simulated odometry drift to test mission tolerance.
2. **Perception False Positives in Outdoor Shadows (Medium)**: Natural outdoor environments feature high-contrast shadows. If traversability masks classify shadows as non-traversable, the UGV may trigger unnecessary deadlocks. Safety hysteresis in `SpatialFusionEngine` must filter single-frame anomalies.
3. **Computational Resource Contention (Low/Medium)**: Running Gazebo Sim + RViz2 + PyTorch Perception + C++ DWA simultaneously on the Intel Core Ultra 7 155H requires thread partitioning (OpenMP threads for DWA, separate thread for perception) to prevent frame drops in the 20 Hz command loop.

---

## 16. Recommended Phase 14.1 Scope

### BOUNDED SPECIFICATION FOR PHASE 14.1

#### 16.1 Exact Objective
Construct the high-level **Mission Executive & Sequential Waypoint Manager** in `src/ugv_mission/` and expose it to the ROS 2 navigation stack. It will:
- Load and parse sequential waypoint mission files (YAML/JSON: $[W_1, W_2, \dots, W_n]$).
- Track waypoint acceptance radii (e.g., $0.4\text{ m}$ radius arrival check).
- Step through waypoints sequentially, passing the active target waypoint to the global replanner (`EventDrivenGlobalReplanner`).
- Implement the formal Mission State Machine:
  - `IDLE` → `INITIALIZING` → `NAVIGATING_TO_WAYPOINT` → `WAYPOINT_REACHED` → `RECOVERY` → `MISSION_COMPLETE` / `MISSION_ABORTED`.
  - Handle stuck/deadlock events by orchestrating the existing `RecoveryController` without duplicating recovery logic.
  - Abort gracefully to safe stop if maximum recovery attempts are exceeded or path is permanently blocked.

#### 16.2 Exact Files / Modules Expected to Change
- **New Files to Create**:
  - `src/ugv_mission/mission_manager.py` (Core mission executive & waypoint progression engine).
  - `src/ugv_mission/mission_state.py` (Mission state enums, contracts, data models).
  - `src/ugv_mission/waypoint.py` (Waypoint definitions, tolerance checks, attributes).
  - `tests/test_mission_executive.py` (Unit tests for waypoint sequencing, state transitions, timeout handling).
  - `tests/test_mission_recovery_integration.py` (Integration tests connecting `MissionExecutive` with `RecoveryController` and `UnifiedDWAPlanner`).
- **Existing Files to Adapt (Additive & Isolated Only)**:
  - `src/ugv_mission/__init__.py` (Expose public mission classes).
  - `simulation/nodes/interactive_ugv_navigation_node.py` (Add optional multi-waypoint mission mode parameter while preserving single-goal judge demo behavior).

#### 16.3 Exact Interfaces
- Inputs:
  - Current UGV pose: `(x, y, yaw)` from odometry / localization.
  - Current planner status: `PlannerStatus` from `UnifiedDWAPlanner`.
  - Recovery state: `RecoveryState` from `RecoveryController`.
- Outputs:
  - Active goal pose: `(target_x, target_y)` to `EventDrivenGlobalReplanner`.
  - Mission status: `MissionState`, active waypoint index, distance to waypoint, total progress %.
  - Safe abort command: Dispatches safe stop to `SafeCommandBuffer` if mission aborts.

#### 16.4 Exact Acceptance Criteria
1. **Sequential Progression**: Successfully sequences through a 5-waypoint course ($W_1 \to W_2 \to W_3 \to W_4 \to W_5$) in SIL simulation, transitioning to $W_{i+1}$ only when within acceptance radius ($0.4\text{ m}$).
2. **Deadlock Recovery Handoff**: When DWA/Safety reports deadlock near an obstacle at $W_i$, the executive enters `MISSION_RECOVERY`, allows `RecoveryController` to execute safe reverse, triggers replan, and resumes navigation toward $W_i$.
3. **Safe Abort Under Permanent Blockage**: If an obstacle permanently encloses a waypoint and recovery cycles exceed threshold (e.g. 3 attempts), the executive enters `MISSION_ABORTED` and brings the UGV to a complete, latched halt.
4. **Zero Phase 13 Regressions**: All 800+ existing tests must pass with 0 regressions. DWA C++ binaries and contracts remain untouched.

#### 16.5 What Is Explicitly OUT OF SCOPE for Phase 14.1
- **NO** changes to C++ DWA mathematics or binaries (`ai/planning/native_cpp/`).
- **NO** modifications to `SafetyGovernor` or `RecoveryController` algorithms.
- **NO** new neural network training or architecture changes (No Fast-SCNN / MobileNetV3-UNet).
- **NO** inverse perspective mapping (IPM) implementation (deferred to 14.2).
- **NO** vision-based dynamic obstacle detector training (deferred to 14.3).
- **NO** `robot_localization` EKF fusion or stereo VIO integration (deferred to 14.4).
- **NO** embedded Jetson Orin deployment (deferred to 14.6).

---

## 17. "Do Not Start Yet" Conditions

Phase 14.1 must **NOT** begin until:
1. The user explicitly reviews this audit report (`reports/PHASE_14_READINESS_AUDIT.md`) and authorizes the bounded Phase 14.1 scope.
2. Working tree remains clean on branch `demo-01-interactive-ugv`.
3. The scope boundaries outlined in Section 16 are formally acknowledged.

Subsequent milestones (14.2–14.6) must **NOT** begin until Phase 14.1 is fully implemented, tested, and accepted.

---

## 18. Final Decision

### Final Classification:
# **`READY_TO_START_PHASE_14_1_ONLY`**

### Summary Evidence:
- **Phase 13 Baseline**: Frozen at commit `c748fb8`, operating with verified C++ AVX2 DWA (~1ms), authoritative safety governor, latched hazard memory, and safe command buffering.
- **Why NOT `READY_TO_START_PHASE_14` (Full)**: Milestones 14.2–14.6 have critical prerequisites missing (calibrated IPM projection, camera costmap wiring, real vision obstacle detection, active EKF node, large outdoor world, and physical Jetson hardware).
- **Why NOT `BLOCKED_PENDING_PREREQUISITES`**: Phase 14.1 has **zero** unmet dependencies. All lower-level controllers (`UnifiedDWAPlanner`, `RecoveryController`, `AStarPlanner`, `SafeCommandBuffer`) exist, are fully tested, and provide clean decoupled interfaces for a mission executive.
- **Why `READY_TO_START_PHASE_14_1_ONLY`**: Milestone 14.1 is self-contained, purely algorithmic, fully executable on the dev machine, and directly extends the repository's mission capabilities while strictly preserving Phase 13 safety guarantees.
