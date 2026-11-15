# SIH26126 — Vision-Based Autonomous Navigation for Unmanned Ground Vehicle (UGV) in Outdoor Environments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B17-AVX2%20SIMD%20%2B%20OpenMP-00599C.svg)]()
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-INT8%20%2F%20FP32-blueviolet.svg)]()
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy%20%2F%20Humble-orange.svg)](https://docs.ros.org/)
[![Gazebo](https://img.shields.io/badge/Simulation-Gazebo%20Harmonic-purple.svg)](https://gazebosim.org/)
[![Tests](https://img.shields.io/badge/Automated_Tests-177%2F177%20PASS-brightgreen.svg)](reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md)
[![SIH_Benchmark](https://img.shields.io/badge/SIH_10--Scenario_Suite-10%2F10%20PASS%20(100%25)-brightgreen.svg)](reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md)

A production-grade, **camera-first, GPS-denied, 2.5D terrain-aware autonomous navigation stack** for outdoor Unmanned Ground Vehicles (UGVs), developed for **Smart India Hackathon (SIH 2026 — Problem Statement SIH26126)**.

---

## 🎯 Executive Summary & Key Performance Highlights

This repository delivers a complete **17-Phase (`Phase 0` to `Phase 16`)** autonomous navigation system engineered to traverse unstructured outdoor trails, steep embankments ($32^\circ$), lateral cross-slopes ($24^\circ$), vertical rock steps ($+20\,\text{cm}$), negative ditches ($-15\,\text{cm}$), high-centering boulder mounds, harsh sun glare ($+60\%$), deep tree shadows, and moving obstacles **strictly without GPS**.

| Subsystem | Architecture & Engineering Breakthrough | Measured Performance |
| :--- | :--- | :--- |
| **Photometric & Semantic Perception** | Adaptive CIE-LAB CLAHE + Specular Glare/Shadow Confidence Filter ($C_{\text{photo}}$) + Mixed-Precision **INT8 ONNX** `LightweightTraversabilityNet` + `AsyncPerceptionWorker` | **`0.92 ms`** CLAHE, **`12.52 ms`** INT8 ONNX, **`0.0%`** glare/shadow false obstacles |
| **2.5D Terrain Intelligence** | `ElevationMapper` (Kalman/EMA height fusion) + `VectorizedTerrainGeometryEstimator` (Normals $\hat{\mathbf{n}}$, Slope $\alpha$, Steps $\Delta Z^+$, Ditches $\Delta Z^-$, Belly Crown $h_{\text{crown}}$) + Dynamic **SE(3) Attitude IPM** | **`0.99 ms`** elevation, **`1.86 ms`** 2.5D geometry, **`1.61 ms`** SE(3) IPM ($\le 0.07\,\text{m}$ tilt drift) |
| **GPS-Denied 3D Localization** | 15-State `MultiSensorEKF` fusing Wheel Odometry + 200 Hz 6-Axis IMU (3D Euler $E(\phi,\theta)$ + Gravity Tilt Observability) + Stereo VIO with dead-reckoning fallback | **`0.01%–0.025%` drift** over $100\text{--}1,280\,\text{m}$; **`< 0.03 m` drift** during 3s VIO outage |
| **Global & Local Motion Planning** | **Native C++17 Global $A^*$** (`astar_cpp.dll`/`.so` with JPS & switchback-preserving smoothing) + **Native C++17 AVX2 DWA** (`dwa_cpp.dll`/`.so`, 375 rollouts, 8-axis SAT, Rollover Margin $\mathcal{M}_{\text{roll}} \ge 0.20$) | **`1.27 ms`** C++17 $A^*$ ($252\times$ speedup), **`1.32 ms`** C++17 AVX2 DWA ($4,298\times$ speedup) |
| **Safety Governance & Recovery** | Downhill-Compensated `AuthoritativeSafetyGovernor` ($d_{\text{stop}}(\theta)$ expansion, $18^\circ/26^\circ$ IMU rollover clamp) + `RecentHazardMemory` (`"NOT DETECTED != SAFE"`) + Reverse-and-Turn `RecoveryController` + 20 Hz `SafeCommandBuffer` | **`0` collisions, `0` rollovers, `0` high-centerings** across all 10 SIH Grand Finale scenarios |

---

## 🏗️ End-to-End Unified System Architecture

```text
+---------------------------------------------------------------------------------------------------------------+
|                     PHASE 16: UNIFIED SIH26126 GRAND FINALE AUTONOMOUS NAVIGATION STACK                       |
|                                                                                                               |
|  [Input Mode A: Gazebo Sim 100m 2.5D Uneven World]   [Input Mode B: Live USB Webcam / Stereo HIL Table Demo]  |
|    ├── RGB Stream (/camera/image_raw @ 30 Hz)         ├── OpenCVCameraSource / SyntheticCameraSource          |
|    ├── Metric DepthSource / 2.5D Terrain Raycast      └── Interactive IMU Tilt / Obstacle Injection Controls  |
|    └── 200 Hz IMU (/imu) + Wheel Odometry (/odom)                                                             |
|                           │                                                   │                               |
|                           └─────────────────────────┬─────────────────────────┘                               |
|                                                     ▼                                                         |
|                [Photometric Resilience Preprocessor (production/perception/photometric_resilience.py)]        |
|                  ├── Adaptive LAB-CLAHE Illumination Normalization (< 1.0 ms, +/-60% Glare/Shadow Recovery)   |
|                  ├── Specular Glare & Cast Shadow Confidence Map C_photo(u,v) in [0.1, 1.0]                   |
|                  └── Frame Blackout / Camera Dropout Watchdog (Triggers 0.35 m/s Degraded Speed Clamp)        |
|                                                     │                                                         |
|                           ┌─────────────────────────┴─────────────────────────┐                               |
|                           ▼                                                   ▼                               |
|        [AsyncPerceptionWorker + ONNX INT8/FP32]                 [15-State MultiSensorEKF (50-200 Hz)]         |
|          ├── Traversability τ(u, v) in ~12.5 ms                   ├── 3D Pose (x, y, z) & Vel (vx, vy, vz)    |
|          └── Dynamic SE(3) Attitude IPM (1.61 ms)                 ├── 3D Euler Attitude (roll φ, pitch θ, ψ)  |
|                           │                                       └── VIO-Dropout Dead-Reckoning Fallback     |
|                           └─────────────────────────┬─────────────────────────┘                               |
|                                                     ▼                                                         |
|                    [2.5D ElevationMapper + VectorizedTerrainGeometryEstimator (< 2.9 ms)]                     |
|                      ├── Kalman/EMA Elevation Z(x, y) & Flying-Pixel Spike Rejection                          |
|                      ├── Boundary-Aware Normals [nx, ny, nz] & Slope α(x, y)                                  |
|                      ├── Positive Steps ΔZ+ >= 0.12m, Negative Ditches ΔZ- <= -0.12m                          |
|                      └── Undercarriage High-Centering Crown Guard h_crown >= 0.12m                            |
|                                                     │                                                         |
|            ┌────────────────────────────────────────┼───────────────────────────────────────┐                 |
|            ▼                                        ▼                                       ▼                 |
|  [Terrain-Constrained C++17 A*]         [Rollover-Aware C++17 AVX2 DWA]         [Authoritative Safety         |
|   ├── 1.27 ms mean latency               ├── 375 rollouts in 1.32 ms             Governor & Recovery]         |
|   ├── Detours >28° embankments           ├── Rollover Margin >= 0.20             ├── Downhill d_stop(θ)       |
|   └── Preserves 10° switchbacks          ├── v_eff <= v_max * cos(α)             ├── 18°/26° IMU Tilt Clamp   |
|                                          └── Path-Normal Dynamic Evasion         └── Reverse-and-Turn Escape  |
|            │                                        │                                       │                 |
|            └────────────────────────────────────────┼───────────────────────────────────────┘                 |
|                                                     ▼                                                         |
|             [SafeCommandBuffer (/cmd_vel @ 20 Hz)] + [Judge RViz2 /telemetry/judge_dashboard HUD]             |
+---------------------------------------------------------------------------------------------------------------+
```

---

## 📊 10-Scenario SIH Grand Finale Benchmark Results

Reproducible via `python benchmarks/run_phase16_sih_grand_finale_benchmark.py` (telemetry sealed in [`reports/phase16_grand_finale_benchmark.json`](reports/phase16_grand_finale_benchmark.json) and [`reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md`](reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md)):

| # | Scenario ID | Outdoor Challenge Tested | Waypoints | FPS | C++17 A* (ms) | C++17 DWA (ms) | Control Loop (ms) | EKF Drift (%) | Min Rollover Margin | Collisions / Rollovers / High-Centerings | Result |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `S01_FLAT_MEADOW_BASELINE` | Benign flat trail baseline | `1/1` | `168.8` | `0.55` | `1.94` | `5.92` | `0.010%` | `1.000` | `0 / 0 / 0` | **PASS** |
| **2** | `S02_GENTLE_10DEG_TRAIL` | Continuous $10^\circ$ incline & descent | `1/1` | `227.3` | `1.51` | `1.25` | `4.40` | `0.010%` | `0.642` | `0 / 0 / 0` | **PASS** |
| **3** | `S03_STEEP_32DEG_EMBANKMENT` | $32^\circ$ embankment + $10^\circ$ switchback detour | `1/1` | `265.4` | `2.07` | `0.97` | `3.77` | `0.010%` | `0.631` | `0 / 0 / 0` | **PASS** |
| **4** | `S04_SEVERE_24DEG_CROSS_SLOPE` | $24^\circ$ lateral cross-slope ridge | `1/1` | `229.8` | `1.10` | `1.43` | `4.35` | `0.010%` | `0.988` | `0 / 0 / 0` | **PASS** |
| **5** | `S05_BOULDER_STEP_FIELD_18CM` | $+18\,\text{cm}$ vertical rock ledges ($\Delta Z^+ \ge 0.12\,\text{m}$) | `1/1` | `267.2` | `1.15` | `1.09` | `3.74` | `0.010%` | `0.988` | `0 / 0 / 0` | **PASS** |
| **6** | `S06_NEGATIVE_DITCH_TRENCH_15CM` | $-15\,\text{cm}$ sunken trench ($\Delta Z^- \le -0.12\,\text{m}$) | `1/1` | `135.9` | `1.79` | `3.38` | `7.36` | `0.010%` | `0.988` | `0 / 0 / 0` | **PASS** |
| **7** | `S07_BELLY_CROWN_MOUND_14CM` | $+14\,\text{cm}$ high-centering mound ($h_{\text{crown}} \ge 0.12\,\text{m}$) | `1/1` | `224.8` | `0.66` | `1.59` | `4.45` | `0.010%` | `0.988` | `0 / 0 / 0` | **PASS** |
| **8** | `S08_UNDULATING_HILLS_ATTITUDE_IPM` | $\pm 12^\circ$ pitching hills + SE(3) attitude IPM | `1/1` | `172.3` | `1.40` | `1.15` | `5.80` | `0.010%` | `0.583` | `0 / 0 / 0` | **PASS** |
| **9** | `S09_ADVERSE_GLARE_SHADOW_DROPOUT` | $+60\%$ sun glare, shadow bands & 3s VIO dropout | `1/1` | `279.2` | `0.69` | `1.43` | `3.58` | `0.025%` | `0.999` | `0 / 0 / 0` | **PASS** |
| **10** | `S10_FULL_100M_5WP_GRAND_FINALE` | Full 100m 5-waypoint course ($W_1\to W_5$) + 5 moving obstacles | `5/5` | `111.0` | `1.74` | `1.38` | `9.01` | `0.010%` | `0.682` | `0 / 0 / 0` | **PASS** |
| **Total** | **10 Canonical SIH Scenarios** | **All Outdoor Hazards Combined** | **`14/14`** | **`208.2`** | **`1.27`** | **`1.56`** | **`5.24`** | **`<=0.025%`** | **`>=0.583`** | **`0 / 0 / 0`** | **`100.0%` (10/10)** |

---

## 🚀 Quickstart & 1-Command Judge Runbook

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/umangkaushik17-bit/Vision-Based-Autonomous-Navigation-for-UGV.git
cd Vision-Based-Autonomous-Navigation-for-UGV
pip install -e .
```
*(Pre-compiled Windows `.dll` and Linux `.so` C++17 binaries for $A^*$ and DWA, along with INT8/FP32 ONNX and PyTorch weights, are included in the repository for immediate zero-compile execution. To recompile C++ binaries on a new CPU architecture, run `python scripts/build_astar_cpp.py && python scripts/build_dwa_cpp.py`.)*

### 2. Run the Automated Verification & Benchmark Suite (No Gazebo Required)
```bash
# Run the Phase 15 (2.5D Terrain) & Phase 16 (SIH Grand Finale) verification suites:
pytest tests/test_phase15_*.py tests/test_phase16_*.py -v

# Execute the 10-Scenario SIH Grand Finale Benchmark Runner directly:
python benchmarks/run_phase16_sih_grand_finale_benchmark.py
```

### 3. Run the Live Webcam / Stereo Hardware-in-the-Loop (HIL) Table Demo
```bash
# Live USB webcam GUI mode (hold real objects in front of camera or tilt camera):
python scripts/run_live_camera_hil_demo.py --live-webcam --camera-id 0

# Deterministic headless verification mode (30 frames):
python scripts/run_live_camera_hil_demo.py --headless --frames 30
```

### 4. Launch the 100m 2.5D Gazebo Harmonic + Hero RViz2 Simulation
```bash
# On ROS 2 Linux / WSL2 (Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic):
ros2 launch simulation/launch/prototype_presentation.launch.py \
  world:=simulation/worlds/outdoor_uneven_terrain_100m.sdf \
  enable_2_5d_terrain:=true \
  terrain_world_mode:=uneven_100m

# On Windows (1-click launchers):
launch_presentation.bat        # Autonomous 5-obstacle patrol mode
launch_presentation_wasd.bat   # Interactive WASD dynamic obstacle teleop mode
```

---

## 📁 Systematic Repository Structure

```text
Vision-Based-Autonomous-Navigation-for-UGV/
├── ai/                                # ROS-Independent Core AI, Geometry & Motion Planning
│   ├── costmap/                       # 2D BEV rolling terrain costmap & temporal fusion
│   ├── datasets/                      # RUGD & RELLIS-3D streaming dataset loaders
│   ├── localization/                  # Stereo Visual Odometry, VIO & loop closure
│   ├── models/                        # Semantic segmentation backbones
│   ├── obstacles/                     # Water reflection, dark obstacle & negative ditch detectors
│   ├── perception_fusion/             # Multi-factor 2.5D TraversabilityCostModel
│   ├── planning/                      # Motion planning subsystem
│   │   ├── native_cpp/                # C++17 AVX2 & Scalar A* (astar_core.cpp) & DWA (dwa_core.cpp)
│   │   ├── global_planners/           # AStarPlanner & CPPAStarBackend FFI
│   │   ├── local_planners/            # DWAPlanner & Pure Pursuit
│   │   ├── backends/                  # UnifiedDWAPlanner (CPP_SIMD, CUDA, INTEL_GPU, CPU)
│   │   ├── dynamic/                   # Hungarian/Kalman Tracker, CTRV Predictor, SAT Collision, FSM
│   │   ├── safety_governor.py         # Authoritative 3D tilt, downhill braking & hazard latch governor
│   │   └── recovery_controller.py     # Reverse-and-turn deadlock recovery controller
│   ├── spatial_fusion/                # 10-layer WorldTerrainMap & vectorized SE(3) fusion
│   └── traversability/                # LightweightTraversabilityNet & AnalyticalTerrainDepthSource
├── production/                        # Production Real-Time Perception, Localization & Safety
│   ├── localization/                  # 15-state MultiSensorEKF (ekf_fusion.py) & StereoVIOAdapter
│   ├── perception/                    # ElevationMapper, VectorizedTerrainGeometryEstimator,
│   │                                  # PhotometricResilienceProcessor, VisionCostmapIntegrator,
│   │                                  # AsyncPerceptionWorker & HILCameraNavigationBridge
│   ├── runtime/                       # RuntimeAdaptationManager compute-latency governor
│   └── safety/                        # AuthoritativeSafetyGovernor
├── src/                               # Core System & Mission Executive Packages
│   ├── ugv_core/                      # SystemConfig, device detection, types & logger
│   └── ugv_mission/                   # MissionExecutive, WaypointManager & RecoveryOrchestrator
├── simulation/                        # Gazebo Harmonic 100m Worlds, 4WD UGV Model & ROS 2 Nodes
│   ├── worlds/                        # outdoor_uneven_terrain_100m.sdf & outdoor_presentation_100m.sdf
│   ├── models/                        # ugv/ (URDF & SDF) + outdoor_assets/ (beacons, boulders, trees)
│   ├── nodes/                         # interactive_ugv_navigation_node.py & presentation_visualization_adapter.py
│   ├── launch/                        # prototype_presentation.launch.py
│   └── rviz/                          # hero_presentation.rviz (includes /telemetry/judge_dashboard)
├── ros2_ws/src/                       # Native ROS 2 Packages (Colcon Workspace)
│   ├── ugv_navigation_msgs/           # 13 canonical .msg definitions
│   └── ugv_navigation/                # ROS 2 nodes, converters, TF2 tree & zero_copy_ipc.py
├── benchmarks/                        # Automated Benchmark Harnesses (Phase 14.5, 14.7.4, 15, 16)
├── configs/                           # All 20 YAML configurations (terrain_safety.yaml, planning.yaml, etc.)
├── docs/                              # Consolidated Master Documentation (Phases 0–16 & Optimization)
├── reports/                           # Final Benchmark Reports & JSON Telemetry Artifacts
├── scripts/                           # Build scripts, ONNX quantization & run_live_camera_hil_demo.py
├── matlab/                            # Independent MATLAB (.m) Mathematical Parity Suite
└── tests/                             # Automated Pytest Suite (177/177 tests passing)
```

---

## 📚 Master Documentation & Reports Index

### Top-Level Master Documents (`docs/`)
- **[`docs/README_DOCS_INDEX.md`](docs/README_DOCS_INDEX.md)** — Complete Navigator for all files in `docs/` and `reports/`.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — Official 17-Phase Development Roadmap (`Phase 0` to `Phase 16` — All **COMPLETE & VERIFIED**).
- **[`docs/PHASES_0_TO_16_MASTER_ENCYCLOPEDIA.md`](docs/PHASES_0_TO_16_MASTER_ENCYCLOPEDIA.md)** — Unified Encyclopedia of all 17 Phases (`Phase 0` through `Phase 16`).
- **[`docs/OPTIMIZATION_AND_ACCELERATION_MASTER.md`](docs/OPTIMIZATION_AND_ACCELERATION_MASTER.md)** — Master Guide for Optimization Tiers 1–4 (`WS-1A` to `WS-4A`), C++17 SIMD, ONNX INT8, Zero-Copy IPC, and GPU Acceleration.
- **[`docs/PHASE_16_MASTER_IMPLEMENTATION_PLAN.md`](docs/PHASE_16_MASTER_IMPLEMENTATION_PLAN.md)** — Phase 16 Master Specification (`Subphases 16.0–16.5`).
- **[`docs/PHASE_15_MASTER_IMPLEMENTATION_PLAN.md`](docs/PHASE_15_MASTER_IMPLEMENTATION_PLAN.md)** — Phase 15 2.5D Terrain-Aware Navigation Specification (`Subphases 15.0–15.9`).
- **[`docs/PHASE_14_CONSOLIDATED_MASTER.md`](docs/PHASE_14_CONSOLIDATED_MASTER.md)** — Consolidated Phase 14 Reference (`Subphases 14.1–14.7.4`).
- **[`docs/PHASE_13_CONSOLIDATED_MASTER.md`](docs/PHASE_13_CONSOLIDATED_MASTER.md)** — Consolidated Phase 13 Reference (`Milestones 13.0–13.26`).

### Top-Level Benchmark Reports (`reports/`)
- **[`reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md`](reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md)** — **SIH Grand Finale Judge Dossier** (`10/10` scenarios PASS).
- **[`reports/PHASE_15_BENCHMARK_REPORT.md`](reports/PHASE_15_BENCHMARK_REPORT.md)** — **Phase 15 Differential 2.5D Terrain Benchmark Report** (`7/7` scenarios PASS).
- **[`reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md`](reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md)** — **Phase 14 100m Corridor Validation Report** (`12/12` runs PASS).
- **[`reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md`](reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md)** — **P0/P1 Architecture Hardening & C++/ROS 2 Optimization Report**.
