# SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (UGV) in Outdoor Environments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy%20%2F%20Humble-orange.svg)](https://docs.ros.org/)
[![Gazebo](https://img.shields.io/badge/Simulation-Gazebo%20Harmonic-blueviolet.svg)](https://gazebosim.org/)
[![Compute](https://img.shields.io/badge/Compute-CPU%20%2B%20NVIDIA%20A100-purple.svg)](https://nvidia.com)
[![Tests](https://img.shields.io/badge/Tests-461%20Passed%20(100%25)-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Phase%2013.12-Complete%20(Baseline%2045397f8)-blue.svg)](docs/PHASE_13_MASTER_CONTEXT.md)

A production-grade, camera-first, GPS-denied autonomous robotics stack for outdoor Unmanned Ground Vehicles (UGVs) developed for the **Smart India Hackathon (SIH 2026)**.

---

## 📌 Project Overview

This repository houses an end-to-end autonomous navigation system designed to traverse semi-structured and unstructured outdoor environments (dirt roads, trails, tall vegetation, rocky inclines, water reflections) strictly without GPS signals.

The stack unifies deep semantic traversability estimation, visual-inertial odometry, dynamic spatial costmaps, predictive Dynamic Window Approach (DWA) local planning, event-driven global replanning ($A^*$), dynamic obstacle tracking/prediction, and fail-safe behavioral arbitration, cleanly packaged behind native **ROS 2** interfaces.

> [!IMPORTANT]
> **NATIVE ROS 2 EXECUTION STATUS:**
> Architecture and ROSIDL integration are implemented and structurally tested. Native ROS 2 execution/Gazebo validation is performed separately on the Linux runtime environment and remains a runtime-validation task unless explicitly marked validated by the runtime report. See [docs/ROS2_RUNTIME_INTEGRATION_PLAN.md](docs/ROS2_RUNTIME_INTEGRATION_PLAN.md).

---

## 🚀 Core Architectural Capabilities

1. **Camera-First Traversability Intelligence (Phases 1–4):**
   - Deep semantic terrain segmentation (traversable trail, rough soil, vegetation, water hazards, obstacles).
   - Negative obstacle (ditch, rut, drop-off) and positive obstacle (boulder, trunk, fence) detection.
   - Continuous cost formulation penalizing terrain roughness, slope, and optical uncertainty without ground-truth cheating.

2. **GPS-Denied Visual Odometry & Spatial Fusion (Phases 8–10):**
   - Stereo/monocular visual-inertial state estimation providing drift-mitigated metric pose estimates.
   - Recursive Bayesian spatial fusion projecting camera-frame perceptions into a global rolling/static `WorldTerrainMap`.
   - Complete TF2 coordinate frame tree (`odom -> base_link -> camera_link -> camera_optical`) with sub-millisecond lookup latency.

3. **Terrain-Aware Global & Local Planning (Phase 12):**
   - Global $A^*$ pathfinding with clearance penalties, octile heuristics, and footprint-validated path smoothing.
   - Local Dynamic Window Approach (DWA) evaluating exact rectangular vehicle footprints ($0.90\,\text{m} \times 0.65\,\text{m}$).
   - Hierarchical safety supervisory controller enforcing emergency stops ($d < 0.35\,\text{m}$) and speed clamping ($0.35\,\text{m} \le d < 0.60\,\text{m}$).

4. **Dynamic Obstacle Tracking & Predictive Replanning (Phase 13.1–13.12):**
   - Multi-frame data association via Hungarian bipartite matching and Kalman state filtering (constant-velocity and coordinated-turn models).
   - Future trajectory prediction over $1.0\text{--}3.0\,\text{s}$ horizons with time-growing positional covariance ellipses.
   - Dynamic spatio-temporal reservation grids (`DynamicOccupancyLayer`) separate from static terrain maps.
   - Predictive DWA with swept-segment broad-phase filtering and time-parameterized footprint collision checking.
   - Four-state behavioral arbitration finite state machine (`PROCEED`, `AVOID`, `WAIT`, `STOP`).
   - Event-driven global replanning triggered by persistent blockages, corridor occlusions, and dynamic deadlocks.

5. **Decoupled ROS 2 Architecture & Native ROSIDL Interfaces (Phase 13.12):**
   - Standalone `ugv_navigation_msgs` pure interface package providing 13 canonical `.msg` definitions.
   - Isolated bidirectional conversion layer (`dynamic_converters.py`) keeping all mathematical and AI models 100% ROS-independent.
   - Unified launch orchestrator (`dynamic_navigation.launch.py`) with validated QoS contracts and topic ownership.

6. **Dynamic SIL Scenario Validation (Phase 13.13 Revalidated):**
   - Comprehensive Software-in-the-Loop closed-loop validation revalidated under canonical termination semantics.
   - Strict ground-truth isolation, zero simulated collisions, full-horizon goal arrival validation, and legacy evidence separation.
   - Full quantitative characterization documented in [docs/PHASE_13_13_SIL_RESULTS.md](docs/PHASE_13_13_SIL_RESULTS.md).

---

## 📁 Repository Structure

```
SIH2026/
├── ai/                                # Pure Mathematical & Algorithmic Core (ROS-Independent)
│   ├── datasets/                      # Dataset ingestion (RUGD, RELLIS-3D)
│   ├── models/                        # Semantic segmentation backbones
│   ├── obstacles/                     # Explicit positive & negative obstacle perception
│   ├── traversability/                # Continuous traversability scoring & risk evaluation
│   ├── costmap/                       # BEV local rolling occupancy costmaps
│   ├── localization/                  # GPS-denied visual odometry & SLAM
│   ├── spatial_fusion/                # WorldTerrainMap spatial fusion engine
│   └── planning/                      # Planning subsystem
│       ├── global_planners/           # A*, Hybrid A*, PF-RRT*
│       ├── local_planners/            # Predictive DWA, Pure Pursuit
│       └── dynamic/                   # Phase 13 Dynamic Obstacle & Prediction Core
│           ├── data_model.py          # Track, Prediction, & Obstacle models
│           ├── tracker.py             # Hungarian association & Kalman tracker
│           ├── prediction.py          # Constant-velocity / Turn-rate motion predictor
│           ├── occupancy_layer.py     # Spatio-temporal dynamic occupancy slices
│           ├── predictive_collision.py# Time-parameterized swept-footprint checker
│           ├── dwa_filter.py          # Continuous swept-segment broad-phase filter
│           ├── behavioral_state.py    # PROCEED / AVOID / WAIT / STOP State Machine
│           └── global_replanner.py    # Event-driven A* trigger & replan evaluator
├── ros2_ws/src/                       # ROS 2 Workspace (Colcon Build Tree)
│   ├── ugv_navigation_msgs/           # Pure ROSIDL Interface Package (ament_cmake)
│   │   ├── msg/                       # 13 Canonical ROS2 .msg interface definitions
│   │   ├── CMakeLists.txt             # rosidl_generate_interfaces configuration
│   │   └── package.xml
│   └── ugv_navigation/                # Application Package (ament_python)
│       ├── ugv_navigation/
│       │   ├── nodes/                 # 7 dynamic nodes + base navigation nodes
│       │   ├── converters/            # dynamic_converters.py (Boundary Converter Layer)
│       │   ├── adapters/              # DynamicNavigationAdapter, PlannerAdapter
│       │   ├── tf/                    # TF2 Tree buffer & broadcaster
│       │   └── diagnostics/           # Node diagnostics & performance trackers
│       ├── config/                    # ugv_params.yaml, qos_profiles.yaml
│       ├── launch/                    # dynamic_navigation.launch.py, ugv_core.launch.py
│       ├── package.xml
│       └── setup.py                   # 13 console_scripts entrypoints
├── simulation/                        # Gazebo Simulation Subsystem
│   ├── models/ugv/                    # 4WD UGV URDF, SDF, and mesh descriptions
│   ├── worlds/                        # outdoor_terrain.sdf (elevation, rocks, lighting)
│   ├── config/                        # bridge_params.yaml (ros_gz parameter bridge)
│   └── launch/                        # ugv_simulation.launch.py
├── matlab/                            # Analytical Verification & Math Ground-Truth
│   ├── calibration/                   # Camera calibration verification
│   ├── geometry/                      # Parity checks for rigid SE(3) transforms
│   └── docs/                          # MATLAB validation reports (discrepancy < 10^-12)
├── configs/                           # Central system YAML configurations
├── docs/                              # Project specifications, contexts, and plans
├── tests/                             # Automated test suite (461 tests passing)
├── pyproject.toml                     # Python packaging configuration
└── README.md
```

---

## 🛠️ Quickstart & Local Verification

### 1. Python Environment Setup
```powershell
# Clone and enter directory
cd c:\Users\shres\Desktop\SIH2026

# Install editable package with development dependencies
pip install -e .
```

### 2. Run Comprehensive Test Suite
The entire repository is validated by an extensive test suite verifying algorithm purity, coordinate transforms, ROS 2 conversion boundaries, and scenario integration:

```powershell
# Run the complete test suite (461 tests)
pytest -v tests/

# Run Phase 13 dynamic obstacle core tests (239 tests)
pytest -v tests/test_dwa_dynamic.py tests/test_behavioral_state.py tests/test_global_replanning.py

# Run ROS 2 boundary, interface, and topic contract tests (180 tests)
pytest -v tests/test_dynamic_converters.py tests/test_ros2_topic_contract.py tests/test_ros2_node_graph.py tests/test_phase13_ros2_end_to_end.py
```

---

## 📊 Development Roadmap & Status

| Phase | Subsystem Description | Status | Verification Evidence |
| :---: | :--- | :---: | :--- |
| **Phase 0** | Project Audit, Foundation & Hybrid Compute Strategy | **Complete** | Scaffolding, types, device detection |
| **Phase 1** | Computer Vision Preprocessing (CLAHE, Horizon, Texture) | **Complete** | $>30\,\text{FPS}$ CPU image preprocessing |
| **Phase 2** | Outdoor Dataset Ingestion & Augmentation (RUGD, RELLIS) | **Complete** | Multi-threaded PyTorch streaming loaders |
| **Phase 3** | Traversability Foundation & SAM-TP Neural Integration | **Complete** | False-safe pixel ratio $< 3\%$, 81 tests |
| **Phase 4** | Explicit Obstacle, Ditch & Negative Hazard Detection | **Complete** | False-safe water error reduced to $2.73\%$, 101 tests |
| **Phase 5** | 2D BEV Rolling Terrain Costmap & Temporal Fusion | **Complete** | $< 2\,\text{ms}$ cycle, zero hazard dilution, 117 tests |
| **Phase 6** | Global $A^*$ & Local DWA Terrain-Aware Motion Planning | **Complete** | $100\%$ route completion, emergency stop, 134 tests |
| **Phase 7** | MATLAB Mathematical Parity & Analytical Ground-Truth | **Complete** | Parity error $< 10^{-12}\,\text{m}$, 9 experiments |
| **Phase 8** | GPS-Denied Visual Odometry, Stereo VIO & SLAM | **Complete** | $96\,\text{FPS}$ stereo tracking, frozen API, 153 tests |
| **Phase 9** | Perception + Localization World Spatial Fusion | **Complete** | `WorldTerrainMap` engine, zero-trust safety, 188 tests |
| **Phase 10** | ROS 2 Architecture, Custom Messages & TF2 Tree | **Complete** | Canonical TF2 tree, sub-ms lookups, 205 tests |
| **Phase 11** | Gazebo Outdoor World & 4WD UGV Simulation | **Complete** | Gazebo Harmonic world, ros_gz bridge, 221 tests |
| **Phase 12** | Closed-Loop Navigation, DWA Planning & Safety Clamps | **Complete** | 8 deterministic scenarios, 4 SIL missions, 250 tests |
| **Phase 13** | Dynamic Obstacle Tracking & Predictive Replanning | **Complete** | **M13.1–M13.12 Complete** (461/461 tests passing) |
| **Phase 14** | GPS-Denied Full Long-Distance Mission Execution | Planned | Multi-waypoint outdoor state machine |
| **Phase 15** | Advanced Terrain-Aware Navigation (Slip/Slope/Roll) | Planned | 3D pitch/roll compensation & terrain slip modeling |
| **Phase 16** | Full System Benchmarking & SIH Demonstration | Planned | Standardized 10-scenario demonstration dashboard |

---

## 👥 Team Collaboration & Branching Strategy

To facilitate parallel development without merge conflicts:
- **Baseline Git Commit:** `45397f8`
- **Baseline Git Tag:** `phase-13.12-complete`

### Work Streams:
1. **Algorithm Development Stream (Lead):** Continues on `main` / `feature/phase-13-algorithms` developing advanced multi-agent tracking and Phase 14 waypoint mission sequencing.
2. **ROS 2 Runtime Integration Stream (Teammate):** Branches from `phase-13.12-complete` to `feature/ros2-runtime-jazzy` on an Ubuntu 24.04 Linux workstation to execute native builds, DDS networking, Gazebo Harmonic simulation, and closed-loop runs.

---

## 📚 Technical Documentation Index

- [docs/PHASE_13_MASTER_CONTEXT.md](docs/PHASE_13_MASTER_CONTEXT.md) — **Single Source of Truth for Phase 13** (algorithms, math models, ROSIDL interfaces, contracts, test evidence).
- [docs/ROS2_RUNTIME_INTEGRATION_PLAN.md](docs/ROS2_RUNTIME_INTEGRATION_PLAN.md) — **ROS 2 Teammate Runtime Guide** (gated sequence ROS2-R0 to ROS2-R14, Gazebo Harmonic integration, commands, and reporting templates).
- [docs/ROADMAP.md](docs/ROADMAP.md) — Comprehensive technical roadmap across all project phases.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture, pipeline flows, and component responsibilities.
- [docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) — Current repository audit, pass rates, and environment status.
- [docs/PHASE_12_PATH_PLANNING.md](docs/PHASE_12_PATH_PLANNING.md) — Phase 12 static planning, collision checking, and SIL simulation baseline.
- [docs/PHASE_10_ROS2_ARCHITECTURE.md](docs/PHASE_10_ROS2_ARCHITECTURE.md) — Foundational ROS 2 architecture and TF2 integration.

---

## ⚠️ Known Limitations & Disclosures

1. **Host Native Toolchain Limitation:** The current development machine is Windows; native `colcon build`, native `rclpy` binaries, and DDS daemons are unverified on Windows and are scheduled for validation on Ubuntu 24.04 per [docs/ROS2_RUNTIME_INTEGRATION_PLAN.md](docs/ROS2_RUNTIME_INTEGRATION_PLAN.md).
2. **Pure Python DWA Latency:** The unoptimized Python implementation of DWA evaluates multi-horizon trajectory rollouts in interpreted loops ($1.36\,\text{s}\text{--}4.5\,\text{s}$ latency). Production deployment will target C++/Cython acceleration in subsequent milestones.
3. **Visual Odometry Scale Factor:** The monocular visual odometry pipeline utilizes an empirical scale compensation factor ($1.167$) calibrated to the specific simulated camera mount.
4. **Physical Hardware Demonstration:** All tests to date are evaluated in high-fidelity Software-In-The-Loop (SIL) simulation and unit/integration harnesses. Physical UGV outdoor trials are scheduled for Phase 16.
