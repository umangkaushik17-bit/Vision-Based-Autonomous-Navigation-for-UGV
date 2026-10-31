# SIH26126 — Master Development Roadmap (Phases 0 to 16)

This roadmap outlines the systematic, phased implementation plan for the autonomous UGV navigation software.

> **CRITICAL RULE:** Never advance to a subsequent phase until the exit criteria of the current phase are fully validated and passing.

---

```
  [Phase 0: Setup & Audit]
             │
             ▼
  [Phase 1: Python CV Foundation]
             │
             ▼
  [Phase 2: Dataset & Perception Pipeline]
             │
             ▼
  [Phase 3: Traversability Model]
             │
             ▼
  [Phase 4: Obstacle Detection]
             │
             ▼
  [Phase 5: Costmap & Planner in Pure Python]
             │
             ▼
  [Phase 6: Visual Localization / SLAM Experiments]
             │
             ▼
  [Phase 7: MATLAB Validation & Benchmarking]
             │
             ▼
  [Phase 8: GPS-Denied Visual Localization & SLAM]
             │
             ▼
  [Phase 9: Perception + Localization Spatial Fusion]
             │
             ▼
  [Phase 10: ROS 2 Architecture & TF2 Integration]
             │
             ▼
  [Phase 11: Gazebo UGV Simulation]
             │
             ▼
   [Phase 12: Global + Local Path Planning]
              │
              ▼
   [Phase 13: Dynamic Obstacles & Predictive Planning (COMPLETE WITH DOCUMENTED LIMITATIONS)]
              │  (Milestones 13.0-13.19 Complete; Frozen)
              ▼
   [Phase 14: GPS-Denied Outdoor Navigation (Planned / Not Started)]
              │
              ▼
   [Phase 15: Advanced Terrain-Aware Navigation]
              │
              ▼
  [Phase 16: Full Benchmarking & SIH Demonstration]
```

---

## Phase Breakdown

### Phase 0: Project Audit and Foundation (Current Phase)
- **Goal:** Audit workspace, dependencies, establish clean modular scaffolding, configure hybrid CPU/A100 workflow, and document technical architecture.
- **Deliverables:** Directory structure, `pyproject.toml`, `requirements.txt`, `ugv_core` library, synthetic data generator, `docs/` technical specifications.
- **Exit Criteria:** All unit tests passing, clean git repository, dependencies identified and documented.

### Phase 1: Python Computer-Vision Foundation
- **Goal:** Build robust image preprocessing, edge detection, color spaces (HSV, LAB), texture filters, and adaptive illumination correction (CLAHE) for outdoor environments.
- **Deliverables:** `ugv_perception/preprocessing/` modules (denoising, shadow removal, adaptive thresholding, horizon detection).
- **Exit Criteria:** Preprocessing pipeline processes 640x480 frames at $>30\text{ FPS}$ on CPU with test coverage.

### Phase 2: Dataset and Perception Pipeline
- **Goal:** Ingest outdoor datasets (RUGD, RELLIS-3D, and synthetic feeds). Build standardized PyTorch DataLoaders with data augmentation (random lighting, blur, tilt).
- **Deliverables:** `ai/datasets/` loaders, annotation converters, visualization utilities.
- **Exit Criteria:** Fast streaming batch loaders with $<5\text{ms}$ batch overhead and data integrity verification.

### Phase 3: Traversability Foundation, SAM-TP Integration, GPU Adaptation & Terrain-Aware Perception (Completed)
- **Goal:** Operationalize pretrained SAM-TP Mini+ Traversability (SAM 2.1 Hiera-Tiny `checkpoint_finetuned_v2.pt`), evaluate multi-domain generalization, conduct false-safe risk analysis, build BEV projection with modular depth sources, and construct 3D terrain geometry abstractions (MoveBase3D/Nav4D).
- **Deliverables:** `ai/traversability/` production adapter, `ai/perception_fusion/` terrain geometry and cost models, BEV projection pipeline, failure analysis test harness, video benchmark suite, external GPU fine-tuning configs.
- **Exit Criteria:** Zero-leakage empirical evaluation on Synthetic, RUGD, and Own-real splits; False-Safe pixel ratio $< 3\%$; empirical execution of `from rover_traversability import TraversabilityPredictor`; 100% test pass rate across 81 unit tests.

### Phase 4: Explicit Obstacle & Hazard Perception + RUGD Model Comparison + Safety-Aware Perception (Completed)
- **Goal:** Import and audit the RUGD-trained baseline model (Model B), execute fair comparative evaluations against SAM-TP (Model A), implement explicit hazard and obstacle perception (neural detector + water reflection, low-dark obstacle, and negative ditch analyzers), enforce configurable hazard policy, and construct safety-aware perception fusion with first-class UNKNOWN handling.
- **Deliverables:** `scripts/import_rugd_model.py`, `ai/traversability/rugd_model.py`, `ai/traversability/comparison.py`, `ai/obstacles/` subsystem (`base.py`, `detector.py`, `model.py`, `terrain_hazards.py`, `postprocess.py`, `visualization.py`), `configs/hazard_policy.yaml`, `ai/perception_fusion/safety_fusion.py`, `scripts/evaluate_perception.py`, `scripts/benchmark_perception.py`.
- **Exit Criteria:** Measurable false-safe reduction on creek water reflections (reduced from 23.78% to 7.64%); RUGD domain false-safe area error reduced from 8.06% to 2.73%; 101/101 unit tests passing with zero regressions.

### Phase 5: BEV Terrain Costmap, Temporal Fusion & Planner-Ready Terrain Representation (Completed)
- **Goal:** Convert multi-modal perception outputs and 3D terrain geometry into a unified, robot-centric, terrain-aware, temporally stable, and planner-ready BEV Costmap (`BEVTerrainCostmap`).
- **Deliverables:** `ai/costmap/` subsystem (`base.py`, `bev_costmap.py`, `layers.py`, `fusion.py`, `temporal.py`, `motion_compensation.py`, `inflation.py`, `visualization.py`, `serialization.py`), `configs/costmap.yaml`, `scripts/build_costmap.py`, `scripts/benchmark_costmap.py`, `scripts/visualize_costmap.py`, `scripts/evaluate_costmap.py`.
- **Exit Criteria:** Zero false-safe BEV cells on water/hazard scenarios; $13.0\%$ temporal flicker reduction; $< 2\text{ ms}$ postprocessing and fusion cycle at $10\text{ cm}$ resolution ($115.5\text{ KB}$ memory footprint); 117/117 unit tests passing with zero regressions.

### Phase 6: Terrain-Aware Global & Local Path Planning + Planner Selection (Completed)
- **Goal:** Implement, benchmark, and select optimal terrain-aware global path planners (A*, PF-RRT*, Hybrid A*) and local trajectory planners (DWA, Pure Pursuit) operating purely on the canonical `BEVTerrainCostmap`.
- **Deliverables:** `ai/planning/` subsystem (`base.py`, `path.py`, `trajectory.py`, `collision.py`, `costs.py`, `validation.py`, `metrics.py`, `visualization.py`), `global_planners/` (`astar.py`, `pf_rrt_star.py`, `hybrid_astar.py`), `local_planners/` (`dwa.py`, `path_follower.py`), `configs/planning.yaml`, `scripts/plan_route.py`, `scripts/benchmark_global_planners.py`, `scripts/benchmark_local_planners.py`, `scripts/simulate_navigation.py`, `scripts/evaluate_planners.py`.
- **Exit Criteria:** A* selected as primary global planner ($100\%$ success rate on traversable scenarios, $16 - 59\text{ ms}$ latency); DWA selected as primary local planner (sub-millisecond evaluation, complete dynamic obstacle avoidance, verified zero-velocity emergency stop); Section 60 dynamic obstacle and Section 61 terrain detour experiments verified; 134/134 unit tests passing with zero regressions.

### Phase 7: MATLAB Validation, Camera Geometry Verification & Independent Benchmarking (Completed)
- **Goal:** Develop an independent, reproducible MATLAB analysis environment to cross-validate camera intrinsics/extrinsics, coordinate frames, BEV projection geometry, terrain estimation, costmaps, planner outputs, and trajectory tracking metrics, while establishing the ATE/RPE visual SLAM validation infrastructure for Phase 8.
- **Deliverables:** `matlab/` suite (`calibration/`, `geometry/`, `bev/`, `terrain/`, `costmap/`, `planning/`, `slamming/`, `metrics/`, `io/`, `utils/`, `tests/`, `scripts/`, `data/`, `results/`, `docs/`), `scripts/export_matlab_data.py`, `scripts/verify_matlab_math.py`, `tests/test_matlab_validation.py`, `matlab/docs/PHASE_7_MATLAB_VALIDATION.md`, `matlab/docs/SLAM_VALIDATION_INTERFACE.md`.
- **Exit Criteria:** All 9 required experiments (A through I) mathematically verified with zero discrepancies down to $10^{-12}$; ATE/RPE evaluation engine tested on synthetic perturbed trajectories; physical calibration and real SLAM properly marked `PENDING PHYSICAL DATA` / `PENDING PHASE 8`; 137/137 tests passing with zero regressions.

### Phase 8: GPS-Denied Visual Localization, Visual Odometry & SLAM (Completed)
- **Goal:** Implement, evaluate, and benchmark camera-first, GPS-denied visual odometry, VIO, and SLAM pipelines operating independently of ROS2, evaluating Monocular, Stereo, and Stereo+IMU configurations, and feeding the Phase 7 MATLAB ATE/RPE evaluation suite.
- **Deliverables:** `ai/localization/` subsystem (`base.py`, `trajectory.py`, `result.py`, `datasets.py`, `preprocessing.py`, `mono_vo.py`, `stereo_vo.py`, `visual_inertial.py`, `loop_closing.py`, `evaluation.py`, `diagnostics.py`), `configs/slam.yaml`, `scripts/run_slam.py`, `scripts/evaluate_slam.py`, `scripts/benchmark_slam.py`, `scripts/export_slam_trajectory.py`, `tests/test_localization.py`, `test_trajectory_contract.py`, `test_slam_adapter.py`, `test_slam_metrics.py`, `docs/PHASE_8_SLAM.md`, `docs/SLAM_HANDOFF.md`, `docs/SLAM_CONFIGURATION.md`, `docs/SLAM_ENVIRONMENT_AUDIT.md`.
- **Exit Criteria:** Stereo RGB (with optional IMU preintegration) selected as Primary Localization System (metric scale observable, $92-96\text{ FPS}$, $< 11\text{ ms}$ latency); full ATE/RPE quantitative evaluation verified; frozen `LocalizationEstimate` API established; 153/153 unit tests passing across repository with zero regressions.

### Phase 9: Perception + Localization Spatial Fusion (Completed)
- **Goal:** Build and validate the spatial fusion layer connecting Phase 4/5 perception-costmap outputs with Phase 8 visual-inertial localization estimates into a world-referenced, planar terrain map (`WorldTerrainMap`).
- **Deliverables:** `ai/spatial_fusion/` subsystem (`transforms.py`, `synchronization.py`, `world_map.py`, `fusion_engine.py`, `diagnostics.py`), `configs/spatial_fusion.yaml`, `scripts/evaluate_spatial_fusion.py`, `scripts/visualize_spatial_fusion.py`, `scripts/run_spatial_fusion.py`, `matlab/scripts/validate_phase_9_transforms.m`, `tests/test_coordinate_transforms.py`, `test_temporal_sync.py`, `test_world_map.py`, `test_spatial_fusion.py`, `test_phase_9_safety.py`, `docs/PHASE_9_SPATIAL_FUSION.md`.
- **Exit Criteria:** Rigid SE(3) transform parity with MATLAB ($< 10^{-12}\text{ m}$); deterministic safety precedence (zero hazard dilution); zero-trust tracking state protection (`LOST` strictly blocks fusion); $29.18\text{ ms}$ fusion cycle; $6.41\text{ MB}$ bounded memory footprint; 188/188 unit tests passing across repository with zero regressions.

### Phase 10: ROS2 Architecture + TF2 Integration (Completed)
- **Goal:** Convert the validated Python core into a clean, modular ROS2 package architecture and canonical TF2 coordinate frame tree (`odom` -> `base_link` -> `camera_link` -> `camera_optical`) without modifying core perception, localization, or spatial fusion algorithms.
- **Deliverables:** `ros2_ws/src/ugv_navigation/` package (`package.xml`, `setup.py`, `msg/WorldTerrainMap.msg`, `msg/LocalizationStatus.msg`, `launch/ugv_core.launch.py`, `launch/replay_pipeline.launch.py`, `config/ugv_params.yaml`, `config/qos_profiles.yaml`), adapters (`perception_adapter.py`, `localization_adapter.py`, `spatial_fusion_adapter.py`), nodes (`replay_node.py`, `perception_node.py`, `localization_node.py`, `spatial_fusion_node.py`, `static_tf_node.py`), TF2 tree engine (`tf_tree.py`, `tf_broadcaster.py`), diagnostics (`node_diagnostics.py`), scripts (`evaluate_ros2_pipeline.py`), unit/integration tests (`test_ros2_messages.py`, `test_ros2_tf.py`, `test_ros2_adapters.py`, `test_ros2_safety.py`, `test_ros2_replay.py`), `docs/PHASE_10_ROS2_ARCHITECTURE.md`.
- **Exit Criteria:** Exact numerical parity between TF2 and Phase 9 transforms ($< 10^{-12}\text{ m}$); zero-trust enforcement (LOST blocks TF broadcast and map updates); DEGRADED conservative confidence scaling; sub-millisecond TF lookup ($0.036\text{ ms}$); 205/205 unit tests passing across repository with zero regressions.

### Phase 11: Gazebo UGV Simulation (Completed)
- **Goal:** Build outdoor Gazebo worlds with terrain elevation, rocks, vegetation, dynamic sunlight, and differential-drive 4WD UGV robot URDF/SDF model with camera and IMU sensors connected to Phase 10 ROS2 topics.
- **Deliverables:** `simulation/worlds/outdoor_terrain.sdf`, `simulation/models/ugv/` (`model.sdf`, `model.config`, `ugv.urdf`), `simulation/config/` (`simulation_params.yaml`, `bridge_params.yaml`, `ground_truth_obstacles.json`), `simulation/launch/ugv_simulation.launch.py`, `simulation/sim_engine.py`, `simulation/motion_controller.py`, `scripts/evaluate_gazebo_simulation.py`, `tests/test_simulation_*.py`, `outputs/simulation/benchmarks/simulation_audit.json`, `docs/PHASE_11_GAZEBO_UGV_SIMULATION.md`.
- **Exit Criteria:** APPROVED WITH SPECIFIC CONDITIONS; UGV controllable via `/cmd_vel` with realistic sensor simulation feeds publishing to ROS2 `/camera/image_raw`, `/camera/camera_info`, `/imu/data`, `/joint_states`; ground truth isolated; 5 scenarios evaluated across 150 frames with position RMSE $0.1536\text{ m}$; 221/221 unit tests passing across repository with zero regressions.

### Phase 12: Global + Local Path Planning & Closed-Loop Simulation Integration (Completed)
- **Goal:** Implement and validate the primary navigation planning layer consuming Phase 9 `WorldTerrainMap` via Phase 10 ROS2 interfaces (`WorldTerrainMapMsg`, `LocalizationStatusMsg`, `PoseStampedMsg`) to generate collision-free `/cmd_vel` velocity commands for the Phase 11 simulated 4WD outdoor UGV.
- **Deliverables:** `AStarPlanner` (octile heuristic, multi-layer cost function, footprint-validated path smoothing), `DWAPlanner` (dynamic window velocity sampling, kinematic rollout, 2D footprint collision checking, polyline cross-track scoring), `FootprintCollisionChecker` (0.65m x 0.45m oriented bounding box, inscribed circle shortcut, 35 interior samples), `SafetyController` (emergency stop arbitration, tracking-state gating, staleness watchdogs, degraded velocity cap), `PlannerAdapter` and `PlannerNode` (`ugv_navigation`), `configs/planning.yaml`, `scripts/evaluate_path_planning.py`, `tests/test_planning_*.py`, `outputs/planning/benchmarks/path_planning_metrics.json`, `docs/PHASE_12_PATH_PLANNING.md`.
- **Exit Criteria:** APPROVED WITH SPECIFIC CONDITIONS; 8 deterministic scenarios passing (open flat, obstacle detour, water hazard avoidance, narrow passage, blocked goal, localization loss, map stale, invalid goal); 4 closed-loop SIL simulation missions executed with Phase 11 UGV with zero collisions and strictly isolated ground truth; scaling benchmarks measured from 50x50 to 400x400 grids with P95 latencies reported; 29 Phase 12 tests passed; all 250 repository tests passing with zero regressions.

### Phase 13: Dynamic Obstacle Tracking, Predictive Replanning & Acceleration (COMPLETE WITH DOCUMENTED LIMITATIONS — Frozen)
- **Goal:** Implement dynamic hazard tracking (moving obstacles, crossing pedestrians, moving vehicles) with Kalman state estimation, multi-horizon velocity obstacle prediction ($1.0\text{--}3.0\,\text{s}$), dynamic occupancy grid slices, predictive DWA local planning with swept-segment broad-phase filtering, four-state behavioral arbitration (`PROCEED`, `AVOID`, `WAIT`, `STOP`), event-driven global replanning ($A^*$), clean native ROS 2 package architecture (`ugv_navigation_msgs` + `ugv_navigation`), and GPU-accelerated real-time execution validated on NVIDIA Tesla T4.
- **Deliverables:**
  - `ai/planning/dynamic/`: Frozen CPU reference stack (`data_model.py`, `tracker.py`, `prediction.py`, `occupancy_layer.py`, `predictive_collision.py`, `dwa_filter.py`, `behavioral_state.py`, `global_replanner.py`).
  - `ai/planning/dynamic_gpu/`: GPU-accelerated local planner (`gpu_backend.py`, `gpu_rollout.py`, `gpu_collision.py`, `gpu_scoring.py`, `gpu_dwa.py`).
  - `ros2_ws/src/ugv_navigation_msgs/`: Pure ROSIDL interface package (13 canonical `.msg` definitions).
  - `ros2_ws/src/ugv_navigation/`: ROS 2 integration package (converters, 7 nodes, launch files).
  - Validation Suites: Dynamic SIL scenarios (13.13), multi-obstacle failure testing (13.14), MATLAB numerical parity (13.15), 21-mission CPU benchmark (13.16), and Kaggle Tesla T4 GPU acceleration workstream (hist. 13.17.1–13.17.4).
- **Status:** **COMPLETE WITH DOCUMENTED LIMITATIONS (Phase 13 Frozen)**.
  - **Completed Engineering Milestones (13.0–13.16 & GPU Acceleration):**
    - 13.0–13.12: Core Perception, Tracking, Prediction, Dynamic DWA, ROS2 Boundary (**PASS**).
    - 13.13: Dynamic SIL Scenarios (24 closed-loop scenarios, 0 collisions) (**PASS**).
    - 13.14: Multi-Obstacle & Failure Testing (17 CPU scenarios, 0 collisions) (**PASS**).
    - 13.15: MATLAB Independent Numerical Parity (1,055/1,055 checks pass) (**PASS**).
    - 13.16: Full Closed-Loop CPU Benchmark (21 missions, 3,008 cycles, identified 5.7s latency) (**COMPLETE**).
    - **GPU Acceleration Workstream (Hist. 13.17.1–13.17.4):** Slashed DWA latency from $5,673.7\,\text{ms} \to 58.31\,\text{ms}$ ($97.32\times$ speedup, $1.235\times$ RTF) on NVIDIA Tesla T4; 1,210-case parity (**PASS**); 250-cycle stability (**PASS**); Phase 13.16-R2 benchmark (**COMPLETE**).
  - **Finalization Milestones (13.17 to 13.19):**
    - **Milestone 13.17: Visualization + Documentation** (**PASS**)
      - Multi-panel diagnostic plots (Trajectory vs Time, Velocity vs Time, Clearance vs Time, Prediction Fan).
      - Trajectory animations and video/GIF recordings demonstrating dynamic obstacle avoidance (`07_avoidance_animation.gif`).
      - Dedicated RViz marker configs for dynamic tracking and predictive occupancy (`simulation/rviz/judge_demo.rviz`).
      - User/developer documentation: `docs/DYNAMIC_OBSTACLES.md` and `docs/VISUALIZATION_GUIDE.md`.
    - **Milestone 13.18: Full Regression + CPU/GPU Parity Audit** (**CONDITIONAL PASS**)
      - Full 809-test backward regression executed with 99.51% pass rate (805/809) and zero genuine production regressions.
      - Phase 12 static navigation baseline verified with 100% numerical and behavioral reproduction.
      - CPU vs GPU parity verified across 1,210 cases with zero false negatives and zero safety divergences.
    - **Milestone 13.19: Strict Engineering Audit** (**COMPLETE**)
      - Final safety hierarchy, data contract, ground-truth isolation audit, and Phase 14 handoff package.
      - Full adversarial audit report published in `docs/PHASE_13_19_STRICT_ENGINEERING_AUDIT.md`.

### Phase 14: GPS-Denied Outdoor Navigation & Mission Executive (Planned / Not Started)
- **Goal:** Full mission execution across long-distance outdoor scenarios under changing daylight, shadows, and high-contrast environments without GPS signals, leveraging the real-time GPU-accelerated local planner. Introduce hierarchical recovery behaviors and multi-waypoint mission management.
- **Deliverables:** `ugv_mission/` mission executive managing sequential waypoint navigation, stuck detection, escape maneuvers, and topological recovery.
- **Status:** **PLANNED / NOT STARTED** (Phase 13 frozen with documented limitations).
- **Scope Note:** *Phase 14 is the planned next major phase. Detailed milestone breakdown requires a dedicated Phase 14 architecture and planning task.*

### Phase 15: Advanced 2.5D Terrain-Aware Navigation
- **Goal:** Extend planar traversability scoring with a 2.5D elevation representation to evaluate terrain slope, surface roughness, and step heights.
- **Deliverables:** Subphase 15.0 metric depth foundation, 2.5D elevation & surface normal costmap layers, dynamic SE(3) camera attitude compensation, terrain-constrained Global A*, terrain-aware native C++ DWA, and authoritative terrain safety governor.
- **Exit Criteria:** Rover reliably avoids terrain-specific failure modes (steep slopes $>28^\circ$, high-centering step obstacles $>0.12\text{ m}$) while maintaining parity on flat terrain.

### Phase 16: Full Benchmarking and SIH Demonstration
- **Goal:** Execute comprehensive system benchmarks across 10 standardized scenarios. Generate metric reports, demo recordings, and presentation assets.
- **Deliverables:** `benchmarks/` automated evaluation harness, comprehensive final report, and demonstration dashboard.
- **Exit Criteria:** Published metrics table (FPS, latency, drift %, clearance, success rate) and video demonstration.
