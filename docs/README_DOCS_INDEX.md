# SIH26126 — Master Documentation & Reports Index

**Project**: SIH26126 — Vision-Based Autonomous Navigation for Unmanned Ground Vehicle (UGV) in Outdoor Environments  
**Document**: `docs/README_DOCS_INDEX.md`  

This directory and `reports/` are organized so judges and developers can immediately find top-level master references while keeping all detailed phase and optimization sub-reports cleanly grouped in subfolders.

---

## 1. Top-Level Master Documents (`docs/`)

| File | Purpose |
| :--- | :--- |
| **[`docs/ROADMAP.md`](ROADMAP.md)** | Official 17-Phase Development Roadmap (`Phase 0` to `Phase 16` — All Complete & Verified) |
| **[`docs/PHASES_0_TO_16_MASTER_ENCYCLOPEDIA.md`](PHASES_0_TO_16_MASTER_ENCYCLOPEDIA.md)** | Unified Encyclopedia of all 17 Phases (`Phase 0` through `Phase 16`), architectures, and metrics |
| **[`docs/OPTIMIZATION_AND_ACCELERATION_MASTER.md`](OPTIMIZATION_AND_ACCELERATION_MASTER.md)** | Consolidated Master Guide for Optimization Tiers 1–4, C++17 SIMD, ONNX INT8, Zero-Copy IPC, and GPU Acceleration |
| **[`docs/OPTIMIZATION_WORKSTREAMS_MASTER.md`](OPTIMIZATION_WORKSTREAMS_MASTER.md)** | Authoritative Ledger Table for Optimization Workstreams `WS-1A` through `WS-4A` |
| **[`docs/PHASE_16_MASTER_IMPLEMENTATION_PLAN.md`](PHASE_16_MASTER_IMPLEMENTATION_PLAN.md)** | Phase 16 Master Plan: Live 2.5D Node, Photometric Resilience, RViz2 Judge HUD, Webcam HIL & 10-Scenario Suite |
| **[`docs/PHASE_15_MASTER_IMPLEMENTATION_PLAN.md`](PHASE_15_MASTER_IMPLEMENTATION_PLAN.md)** | Phase 15 Master Plan: 2.5D Elevation Mapping, Normals/Slope/Steps/Ditches, Belly Clearance & Rollover DWA |
| **[`docs/PHASE_14_CONSOLIDATED_MASTER.md`](PHASE_14_CONSOLIDATED_MASTER.md)** | Consolidated Master Reference for Phase 14 (`14.1`–`14.7.4`: 100m Mission Executive, Vision Loop & 15D EKF) |
| **[`docs/PHASE_13_CONSOLIDATED_MASTER.md`](PHASE_13_CONSOLIDATED_MASTER.md)** | Consolidated Master Reference for Phase 13 (`13.0`–`13.19` Dynamic Planning + `13.21`–`13.26` Post-DWA Safety) |
| **[`docs/ARCHITECTURE.md`](ARCHITECTURE.md)** | Core System Architecture & Data Contracts |
| **[`docs/DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md)** | Repository Audit & Phase Completion Ledger |

---

## 2. Top-Level Final Benchmark Reports (`reports/`)

| File | Purpose |
| :--- | :--- |
| **[`reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md`](../reports/PHASE_16_SIH_FINAL_BENCHMARK_REPORT.md)** | **SIH Grand Finale Judge Dossier**: 10-Scenario Metrics Table (`10/10 PASS`), Phase 0–16 Matrix & Judge Runbook |
| **[`reports/PHASE_15_BENCHMARK_REPORT.md`](../reports/PHASE_15_BENCHMARK_REPORT.md)** | Phase 15 Differential Terrain Benchmark Report (`Phase 14 2D` vs `Phase 15 2.5D` across 7 scenarios) |
| **[`reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md`](../reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md)** | Phase 14.7.4 100m 5-Waypoint Corridor Validation Report (`12/12` runs PASS) |
| **[`reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md`](../reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md)** | Detailed Mathematical & Engineering Report for `P0-1`–`P0-4` and `P1-1`–`P1-3` Optimizations |

---

## 3. Organized Archive Subfolders

### Inside `docs/`:
- **[`docs/phases/phases_01_to_12/`](phases/phases_01_to_12/)**: Individual specifications for Phases 1–12 (`PHASE_1.md` through `PHASE_12_PATH_PLANNING.md`, `PHASE_3_HANDOFF.md`, `SLAM_*.md`).
- **[`docs/phases/phase_13/`](phases/phase_13/)**: All 14 detailed Phase 13 milestone documents (`PHASE_13_MASTER_CONTEXT.md`, `DYNAMIC_OBSTACLES.md`, `PHASE_13_13_SIL_RESULTS.md` through `PHASE_13_19_STRICT_ENGINEERING_AUDIT.md`).
- **[`docs/optimization/`](optimization/)**: Hardware acceleration and strategic review documents (`DWA_ACCELERATION_AUDIT_REPORT.md`, `KAGGLE_GPU_INTEGRATION.md`, `PHASE_13_PERFORMANCE_ACCELERATION.md`, `PHASE_13_TO_14_STRATEGIC_REVIEW.md`).
- **[`docs/guides_and_demos/`](guides_and_demos/)**: Operator and runtime guides (`DEMO_01_*.md`, `VISUALIZATION_GUIDE.md`, `ROS2_RUNTIME_INTEGRATION_PLAN.md`, `TECH_STACK.md`, `DATASET_STRATEGY.md`).

### Inside `reports/`:
- **[`reports/phase_13/`](../reports/phase_13/)**: Phase 13.21–13.26 engineering reports (`PHASE_13_21_DWA_CPP_ENGINEERING_REPORT.md` through `PHASE_13_26_PRE_RUNTIME_PIPELINE_AUDIT.md`).
- **[`reports/phase_14/`](../reports/phase_14/)**: Phase 14.1–14.7.3 implementation and readiness reports.
- **[`reports/phase_15_16/`](../reports/phase_15_16/)**: Phase 15 strategic audit and Phase 16 stability/collision fix reports.
- **[`reports/optimization/`](../reports/optimization/)**: Post-DWA vehicle safety workstream plans and final reports (`POST_DWA_*.md`).
- **[`reports/datasets/`](../reports/datasets/)**: RUGD and RELLIS-3D dataset audit and dataloader benchmark reports.
