# M13.13.1 SIL Architecture Audit Report

**Milestone:** Phase 13.13.1 — Software-in-the-Loop (SIL) Architecture Audit  
**Status:** COMPLETE  
**Date:** September 2026  
**Scope:** Reusable infrastructure identification, capability gap analysis, and minimal additions design for dynamic obstacle SIL validation.

---

## 1. Existing Infrastructure Audit

We audited the Phase 11/12 SIL simulation framework and Phase 13 dynamic planning stack:
- `simulation/motion_controller.py`: `UGVMotionController`, `RobotState2D`, `UGVCommand`.
- `simulation/sim_engine.py`: `GazeboSimEngine`, `SimulatedSensorStream`, terrain landmarks.
- `ai/planning/base.py`: `RobotState`, `GoalPose`, `VehicleKinematics` ($0.65\,\text{m} \times 0.45\,\text{m}$).
- `ai/planning/collision.py`: `FootprintCollisionChecker` (OBB, 35 interior samples, inscribed circle $r=0.225\,\text{m}$).
- `ai/spatial_fusion/world_map.py`: `WorldTerrainMap` (2D cost, obstacle, hazard, unknown layers).
- `ai/planning/dynamic/*`: Data model, tracker, predictor, dynamic occupancy, predictive collision checker, DWA integration, behavioral state machine, event-driven replanner.
- `ros2_ws/src/ugv_navigation/ugv_navigation/adapters/dynamic_navigation_adapter.py`: Full pipeline coordinator.

### Reusable Components:
1. **Kinematics Engine (`UGVMotionController`):**
   - Exact differential-drive kinematics with midpoint numerical integration.
   - Command velocity saturation ($v_{\max} = 1.5\,\text{m/s}$, $\omega_{\max} = 1.0\,\text{rad/s}$).
   - Reusable directly for SIL vehicle state propagation.
2. **Vehicle Footprint Collision Geometry (`FootprintCollisionChecker`):**
   - Oriented Bounding Box calculation `get_footprint_corners(x, y, yaw)`.
   - Used for objective ground-truth collision auditing between the UGV and moving obstacle bounding boxes.
3. **World Representation (`WorldTerrainMap`):**
   - Canonical 2D grid representation for static background terrain and global A* planning.
4. **Dynamic Algorithmic Core (`ai/planning/dynamic/*`):**
   - All 10 core modules are frozen, unit-tested (461 tests pass), and ready for end-to-end composition.

---

## 2. Missing Capabilities for Dynamic SIL

The existing Phase 12 SIL simulation was designed primarily for static obstacle detours:
1. **No Ground-Truth Dynamic Obstacle Engine:** `sim_engine.py` only maintains static spherical/cylindrical obstacles (`gt_registry["obstacles"]`). It lacks continuous time-parameterized kinematics for moving actors.
2. **No Observation Generation with Noise/Dropout:** The static SIL injected static obstacles directly into the camera or map. Dynamic SIL requires realistic sensor observation generation from ground truth, incorporating:
   - Measurement noise ($\sigma_x, \sigma_y$).
   - Sensor detection dropout probability ($p_{\text{drop}}$).
   - Ephemeral detection IDs (stripping ground truth IDs and velocities).
3. **No Closed-Loop Dynamic Orchestrator:** Need a clean simulation runner that steps:
   $\text{GT Obstacles} \to \text{Noisy Detections} \to \text{Tracker} \to \text{Predictor} \to \text{Occupancy} \to \text{Predictive Collision} \to \text{DWA} \to \text{Behavior} \to \text{Kinematics}$.
4. **No Formal Ground-Truth Isolation Guard:** Need an architectural barrier guaranteeing that ground truth state is never leaked into the navigation pipeline.

---

## 3. Proposed Minimum Additions (M13.13.2 – M13.13.5)

To address these gaps without modifying frozen core algorithms (`ai/planning/dynamic/*`):
1. **`simulation/scenarios/dynamic/data_model.py` (M13.13.2):**
   - `DynamicObstacleSpec`: Configuration schema for dynamic obstacles (ID, dimensions, motion model, initial conditions, noise/dropout params).
   - `DynamicScenarioConfig`: Complete scenario definition (ID, world bounds, start, goal, obstacle list, duration, control dt, seed).
2. **`simulation/scenarios/dynamic/motion_models.py` (M13.13.2):**
   - Deterministic trajectory generators: Constant Velocity (CV), Constant Turn Rate (CT), Accelerated/Decelerated, Waypoint schedule, and Stationary blockage.
3. **`simulation/scenarios/dynamic/observation_model.py` (M13.13.3):**
   - Observation synthesis with Gaussian noise, dropout, and ephemeral ID generation (`det_<frame>_<idx>`).
4. **`simulation/runners/dynamic_sil_runner.py` (M13.13.4):**
   - Closed-loop simulation engine advancing kinematics, feeding observations into the real Phase 13 stack, executing DWA and Behavioral State Machine, and recording metrics.
5. **`tests/test_dynamic_sil_scenarios.py` (M13.13.5):**
   - Automated regression test suite validating isolation, footprint collision calculation, determinism, and safety invariance.
