# Phase 13.25A — Runtime Evidence, Benchmark Provenance & Measurement Audit

**Project:** SIH2026 — Vision-Based Autonomous Navigation for UGV  
**Audit Target:** Post-DWA Vehicle Safety & Recovery Workstream (Stages 1–4 Reports & JSON Telemetry)  
**Date:** 2026-09-30T16:45:00+05:30  
**Branch:** `demo-01-interactive-ugv`  
**Audit Status:** COMPLETE & VERIFIED  

---

## 1. Audit Objective & Scope

This audit examines the exact provenance, measurement methodologies, and test harnesses behind the metrics reported in:
- `reports/POST_DWA_VEHICLE_SAFETY_WORKSTREAM_FINAL_REPORT.md`
- `reports/integrated_safety_workstream_measurements.json`
- `reports/PHASE_13_22_CONTINUOUS_MOTION_REPORT.md`
- `reports/PHASE_13_23_LOST_OBSTACLE_SAFETY_REPORT.md`
- `reports/PHASE_13_24_RECOVERY_REPLANNING_REPORT.md`

Specifically, this audit provides verifiable answers to:
1. Exactly how the 2.14 ms perception, 0.65 ms C++ DWA, and 2.85 ms total cycle latencies were obtained.
2. Proof of whether `LightweightTraversabilityNet` neural inference was executed in B01–B10.
3. The exact execution environment of benchmark matrix B01–B10 (Gazebo vs Python SIL vs synthetic).
4. The exact measurement source of the 20 Hz `/cmd_vel` publication rate and gap metrics.
5. Clarification of which metrics are legitimate to publish as physical/ROS2 measurements versus synthetic/SIL algorithmic benchmarks.

---

## 2. Metric Provenance Audit: 2.14 ms, 0.65 ms, and 2.85 ms

### A. 0.65 ms C++ DWA Latency
- **Source Code:** `scripts/validate_integrated_safety_workstream.py`, lines 237–245:
  ```python
  t_dwa_start = time.perf_counter()
  cand_traj, cand_v, cand_w, planner_status, dwa_diag = self.dwa_planner.compute_trajectory(
      current_state=robot_state,
      global_path=active_path,
      costmap=costmap,
      dynamic_layer=self.occupancy_layer,
      predicted_tracks=active_predictions,
  )
  dwa_latencies.append((time.perf_counter() - t_dwa_start) * 1000.0)
  ```
- **Execution Path:** `UnifiedDWAPlanner(backend=DWABackendType.AUTO)` $\rightarrow$ `NativeCppDWABackend` $\rightarrow$ `libdwa_core.dll` (`dwa_c_api.cpp` / `dwa_core.cpp`).
- **Nature of Measurement:** **REAL C++ EXECUTION**. The compiled native C++ AVX2 shared library was dynamically linked and executed on the host Intel Core Ultra 7 155H CPU evaluating 375 trajectory candidates per cycle.
- **Measured Range:** Scenario means ranged from **0.54 ms to 1.05 ms**, yielding an aggregate mean of **0.65 ms** (P95: **0.88 ms**).

### B. 2.85 ms Total Planning Cycle Latency
- **Source Code:** `scripts/validate_integrated_safety_workstream.py`, lines 182 & 315:
  ```python
  t_cycle_start = time.perf_counter()
  # [Synthetic detection generation + MultiFrameTracker update +
  #  RecentHazardMemory update + MotionPredictor + DynamicOccupancyLayer update +
  #  Global replanner step + C++ DWA trajectory computation +
  #  RecoveryController step + SafetyGovernor evaluation + SafeCommandBuffer set_command]
  cycle_latencies.append((time.perf_counter() - t_cycle_start) * 1000.0)
  ```
- **Nature of Measurement:** **REAL CPU WALL-CLOCK TIME** for the algorithmic decision pipeline executed in Python.
- **Measured Range:** Scenario means ranged from **1.69 ms to 11.86 ms** (higher during A* global replanning ticks), yielding an aggregate mean of **2.85 ms** (P95: **7.50 ms**).

### C. 2.14 ms "Perception & Tracking" Metric
- **Mathematical Derivation:** In Section 9 of the Final Report, 2.14 ms was calculated as:
  $$\text{Total Cycle (2.85 ms)} - \text{C++ DWA (0.65 ms)} - \text{Safety Governor / Recovery (0.18 ms)} \approx 2.02–2.14\text{ ms}$$
- **Components Included:**
  1. Synthetic detection bounding box generation (`crossing_func`, `close_dyn`, `close_pin`).
  2. `MultiFrameTracker.update()` (Hungarian matching + Kalman filter state estimation).
  3. `RecentHazardMemory.update()` (hazard latching & dropout tracking).
  4. `MotionPredictor.predict_tracks()` (constant-velocity spatiotemporal projection).
  5. `DynamicOccupancyLayer.update_predictions()` (2D grid rasterization).
- **Components Excluded:** **NO NEURAL NETWORK INFERENCE WAS EXECUTED.**

---

## 3. Proof: LightweightTraversabilityNet Was NOT Executed in B01–B10

### Proof Points:
1. **Module Imports in `scripts/validate_integrated_safety_workstream.py`:**
   Neither `DenseTraversabilityPredictor` nor `LightweightTraversabilityNet` is imported or referenced anywhere in `scripts/validate_integrated_safety_workstream.py`.
2. **Costmap Generation:**
   The costmap in B01–B10 was constructed using `build_base_map()` (lines 65–76):
   ```python
   def build_base_map(x_bounds=(-2.0, 10.0), y_bounds=(-5.0, 5.0), res=0.10) -> WorldTerrainMap:
       wtm = WorldTerrainMap(resolution_m=res, x_bounds_m=list(x_bounds), y_bounds_m=list(y_bounds), created_timestamp=0.0)
       wtm.cost.fill(0.10)
       wtm.obstacle.fill(False)
       return wtm
   ```
   Obstacles were imprinted analytically using `imprint_circular_obstacle()` (lines 79–89). No camera RGB/depth images or PointClouds were processed.
3. **Comparison with Prior End-to-End Neural Benchmarks:**
   In Phase 13.21 (`scripts/benchmark_end_to_end_cpp.py`, logged in `reports/dwa_end_to_end_cpp_validation.json`), actual PyTorch inference of frozen `LightweightTraversabilityNet` was measured:
   - **Neural Perception Inference (`LightweightTraversabilityNet`):** **118.38 ms mean (160.37 ms P95)**.
   - **Costmap Rasterization:** **0.47 ms**.
   - **Dynamic Obstacle Tracking & Prediction:** **4.86 ms**.
   - **Native C++ DWA:** **1.93 ms**.
   - **True Full End-to-End Navigation Cycle:** **125.70 ms** (~8.0 Hz).
4. **Relabeling Requirement:**
   The 2.14 ms metric MUST be strictly relabeled as:
   **`Synthetic Dynamic Tracking & Occupancy Grid Update (no DNN)`**.
   It must NEVER be presented as neural network perception latency.

---

## 4. Execution Environment of B01–B10: Python SIL Simulation

### Analysis:
- `scripts/validate_integrated_safety_workstream.py` does not launch Gazebo Sim Harmonic, does not call `rclpy.init()`, and does not communicate over ROS2 middleware.
- Vehicle motion is integrated using a 2D unicycle kinematic simulation:
  ```python
  theta_mid = ugv_yaw + 0.5 * dispatched_w * dt_pub_s
  ugv_x += dispatched_v * math.cos(theta_mid) * dt_pub_s
  ugv_y += dispatched_v * math.sin(theta_mid) * dt_pub_s
  ugv_yaw += dispatched_w * dt_pub_s
  ```
- **Conclusion:** B01–B10 are **Python Software-In-the-Loop (SIL) closed-loop algorithmic simulation tests**, NOT live Gazebo physics engine runs.
- **Algorithmic Authenticity:** While the environment and kinematics were simulated in Python, the navigation algorithms executed were the actual production classes:
  - `SafeCommandBuffer` (`ai/control/command_buffer.py`)
  - `UnifiedDWAPlanner` using `libdwa_core.dll` (`ai/planning/backends/cpp_backend.py`)
  - `RecentHazardMemory` and `SafetyGovernor` (`ai/planning/safety_governor.py`)
  - `RecoveryController` (`ai/planning/recovery_controller.py`)
  - `MultiFrameTracker`, `MotionPredictor`, `DynamicOccupancyLayer` (`ai/planning/dynamic/`)
  - `EventDrivenGlobalReplanner` and `AStarPlanner` (`ai/planning/global_planners/astar.py`)

---

## 5. Provenance of the 20 Hz `/cmd_vel` Metric

### Analysis:
1. **In `scripts/validate_integrated_safety_workstream.py`:**
   - The loop advanced simulation time by `dt_pub_s = 0.05 s` (50 ms).
   - In each 50 ms step, `command_buffer.get_dispatch_command(sim_time)` was called.
   - The buffer telemetry recorded the inter-dispatch interval: exactly $50.0\text{ ms}$, corresponding mathematically to $20.0\text{ Hz}$.
2. **In `simulation/nodes/interactive_ugv_navigation_node.py`:**
   - The production ROS2 node implements this exact decoupling using a dedicated ROS2 timer:
     ```python
     dt_cmd = 1.0 / self.cmd_publish_rate_hz  # 0.05s
     self.cmd_timer = self.create_timer(dt_cmd, self._dispatch_cmd_vel_loop)
     ```
   - In `_dispatch_cmd_vel_loop()`, the node queries `self.command_buffer.get_dispatch_command()` and publishes `Twist` to `/cmd_vel`.
3. **Conclusion:**
   - The 20.0 Hz / 50.0 ms gap figures in `reports/integrated_safety_workstream_measurements.json` are **measurements from the Python SIL simulation harness and internal command buffer telemetry**.
   - They were **NOT** recorded via `ros2 topic hz /cmd_vel` during an active Gazebo execution session.

---

## 6. Physical vs. Simulated Velocity (0.493 m/s)

- **Was 0.493 m/s physically measured in Gazebo?**
  **NO.** 0.493 m/s was measured in the Python SIL unicycle kinematic simulation during scenario B01 (Open Path).
- **Physical Context in Gazebo:**
  In prior Gazebo runs, the robot moved at 0.06–0.09 m/s because the legacy synchronous 4 Hz control loop starved the Gazebo `DiffDrive` watchdog. The decoupling implemented in `interactive_ugv_navigation_node.py` architecturally resolves this starvation by delivering 20 Hz `/cmd_vel` packets, but the 0.493 m/s number in `integrated_safety_workstream_measurements.json` is a SIL kinematic result.

---

## 7. True End-to-End Neural Navigation vs. Algorithmic Planning Latency

- **Is 2.85 ms a true end-to-end neural navigation measurement?**
  **NO.** 2.85 ms is the **algorithmic planning and tracking latency** (excluding deep learning perception).
- **True End-to-End Neural Measurement:**
  The true end-to-end latency incorporating the frozen `LightweightTraversabilityNet` neural network is **~125.7 ms** (~8.0 Hz), as verified in Phase 13.21:
  - Neural Traversability Inference: **118.4 ms**
  - Dynamic Obstacle Tracking: **4.9 ms**
  - Native C++ DWA: **1.9 ms**
  - Safety Governor & Overheads: **0.5 ms**
  - Total End-to-End: **~125.7 ms**

---

## 8. Summary of Metric Classification (Publishable vs. Relabeled)

| Metric | Reported Value | Original Report Label | Verified Actual Source | Mandatory Relabeling |
|---|---|---|---|---|
| **C++ DWA Latency** | 0.65 ms | DWA Trajectory Evaluation (C++ CPU) | Host execution of `libdwa_core.dll` via C API | **LEGITIMATE TO PUBLISH** (Native C++ core evaluation) |
| **Tracking/Occupancy Latency** | 2.14 ms | Perception & Tracking Update | Python Kalman tracker + predictor + grid rasterization | **MUST RELABEL:** `Synthetic Dynamic Tracking & Occupancy Update (no DNN)` |
| **Total Planning Cycle** | 2.85 ms | Total Navigation Planning Cycle | Python loop enclosing tracking, C++ DWA, governor, and replanner | **MUST RELABEL:** `Algorithmic Planning & Safety Cycle (excluding DNN perception)` |
| **Command Dispatch Rate** | 20.0 Hz (50ms gap) | Actuator Command Dispatch Rate | `SafeCommandBuffer` telemetry in Python SIL loop | **MUST RELABEL:** `Command Buffer Dispatch Rate (Python SIL simulation)` |
| **UGV Velocity Profile** | 0.493 m/s | Commanded Vehicle Velocity | Unicycle forward kinematic model in Python SIL | **MUST RELABEL:** `Simulated Kinematic Velocity (Python SIL)` |
| **Benchmark Matrix B01–B10** | 10 Scenarios Passed | Integrated Validation Suite | Python SIL simulation harness with analytical obstacle trajectories | **MUST RELABEL:** `Closed-Loop Python SIL Simulation Matrix` |
| **Safety Invariants (Collisions, Inversions)** | 0 Violations | Safety Metrics Summary | Evaluated against kinematic bounding box models in Python SIL | **LEGITIMATE TO PUBLISH as SIL Safety Verification** |
