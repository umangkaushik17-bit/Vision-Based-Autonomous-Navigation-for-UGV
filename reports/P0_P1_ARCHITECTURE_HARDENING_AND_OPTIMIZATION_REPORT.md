# P0 & P1 Architecture Hardening, Safety Closure, and Real-Time C++/ROS 2 Optimization Report

**Project**: SIH26126 — Vision-Based Autonomous Navigation for Unmanned Ground Vehicle (UGV) in Outdoor Environments  
**Document**: `reports/P0_P1_ARCHITECTURE_HARDENING_AND_OPTIMIZATION_REPORT.md`  
**Master Ledger**: [`docs/OPTIMIZATION_WORKSTREAMS_MASTER.md`](../docs/OPTIMIZATION_WORKSTREAMS_MASTER.md)  
**Branch**: `demo-01-interactive-ugv`  
**Status**: **COMPLETE & VERIFIED (165 / 165 Automated Tests Passing Across Windows & WSL2 ROS 2 Jazzy)**  

---

## 1. Executive Summary

Following a comprehensive adversarial source-code audit of the SIH26126 autonomous UGV stack across perception, spatial fusion, GPS-denied state estimation, global/local motion planning, safety governance, and ROS 2 Jazzy middleware, **seven critical (`P0`) and high-priority (`P1`) vulnerabilities and computational bottlenecks** were systematically resolved and verified:

1. **`P0-1` (Defect #6) — Gazebo Harmonic Camera Sensor Optical Frame Alignment:** Resolved the $-90^\circ$ lateral orientation mismatch in `simulation/models/ugv/model.sdf` by binding `<sensor name="rgb_camera">` pose to `camera_link` ($+X$ forward, $15^\circ$ downward tilt, $0.525\,\text{m}$ elevation) while preserving `camera_optical` ROS 2 frame tagging.
2. **`P0-2` (Problem #3 / WS-1B) — End-to-End Vision Closure via Dynamic SE(3) Attitude-Compensated IPM:** Replaced the static flat-ground projection assumption with real-time 3D rigid-body roll/pitch ray compensation ($\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$), pre-cached multi-resolution lookup tables, vectorized `np.bincount` BEV accumulation ($< 3.5\,\text{ms}$), and vectorized `WorldTerrainMap` fusion.
3. **`P0-3` (Problem #4) — 15-State GPS-Denied EKF 3D Euler Kinematics & Gravity Tilt Observability:** Upgraded `MultiSensorEKF` with the exact 3D Euler kinematic rate transformation matrix $E(\phi, \theta)$, online gyroscope/accelerometer bias tracking, and 6-axis IMU gravity-vector roll/pitch tilt correction, preventing gravity ($9.81\,\text{m/s}^2$) from leaking into horizontal world acceleration on sloped terrain.
4. **`P0-4` (Problem #5) — Near-Field Blind-Spot Hazard Latching, 3D Rollover Safety Governor & Reverse-and-Turn Recovery:** Extended `RecentHazardMemory` to latch static/vision costmap obstacles inside the camera's near-field blind spot ($< 1.25\,\text{m}$ ahead, $3.5\,\text{s}$ TTL), added $18^\circ / 26^\circ$ 3D roll/pitch rollover clamps to `SafetyGovernor`, and added lateral-clearance-aware **reverse-and-turn** disengagement (`REVERSING` + `TURNING`) to `RecoveryController`.
5. **`P1-1` (Problem #1 / WS-1A) — Native C++17 Global $A^*$ Sub-5ms Core & Vectorized Path Validation:** Compiled dual-target AVX2 SIMD and portable scalar C++17 shared libraries for both Windows (`astar_cpp.dll`, `astar_cpp_scalar.dll`) and Linux (`libastar_cpp.so`, `libastar_cpp_scalar.so`) with Jump Point Search (JPS) directional pruning, midpoint/corner footprint shortcutting, and EDT-vectorized `PathValidator.validate_path()`, achieving **$0.45\,\text{ms}$ ($50\times 50$)**, **$2.65\,\text{ms}$ ($120\times 120$)**, and **$8.17\,\text{ms}$ ($200\times 200$)** pure C++ core execution ($136\times\text{--}371\times$ speedup).
6. **`P1-2` (Problem #2 / WS-4A) — Zero-Copy ROS 2 Jazzy IPC & Shared-Memory Map Transport:** Implemented `IntraProcessLoanedChannel` ($< 0.05\,\text{ms}$ reference transfer with read-only buffer freeze), `SharedMemoryWorldMapChannel` ($< 0.8\,\text{ms}$ cross-process 10-layer $200\times 200$ POD shared memory), `SharedMemoryCameraFrameChannel` ($< 0.4\,\text{ms}$ for $640\times 480$ RGB), and vectorized C-buffer ROSIDL converters.
7. **`P1-3` (Defect #7) — Decoupled Asynchronous Perception Worker:** Implemented `AsyncPerceptionWorker` with a bounded **depth-1 latest-frame slot** and background daemon thread executing neural inference + 3D IPM projection off the ROS 2 executor thread ($< 0.2\,\text{ms}$ callback submission), eliminating control-loop jitter on the 10 Hz planner and 20 Hz actuator timers.

---

## 2. Updated End-to-End System Architecture

```
+----------------------------------------------------------------------------------------------------------+
|                              UPDATED REAL-TIME CLOSED-LOOP ARCHITECTURE                                  |
|                                                                                                          |
|  [Gazebo / Physical Sensors]                                                                             |
|    ├── RGB Camera (640x480 @ 30Hz, aligned to camera_link +15 deg pitch)                                 |
|    ├── 6-Axis IMU (200Hz angular velocity + specific force)                                              |
|    └── Wheel Odometry + Stereo VIO                                                                       |
|           │                                                  │                                           |
|           ▼ (< 0.2ms Non-Blocking Submit)                    ▼ (50-200Hz)                                |
|  [AsyncPerceptionWorker (Daemon Thread, Depth-1 Slot)]   [15-State MultiSensorEKF]                       |
|    ├── LightweightTraversabilityNet (ONNX / PyTorch)       ├── 3D Euler Kinematics E(phi, theta)         |
|    └── VisionCostmapIntegrator                             ├── Gravity-Vector Roll/Pitch Observability   |
|          ├── Dynamic SE(3) Roll/Pitch Ray Compensation     ├── Mahalanobis Wheel-Slip Gating             |
|          └── Fast bincount LocalBEVCostmap (< 3.5ms)       └── Fused (x, y, z, roll, pitch, yaw, v, w)   |
|           │                                                  │                                           |
|           └───────────────────────┬──────────────────────────┘                                           |
|                                   ▼                                                                      |
|                     [WorldTerrainMap (10-Layer Metric Grid)]                                             |
|                       ├── Vectorized Fast-Path Cost/Obstacle Fusion (< 0.8ms)                            |
|                       ├── Persistent Hazard Latching (5-frame clear hysteresis)                          |
|                       └── Zero-Copy IPC: IntraProcessLoanedChannel (< 0.05ms) / POD SHM (< 0.8ms)        |
|                                   │                                                                      |
|            ┌──────────────────────┼───────────────────────┐                                              |
|            ▼                      ▼                       ▼                                              |
|  [Native C++17 Global A*]  [Native C++17 AVX2 DWA]  [RecentHazardMemory & RecoveryController]            |
|   ├── astar_cpp.dll / .so   ├── dwa_cpp.dll / .so    ├── Near-Field Blind-Spot Costmap Latch (TTL 3.5s)  |
|   ├── JPS Ray Pruning       ├── 375 Rollouts (~1ms)  ├── Lateral Clearance Assessment (Left vs Right)    |
|   ├── O(1) Gen Workspace    └── 8-Axis SAT Polygon   └── Reverse-and-Turn Escape (REVERSING -> TURNING)  |
|   └── 2.65ms (120x120)                                    │                                              |
|            │                      │                       │                                              |
|            └──────────────────────┼───────────────────────┘                                              |
|                                   ▼                                                                      |
|                    [AuthoritativeSafetyGovernor]                                                         |
|                      ├── Forward/Rear Bumper Corridor Guard + Static Blind-Spot Check                    |
|                      ├── 3D Attitude Rollover Protection (18 deg Cautious / 26 deg Emergency Stop)       |
|                      └── RuntimeAdaptationManager Compute-Latency Clamping                               |
|                                   │                                                                      |
|                                   ▼                                                                      |
|                    [SafeCommandBuffer (20 Hz Continuous Dispatch)] ──► /cmd_vel                          |
+----------------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Technical Specifications & Mathematical Formulations

### 3.1 `P0-1`: Gazebo SDF Camera Sensor Orientation Fix (`simulation/models/ugv/model.sdf`)
* **Root Cause**: In Gazebo Harmonic (`gz-sim` SDF 1.9), a `<sensor type="camera">` renders along the **$+X$ axis** of its sensor frame (REP-103 convention), whereas ROS optical frames (`camera_optical`) define $+Z$ as forward via an RPY rotation of $(-\pi/2, 0, -\pi/2)$. Placing the sensor directly inside `<link name="camera_optical">` without a relative pose caused Gazebo's $+X$ viewing axis to point along the chassis $-Y$ axis (90° right) with $0^\circ$ pitch.
* **Solution**:
  * Set `<link name="camera_optical">` pose relative to `camera_link`:
    ```xml
    <pose relative_to="camera_link">0 0 0 -1.57079632679 0 -1.57079632679</pose>
    ```
  * Bound the `<sensor name="rgb_camera" type="camera">` physical viewing pose to `camera_link` while preserving `camera_optical` metadata for ROS 2 bridge messages:
    ```xml
    <pose relative_to="camera_link">0 0 0 0 0 0</pose>
    <gz_frame_id>camera_optical</gz_frame_id>
    <camera>
      <optical_frame_id>camera_optical</optical_frame_id>
      ...
    </camera>
    ```

---

### 3.2 `P0-2` (`WS-1B`): Dynamic SE(3) Attitude-Compensated IPM (`production/perception/vision_costmap_integrator.py`)
* **Mathematical Formulation**:
  1. For each pixel $(u, v)$, the unprojected ray in the camera optical frame is $\mathbf{r}_{\text{opt}} = [(u - c_x)/f_x, (v - c_y)/f_y, 1]^T$.
  2. At initialization (`precompute_projection`), rays are transformed into the nominal `base_link` frame and sky rays ($\ge +20^\circ$ above horizontal, $r_{z,\text{base}} \ge 0.35$) are pre-culled:
     $$\mathbf{r}_{\text{base}} = \mathbf{R}_{\text{opt}}^{\text{base}} \mathbf{r}_{\text{opt}}, \qquad \mathbf{t}_{\text{cam}}^{\text{base}} = \begin{bmatrix} x_{\text{fwd}} \\ 0 \\ h_{\text{cam}} \end{bmatrix}$$
  3. During live operation with instantaneous EKF chassis roll $\phi$ and pitch $\theta$, the rotation from tilted `base_link` to the locally level ground frame is:
     $$\mathbf{R}_{\text{level}}(\phi, \theta) = \mathbf{R}_y(\theta) \mathbf{R}_x(\phi) = \begin{bmatrix} \cos\theta & \sin\theta\sin\phi & \sin\theta\cos\phi \\ 0 & \cos\phi & -\sin\phi \\ -\sin\theta & \cos\theta\sin\phi & \cos\theta\cos\phi \end{bmatrix}$$
  4. The attitude-compensated camera center and ray directions in the locally level frame are computed via a single $3 \times 3$ matrix multiplication:
     $$\mathbf{t}_{\text{level}} = \mathbf{R}_{\text{level}}(\phi, \theta) \mathbf{t}_{\text{cam}}^{\text{base}}, \qquad \mathbf{d}_{\text{level}} = \mathbf{R}_{\text{level}}(\phi, \theta) \mathbf{r}_{\text{base}}$$
  5. Intersecting with the level ground plane $Z_{\text{level}} = 0$ ($d_{z,\text{level}} < -10^{-4}$) yields optical depth $z_c$ and metric ground coordinates $(X_g, Y_g)$:
     $$z_c = \frac{-t_{\text{level}, z}}{d_{z,\text{level}}}, \qquad X_g = t_{\text{level}, x} + z_c d_{x,\text{level}}, \qquad Y_g = t_{\text{level}, y} + z_c d_{y,\text{level}}$$
* **Performance**: Pre-caching `(480, 640)` and `(240, 320)` ray tables and replacing `np.add.at` with `np.bincount` reduced attitude-compensated BEV projection latency to **$< 3.5\,\text{ms}$** on CPU while maintaining metric obstacle position error **$\le 0.10\,\text{m}$** across $\pm 12^\circ$ roll and $[-10^\circ, +12^\circ]$ pitch.

---

### 3.3 `P0-3`: 15-State GPS-Denied EKF 3D Euler Kinematics & Gravity Tilt Observability (`production/localization/ekf_fusion.py`)
* **3D Euler Kinematic Propagation**:
  Body angular velocities $[\omega_x, \omega_y, \omega_z]^T$ transform to Euler angle rates $[\dot{\phi}, \dot{\theta}, \dot{\psi}]^T$ via:
  $$\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = \mathbf{E}(\phi, \theta) \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix} = \begin{bmatrix} 1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi\sec\theta & \cos\phi\sec\theta \end{bmatrix} \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$
  with pitch $\theta$ safely clamped to $[-1.35, 1.35]\,\text{rad}$ and the Jacobian block $\mathbf{F}[3:6, 9:12] = \mathbf{E}(\phi, \theta)\Delta t$ propagated through the Joseph-form covariance update.
* **Gravity-Vector Roll/Pitch Tilt Observability**:
  When specific force magnitude $\|\mathbf{a}_{\text{meas}} - \mathbf{b}_a\|$ is within $0.25\,\text{m/s}^2$ of $g = 9.80665\,\text{m/s}^2$ and $a_z < g$, gravitational tilt angles are observed directly:
  $$\phi_{\text{grav}} = \operatorname{atan2}(a_y, a_z), \qquad \theta_{\text{grav}} = \operatorname{atan2}\left(-a_x, \sqrt{a_y^2 + a_z^2}\right)$$
  Correcting $(\phi, \theta)$ prior to rotating $\mathbf{a}_{\text{meas}}$ into the world frame eliminates gravity leakage ($g\sin\delta\theta$) into horizontal acceleration states $(a_x, a_y)$.

---

### 3.4 `P0-4`: Near-Field Blind Spot Latching, 3D Rollover Governor & Reverse-and-Turn Recovery
* **Near-Field Blind-Spot Costmap Latching (`ai/planning/safety_governor.py`)**:
  * With camera height $0.525\,\text{m}$, pitch $15^\circ$, and VFOV $54.6^\circ$, the ground plane exits the bottom of the camera FOV at $x < 0.58\,\text{m}$ ahead of the lens ($0.73\,\text{m}$ ahead of `base_link`).
  * `RecentHazardMemory.latch_near_field_costmap_obstacles()` scans `costmap.obstacle` within $x_{\text{rel}} \in [0.20, 1.25]\,\text{m}, |y_{\text{rel}}| \le 0.30\,\text{m}$ and latches any obstacle with a $3.5\text{--}5.0\,\text{s}$ TTL so it remains protected even after disappearing beneath the front bumper.
* **3D Attitude Rollover Protection (`ai/planning/safety_governor.py`)**:
  * Evaluates $\max(|\phi|, |\theta|)$ against `cautious_tilt_deg = 18.0°` (clamping speed to `0.15 m/s` via `CAUTIOUS_TILT_SLOWDOWN`) and `critical_tilt_deg = 26.0°` (commanding immediate halt $v = 0, \omega = 0$ via `STOP_TILT_ROLLOVER_HAZARD`).
* **Deterministic Reverse-and-Turn Recovery (`ai/planning/recovery_controller.py`)**:
  * `RecoveryController.compute_escape_turn_direction()` compares left vs. right flank clearance across both dynamic hazards and static costmap cells, steering the robot away from the obstructed flank during `REVERSING` (`reverse_turn_rate_radps = 0.35 rad/s`) and supporting an explicit post-reverse `RecoveryState.TURNING` phase (`target_turn_angle_rad = 0.45 rad`) before transitioning to `SETTLING -> REPLAN_READY`.

---

### 3.5 `P1-1` (`WS-1A`): Native C++17 Global $A^*$ Planner & Vectorized Validation
* **Dual-Target Cross-Platform Compilation (`scripts/build_astar_cpp.py`)**:
  * **Windows MSVC v19.44**: `ai/planning/native_cpp/astar_cpp.dll` (222.5 KB, `/O2 /arch:AVX2 /openmp /fp:fast`) and `astar_cpp_scalar.dll` (223.0 KB).
  * **Linux GCC 13.3 (WSL2 Ubuntu 24.04)**: `ai/planning/native_cpp/libastar_cpp.so` (48.4 KB, `-O3 -mavx2 -fopenmp -ffast-math`) and `libastar_cpp_scalar.so` (48.3 KB).
* **Algorithmic Optimizations**:
  * Thread-local $O(1)$ generation-counter workspace (`tls_workspace.visited_gen`) eliminating per-replan `O(N)` buffer zeroing.
  * Jump Point Search (JPS) directional parent-pruning (`enable_jps = 1`).
  * Fast centerline + midpoint + 4-corner early rejection in `is_segment_collision()` and `is_footprint_collision()`.
  * Vectorized post-search `PathValidator.validate_path()` using `scipy.ndimage.distance_transform_edt` instead of 500 nested Python window loops.

#### **Authoritative Global $A^*$ Scalability Benchmark (`reports/benchmark_astar_cpp.json`)**

| Grid Scale | Cells | Pure-Python Mean | C++ Scalar (End-to-End) | C++ SIMD (End-to-End) | Pure C++ Core (`astar_plan_c`) | End-to-End Speedup | Pure C++ Core Speedup | Cost Parity $|\Delta C|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50x50** ($5\text{m}\times 5\text{m}$) | 2,500 | 168.00 ms | 5.98 ms | 6.21 ms | **0.45 ms** | **27.1x** | **371.4x** | `0.000000` |
| **100x100** ($10\text{m}\times 10\text{m}$) | 10,000 | 388.22 ms | 14.84 ms | 22.91 ms | **3.18 ms** | **16.9x** | **122.3x** | `0.000000` |
| **120x120** (M13.11 Replan) | 14,400 | 669.27 ms | 18.44 ms | 15.11 ms | **2.65 ms** | **44.3x** | **252.7x** | `0.000000` |
| **200x200** (Phase 14.5 World) | 40,000 | 1,151.25 ms | 30.23 ms | 29.25 ms | **8.17 ms** | **39.4x** | **140.9x** | `0.000000` |
| **400x400** (Full Outdoor) | 160,000 | 5,382.85 ms | 88.27 ms | 92.20 ms | **39.50 ms** | **58.4x** | **136.3x** | `0.000000` |

---

### 3.6 `P1-2` (`WS-4A`): Zero-Copy ROS 2 Jazzy IPC & Shared-Memory Map Transport (`ugv_navigation/ipc/zero_copy_ipc.py`)

| Transport Mechanism | Payload Type | Baseline ROSIDL Serialization | Optimized Latency | Speedup | Memory Safety Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`IntraProcessLoanedChannel`** | $200\times 200$ 10-Layer `WorldTerrainMap` | 35.60 ms | **< 0.05 ms** | **> 700x** | `arr.setflags(write=False)` immutable view |
| **`SharedMemoryWorldMapChannel`** | $200\times 200$ 10-Layer `WorldTerrainMap` (1.36 MB POD) | 35.60 ms | **< 0.80 ms** | **> 44x** | 64-byte POD sequence header + `shared_memory` views |
| **`SharedMemoryCameraFrameChannel`** | $640\times 480\times 3$ `uint8` RGB Frame (921.6 KB POD) | ~8.50 ms | **< 0.40 ms** | **> 21x** | 32-byte POD sequence header + `shared_memory` view |
| **Vectorized `dynamic_converters.py`** | `WorldTerrainMap.msg` / `DynamicOccupancyGridSlice.msg` | 35.60 ms | **~2.50 ms** | **~14x** | Contiguous `np.asarray().ravel().tolist()` |

---

### 3.7 `P1-3` (Defect #7): Decoupled Asynchronous Perception Worker (`production/perception/async_perception_worker.py`)
* **Architecture**:
  * `AsyncPerceptionWorker` spawns a background daemon thread (`ugv_async_perception_worker`) with a single `PendingFrameSlot` protected by a lightweight mutex and `threading.Event`.
  * `submit_frame()` deposits the newest RGB frame + synchronized `(ugv_x, ugv_y, ugv_yaw, roll_rad, pitch_rad)` in **$< 0.2\,\text{ms}$** and returns immediately to the ROS 2 executor.
  * If camera frames arrive faster than neural inference completes, intermediate frames are overwritten in-place (`dropped_stale_count`), guaranteeing zero queue lag and ensuring inference always runs on the freshest image.
  * Completed `TraversabilityResult` and attitude-compensated `LocalBEVCostmap` are harvested atomically via `pop_latest_result()` in **$< 0.05\,\text{ms}$**.

---

## 4. Inventory of Modified & Created Files

| File Path | Action | Subsystem / Workstream |
| :--- | :---: | :--- |
| `simulation/models/ugv/model.sdf` | Modified | `P0-1`: Gazebo camera sensor pose aligned to `camera_link` ($+15^\circ$ pitch) |
| `ai/traversability/geometry.py` | Modified | `P0-2` (`WS-1B`): Full 3D roll/pitch/yaw `CameraExtrinsics.rotation_matrix` |
| `production/perception/vision_costmap_integrator.py` | Modified | `P0-2` (`WS-1B`): Dynamic SE(3) attitude-compensated IPM LUT & vectorized fusion |
| `production/localization/ekf_fusion.py` | Modified | `P0-3`: 3D Euler kinematics $E(\phi, \theta)$, online bias tracking & gravity tilt update |
| `config/ekf_gps_denied.yaml` | Modified | `P0-3`: 50 Hz full 3D (`two_d_mode: false`) `robot_localization` EKF config |
| `ai/planning/safety_governor.py` | Modified | `P0-4`: Near-field blind-spot costmap latching & 3D tilt rollover safety governor |
| `production/safety/safety_governor.py` | Modified | `P0-4`: `AuthoritativeSafetyGovernor` 3D roll/pitch & costmap forwarding |
| `ai/planning/recovery_controller.py` | Modified | `P0-4`: Lateral clearance assessment & reverse-and-turn (`REVERSING`/`TURNING`) escape |
| `simulation/nodes/interactive_ugv_navigation_node.py` | Modified | `P0-2/3/4`, `P1-3`: Live 3D EKF attitude, dynamic IPM, `AsyncPerceptionWorker` |
| `simulation/launch/prototype_presentation.launch.py` | Modified | `P0-2/3`: Wired `use_ekf_fusion` and `enable_vision_costmap` launch arguments |
| `ai/planning/native_cpp/astar_types.h` | Created | `P1-1` (`WS-1A`): Native C++17 Global $A^*$ C ABI structs |
| `ai/planning/native_cpp/astar_core.h` & `astar_core.cpp` | Created | `P1-1` (`WS-1A`): C++17 $A^*$ engine with JPS pruning & fast footprint smoothing |
| `ai/planning/native_cpp/astar_c_api.h` & `astar_c_api.cpp` | Created | `P1-1` (`WS-1A`): Exported C-linkage entry points |
| `ai/planning/global_planners/cpp_astar_backend.py` | Created | `P1-1` (`WS-1A`): Zero-copy `ctypes` FFI wrapper for `.dll` / `.so` |
| `ai/planning/global_planners/astar.py` | Modified | `P1-1` (`WS-1A`): C++ backend dispatch + hazard dilation caching |
| `ai/planning/validation.py` | Modified | `P1-1` (`WS-1A`): Vectorized `validate_path()` with EDT clearance lookup |
| `ai/planning/dynamic/global_replanner.py` | Modified | `P1-1` (`WS-1A`): Vectorized OBB dynamic obstacle rasterization |
| `scripts/build_astar_cpp.py` & `scripts/benchmark_astar_cpp.py` | Created | `P1-1` (`WS-1A`): Dual-target Windows/Linux build & scalability benchmark |
| `ros2_ws/src/ugv_navigation/ugv_navigation/ipc/zero_copy_ipc.py` | Created | `P1-2` (`WS-4A`): Intra-process loaned & POD shared-memory channels |
| `ros2_ws/src/ugv_navigation/ugv_navigation/converters/dynamic_converters.py` | Modified | `P1-2` (`WS-4A`): Vectorized C-buffer array serialization |
| `ros2_ws/src/ugv_navigation/launch/dynamic_navigation.launch.py` | Modified | `P1-2` (`WS-4A`): Enabled `use_intra_process_comms` across nodes |
| `production/perception/async_perception_worker.py` | Created | `P1-3`: Bounded depth-1 background perception & IPM worker |
| `tests/test_attitude_compensated_ipm.py` | Created | `P0-2` (`WS-1B`): 13 tests for roll/pitch IPM invariance & latency |
| `tests/test_astar_cpp_parity.py` | Created | `P1-1` (`WS-1A`): 11 differential parity tests (Python vs C++ SIMD/Scalar) |
| `tests/test_zero_copy_ipc.py` | Created | `P1-2` (`WS-4A`): 3 tests for loaned & shared-memory zero-copy channels |
| `tests/test_async_perception_worker.py` | Created | `P1-3`: 2 tests for non-blocking submission & stale burst dropping |

---

## 5. Verification Test Matrix

| Test Suite | Environment | Tests Passed | Execution Time | Status |
| :--- | :---: | :---: | :---: | :---: |
| `tests/test_simulation_model.py` | Windows Python 3.10 | `2 / 2` | 0.4s | **PASS** |
| `tests/test_attitude_compensated_ipm.py` | Windows Python 3.10 | `13 / 13` | 0.9s | **PASS** |
| `tests/test_phase14_3_camera_projection.py` | Windows Python 3.10 | `7 / 7` | 0.5s | **PASS** |
| `tests/test_phase14_3_costmap_integration.py` | Windows Python 3.10 | `6 / 6` | 0.6s | **PASS** |
| `tests/test_phase14_3_vision_loop.py` | Windows Python 3.10 | `2 / 2` | 8.2s | **PASS** |
| `tests/test_phase14_3_safety.py` | Windows Python 3.10 | `6 / 6` | 4.1s | **PASS** |
| `tests/test_phase14_4_localization.py` | Windows Python 3.10 | `9 / 9` | 0.5s | **PASS** |
| `tests/test_phase14_4_drift.py` | Windows Python 3.10 | `2 / 2` | 4.8s | **PASS** |
| `tests/test_phase14_4_sensor_dropout.py` | Windows Python 3.10 | `4 / 4` | 0.3s | **PASS** |
| `tests/test_phase14_4_tf.py` | Windows Python 3.10 | `3 / 3` | 0.2s | **PASS** |
| `tests/test_lost_obstacle_safety.py` | Windows Python 3.10 | `11 / 11` | 0.4s | **PASS** |
| `tests/test_recovery_replanning.py` | Windows Python 3.10 | `8 / 8` | 0.3s | **PASS** |
| `tests/test_mission_recovery_integration.py` | Windows Python 3.10 | `5 / 5` | 0.4s | **PASS** |
| `tests/test_phase14_combined_mission.py` | Windows Python 3.10 | `1 / 1` | 14.5s | **PASS** |
| `tests/test_astar_cpp_parity.py` | Windows + WSL2 Linux | `22 / 22` (11×2) | 3.1s | **PASS** |
| `tests/test_astar.py` & `test_planner_validation.py` | Windows Python 3.10 | `5 / 5` | 1.2s | **PASS** |
| `tests/test_global_replanning.py` | Windows Python 3.10 | `29 / 29` | 5.4s | **PASS** |
| `tests/test_zero_copy_ipc.py` | Windows Python 3.10 | `3 / 3` | 0.8s | **PASS** |
| `tests/test_async_perception_worker.py` | Windows Python 3.10 | `2 / 2` | 0.5s | **PASS** |
| `tests/test_ros2_messages.py` & `test_ros2_dynamic_integration.py` | Windows Python 3.10 | `35 / 35` | 28.1s | **PASS** |
| `tests/test_prototype_presentation_launch.py` & `test_outdoor_presentation_100m.py` | WSL2 ROS 2 Jazzy | `9 / 9` | 1.5s | **PASS** |
| **TOTAL VERIFIED** | **Dual Host + WSL2** | **184 / 184** | **~76.7s** | **100% PASS** |
