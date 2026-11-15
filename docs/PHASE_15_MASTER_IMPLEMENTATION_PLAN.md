# Phase 15: Advanced 2.5D Terrain-Aware Autonomous Navigation — Master Implementation Plan

**Project**: SIH26126 — Vision-Based Autonomous Navigation for Unmanned Ground Vehicle (UGV) in Outdoor Environments  
**Document**: `docs/PHASE_15_MASTER_IMPLEMENTATION_PLAN.md`  
**Strategic Audit Reference**: [`reports/PHASE_15_STRATEGIC_AUDIT.md`](../reports/PHASE_15_STRATEGIC_AUDIT.md)  
**Optimization Ledger Reference**: [`docs/OPTIMIZATION_WORKSTREAMS_MASTER.md`](OPTIMIZATION_WORKSTREAMS_MASTER.md)  
**Branch**: `demo-01-interactive-ugv`  
**Phase Status**: **SUBPHASES 15.0–15.9 COMPLETE (`72/72` Unit & Benchmark Tests PASS; `100%` 7-Scenario Differential Benchmark Success)**  

---

## 1. Phase 15 Core Objective & Capability Leap

In Phase 14, the navigation stack operated under a **flat 2D planar ground assumption ($Z = 0$)**. While visually traversable trails, boulders, and moving obstacles were handled effectively on flat ground, a purely 2D stack cannot distinguish between:
- Green grass on a safe $8^\circ$ trail vs. green grass on a lethal **$32^\circ$ rollover embankment**,
- A flat dirt patch vs. an **$18\,\text{cm}$ rock ledge** that will high-center a UGV with $12\,\text{cm}$ ground clearance,
- A safe turn on level ground vs. a high-speed turn across a **$24^\circ$ lateral cross-slope** that induces dynamic tip-over,
- A shadow across the trail vs. a **$-20\,\text{cm}$ negative ditch / drop-off**.

### Concrete Capability Statement
> **Phase 15** upgrades the SIH26126 UGV stack from a 2D planar navigator to a **2.5D Manifold Terrain-Aware Navigation System**. It ingests metric depth via the hardware-agnostic `DepthSource` abstraction, compensates camera ray projections for instantaneous 3D chassis roll and pitch ($\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$), maintains a 2.5D world elevation map $Z(x,y)$, extracts surface normals $\hat{\mathbf{n}}$, slope $\alpha$, roughness $\sigma_z$, positive step heights $\Delta Z^+$, and negative drop-offs $\Delta Z^-$, enforces undercarriage belly clearance and quasi-static rollover stability margins in Global $A^*$ and Native C++ AVX2 DWA, and validates performance across a 7-scenario differential benchmark comparing Phase 14 (2D Planar) vs. Phase 15 (2.5D Terrain-Aware).

---

## 2. Architectural Scope & Explicit Non-Goals

### Included in Phase 15 Scope:
1. **Camera-Agnostic Metric Depth Ingestion (`DepthSource`):** Supports simulated Gazebo/analytical uneven terrain depth, stereo disparity depth, and RGB-D sensors without vendor SDK lock-in.
2. **2.5D Multi-Layer World Map (`WorldTerrainMap`):** Elevation $Z(x,y)$, unit surface normals $[n_x, n_y, n_z]$, slope angle $\alpha(x,y)$, roughness $\sigma_z(x,y)$, step height $\Delta Z(x,y)$, and negative ditch hazard masks.
3. **3D SE(3) Attitude-Compensated Projection:** Real-time EKF roll/pitch ray compensation eliminating ghost obstacles during vehicle pitching/rolling.
4. **Undercarriage High-Centering & Rollover Decision Models:** Configured engineering safety thresholds (`20.0°` max safe slope, `28.0°` critical rollover slope, `0.12 m` max step / belly clearance) enforced in costmaps, Global $A^*$, Native C++ DWA, and the Authoritative Safety Governor.

### Explicit Non-Goals (Excluded to Prevent Scope Creep):
- **NO Full 3D/4D Voxel or Lattice Planners (OctoMap / MoveBase3D $O(N^6)$ search):** Computationally unfeasible for real-time CPU execution; the UGV is holonomically constrained to the 2.5D ground surface $Z = f(x, y)$.
- **NO Overhang / Cave / Multi-Level Canopy Modeling:** Arbitrary 3D volumetric geometry is outside the outdoor trail operating envelope.
- **NO Vision-Inferred Physical Friction ($\mu$) or Bekker-Wong Terramechanics:** Visual semantic labels cannot reliably measure soil shear strength or tire friction coefficients.
- **NO Physical Hardware / Jetson Flashing Dependency:** All milestones are 100% verifiable in deterministic Software-in-the-Loop (SIL) and Gazebo simulation on host CPU.

---

## 3. End-to-End 2.5D Terrain-Aware Architecture

```
+----------------------------------------------------------------------------------------------------------+
|                         PHASE 15: 2.5D TERRAIN-AWARE NAVIGATION ARCHITECTURE                             |
|                                                                                                          |
|  [RGB Camera] ──► [RealtimeTraversabilityPipeline (ONNX INT8/FP32)] ──► Drivability τ(u, v)              |
|                                                                                │                         |
|  [DepthSource Abstraction] ──► Metric Depth D(u, v)                            ▼                         |
|    ├── GazeboSimulatedDepthSource                           [3D SE(3) Attitude Ray Unprojector]          |
|    ├── AnalyticalTerrainDepthSource                           P_level = t_cam(φ,θ) + z_c * R(φ,θ) r_base |
|    └── StereoDisparityDepthSource                                              │                         |
|                                                                                ▼                         |
|  [15-State MultiSensorEKF] ───────────────────────────────► [2.5D ElevationMapper & WorldTerrainMap]     |
|    ├── 3D Pose (x, y, z) & Attitude (roll φ, pitch θ, yaw ψ)  ├── Layer 0: Elevation Z(x, y) [Kalman EMA]|
|    └── Gravity-Vector Roll/Pitch Tilt Observability           ├── Layer 1: Surface Normal [nx, ny, nz]   |
|                                                               ├── Layer 2: Slope Angle α = arccos(nz)    |
|                                                               ├── Layer 3: Step Height ΔZ+ & Ditch ΔZ-   |
|                                                               ├── Layer 4: Roughness σ_z & Belly Crown   |
|                                                               └── Layer 5: Fused Master Cost C_total     |
|                                                                      │                                   |
|                          ┌───────────────────────────────────────────┼──────────────────────────┐        |
|                          ▼                                           ▼                          ▼        |
|        [Terrain-Constrained C++17 Global A*]         [Terrain-Aware C++17 AVX2 DWA]    [Terrain Safety   |
|         ├── Penalizes uphill/downhill grade           ├── Evaluates 375 rollouts        Governor]        |
|         ├── Blocks α >= 28° & ΔZ >= 0.12m             ├── Quasi-Static Rollover Margin  ├── 18°/26° IMU  |
|         └── Discovers gentle switchback trails        ├── Cross-slope arc rejection     ├── Downhill     |
|                                                       └── Speed derating v <= vmax*cosα │   stop margin  |
|                          │                                           │                  └── Ditch brake  |
|                          └───────────────────────────────────────────┼──────────────────────────┘        |
|                                                                      ▼                                   |
|                                                       [SafeCommandBuffer (20 Hz)] ──► /cmd_vel           |
+----------------------------------------------------------------------------------------------------------+
```

---

## 4. Detailed 10-Subphase Implementation Plan (Subphases 15.0 to 15.9)

| Subphase | Title | Primary Files | Target Budget / Exit Criteria | Status |
| :---: | :--- | :--- | :--- | :---: |
| **15.0** | Metric Depth Foundation & Uneven Terrain Generator | `ai/traversability/depth_source.py`, `production/perception/depth_unprojector.py`, `tests/test_phase15_0_depth_foundation.py` | Depth-to-3D unprojection error $< \pm 0.02\,\text{m}$ over $0.5\text{--}5.0\,\text{m}$ across flat, ramp, step, and ditch geometries | **`[x] COMPLETE` (`0.22 ms` mean, `2/2 PASS`)** |
| **15.1** | 2.5D Elevation Mapping & Temporal Height Fusion | `production/perception/elevation_mapper.py`, `ai/spatial_fusion/world_map.py`, `tests/test_phase15_1_elevation_mapper.py` | Variance-weighted / EMA elevation grid $Z(x,y)$ accuracy $< \pm 0.03\,\text{m}$, outlier rejection, $< 2.0\,\text{ms}$ update | **`[x] COMPLETE` (`0.99 ms` mean, `10/10 PASS`)** |
| **15.2** | Vectorized Surface Normals, Slope, Steps & Negative Ditch Detection | `production/perception/terrain_geometry.py`, `tests/test_terrain_geometry_vectorized.py` | Normals $\hat{\mathbf{n}}$, slope $\alpha$, roughness $\sigma_z$, positive steps $\Delta Z^+ \ge 0.12\,\text{m}$, and negative ditches $\Delta Z^- \le -0.15\,\text{m}$ in $< 3.0\,\text{ms}$ | **`[x] COMPLETE` (`1.86 ms` mean, `14/14 PASS`)** |
| **15.3** | Dynamic SE(3) Attitude-Compensated Camera Projection | `production/perception/vision_costmap_integrator.py`, `tests/test_attitude_compensated_ipm.py` | Real-time $\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$ ray projection in $< 3.5\,\text{ms}$; stationary obstacle drift $\le 0.10\,\text{m}$ under $\pm 12^\circ$ tilt | **`[x] COMPLETE` (`1.61 ms` mean, `17/17 PASS`)** |
| **15.4** | Multi-Factor 2.5D Costmap Fusion & Belly Clearance Guard | `production/perception/terrain_geometry.py`, `ai/perception_fusion/cost_model.py`, `tests/test_phase15_4_terrain_costmap.py` | Lethal gates ($\alpha \ge 28^\circ$, $\Delta Z^+ \ge 0.12\,\text{m}$, $\Delta Z^- \le -0.15\,\text{m}$) + undercarriage high-centering crown check | **`[x] COMPLETE` (`1.50 ms` mean, `4/4 PASS`)** |
| **15.5** | Terrain-Constrained C++17 Global $A^*$ Planning | `ai/planning/native_cpp/astar_core.cpp`, `ai/planning/global_planners/astar.py`, `tests/test_phase15_5_terrain_astar.py` | $A^*$ routes around $>28^\circ$ embankments and boulder step fields via gentle trails in $< 5.0\,\text{ms}$ (120x120) | **`[x] COMPLETE` (`1.74 ms` mean, `5/5 PASS`)** |
| **15.6** | Terrain-Aware C++17 AVX2 DWA & Quasi-Static Rollover Margin | `ai/planning/native_cpp/dwa_core.cpp`, `ai/planning/backends/cpp_backend.py`, `tests/test_phase15_6_terrain_dwa.py` | C++ DWA evaluates cross-slope rollover margin $\ge 0.20$, rejects tip-over arcs, and derates speed $v \le v_{\max}\cos\alpha$ in $< 3.5\,\text{ms}$ | **`[x] COMPLETE` (`1.32 ms` mean, `11/11 PASS`)** |
| **15.7** | Authoritative 3D Terrain & Downhill Braking Safety Governor | `ai/planning/safety_governor.py`, `production/safety/safety_governor.py`, `tests/test_phase15_7_terrain_governor.py` | $18^\circ / 26^\circ$ IMU rollover clamp + slope-adjusted stopping distance $d_{\text{stop}}(\theta)$ and forward cliff/ditch halt | **`[x] COMPLETE` (`5/5 PASS`)** |
| **15.8** | Uneven Gazebo Terrain World & RViz 2.5D Telemetry | `simulation/worlds/outdoor_uneven_terrain_100m.sdf`, `simulation/nodes/presentation_visualization_adapter.py` | SDF uneven terrain features (ramps, cross-slopes, rock steps) + RViz slope/elevation marker telemetry | **`[x] COMPLETE` (`3/3 PASS`)** |
| **15.9** | 7-Scenario Differential Terrain Benchmark Suite | `benchmarks/run_phase15_terrain_benchmark.py`, `reports/PHASE_15_BENCHMARK_REPORT.md` | Quantitative comparison of Phase 14 (2D Planar) vs. Phase 15 (2.5D Stack) across all 7 scenarios (0 rollovers, 0 high-centerings) | **`[x] COMPLETE` (`7/7 Scenarios PASS`, `1/1 Suite PASS`)** |

---

### Subphase Breakdown & Engineering Specifications

#### **Subphase 15.0: Metric Depth Foundation & Uneven Terrain Ground-Truth Generator**
* **Goal**: Establish a deterministic, calibrated source of metric depth $D(u,v)$ and 3D point clouds $\mathbf{P}_{\text{level}}(u,v) = [X, Y, Z]^T$ over known geometric ground-truth surfaces before elevation mapping.
* **Deliverables**:
  1. `AnalyticalTerrainDepthSource` in `ai/traversability/depth_source.py`: Ray-casts camera pixels against parametric 2.5D heightfields $Z = f(x, y)$ (flat planes, inclined ramps, cross-slopes, discrete step ledges, ditches, and Gaussian hills) under arbitrary vehicle pose $(x, y, z, \phi, \theta, \psi)$.
  2. `DepthUnprojector` in `production/perception/depth_unprojector.py`: Vectorized unprojection of any `(H, W)` depth map + camera intrinsics/extrinsics + EKF attitude $(\phi, \theta)$ into 3D points $[X_{\text{level}}, Y_{\text{level}}, Z_{\text{level}}]^T$ in $< 1.5\,\text{ms}$.
* **Exit Criteria**: Reconstructed 3D points match analytical terrain elevation $Z(x,y)$ within $< \pm 0.02\,\text{m}$ across $0.5\text{--}5.0\,\text{m}$ range (`tests/test_phase15_0_depth_foundation.py`).

---

#### **Subphase 15.1: 2.5D Elevation Mapping & Temporal Height Fusion (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Continuously fuse unprojected 3D terrain points into the metric `elevation` layer $Z(x,y)$ of `LocalBEVCostmap` and `WorldTerrainMap`.
* **Deliverables**:
  1. `ElevationMapper` (`production/perception/elevation_mapper.py`) and extended `WorldTerrainMap` (`ai/spatial_fusion/world_map.py`):
     * Bins 3D points $(X_w, Y_w, Z_w)$ into `WorldTerrainMap` cells using fast bounded-span flat-index `np.bincount` and tracks per-cell `elevation`, `elevation_valid`, `elevation_var`, `elevation_min`, and `elevation_max`.
     * Applies vectorized leave-one-out per-cell outlier rejection (`outlier_deviation_m = 0.18 m`, `outlier_zscore_thresh = 1.15`) to discard flying-pixel depth spikes before cell averaging.
     * Applies minimum point-count gating (`min_points_per_cell = 2`) and variance-weighted Kalman / exponential moving average (EMA) temporal smoothing (`alpha = 0.35`) to suppress depth sensor noise while preserving sharp step edges.
     * Tracks `elevation_valid` mask on `WorldTerrainMap` and `LocalBEVCostmap` so unobserved cells are never hallucinated as $Z = 0$ next to elevated terrain (preventing artificial cliff artifacts at the FOV boundary).
* **Verification Evidence**: `tests/test_phase15_1_elevation_mapper.py` (**`10/10 PASS`**) — ramp/planar MAE $< 0.005\,\text{m}$ (max error $< 0.03\,\text{m}$), multi-frame noisy RMSE $< 0.015\,\text{m}$, $100\%$ flying-pixel spike rejection, zero FOV-boundary cliff artifacts on $+0.65\,\text{m}$ elevated terrain, and **`0.99 ms` mean update latency** ($1.67\,\text{ms}$ P95, well within $< 2.0\,\text{ms}$ budget).

---

#### **Subphase 15.2: Vectorized Surface Normals, Slope, Steps & Negative Ditch Detection (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Extract complete 2.5D geometric risk attributes from `WorldTerrainMap.elevation` in $< 3.0\,\text{ms}$.
* **Deliverables**:
  1. Extended `VectorizedTerrainGeometryEstimator` (`production/perception/terrain_geometry.py`):
     * Boundary-aware finite-difference gradient and morphological extrema computation restricted strictly to `elevation_valid` cells (preventing false step/slope/ditch spikes at the boundary between observed and unobserved regions).
     * **Negative Obstacle / Ditch Detection (`negative_step_mask` & `ditch_depth_m`):** Distinguishes upward positive steps ($\Delta Z^+ \ge 0.12\,\text{m}$) from downward negative ditches ($\Delta Z^- \le -0.12\,\text{m}$ below the local ambient ground plane) and enforces hard lethal gates (`obstacle = True`, `hazard = 1.0`, `cost = 1.0`).
* **Verification Evidence**: `tests/test_terrain_geometry_vectorized.py` (**`14/14 PASS`**) — $< 0.05^\circ$ slope error across $10^\circ\text{--}32^\circ$ ramps, accurate $+0.15\,\text{m} / +0.25\,\text{m}$ positive step and $-0.15\,\text{m} / -0.22\,\text{m}$ negative ditch detection, zero false FOV-boundary spikes on partially observed $14^\circ$ elevated ramps, and **`1.86 ms` mean latency** on $200 \times 200$ grids ($2.76\,\text{ms}$ P95, within $< 3.0\,\text{ms}$ budget).

---

#### **Subphase 15.3: Dynamic SE(3) Attitude-Compensated Camera Projection (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified (`P0-2 / WS-1B` + Subphase 15.3 Combined Roll/Pitch & Latency Hardening)**.
* **Deliverables**:
  1. `VisionCostmapIntegrator` (`production/perception/vision_costmap_integrator.py`):
     * Contiguous 1D base-ray buffers (`_rx_base`, `_ry_base`, `_rz_base`) with dynamic $\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$ SE(3) ray transformation and $O(1)$-seeded conservative min-pooling.
     * Supports 2.5D `elevation` and `elevation_valid` fusion from `LocalBEVCostmap` into `WorldTerrainMap`.
* **Verification Evidence**: `tests/test_attitude_compensated_ipm.py` (**`17/17 PASS`**) — stationary ground obstacle drift $\le 0.10\,\text{m}$ across pitch $[-10^\circ, +12^\circ]$, roll $[-12^\circ, +12^\circ]$, and simultaneous combined roll + pitch tilts, with **`1.61 ms` mean latency** ($2.06\,\text{ms}$ P95, well within $< 3.5\,\text{ms}$ budget).

---

#### **Subphase 15.4: Multi-Factor 2.5D Costmap Fusion & Undercarriage Belly Clearance Guard (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Fuse learned semantic drivability with 2.5D geometric hazards and verify undercarriage ground clearance ($0.12\,\text{m}$).
* **Deliverables**:
  1. **Undercarriage High-Centering Crown Check (`check_belly_clearance_grid`)** in `production/perception/terrain_geometry.py` and multi-factor cost fusion in `ai/perception_fusion/cost_model.py`:
     * Evaluates local terrain crown height relative to the surrounding wheel support plane ($5 \times 5$ / $7 \times 7$ outer wheel perimeter ring vs. $3 \times 3$ inner chassis belly):
       $$h_{\text{crown}}(r, c) = Z_{\text{belly}}(r, c) - \bar{Z}_{\text{wheels}}(r, c) - \text{smooth\_slope\_allowance}$$
     * If $h_{\text{crown}} \ge 0.12\,\text{m}$ (`ground_clearance_m`), the cell is marked lethal (`cost = 1.0`, `obstacle = True`, `hazard = 1.0`) to prevent chassis high-centering on boulders or sharp ridge crests.
     * Extended `CostModelWeights` and `TraversabilityCostModel` (`compute_cell_cost`, `compute_grid_cost`, `fuse_world_map_cost`) to enforce all four 2.5D lethal gates ($\alpha \ge 28^\circ$, $\Delta Z^+ \ge 0.12\,\text{m}$, $\Delta Z^- \le -0.12\,\text{m}$, and $h_{\text{crown}} \ge 0.12\,\text{m}$).
* **Verification Evidence**: `tests/test_phase15_4_terrain_costmap.py` (**`4/4 PASS`**) and `tests/test_perception_fusion.py` (**`5/5 PASS`**) — $30^\circ$ slopes, $+15\,\text{cm}$ steps, $-15\,\text{cm}$ ditches, and $+14\,\text{cm}$ belly-high-centering mounds are all marked lethal (`C = 1.0`, `obstacle = True`) even at $0.98$ visual traversability, with **`1.50 ms` mean crown-check latency** ($1.79\,\text{ms}$ P95 on $200 \times 200$).

---

#### **Subphase 15.5: Terrain-Constrained C++17 Global $A^*$ Planning (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Ensure Global $A^*$ discovers low-slope switchbacks and gentle trails around steep embankments and rough boulder fields.
* **Deliverables**:
  1. Extended `AStarPlannerCore::evaluate_cell_cost` and `smooth_path` (`ai/planning/native_cpp/astar_core.cpp`), `TerrainAwareCostEvaluator` (`ai/planning/costs.py`), `AStarPlanner` (`ai/planning/global_planners/astar.py`), and `CPPAStarBackend` (`ai/planning/global_planners/cpp_astar_backend.py`):
     * Enforces hard impassable cutoffs on critical slopes ($\ge 28.0^\circ$) and step heights ($\ge 0.12\,\text{m}$) with full vehicle footprint inflation.
     * Terrain-aware line-of-sight smoothing (`smooth_path`) bounds shortcut segment cost by $\min(C_{\text{max\_safe}}, \max(C(p_1), C(p_2)) + 0.12)$ so low-slope switchback detours around moderate hills ($18^\circ\text{--}25^\circ$) are preserved without shortcutting back across the hill.
     * Rebuilt `astar_cpp.dll` / `astar_cpp_scalar.dll` (Windows) and `libastar_cpp.so` / `libastar_cpp_scalar.so` (Linux WSL2).
* **Verification Evidence**: `tests/test_phase15_5_terrain_astar.py` (**`5/5 PASS`**) and `tests/test_astar_cpp_parity.py` (**`11/11 PASS`**) — $A^*$ consistently selects longer $8^\circ\text{--}10^\circ$ gentle detours around $32^\circ$ embankments, $18\,\text{cm}$ boulder fields, and $22^\circ$ rough hills with $100\%$ Python/C++ parity in **`1.74 ms` mean latency** ($2.11\,\text{ms}$ P95 on $120 \times 120$, well within $< 5.0\,\text{ms}$ budget).

---

#### **Subphase 15.6: Terrain-Aware Native C++17 AVX2 DWA Local Planning & Quasi-Static Rollover Margin (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Upgrade the local trajectory planner (`dwa_core.cpp` and `DWAPlanner`) to evaluate 3D terrain orientation, cross-slope rollover stability, and incline velocity derating along every candidate trajectory arc.
* **Deliverables**:
  1. **Quasi-Static Rollover Stability Margin** in `ai/planning/native_cpp/dwa_core.cpp` and `ai/planning/local_planners/dwa.py`:
     * For each candidate $(v, \omega)$ rollout step at $(x_s, y_s, \psi_s)$, evaluates local terrain slope $\alpha$, yaw-rate cross-slope coupling $\alpha_{\text{eff}}$, and centrifugal lateral acceleration:
       $$\text{Rollover Margin} = 1.0 - \frac{\alpha_{\text{eff}}}{\alpha_{\text{crit}}} - \frac{v^2 |\omega / \max(|v|, 0.1)|}{g \cdot \frac{W}{2 h_{\text{cg}}}}$$
       where $\alpha_{\text{crit}} = 28.0^\circ$, $W = 0.45\,\text{m}$, $h_{\text{cg}} = 0.18\,\text{m}$.
     * Any candidate trajectory where $\text{Rollover Margin} < 0.20$ is **rejected** as a rollover hazard.
  2. **Slope-Adaptive Velocity Derating**:
     * Automatically caps candidate linear velocity on inclines:
       $$v_{\text{eff\_max}}(\alpha) = v_{\max} \cdot \max(0.25, \cos\alpha)$$
  3. Rebuilt `dwa_cpp.dll` / `dwa_cpp_scalar.dll` (Windows) and `libdwa_cpp.so` / `libdwa_cpp_scalar.so` (Linux WSL2).
* **Verification Evidence**: `tests/test_phase15_6_terrain_dwa.py` (**`11/11 PASS`**) and `tests/test_dwa_cpp_parity.py` / `test_dwa_scalar_parity.py` / `test_dwa_backends.py` (**`21/21 PASS`**) — C++ DWA rejects high-speed turns on $22^\circ\text{--}24^\circ$ slopes while allowing low-speed aligned traversal, derates velocity by $\cos\alpha$, steers away from $25^\circ$ cross-slopes toward flat valley floors, and executes 375 candidates in **`1.32 ms` mean latency** ($1.63\,\text{ms}$ P95, well within $< 3.5\,\text{ms}$ mean / $< 5.0\,\text{ms}$ P95 budget).

---

#### **Subphase 15.7: Authoritative 3D Terrain & Downhill Braking Safety Governor (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Extend `SafetyGovernor` (`ai/planning/safety_governor.py`) with slope-compensated stopping distance and forward terrain hazard checks.
* **Deliverables**:
  1. **Downhill Stopping Distance Compensation (`compute_downhill_stopping_distance`)**:
     * On a downhill pitch ($\theta_{\text{downhill}} > 0$), gravitational component $g\sin\theta$ opposes braking deceleration $a_{\max}$, increasing required stopping distance:
       $$d_{\text{stop}}(\theta) = d_{\text{stop}, 0} \cdot \frac{a_{\max}}{\max(0.30, a_{\max} - g\sin\theta_{\text{downhill}})}$$
     * `SafetyGovernor.evaluate()` dynamically expands `stop_distance_m` and `cautious_distance_m` when descending slopes and inspects `costmap.slope` ($\ge 28.0^\circ$), `costmap.height_var` ($\ge 0.12\,\text{m}$), and `costmap.obstacle` in the immediate forward corridor.
* **Verification Evidence**: `tests/test_phase15_7_terrain_governor.py` (**`5/5 PASS`**) and `tests/test_lost_obstacle_safety.py` (**`11/11 PASS`**) — vehicle initiates braking earlier on downhill grades, clamps speed at $18^\circ$ IMU tilt, and halts before $\ge 28^\circ$ drop-offs, $\ge 0.12\,\text{m}$ steps/ditches, or $\ge 26^\circ$ IMU tilt.

---

#### **Subphase 15.8: Uneven Gazebo Terrain World & RViz 2.5D Telemetry (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Provide visual simulation assets and RViz telemetry for uneven terrain navigation.
* **Deliverables**:
  1. `simulation/worlds/outdoor_uneven_terrain_100m.sdf`: Extends the 100m presentation corridor with `steep_embankment_32deg` ($32^\circ$), `gentle_bypass_ramp_10deg` ($10^\circ$), `severe_cross_slope_24deg` ($24^\circ$), `step_boulder_cluster_20cm` ($18\text{--}20\,\text{cm}$), `negative_ditch_trench_15cm` ($-15\,\text{cm}$), and `undulating_pitch_hills_12deg` ($\pm 12^\circ$).
  2. Extended `CostmapOccupancyAdapter` (`simulation/nodes/presentation_visualization_adapter.py`) to reflect 2.5D `slope`, `height_var`, and `roughness` costs in `/costmap/occupancy` and generate 2.5D elevation/slope/step RViz telemetry markers (`generate_terrain_telemetry_markers`).
* **Verification Evidence**: `tests/test_phase15_8_uneven_world.py` (**`3/3 PASS`**) — validated SDF 1.9 XML syntax, world terrain elements, OccupancyGrid 2.5D cost mapping, and RViz HUD marker telemetry.

---

#### **Subphase 15.9: 7-Scenario Differential Terrain Benchmark Suite (`[x] COMPLETE`)**
* **Status**: **Implemented and Verified**.
* **Goal**: Quantitatively prove that the Phase 15 (2.5D Terrain-Aware) stack eliminates terrain-induced failures (rollovers, high-centerings, projection ghost stops) compared to the Phase 14 (Flat 2D Baseline) stack while maintaining parity on flat terrain.
* **Deliverables**:
  1. `benchmarks/run_phase15_terrain_benchmark.py`: Automated runner executing both **Baseline (Phase 14 2D)** and **Evaluation (Phase 15 2.5D)** stacks across all 7 canonical scenarios:
     * **Scenario 1 (Flat Baseline):** 100% completion parity (`8.00 m`), 0 latency regression.
     * **Scenario 2 (Gentle $10^\circ$ Slope):** Smooth traversal with adaptive speed derating (`0` rollovers, `0` high-centerings).
     * **Scenario 3 ($32^\circ$ Steep Embankment Detour):** Phase 14 attempts direct climb and suffers `25` rollover violations ($32.0^\circ$); Phase 15 detours via $10^\circ$ side trail (`10.27 m`) with **`0` rollover violations**.
     * **Scenario 4 ($24^\circ$ Severe Cross-Slope):** Phase 14 suffers `52` rollover violations; Phase 15 routes safely along the flat valley floor (`9.45 m`) with **`0` rollover violations**.
     * **Scenario 5 ($18\,\text{cm}$ Step / Boulder Field):** Phase 14 suffers `5` high-centering events on the $0.18\,\text{m}$ ledge; Phase 15 detects $\Delta Z \ge 0.12\,\text{m}$ and routes around (`9.89 m`) with **`0` high-centering events**.
     * **Scenario 6 (Undulating $12^\circ$ Pitching Hills):** Phase 14 flat IPM suffers `1.15 m` projection drift; Phase 15 SE(3) attitude IPM stabilizes projection drift to **`0.0707 m`** ($\le 0.10\,\text{m}$).
     * **Scenario 7 (Compound Outdoor Course + Dynamic Obstacles):** Simultaneous 2.5D terrain avoidance and dynamic obstacle evasion with **`0` collisions, `0` rollovers, `0` high-centerings**.
  2. Generated telemetry (`reports/phase15_benchmark.json`) and final report (`reports/PHASE_15_BENCHMARK_REPORT.md`).
* **Verification Evidence**: `tests/test_phase15_9_benchmark.py` (**`1/1 PASS`**) — **100.0% mission success (7/7 scenarios)** for Phase 15 vs. 28.6% (2/7) for Phase 14, with `0` collisions, `0` rollover violations, `0` high-centering events, `1.53 ms` mean C++17 A* latency, and `1.36 ms` mean C++17 AVX2 DWA latency.
