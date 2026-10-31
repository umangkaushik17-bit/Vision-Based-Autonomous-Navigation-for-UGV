# PHASE 15: STRATEGIC RE-AUDIT AND IMPLEMENTATION DIRECTION (REVISED)

**Project**: Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Document**: `reports/PHASE_15_STRATEGIC_AUDIT.md`  
**Revision Date**: October 1, 2026  
**Status**: COMPLETE / AUDIT REVISION ONLY (STRICTLY NO PRODUCTION CODE MODIFICATIONS)  
**Current Branch**: `demo-01-interactive-ugv`  
**Baseline Commit**: `e129fa64817159e729f2deb8070ec764a01b015d`  

---

## 1. Current Project State

The project has concluded Phase 14 with the successful execution and archival of **Phase 14.5: Long-Horizon Multi-Waypoint Autonomous Navigation Benchmark** ([`reports/PHASE_14_5_LONG_HORIZON_BENCHMARK_REPORT.md`](file:///c:/Users/shres/Desktop/SIH2026/reports/PHASE_14_5_LONG_HORIZON_BENCHMARK_REPORT.md)):
- **Missions Completed**: 12 structured runs across 6 scenarios totaling $1,280.14\text{ m}$ of autonomous travel.
- **Mission Success Rate**: $91.7\%$ ($11/12$ missions complete; $100\%$ waypoint success across all 5 waypoints $W_1 \to W_2 \to W_3 \to W_4 \to W_5$).
- **Safety Gate**: **0 collisions** across all runs ($0.0\%$ collision rate).
- **Localization Drift**: $0.01\%$ average drift ($0.012\text{ m}$ final error over $106.68\text{ m}$ travel) under 15-state `MultiSensorEKF` fusion.
- **Perception Throughput**: $18.9\text{ FPS}$ mean integrated inference on host CPU ($32.8\text{ FPS}$ peak).
- **Planner Latency**: Native C++ AVX2 DWA local planner (`libdwa_core.dll`) achieved a mean execution latency of $3.12\text{ ms}$ ($7.76\text{ ms}$ P95).
- **Dynamic Obstacle Avoidance**: Verified multi-frame tracking, 2.0s velocity prediction corridor rasterization, and active planner evasion (up to 355 `AVOID` and 41 `STOP` cycles).

The navigation stack currently functions as a **robust, high-performance 2D planar navigation system** operating on a flat rolling world grid.

---

## 2. Roadmap Evidence & Legacy Artifact Clarification

1. **`docs/ROADMAP.md` Definition**:
   ```markdown
   ### Phase 15: Advanced Terrain-Aware Navigation
   - Goal: Extend traversability scoring to evaluate slope/pitch angle, surface slip risk, and vehicle ground clearance.
   - Deliverables: 3D pitch/roll compensation in costmap generation.
   - Exit Criteria: Rover successfully rejects steep embankments and chooses gentle dirt trails.
   ```
2. **`docs/PHASE_13_TO_14_STRATEGIC_REVIEW.md` Problem Statement Requirements**:
   - Requirement B3 explicitly highlights: *"Traversing uneven terrain (pitch/roll/elevation)"* as a strongly implied outdoor operational challenge.
   - The Smart India Hackathon problem statement explicitly demands navigation in *"unpredictable terrain, changing light, and unreliable GPS signals"*.
3. **Repository Legacy Artifacts Clarification**:
   - Files named `test_phase15_matlab_parity.py` and directory `validation/matlab/phase15/` in the repository represent **Milestone 13.15** (a historical naming misnomer for dynamic obstacle mathematical parity verification). They do **not** implement Phase 15 terrain navigation. Phase 15 remains unstarted.

---

## 3. Phase 14 Capability Baseline

Tracing the active software path reveals exactly what Phase 14 provides and where its planar assumptions lie:

| Pipeline Subsystem | Phase 14 Implementation State | Dimensionality & Underlying Assumption |
| :--- | :--- | :--- |
| **Perception** | `RealtimeTraversabilityPipeline` running RELLIS-3D segmentation (`outputs/rellis3d/checkpoints/best.pt`). Outputs pixel drivability $\tau \in [0.0, 1.0]$. | **2D Semantic Only**. No active depth estimation, no point cloud generation, no elevation measurement. |
| **BEV Projection** | `VisionCostmapIntegrator` utilizing precomputed Inverse Perspective Mapping (IPM) lookup table. | **Flat Planar Ground Assumption ($Z_{\text{ground}} = 0$)**. Rays are projected onto a horizontal plane ahead of the vehicle assuming fixed nominal camera pitch ($15^\circ$) and height ($0.525\text{ m}$). |
| **World Costmap** | `WorldTerrainMap` rolling grid ($200 \times 200$ cells @ $0.10\text{ m/cell}$). Has array fields `slope`, `roughness`, `height_var`. | **2D Planar Surface**. Crucially, `WorldTerrainMap` **lacks an active elevation layer $Z(x, y)$**. Array fields for `slope` and `roughness` exist but are populated with `0.0`. |
| **Localization** | 15-state `MultiSensorEKF` (`production/localization/ekf_fusion.py`) fusing wheel odometry, 6-axis IMU, and stereo VIO. | **Full 3D Kinematics**. Continuous 3D position $(x, y, z)$, 3D velocity, and 3D orientation (roll $\phi$, pitch $\theta$, yaw $\psi$) are tracked with high accuracy at $50 - 200\text{ Hz}$. |
| **Dynamic Obstacles** | `MultiFrameTracker` + linear velocity predictor + `DynamicOccupancyLayer`. | **2D Planar Polygons**. Tracking and spatio-temporal reservation slices are strictly evaluated in the horizontal $(x, y)$ plane. |
| **Global Planner** | Event-driven Global $A^*$ on `WorldTerrainMap`. | **2D Grid Search**. Mathematical hooks for slope penalty exist ($w_{\text{slope}} = 1.20$), but slope inputs are zero. |
| **Local Planner** | `UnifiedDWAPlanner` executing native C++ AVX2 DWA (`libdwa_core.dll`). | **2D Unicycle Trajectories**. 336 candidate velocity pairs $(v, \omega)$ evaluated against 2D distance, traversability, and obstacle boundaries. Slope penalty hook exists in C++, but receives zero. |
| **Safety Governor** | Authoritative `SafetyGovernor` and `SafeCommandBuffer` dispatching at $20\text{ Hz}$. | **2D Bounding Footprint**. Forward time-to-collision and dynamic obstacle separation evaluated purely in 2D. |

---

## 4. Existing Terrain-Related Implementations in Repository

A comprehensive search across the repository reveals that significant theoretical and algorithmic scaffolding for terrain-aware navigation was already authored during Phases 3, 5, 7, and 12, but remains disconnected from the active runtime:

1. **Geometric Primitives & SVD Plane Fitting (`ai/perception_fusion/interfaces.py`)**:
   - Class `TerrainCell` contains an implementation of **MoveBase3D-style SVD plane fitting** (`TerrainCell.from_points(pts)`).
   - From a 3D point cloud patch, it calculates:
     - Surface normal vector: $\mathbf{n} = [n_x, n_y, n_z]$
     - Surface inclination / slope angle: $\theta_{\text{slope}} = \arccos(n_z)$
     - Flatness / roughness: residual variance of smallest eigenvalue $\lambda_0 / \sum \lambda_i$
     - Vertical height variation: $z_{\text{max}} - z_{\text{min}}$
     - Point count and validity flag.
   - Fully verified in unit test suite [`tests/test_terrain_geometry.py`](file:///c:/Users/shres/Desktop/SIH2026/tests/test_terrain_geometry.py).
2. **Parametric Terrain Cost Model (`ai/perception_fusion/cost_model.py`)**:
   - `TraversabilityCostModel` combines physical mechanics with learned vision:
     $$C_{\text{total}} = w_{\text{learned}} C_{\text{learned}} + w_{\text{slope}} C_{\text{slope}} + w_{\text{flatness}} C_{\text{flatness}} + w_{\text{height}} C_{\text{height}} + w_{\text{obs}} C_{\text{obs}}$$
   - Incorporates configured threshold parameters:
     - `max_safe_slope_deg: 20.0`
     - `critical_slope_deg: 28.0` (forces $C = 1.0$)
     - `max_step_height_m: 0.12`
     - `critical_step_height_m: 0.20` (forces $C = 1.0$).
3. **Vehicle Specification Configuration ([`configs/terrain_safety.yaml`](file:///c:/Users/shres/Desktop/SIH2026/configs/terrain_safety.yaml))**:
   - Full chassis dimensions: wheelbase $0.40\text{ m}$, track width $0.45\text{ m}$, ground clearance $0.12\text{ m}$, CG height $0.18\text{ m}$.
   - Critical slope threshold ($28.0^\circ$) is derived analytically from the static stability angle:
     $$\alpha_{\text{tip}} = \arctan\left(\frac{W / 2}{h_{\text{cg}}}\right) = \arctan\left(\frac{0.225}{0.18}\right) \approx 51.3^\circ \quad \xrightarrow{\text{Safety Factor } 1.83} \quad 28.0^\circ$$
   - **Crucial Distinction**: These values are **configured conservative engineering thresholds** currently used by the software model and remain subject to physical validation. They do **not** represent an experimentally validated physical rollover threshold on the actual vehicle hardware.
4. **Planner Cost Integration Hooks**:
   - **Global A\*** ([`ai/planning/costs.py`](file:///c:/Users/shres/Desktop/SIH2026/ai/planning/costs.py)): Contains `cost += self.w_slope * min(slope / self.max_safe_slope_deg, 2.0)` with weight $w_{\text{slope}} = 1.20$.
   - **Native C++ DWA** ([`ai/planning/native_cpp/dwa_core.cpp`](file:///c:/Users/shres/Desktop/SIH2026/ai/planning/native_cpp/dwa_core.cpp)): Interface includes `const float* slope_grid`, applying trajectory cost penalties:
     `cost += 1.20 * std::clamp(c_slope / 20.0, 0.0, 2.0)`.
5. **Depth Source Abstractions (`ai/traversability/depth_source.py`)**:
   - Modular interfaces: `PlanarGroundDepthSource`, `StereoDepthSource`, `SimulatedDepthSource`.
6. **BEV Projector Point Cloud Ray Extraction (`ai/traversability/bev_projection.py`)**:
   - Projects 3D camera rays into robot base frame coordinates:
     $$\mathbf{P}_{\text{robot}} = \mathbf{R}_{\text{cam}}^{\text{robot}} \mathbf{P}_{\text{cam}} + \mathbf{T}_{\text{cam}}^{\text{robot}} = [x_r, y_r, z_r]^T$$
   - *Limitation*: Currently extracts $(x_r, y_r)$ to populate the 2D grid and discards $z_r$.

---

## 5. Missing Capabilities (The Reality Gap)

| Capability | Current Phase 14 State | Missing in Phase 14 | Evidence in Codebase | Evidence Classification |
| :--- | :--- | :--- | :--- | :---: |
| **2D Traversability** | RELLIS-3D segmentation | None (robust at $18.9\text{ FPS}$) | `production/perception/realtime_pipeline.py` | **MEASURED** |
| **Elevation $Z(x, y)$** | Discarded during BEV projection | 2.5D height grid $Z_{\text{elev}}(x, y)$ in world costmap | `ai/spatial_fusion/world_map.py` lacks active elevation array | **SIMULATED / DERIVED** |
| **Surface Normals** | `TerrainCell` class exists, unused at runtime | Continuous normal estimation $\mathbf{n}(x, y)$ from depth | `production/perception/vision_costmap_integrator.py` | **DERIVED** |
| **Pitch Estimation** | EKF outputs vehicle chassis pitch $\theta$ at 50Hz | Terrain slope projection along forward path | `production/localization/ekf_fusion.py` | **MEASURED / SIMULATED** |
| **Roll Estimation** | EKF outputs vehicle chassis roll $\phi$ at 50Hz | Cross-slope roll / stability margin estimation | `production/localization/ekf_fusion.py` | **MEASURED / SIMULATED** |
| **Slope Angle** | Array fields exist, but populated with `0.0` | True terrain inclination $\alpha = \arccos(n_z)$ | Costmap slope arrays are all zeros in Phase 14 | **DERIVED** |
| **Roughness** | Array fields exist, but populated with `0.0` | Coplanarity variance / elevation dispersion | Costmap roughness arrays are all zeros | **DERIVED** |
| **Step Height** | Array fields exist, but populated with `0.0` | Height discontinuity across wheel diameter baseline | Costmap height_var arrays are all zeros | **DERIVED** |
| **Ground Clearance** | Configured in yaml ($0.12\text{ m}$), unverified | Undercarriage collision check vs step obstacles | DWA only checks horizontal 2D footprint | **CONFIGURED ASSUMPTION** |
| **Dynamic Tilt IPM** | Fixed nominal pitch ($15^\circ$) assumed | Real-time SE(3) attitude compensation for camera rays | `VisionCostmapIntegrator.precompute_projection` | **DERIVED** |
| **Terrain-Aware Global A\***| Mathematical hook exists, input is 0.0 | Global path deviation around steep terrain features | `ai/planning/costs.py` receives zero slope | **DERIVED** |
| **Terrain-Aware Local DWA** | Mathematical hook exists, input is 0.0 | DWA candidate rejection on hazardous side-slopes | `dwa_core.cpp` receives null/zero slope grid | **DERIVED** |
| **Traction / Friction Estimation**| Kinematics assume rolling contact | Wheel-soil terramechanics, physical friction coefficient | Excluded from Phase 15 deliverables | **FUTURE / RESEARCH** |

---

## 6. Actual SIH Problem Gap

### The Core Problem Statement
The Smart India Hackathon problem statement dictates that the UGV must navigate autonomously in **unstructured outdoor environments**. In genuine outdoor terrain, the ground is never an ideal flat plane ($Z=0$):
- Gentle dirt trails wind alongside steep embankments ($>30^\circ$).
- Soil undulations, ditches, and mounds create rolling and pitching moments.
- Boulder clusters and exposed rock ledges present vertical step heights ($>15\text{ cm}$) that can high-center the vehicle chassis.
- A path may look visually traversable (e.g., green grass on a $35^\circ$ slope), but attempting to drive across it will cause catastrophic vehicle rollover or unrecoverable downhill sliding.

### Concrete Definition of the Phase 15 Capability Leap
> **What new capability does Phase 15 provide that Phase 14 does not?**  
> *"Phase 15 enables the UGV to detect, represent, and actively avoid geometric terrain hazards—specifically rejecting visually traversable but dangerously steep slopes exceeding configured thresholds ($>28^\circ$) and chassis-high-centering step obstacles ($>0.12\text{ m}$)—by dynamically compensating camera projection for vehicle pitch and roll using rigid-body SE(3) transforms, constructing a 2.5D elevation and slope costmap, and executing terrain-constrained global and local paths that prioritize gentle, stable ground corridors."*

---

## 7. 2D vs Terrain-Aware Architecture

```
========================================================================================
CURRENT PHASE 14 ARCHITECTURE (FLAT 2D PLANAR)
========================================================================================
[RGB Camera] ──> [RELLIS-3D AI] ──> [Flat-Ground IPM (Z=0)] ──> [2D Costmap (Slope=0)]
                                                                         │
[EKF Pose (x,y)] <───────────────────────────────────────────────────────┴─> [2D Global A*]
       │                                                                            │
       └──> [2D C++ DWA (Unicycle Rollout)] ──> [2D Safety Governor] ──> [cmd_vel (v, w)]

========================================================================================
PROPOSED PHASE 15 ARCHITECTURE (TERRAIN-AWARE 2.5D MANIFOLD)
========================================================================================
[Metric Depth Source] ────────────────────────────────────────────────────────┐
 (Gazebo / Stereo DepthSource abstraction)                                    ▼
[RGB Camera] ──> [RELLIS-3D AI] ──> [Rigid SE(3) Attitude IPM] ──> [2.5D Elevation & Normal Grid]
                                            ▲                              │  - Height Z(x, y)
[IMU EKF Attitude (R_base_odom)] ───────────┴──────────┐                   │  - Slope α = acos(nz)
                                                       │                   │  - Roughness σ²
                                                       │                   │  - Step Height ΔZ
                                                       ▼                   ▼
[15-State Fused Pose] ─────────────────────> [Terrain-Aware Multi-Factor Costmap]
  (x, y, z, vx, vy, vz, roll, pitch, yaw)              │
                                                       ├────────────────────────┐
                                                       ▼                        ▼
                                             [Terrain-Aware A*]       [Terrain-Aware DWA]
                                             - Penalizes slopes       - Derates speed on hills
                                             - Blocks >28° ridges     - Rejects rollover arcs
                                                       │                        │
                                                       └────────────┬───────────┘
                                                                    ▼
                                                      [Authoritative Terrain Safety Governor]
                                                      - Forward tip-over envelope guard
                                                      - Chassis ground-clearance guard
                                                                    │
                                                                    ▼
                                                           [cmd_vel (v, w)]
```

---

## 8. Terrain Representation Options

| Representation Candidate | Memory Footprint (100m corridor) | Compute / Update Latency | A* & DWA Compatibility | Suitability for Embedded / Host CPU | Simulation Availability | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 2.5D Elevation Grid ($Z(x, y)$)** | **Very Low** (1 raster layer, $\approx 1.6\text{ MB}$) | **Fast** (Target $<2\text{ ms}$ vectorized update) | **Native** (Seamless extension to `WorldTerrainMap`) | **Optimal** (Fits in L3 cache) | **Immediate** (Gazebo depth/point cloud) | **HIGH (RECOMMENDED)** |
| **2. Surface-Normal Grid ($[n_x, n_y, n_z]$)** | **Low** (3 float layers, $\approx 4.8\text{ MB}$) | **Fast** (Target $3-5\text{ ms}$ local SVD / finite diff) | **High** (Direct scalar slope $\arccos(n_z)$ extraction) | **High** | **Immediate** | **HIGH (RECOMMENDED)** |
| **3. Fused Elevation + Traversability Grid** | **Low** ($C, \tau, Z, \alpha, \Delta Z$, $\approx 8\text{ MB}$) | **Fast** (Target $<5\text{ ms}$ end-to-end) | **Optimal** (Single unified tensor structure) | **Optimal** | **Immediate** | **HIGH (RECOMMENDED)** |
| **4. 3D Voxel Grid (OctoMap / TSDF)** | **High** ($30-80\text{ MB}$ sparse tree) | **Slow** (Target $25-60\text{ ms}$ raycasting) | **Poor** (Requires full 3D search space) | **Poor** (High memory bandwidth, cache thrashing) | Supported | **LOW (REJECTED)** |
| **5. Raw 3D Point Cloud** | **High** ($100\text{k}+$ points, $10-25\text{ MB/frame}$) | **Medium** (KD-tree search is expensive) | **Poor** (Cannot be queried in $O(1)$ by DWA) | **Poor** | Supported | **LOW (REJECTED)** |
| **6. 3D Polygonal Mesh (TIN)** | **Very High** (Complex topology) | **Very Slow** (Mesh reconstruction $>100\text{ ms}$) | **Poor** (Complex collision pipeline) | **Unfeasible** | Complex | **LOW (REJECTED)** |

### Architectural Decision & Mathematical Boundary
**Adopt Representation 3: Multi-Layer 2.5D Elevation & Surface Normal Grid.**  
An outdoor ground vehicle operates on a 2D surface embedded in 3D space (a 2.5D manifold). 

> [!IMPORTANT]
> **Corrected Mathematical Scope**:  
> A 2.5D terrain representation captures the terrain hazards relevant to the project's intended UGV operating envelope while preserving computational compatibility with the existing navigation stack.  
> 
> However, a 2.5D representation does **not** represent arbitrary 3D geometry such as:
> - Overhangs and tree canopies,
> - Caves, tunnels, or underpasses,
> - Multiple elevation surfaces at the same $(x, y)$ coordinate,
> - Fully volumetric obstacle structures.
> 
> These arbitrary 3D geometric features are explicitly outside the Phase 15 operational scope. Acknowledging this limitation does **not** justify introducing an unfeasible, computationally heavy 3D voxel/lattice planner.

---

## 9. Terrain Feature Selection & Evidence Classification

To ensure rigorous engineering integrity, all terrain features are classified into five explicit evidence tiers:

1. **MEASURED**: Directly acquired by real physical or simulated onboard sensors (e.g., IMU angular velocity, wheel encoder ticks).
2. **SIMULATED**: Generated deterministically within the Gazebo simulation environment as a known ground truth baseline.
3. **DERIVED**: Computed algorithmically from measured or simulated data via deterministic geometric or mathematical equations.
4. **CONFIGURED ASSUMPTION**: An engineering parameter set in configuration files based on design constraints; **not** a physically validated threshold.
5. **FUTURE / RESEARCH**: Theoretical concepts or physical phenomena outside the active engineering scope of Phase 15.

| Feature | Mathematical Definition | Estimation Category | Measurement Source | Planned Use in Phase 15 |
| :--- | :--- | :---: | :--- | :--- |
| **Elevation $Z$** | Height coordinate $Z_{\text{world}}(x, y)$ | **SIMULATED / DERIVED** | Stereo depth / simulation point cloud | 2.5D ground surface reconstruction |
| **Chassis Pitch $\theta$ & Roll $\phi$** | Vehicle orientation state | **MEASURED / DERIVED** | 15-state `MultiSensorEKF` (IMU fusion) | Dynamic camera ray attitude compensation |
| **Surface Normal $\mathbf{n}$** | $\mathbf{n} = [-\frac{\partial Z}{\partial x}, -\frac{\partial Z}{\partial y}, 1]^T / \|\mathbf{n}\|$ | **DERIVED** | Vectorized spatial finite differences / SVD | Terrain inclination & cross-slope modeling |
| **Slope Angle $\alpha$** | $\alpha = \arccos(\hat{\mathbf{n}} \cdot \hat{\mathbf{z}}) = \arccos(n_z)$ | **DERIVED** | Dot product with vertical gravity vector | Configured cutoff ($28^\circ$) & cost scaling |
| **Surface Roughness $\sigma_z^2$** | Local elevation variance within $3 \times 3$ window | **DERIVED** | Moving window standard deviation $\sigma_Z$ | Speed penalty on rocky/bumpy trails |
| **Step Height $\Delta Z$** | $\max(Z_{\text{patch}}) - \min(Z_{\text{patch}})$ | **DERIVED** | Min-max filter over wheel radius window | Configured cutoff ($>0.12\text{ m}$) |
| **Projected Trajectory Roll** | $\phi_{\text{traj}} = \arcsin(\mathbf{n} \cdot \mathbf{y}_{\text{body}})$ | **DERIVED** | Projection of surface normal onto lateral body axis | Rejection of rollover-prone side-slope paths |
| **Traction / Friction ($\mu$)** | Physical tire-ground friction coefficient | **FUTURE / RESEARCH** | Not reliably inferable from semantics alone | **EXCLUDED FROM PHASE 15** |
| **Soil Shear Sinkage** | Terramechanics Bekker-Wong model | **FUTURE / RESEARCH** | Requires penetrometer / shear sensors | **EXCLUDED FROM PHASE 15** |

> [!CAUTION]
> **Scope Exclusion: Physical Friction & Terramechanics**  
> Semantic segmentation classes (e.g., RELLIS-3D classes like "mud", "dirt", "grass") do **not** provide physically reliable friction coefficients ($\mu$) for the actual vehicle. Inferring $\mu_{\text{required}} = \tan\alpha \le \mu_{\text{surface}}$ from visual labels alone creates a false sense of physical fidelity. Therefore, physical friction estimation, traction coefficient estimation, terramechanics, and soil sinkage models are **strictly excluded from Phase 15 deliverables** and documented as future research possibilities. Phase 15 focuses strictly on observable geometric terrain constraints, attitude tracking, and conservative velocity derating.

---

## 10. Vehicle Dynamics & Rollover Modeling as an Engineering Decision Model

### Distinguishing Vehicle Physics from Decision Heuristics
It is essential to distinguish between actual vehicle multi-body physics and engineering decision heuristics:
1. **Measured Chassis Attitude**: Real-time roll $\phi_{\text{imu}}$ and pitch $\theta_{\text{imu}}$ provided by the 15-state EKF.
2. **Terrain-Derived Surface Orientation**: Surface normal $\mathbf{n}(x, y)$ computed from the 2.5D elevation grid.
3. **Predicted Trajectory Attitude**: Expected chassis orientation when following candidate trajectory arc $s$.
4. **Configured Safety Thresholds**: Conservative parameters in `configs/terrain_safety.yaml` ($20^\circ$ safe slope, $28^\circ$ critical slope, $0.12\text{ m}$ clearance).
5. **Derived Stability Margin**: A dimensionless decision metric used by planners to rank candidate paths.

### Quasi-Static Rollover Margin Model
The rollover margin equation is implemented as an **engineering decision model**, *not* as an experimentally validated multi-body physical simulator:

$$\text{Rollover Margin} = 1.0 - \frac{|\phi_{\text{traj}}|}{\phi_{\text{crit}}} - \frac{v^2 \cdot |\kappa|}{g \cdot \frac{W}{2 h_{\text{cg}}}} \ge 0.20$$

Where:
- $\phi_{\text{crit}} = 28.0^\circ$ (configured conservative threshold, derived analytically with safety factor 1.83 from static tipover angle $51.3^\circ$).
- $W = 0.45\text{ m}$ (wheel track width), $h_{\text{cg}} = 0.18\text{ m}$ (center of gravity height).
- $\kappa = \omega / v$ (trajectory curvature).

**Operational Role**:
- The Authoritative **Safety Governor** remains the final arbiter of vehicle safety.
- The terrain model supports conservative decision-making in the local planner.
- No claim of physical rollover prediction accuracy is made without physical vehicle validation on instrumented uneven terrain.

---

## 11. 3D/4D Planner Necessity Analysis

### The Question
*Does Phase 15 require a full 3D/4D trajectory planner (such as MoveBase3D, SBPL 3D, or Nav4D lattice planners)?*

### Technical & Architectural Evaluation
1. **Degrees of Freedom**: The UGV has 2 control inputs ($v, \omega$). It is physically constrained to stay on the ground surface. Its altitude $Z$, roll $\phi$, and pitch $\theta$ are not independently controllable state variables; they are holonomic functions of its horizontal position and heading on the terrain manifold:
   $$Z = f(x, y), \quad \theta = g(x, y, \psi), \quad \phi = h(x, y, \psi)$$
2. **Search Space Complexity**:
   - A full 3D grid search ($x, y, z, \phi, \theta, \psi$) scales with $O(N^6)$, requiring massive memory and seconds of planning latency.
   - A 2.5D manifold-constrained search scales with $O(N^2 \log N)$, targeting completion in $<15\text{ ms}$.
3. **Defensible Conclusion**:
   **A full 3D/4D planner is UNNECESSARY, computationally wasteful, and architecturally unjustified.**  
   All objectives of Phase 15 (slope avoidance, rollover prevention, ground clearance enforcement, speed adaptation) are effectively achieved for the UGV's operating envelope by injecting terrain-aware costs, attitude compensation, and kinematic stability constraints into the existing, highly optimized **Global A\* and Native C++ AVX2 DWA** pipeline.

---

## 12. Proposed Phase 15 Architecture

The recommended Phase 15 architecture integrates cleanly into the existing production structure without rewriting functional core layers:

```
[Camera & Sensor Layer]
  ├── Monocular RGB Stream (RELLIS-3D Semantic Traversability τ)
  ├── Metric Depth Source (Decoupled via DepthSource abstraction: Gazebo depth / Stereo)
  └── 6-Axis IMU (EKF Fused Chassis Roll φ, Pitch θ)
           │
           ▼
[Elevation & Attitude Perception Engine]
  ├── Dynamic Attitude Camera Ray Transform:
  │     R_cam_world = R_base_odom(t) * R_cam_base
  │     Ray projection: p_terrain = intersect(Ray(p_cam, R_cam_world * ray_dir), Terrain)
  │     Adheres to REP-103 (X-fwd, Y-left, Z-up) and REP-105 (odom -> base_link -> camera_link)
  └── Depth-to-3D-Point Projector: Ingests metric depth into local point cloud
           │
           ▼
[2.5D World Terrain Costmap (WorldTerrainMap Extension)]
  ├── Layer 0: Elevation Z_elev(x, y) [Running Kalman/alpha height update]
  ├── Layer 1: Surface Normal [nx, ny, nz] (Vectorized spatial gradient)
  ├── Layer 2: Slope Angle α(x, y) = arccos(nz) in degrees
  ├── Layer 3: Surface Roughness σ²(x, y)
  ├── Layer 4: Step Height Discontinuity ΔZ(x, y)
  ├── Layer 5: Learned Traversability τ(x, y)
  ├── Layer 6: Static Obstacle & Dynamic Occupancy Layer
  └── Layer 7: Master Cost C_total(x, y) [Lethal if α ≥ 28° or ΔZ ≥ 0.12m]
           │
           ├───────────────────────────────────────┐
           ▼                                       ▼
[Terrain-Constrained Global A*]          [Terrain-Aware C++ AVX2 DWA]
- Plans path corridor minimizing:        - Evaluates 336 candidate arcs
  Cost = Distance + w_slope * α + C_trav - Slope penalty in scoring
- Refuses to route across >28° slopes   - Cross-slope rollover margin check
- Avoids boulder step fields             - Dynamic velocity derating:
                                           v_cmd ≤ v_max * cos(α)
           │                                       │
           └───────────────────┬───────────────────┘
                               ▼
            [Authoritative Terrain Safety Governor]
            - Continuously monitors IMU pitch & roll
            - Forward envelope rollover watchdog
            - Zero-velocity emergency halt if tilt > 26°
                               │
                               ▼
            [SafeCommandBuffer (cmd_vel @ 20 Hz)]
```

### Decoupled DepthSource Architecture
Phase 15 is strictly camera-agnostic and relies on the modular `DepthSource` abstraction:
```
DepthSource (ai/traversability/depth_source.py)
   ├── GazeboSimulatedDepthSource (Primary validation source for Phase 15 SIL)
   ├── StereoDisparityDepthSource (Metric disparity from stereo cameras)
   ├── RGBDDepthSource (Structured light / Time-of-Flight sensors)
   └── Future Physical Dual-Webcam Stereo (Physical deployment target)
```
The terrain pipeline consumes metric depth strictly through this abstraction. No phase logic may be coupled to vendor-specific SDKs (e.g., ZED SDK, RealSense SDK, OAK-D DepthAI).

---

## 13. Proposed Subphase Sequence

A disciplined, incremental 9-subphase structure is established. Each subphase is strictly bounded, independently testable, and verified before advancing.

> [!IMPORTANT]
> **Subphase 15.0 Precondition**:  
> Phase 15 implementation must **NOT** begin until Subphase 15.0 (Terrain Simulation + Metric Depth Foundation) is fully verified.

### Subphase 15.0: Terrain Simulation & Metric Depth Foundation
- **Goal**: Establish a trustworthy, validated source of metric depth and ground truth terrain elevation in Gazebo simulation before writing terrain navigation algorithms.
- **Scope**:
  - Verify Gazebo uneven terrain world availability (`simulation/worlds/`).
  - Configure simulated depth camera or equivalent metric-depth sensor feed in Gazebo UGV model.
  - Verify RGB and depth frame temporal synchronization.
  - Verify camera intrinsics and extrinsics matrix consistency.
  - Establish depth-to-3D point projection pipeline.
  - Create known geometric ground truth elevations (flat, ramps, steps) and quantify baseline depth error.
- **Deliverables**: Verified sensor simulation feed, ground-truth elevation verification script, calibration validation test.
- **Exit Criteria**: Metric depth error $< \pm 0.02\text{ m}$ against known Gazebo geometry over a $0.5 - 5.0\text{ m}$ range.

### Subphase 15.1: 2.5D Elevation Representation & Point Cloud Ingestion
- **Goal**: Add true elevation $Z(x, y)$ tracking to `WorldTerrainMap` and local BEV maps; ingest metric depth into local 3D point representation.
- **Inputs**: Depth image / simulation point cloud, camera extrinsics.
- **Outputs**: `elevation` raster layer in `WorldTerrainMap`, unit test suite.
- **Production Files**: `ai/spatial_fusion/world_map.py`, `ai/costmap/bev_costmap.py`, `production/perception/elevation_mapper.py`.
- **Tests**: `tests/test_elevation_mapping.py` (planar ground, inclined ramp, noisy height fields).
- **Exit Criteria**: Raster elevation matches ground truth within $\pm 0.03\text{ m}$.

### Subphase 15.2: Geometric Surface Normal, Slope & Step Height Estimator
- **Goal**: Implement fast vectorized spatial gradient and local SVD plane fitting to populate `slope_deg`, `roughness`, and `height_var`.
- **Inputs**: Elevation layer $Z(x, y)$.
- **Outputs**: Vectorized slope and step height maps.
- **Production Files**: `production/perception/terrain_geometry.py`.
- **Tests**: `tests/test_terrain_geometry_vectorized.py` (synthetic $15^\circ, 30^\circ, 45^\circ$ ramps, steps of $0.05\text{ m}, 0.15\text{ m}, 0.25\text{ m}$).
- **Exit Criteria**: Surface normal error $<2.0^\circ$, target execution latency $<5.0\text{ ms}$ for $200 \times 200$ grid on host CPU.

### Subphase 15.3: Dynamic Attitude Camera Extrinsics Compensation
- **Goal**: Replace static pitch assumption in `VisionCostmapIntegrator` with real-time rigid-body SE(3) transforms utilizing EKF orientation.
- **Transforms**: Adhere to REP-103 and REP-105:
  $$\mathbf{R}_{\text{cam}}^{\text{world}} = \mathbf{R}_{\text{base}}^{\text{world}}(t) \cdot \mathbf{R}_{\text{cam}}^{\text{base}}$$
  Projection rays use rotation matrices or quaternions internally rather than scalar Euler angle addition.
- **Inputs**: RGB frame, instantaneous EKF orientation quaternion $\mathbf{q}_{\text{base}}^{\text{world}}$.
- **Outputs**: Tilt-compensated BEV projection.
- **Production Files**: `production/perception/vision_costmap_integrator.py`.
- **Tests**: `tests/test_attitude_compensation.py` (pitching vehicle $+10^\circ, -10^\circ$ preserves metric ground obstacle location).
- **Exit Criteria**: Projected obstacle metric position remains stationary ($\Delta < 0.05\text{ m}$) while vehicle pitches $\pm 15^\circ$.

### Subphase 15.4: Terrain-Aware Multi-Factor Costmap & Configured Hard Gates
- **Goal**: Connect `TraversabilityCostModel` to `WorldTerrainMap`; enforce configured lethal gates ($C = 1.0$) on slopes $\ge 28.0^\circ$ and step heights $\ge 0.12\text{ m}$.
- **Inputs**: Traversability, slope, roughness, step height, obstacles.
- **Outputs**: Fused master costmap reflecting configured vehicle safety thresholds.
- **Production Files**: `ai/spatial_fusion/fusion_engine.py`, `configs/terrain_safety.yaml`.
- **Tests**: `tests/test_terrain_costmap_fusion.py`.
- **Exit Criteria**: $30^\circ$ slope cells and $15\text{ cm}$ steps are marked lethal ($C=1.0$) regardless of high visual traversability.

### Subphase 15.5: Terrain-Constrained Global A* Planning
- **Goal**: Enable Global A* to optimize paths over the multi-factor terrain costmap, discovering low-slope detours around steep embankments.
- **Inputs**: Start, goal, terrain costmap with valid slope.
- **Outputs**: Global path minimizing terrain slope and roughness exposure.
- **Production Files**: `ai/planning/global_planners/a_star.py`, `ai/planning/costs.py`.
- **Tests**: `tests/test_terrain_a_star.py` (corridor with steep direct route vs gentle winding detour).
- **Exit Criteria**: A* automatically chooses longer gentle trail ($10^\circ$) over shorter impassable slope ($32^\circ$).

### Subphase 15.6: Terrain-Aware C++ AVX2 DWA Local Planning
- **Goal**: Pass populated `slope_grid` to native C++ DWA; implement cross-slope rollover trajectory rejection and slope velocity derating:
  $$v_{\text{cmd}} \le v_{\text{max}} \cdot \max(0.20, \cos\alpha_{\text{forward}})$$
- **Inputs**: Candidate trajectories, 2.5D costmap, vehicle stability limits.
- **Outputs**: Safe candidate trajectory avoiding steep side-slopes.
- **Production Files**: `ai/planning/native_cpp/dwa_core.cpp`, `ai/planning/backends/cpp_backend.py`.
- **Tests**: `tests/test_terrain_dwa_cpp.py`.
- **Exit Criteria**: DWA rejects trajectories causing predicted roll $>20^\circ$; target C++ P95 latency remains $<10\text{ ms}$.

### Subphase 15.7: Authoritative Terrain Safety Governor
- **Goal**: Extend `SafetyGovernor` to evaluate real-time IMU tilt angles and forward terrain elevation, triggering conservative deceleration or emergency stop before rollover.
- **Inputs**: Current vehicle $(\phi, \theta)$ from IMU, planned trajectory, command buffer.
- **Outputs**: Clamped $(v, \omega)$ with active rollover guard.
- **Production Files**: `production/safety/safety_governor.py`.
- **Tests**: `tests/test_terrain_safety_governor.py`.
- **Exit Criteria**: Emergency stop triggered if chassis roll exceeds configured threshold ($26.0^\circ$, representing a $2^\circ$ pre-warning buffer before critical $28^\circ$).

### Subphase 15.8: Differential Terrain Benchmark Suite
- **Goal**: Execute a formal, repeatable benchmark comparing Phase 14 (Flat 2D Baseline) vs Phase 15 (Terrain-Aware Stack) across standardized uneven outdoor scenarios.
- **Outputs**: Comprehensive benchmark report, machine-readable telemetry (`reports/phase15/`).
- **Benchmark Objective**: Demonstrate that terrain-aware navigation avoids terrain-specific failure modes while maintaining acceptable performance on terrain where the Phase 14 planar planner is valid.

---

## 14. Validation / Benchmark Strategy

### Benchmark Objective
The primary benchmark objective is:
> *"Demonstrate that terrain-aware navigation avoids terrain-specific failure modes (such as rollover-threshold violations and undercarriage high-centering) while maintaining parity on benign planar terrain where the Phase 14 planar planner is valid."*

Subjective claims of "superiority" or "optimality" are explicitly rejected in favor of quantitative failure-mode differentiation.

### Comparative Architectures
1. **BASELINE (Phase 14 Pure 2D Stack)**: Flat IPM ($Z=0$), zero-slope costmap, standard 2D DWA.
2. **EVALUATION (Phase 15 Terrain-Aware Stack)**: Dynamic attitude IPM, 2.5D elevation/slope costmap, slope-aware A* & DWA.

### Benchmark Matrix (Scenarios)

| Scenario ID | Environment Geometry | Expected Behavior (Phase 14 Baseline) | Target Acceptance Criteria (Phase 15 Terrain-Aware) |
| :--- | :--- | :--- | :--- |
| **SCENARIO 1: FLAT BASELINE** | 100m flat planar trail | Completes mission successfully | **Parity Test**: 100% completion, zero regression in latency or drift. |
| **SCENARIO 2: GENTLE SLOPE** | Continuous $8^\circ - 12^\circ$ incline | Completes mission with slight wheel slip | Completes mission with smooth adaptive speed derating. |
| **SCENARIO 3: STEEP EMBANKMENT** | Direct path crosses $32^\circ$ grassy slope; side trail is $12^\circ$ dirt | Attempts direct climb; violates rollover threshold ($>28^\circ$) or stalls | Global A* detects lethal slope, routes along gentle side trail. 0 rollover violations. |
| **SCENARIO 4: SEVERE CROSS-SLOPE** | Trail tilted laterally at $24^\circ$ cross-slope | Experiences severe lateral tilt and potential roll hazard | DWA rejects high-roll trajectories; routes along ridge or valley floor. |
| **SCENARIO 5: STEP / BOULDER FIELD** | Boulders embedded in ground with $18\text{ cm}$ vertical step height | Attempts traversal; chassis high-centers on step obstacle | Costmap flags step height $>0.12\text{ m}$; navigates around obstacles. 0 high-centerings. |
| **SCENARIO 6: ROUGH / UNDULATING** | Rolling $15^\circ$ hills with continuous pitching undulations | Flat IPM creates projection jitter and ghost obstacles | Attitude compensation stabilizes costmap; continuous forward progress. |
| **SCENARIO 7: COMPOUND OUTDOOR** | 100m course combining slopes, boulders, and dynamic crossing obstacle | Fails due to terrain trap or high-centering | Successfully negotiates terrain corridor while safely avoiding dynamic obstacle. |

### Evaluation Metrics
- **Mission Completion Rate** ($\%$)
- **Collision Count** (static and dynamic obstacle contacts)
- **Rollover-Threshold Violations** (instances where $|\phi| \ge 28^\circ$ or $|\theta| \ge 28^\circ$)
- **High-Centering Events** (immobilization caused by ground clearance violation $\Delta Z \ge 0.12\text{ m}$)
- **Minimum Measured Ground Clearance** ($\text{m}$)
- **Maximum Pitch and Roll Experienced** ($^\circ$)
- **Mean Terrain Slope Exposure** ($^\circ$)
- **Path Length & Detour Ratio** ($\text{Path Length} / \text{Euclidean Distance}$)
- **Replanning Count**
- **Planning Latency (Mean & P95)** ($\text{ms}$)
- **End-to-End Control Loop Latency** ($\text{ms}$)
- **Localization Drift** ($\%$)
- **Host CPU Utilization & Memory Footprint**
- **Safety Governor Interventions** (count of speed clamps and emergency halts)

---

## 15. Complexity vs Value Analysis

| Candidate Feature | SIH Practical Value | Implementation Complexity | Computational Overhead | Validation Difficulty | Strategic Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2.5D Elevation Grid $Z(x, y)$** | **HIGH** | **LOW** | **LOW** (Target $<2\text{ ms}$) | **LOW** | **INCLUDE (Foundation)** |
| **Surface Normal Vector $\mathbf{n}$** | **HIGH** | **MEDIUM** | **LOW** (Target $<3\text{ ms}$) | **LOW** | **INCLUDE (Enables slope)** |
| **Terrain Slope Angle $\alpha$** | **HIGH** | **LOW** | **LOW** (Target $<1\text{ ms}$) | **LOW** | **INCLUDE (Safety threshold)** |
| **Surface Roughness $\sigma_z^2$** | **MEDIUM** | **LOW** | **LOW** (Target $<2\text{ ms}$) | **LOW** | **INCLUDE (Roughness penalty)** |
| **Step Height $\Delta Z$** | **HIGH** | **MEDIUM** | **LOW** (Target $<2\text{ ms}$) | **LOW** | **INCLUDE (High-centering guard)** |
| **Dynamic Attitude IPM (SE3)** | **HIGH** | **MEDIUM** | **LOW** (Target $<1\text{ ms}$) | **MEDIUM** | **INCLUDE (Eliminates projection jitter)** |
| **Quasi-Static Rollover Model** | **HIGH** | **MEDIUM** | **LOW** (Target $<1\text{ ms}$) | **LOW** | **INCLUDE (Decision model)** |
| **Terrain-Aware Global A\*** | **HIGH** | **LOW** | **LOW** (Hook exists) | **LOW** | **INCLUDE (Routes around slopes)** |
| **Terrain-Aware C++ DWA** | **HIGH** | **LOW** | **LOW** (Hook exists) | **LOW** | **INCLUDE (Adaptive speed/rejection)** |
| **Terrain Safety Governor** | **HIGH** | **LOW** | **LOW** (Target $<0.5\text{ ms}$) | **LOW** | **INCLUDE (Authoritative gate)** |
| **Physical Friction Estimation ($\mu$)** | **LOW** | **EXTREME** | **HIGH** | **EXTREME** | **EXCLUDE (Unsupported by vision)** |
| **Full 3D/4D Voxel Planner** | **LOW** | **EXTREME** | **UNFEASIBLE** ($>200\text{ ms}$) | **EXTREME** | **EXCLUDE (Architectural anti-pattern)** |
| **Multi-Body Terramechanics (Bekker)**| **LOW** | **EXTREME** | **HIGH** ($>50\text{ ms}$) | **EXTREME** | **EXCLUDE (No soil sensor data)** |

---

## 16. Risks and Technical Mitigations

1. **Risk: Depth Sensor Noise & Sparsity in Outdoor Sunlight**  
   - *Mitigation*: Vectorized spatial outlier rejection, minimum point-count gating (`point_count >= 5`), and temporal exponential smoothing on cell elevation.
2. **Risk: Vehicle Pitch Oscillations Inducing Mapping Artifacts**  
   - *Mitigation*: The 15-state `MultiSensorEKF` applies continuous accelerometer-gyroscope complementary filtering with estimated biases, suppressing high-frequency chassis vibration. Dynamic SE(3) ray projection compensates for instantaneous pitch.
3. **Risk: Local Minima in Mountainous Saddles or Bowl-Shaped Depressions**  
   - *Mitigation*: The hierarchical combination of Terrain-Aware Global A* (which finds the topological pass around the ridge) with local DWA prevents the local planner from entering dead-end concave bowls.
4. **Risk: Performance Throttling on Host CPU**  
   - *Mitigation*: All terrain spatial calculations are vectorized using NumPy strided windows and C++ AVX2 loops, targeting total costmap update overhead under $10\text{ ms}$.

---

## 17. Hardware-Independent Design & Boundaries

Phase 15 strictly adheres to the project's software-first, hardware-independent principles:
- **CPU-First Execution**: All Phase 15 algorithms must execute deterministically on standard x86_64 host CPU (using AVX2 SIMD) while maintaining algorithmic compatibility with ARM64 (NEON) platforms.
- **Hardware Boundary**:
  - **NO Jetson Orin requirement**.
  - **NO TensorRT requirement**.
  - **NO physical stereo-camera requirement**.
  - **NO physical UGV requirement for software milestone completion**.
  - **NO phone / Android optimization requirement**.
- **Future Physical Target Defined**:
  The long-term physical deployment concept comprises:
  $$\text{Phone CPU} + \text{Two Rigidly Mounted Webcams} + \text{Stereo Depth} + \text{Existing Navigation Software}$$
  However, this represents **future physical integration**, NOT Phase 15 software scope. Phase 15 is executed strictly as a software-in-the-loop (SIL) capability.

---

## 18. Final Revised Phase 15 Scope Statement

> ### Phase 15: Advanced 2.5D Terrain-Aware Navigation
> Phase 15 extends the Phase 14 planar navigation stack with metric terrain geometry. It introduces a 2.5D elevation representation, surface normals, slope, roughness, and step-height estimation, attitude-aware camera projection via rigid-body SE(3) transforms, terrain-aware cost fusion, terrain-constrained global planning, terrain-aware local DWA constraints, and an authoritative terrain safety layer.
>
> The phase deliberately avoids full 3D/4D voxel planning, terramechanics, experimentally unsupported friction estimation, embedded GPU deployment, and phone optimization.

---

## 19. Explicit Non-Goals

To maintain sharp engineering focus and prevent scope creep, the following are declared **EXPLICIT NON-GOALS** for Phase 15:
- **NO Full 3D/4D Voxel / Lattice Planners**: Out of scope and unfeasible for real-time UGV operation.
- **NO Arbitrary 3D Geometric Planning**: Overhangs, caves, vertical tunnels, and multi-surface elevations at the same $(x, y)$ are not modeled.
- **NO Physical Friction / Traction Estimation ($\mu$)**: Semantic classes will not be used to guess physical friction coefficients.
- **NO Terramechanics / Soil Shear Sinkage Modeling**: Bekker-Wong mechanics are excluded due to lack of soil parameter sensors.
- **NO Embedded GPU Flashing (Jetson Orin / TensorRT)**: Reserved for subsequent deployment phases.
- **NO Phone / Android Deployment or Optimization**: Optimization for Android runtime is deferred to system packaging.
- **NO Physical Hardware Dependency**: Software completion does not depend on a physical rover or physical cameras.

---

## 20. Phase 15 Readiness Decision

### Exact Decision State:
$$\mathbf{READY\_TO\_START\_PHASE\_15\_WITH\_PRECONDITIONS}$$

### Verified Preconditions Required Before Phase 15 Implementation Begins:

1. **PRECONDITION 1 (Metric Depth Simulation)**:
   Verify or configure a Gazebo depth camera or metric-depth source in the Gazebo UGV model (`simulation/models/ugv/model.sdf`) to provide reliable depth data via the `DepthSource` abstraction.
2. **PRECONDITION 2 (Ground-Truth Uneven Terrain Worlds)**:
   Establish validated uneven Gazebo worlds with known geometric ground truth:
   - Flat planar baseline ($0^\circ$),
   - Gentle slopes ($10^\circ$),
   - Moderate slopes ($20^\circ$),
   - Critical slopes ($30^\circ$),
   - Cross-slope embankments,
   - Step / boulder obstacles ($0.05\text{ m}, 0.15\text{ m}, 0.25\text{ m}$),
   - Undulating / rolling mounds.
3. **PRECONDITION 3 (Sensor Calibration & Sync)**:
   Verify RGB and depth camera synchronization, camera intrinsics, and extrinsics matrix consistency.
4. **PRECONDITION 4 (Regression Baseline Isolation)**:
   Confirm that all Phase 14 regression suites remain passing and production planar paths remain clean before introducing the elevation layer into `WorldTerrainMap`.
5. **PRECONDITION 5 (Threshold Clarification)**:
   Formally document all terrain safety thresholds ($20^\circ$ safe slope, $28^\circ$ critical slope, $0.12\text{ m}$ clearance) as **configurable engineering assumptions**, not experimentally validated physical vehicle limits.
6. **PRECONDITION 6 (Traction / Slip Scope Boundary)**:
   Formally confirm that physical friction ($\mu$) estimation and terramechanics are excluded from Phase 15 core deliverables.
7. **PRECONDITION 7 (Audit Execution Discipline)**:
   Confirm that zero production code was modified during this strategic audit and revision pass, and that Subphase 15.0 will be executed as an audit-oriented foundation prior to Subphase 15.1.