# Phase 9 — Perception + Localization Spatial Fusion

---

## 1. System Identity & Architecture

Phase 9 implements the **spatial fusion layer** of the autonomous UGV navigation stack. Its primary role is to bridge local computer vision observations (Phase 4 hazard perception and Phase 5 BEV terrain costmaps) with metric visual localization (Phase 8 visual-inertial state estimates) into a globally referenced, planar terrain map:

```text
                     RGB / Stereo Camera
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        SAM-TP         RUGD Model     Hazard Detection
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                     Perception Fusion
                            │
                            ▼
                  Terrain / BEV Projection
                            │
                            ▼
                    Phase 5 Costmap
                            ▲
                            │
                    Phase 8 Localization
                            │
                     Pose + Confidence
                            │
                            ▼
                   *** PHASE 9 ***
                  Spatial Fusion Layer
         (Temporal Sync + Rigid SE(3) Transform + 
          Safety Precedence + Confidence Fusion)
                            │
                            ▼
                     WorldTerrainMap
                            │
                            ▼
                   Phase 10+ Navigation
```

---

## 2. Coordinate Frame Conventions & Mathematical Foundations

Phase 9 establishes a strict right-handed coordinate convention conforming to ISO 8855 / ROS standards:

### Frame Hierarchy:
1. **`world` / `odom` Frame:** Fixed inertial reference coordinate frame.
   - $+X$: East / Initial Forward
   - $+Y$: North / Initial Left
   - $+Z$: Up
2. **`base_link` Frame:** Robot body frame centered between drive wheels on the ground contact plane.
   - $+X$: Forward along longitudinal axis
   - $+Y$: Left along lateral axis
   - $+Z$: Up normal to ground plane
3. **`camera_optical` Frame:** Standard camera sensor coordinate frame.
   - $+X$: Right in image plane
   - $+Y$: Down in image plane
   - $+Z$: Forward along optical axis

### Transform Notation:
$\mathbf{T}_{A\to B}$ converts homogeneous points expressed in frame $B$ to frame $A$:
$$\begin{bmatrix} \mathbf{p}_A \\ 1 \end{bmatrix} = \mathbf{T}_{A\to B} \begin{bmatrix} \mathbf{p}_B \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R}_{A\to B} & \mathbf{t}_{A\to B} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} \mathbf{p}_B \\ 1 \end{bmatrix}$$

### Camera Extrinsics:
Configurable via `configs/spatial_fusion.yaml`:
- Lever-arm translation: $\mathbf{t} = [x_{\text{fwd}}, y_{\text{lat}}, z_{\text{height}}]^T = [0.15, 0.00, 0.40]^T\text{ m}$.
- Mounting rotation: Pitch downward tilt $\theta = 15.0^\circ$ ($\mathbf{R}_{\text{mount}} = \mathbf{R}_y(15^\circ)$).
- Optical to Body rotation:
  $$\mathbf{R}_{\text{opt}\to\text{body}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$$
  $$\mathbf{T}_{\text{body}\to\text{camera}} = \begin{bmatrix} \mathbf{R}_{\text{mount}} \mathbf{R}_{\text{opt}\to\text{body}} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

### Full Transform Chain:
$$\mathbf{p}_{\text{world}} = \mathbf{T}_{\text{world}\to\text{body}} \cdot \mathbf{T}_{\text{body}\to\text{camera}} \cdot \mathbf{p}_{\text{camera}}$$
Vectorized point cloud transformation:
$$\mathbf{P}_{\text{world}} = \mathbf{P}_{\text{body}} \mathbf{R}_{\text{world}\to\text{body}}^T + \mathbf{t}_{\text{world}\to\text{body}}^T$$

---

## 3. Temporal Synchronization & Motion Compensation

Observations and localization updates run asynchronously:
1. **Timestamp Association:** For an observation at timestamp $t_{\text{obs}}$, the `TemporalSynchronizer` retrieves the nearest pose at $t_{\text{pose}}$.
2. **Stale Pose Protection:** If $|t_{\text{obs}} - t_{\text{pose}}| > \text{max\_sync\_delta\_s}$ ($0.10\text{ s}$), the observation is rejected. Old poses are never silently reused.
3. **Motion Compensation:** If $|t_{\text{obs}} - t_{\text{pose}}| \le \text{max\_extrapolation\_s}$ ($0.05\text{ s}$), short-term velocity extrapolation compensates for vehicle displacement:
   $$\mathbf{p}_{\text{comp}} = \mathbf{p} + \mathbf{R}_{\text{wb}} \cdot (\mathbf{v}_{\text{body}} \cdot \Delta t), \quad \psi_{\text{comp}} = \psi + \omega_z \cdot \Delta t$$
   This prevents spatial smearing of obstacles during forward driving and turns.

---

## 4. WorldTerrainMap Representation

The `WorldTerrainMap` represents a planar, multi-layer metric raster grid centered in the world coordinate frame ($0.10\text{ m/cell}$ resolution):

| Layer | Type | Range | Description |
| :--- | :---: | :---: | :--- |
| **`cost`** | `float32` | $[0.0, 1.0]$ | Continuous traversal cost field ($0.0=\text{safe}$, $1.0=\text{lethal}$) |
| **`traversability`** | `float32` | $[0.0, 1.0]$ | Learned semantic drivability ($1.0=\text{high drivability}$) |
| **`obstacle`** | `bool` | `True/False` | Hard impassable physical geometry |
| **`hazard`** | `float32` | $[0.0, 1.0]$ | Water, mud, reflections, drop-offs ($0.70+=\text{critical}$) |
| **`unknown`** | `bool` | `True/False` | Unobserved space (**strictly preserved; never converted to free space**) |
| **`confidence`** | `float32` | $[0.0, 1.0]$ | Spatially fused sensor and localization confidence |
| **`slope`** | `float32` | $[0.0, 90.0]^\circ$ | Surface inclination angle |
| **`roughness`** | `float32` | $\ge 0.0$ | Surface coplanarity variance |
| **`height_var`** | `float32` | $\ge 0.0\text{ m}$ | Maximum step height differential |
| **`observation_count`**| `int32` | $\ge 0$ | Cumulative observation count |
| **`last_observed`** | `float64` | seconds | Unix timestamp of most recent update |

---

## 5. Fusion Policies & Safety Rules

### A. Safety Precedence Rule (Non-Negotiable)
High-risk perception information must **NEVER** be diluted into safe ground by averaging:
$$\text{Critical Hazard} > \text{High Hazard} > \text{Lethal Obstacle} > \text{Restricted Traversability} > \text{Normal Terrain}$$
- If a cell was previously observed as a critical hazard ($\ge 0.70$) or lethal obstacle, and a subsequent frame observes it as safe, the fused world cell **retains the hazard and obstacle flag**:
  $$\text{obstacle}_{\text{fused}} = \text{obstacle}_{\text{existing}} \lor \text{obstacle}_{\text{obs}}$$
  $$\text{hazard}_{\text{fused}} = \max(\text{hazard}_{\text{existing}}, \text{hazard}_{\text{obs}})$$
  $$\text{cost}_{\text{fused}} = \max(\text{cost}_{\text{existing}}, \text{cost}_{\text{obs}})$$
  $$\text{trav}_{\text{fused}} = \min(\text{trav}_{\text{existing}}, \text{trav}_{\text{obs}})$$

### B. Unknown-Space Policy
Unobserved cells remain `unknown = True` and neutral cost ($0.50$). An observation containing unknown regions does not overwrite known world obstacles and never turns unknown world space into free space.

### C. Zero-Trust Tracking State Enforcement & DEGRADED Semantics
- **`TRACKING_OK` / `RECOVERED`:** Normal fusion allowed with full confidence ($c_{\text{eff}} = c_{\text{local}} \cdot c_{\text{loc}}$).
- **`DEGRADED`:** Permitted, but observation confidence is scaled down by an engineering fusion multiplier of $0.50$.
  > **Note on DEGRADED confidence:** The $0.50$ factor is an engineering fusion-weighting / confidence down-scaling policy to ensure conservative EKF/costmap weighting, and NOT a statistically calibrated covariance transformation. In Phase 8, the covariance proxy inflates by $\text{scale} = 1/c^2$ ($2.10\times$ to $11.11\times$); Phase 9 applies the $0.50$ factor to scale down the observation update weight in the raster grid.
- **`LOST` / `RELOCALIZING`:** Fusion is strictly **BLOCKED**. Zero stale poses or corrupted local costmaps are integrated into the world map.

---

## 6. Empirical Benchmarks & Measured Evidence

### A. End-to-End Deterministic Scenario (Section 47):
- **Scenario:** Robot starts at $(0,0)$, drives forward, executes $30^\circ$ turn, enters `DEGRADED`, transitions to `LOST`, recovers to `RECOVERED`.
- **Results:**
  - Obstacle at world $(3.0, 1.0)$: **Verified placed at exact cell**, observed 4 times.
  - Hazard at world $(5.0, -1.0)$: **Severity $0.90$ preserved** without dilution.
  - LOST state: **$100\%$ blocked**; zero false cells mapped during tracking loss.
  - RECOVERED state: **Fusion successfully resumed**.

### B. Controlled Ablation Studies (Section 40):
1. **Ablation C (Safety Precedence vs. Naive Averaging):**
   - Production (Safety Precedence): Hazard $= 0.95$, Cost $= 1.00$ (**Critical Preserved**).
   - Naive Averaging: Hazard $= 0.47$, Cost $= 0.55$ (**Hazard Diluted into False-Safe!**).
2. **Ablation D (Motion Compensation during Turn):**
   - Compensated Pose: $0.0400\text{ m}$ forward projection.
   - Uncompensated: $0.0000\text{ m}$ ($4\text{ cm}$ spatial smearing prevented).

### C. Latency Profiling & Performance Interpretation (200 Iterations):
- **Mean Latency:** **$29.18\text{ ms}$** (below the $33.33\text{ ms}$ budget for a $30\text{ Hz}$ cycle)
- **Median Latency:** **$28.20\text{ ms}$**
- **P95 Latency:** **$38.22\text{ ms}$** (exceeds the $33.33\text{ ms}$ budget)
- **Performance Interpretation:**
  > Mean latency ($29.18\text{ ms}$) is below the $30\text{ Hz}$ cycle budget ($33.33\text{ ms}$), but P95 latency ($38.22\text{ ms}$) exceeds the $33.3\text{ ms}$ budget. The current implementation therefore does not yet demonstrate $30\text{ Hz}$ compliance at the P95 level. This is classified as a performance limitation and an optimization opportunity for compiled C++ or Cython vectorization in future embedded deployment phases.

### D. Memory Growth Benchmark:
- Fixed bounded allocation: $400 \times 400$ grid with 11 layers $= \mathbf{6.41\text{ MB}}$.
- Memory remains bounded $O(1)$ across 100, 500, and 1,000 observations.

### E. Independent MATLAB Verification:
- Exported data: `matlab/data/phase_9_transforms.mat`.
- Script: `matlab/scripts/validate_phase_9_transforms.m`.
- Numerical parity error against MATLAB formulations: **$< 10^{-12}\text{ m}$** (Machine Float Parity).

### F. Real Outdoor RUGD Sequence Execution:
- **Evaluation:** Audited across 20 frames of the RUGD `trail` sequence (unpaved dirt path, dense foliage).
- **Observed Behavior:** 10 valid fusions executed during tracking availability, 10 cycles safely blocked during tracking loss; 4,753 cells mapped with mean confidence $0.923$.
- **Scientific Qualification:**
  > This result represents real outdoor execution and spatial-fusion consistency evidence. It does NOT establish absolute metric world-map positioning accuracy because RUGD lacks metric trajectory ground truth. Real metric trajectory validation remains pending physical rover deployment.

---

## 7. Evidence Classification Table

| Capability | Classification | Evidence / Source |
| :--- | :--- | :--- |
| **SE(3) Spatial Transformation** | **Measured & Tested** | 8 unit tests (A–H) passing; MATLAB parity $< 10^{-12}\text{ m}$ |
| **Timestamp Synchronization** | **Tested** | Monotonic buffer matching within $0.10\text{ s}$ verified in unit tests |
| **Stale-Pose Rejection** | **Tested** | Observations exceeding $0.10\text{ s}$ gap rejected with zero map update |
| **LOST-State Fusion Blocking** | **Tested** | Zero-trust enforcement verified in unit tests and scenario Step 4 |
| **Hazard Safety Precedence** | **Measured in Controlled Ablation** | Critical hazard ($0.95$) retained under subsequent safe observations (Ablation C) |
| **DEGRADED Confidence Weighting** | **Implemented + Tested** | Heuristic $0.50$ fusion-weighting policy verified; scales confidence down |
| **Real RUGD Fusion Execution** | **Real Outdoor Execution Evidence** | Audited on RUGD `trail` (20 frames, 4,753 cells mapped, zero stale poses) |
| **Absolute Metric World-Map Accuracy** | **Pending** | RUGD lacks RTK-GPS ground truth; metric validation pending rover field trials |
| **Physical Camera Calibration** | **Pending** | Simulation uses nominal design extrinsics; physical checkerboard photos pending |
| **Mean Fusion Latency** | **Measured** | $29.18\text{ ms}$ ($< 33.3\text{ ms}$ budget; supports $30\text{ Hz}$ average-cycle target) |
| **P95 30 Hz Compliance** | **Not Demonstrated** | $38.22\text{ ms} > 33.3\text{ ms}$ (P95 exceeds budget; optimization opportunity) |

---

## 8. Phase 10 Handoff Interface Contract

The spatial fusion layer provides the `WorldTerrainMap` object as the official input contract for Phase 10 (Autonomous Path Planning & Waypoint Navigation):

```python
from ai.spatial_fusion.world_map import WorldTerrainMap

def on_planning_cycle(world_map: WorldTerrainMap):
    # 1. Query dimensions & resolution
    res_m = world_map.resolution_m         # 0.10 m/cell
    x_bounds = world_map.x_bounds_m        # (-20.0, 20.0) m
    y_bounds = world_map.y_bounds_m        # (-20.0, 20.0) m
    
    # 2. Extract planning layers
    cost_grid = world_map.cost             # (H, W) float32 in [0.0, 1.0]
    obstacle_mask = world_map.obstacle     # (H, W) bool
    hazard_grid = world_map.hazard         # (H, W) float32 (>= 0.70 critical)
    unknown_mask = world_map.unknown       # (H, W) bool
    confidence_grid = world_map.confidence # (H, W) float32
    
    # 3. Query world coordinate directly
    is_safe = world_map.is_cell_traversable(x_world=3.0, y_world=1.0)
    cost_val = world_map.get_cost(x_world=3.0, y_world=1.0)
    
    # 4. Extract localized submap around vehicle
    submap = world_map.export_submap(center_x_m=robot_x, center_y_m=robot_y, extent_x_m=10.0, extent_y_m=10.0)
```

---

## 9. Final Decision

```text
APPROVED WITH CONDITIONS
```

### Justification & Documented Conditions:
1. **Core Architecture & Safety Validated:** Coordinate transforms, timestamp synchronization, WorldTerrainMap grid representation, safety precedence (zero hazard dilution), and zero-trust tracking state protection are completely implemented, verified, and pass all 28 unit tests and 188 full-project regression tests.
2. **Condition 1 (Performance Limitation):** Mean latency ($29.18\text{ ms}$) satisfies the $30\text{ Hz}$ budget ($33.3\text{ ms}$), but P95 latency ($38.22\text{ ms}$) exceeds it. P95 $30\text{ Hz}$ compliance is not yet demonstrated and is carried forward as a non-blocking optimization opportunity for future embedded compilation.
3. **Condition 2 (DEGRADED Weighting Semantics):** The $0.50$ factor is documented as an engineering fusion-weighting policy, not a statistically calibrated covariance transformation.
4. **Condition 3 (Real Metric Trajectory Validation):** RUGD provides real outdoor execution and consistency evidence; absolute metric world-map positioning accuracy remains pending physical rover field deployment with RTK-GPS ground truth.
