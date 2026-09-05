# Phase 5 — BEV Terrain Costmap, Temporal Fusion & Planner-Ready Terrain Representation

---

## Executive Summary
Phase 4 delivered multi-modal, safety-aware vision perception, reconciling SAM-TP, RUGD, and explicit hazard detection. However, raw perspective vision masks cannot be directly consumed by a mobile robot's path planner. 

**Phase 5 bridges perception and motion planning** by converting these multi-source inputs into a unified, robot-centric, terrain-aware, temporally accumulated, and planner-ready **Bird's-Eye-View (BEV) Terrain Costmap** (`BEVTerrainCostmap`).

> [!IMPORTANT]
> **Strict Phase Boundary Adherence:**
> Phase 5 defines and implements the **spatial cost and terrain representation**. It does **NOT** implement path planning algorithms (A*, RRT*, PF-RRT*, Nav4D planner, DWA, MPC, Pure Pursuit, Nav2, or `/cmd_vel`). Those belong strictly to Phase 6 and later. Phase 5 concludes with a certified **Planner-Ready Input Contract Stub**.

---

## Section A: Architecture & Data Flow

```
                      RGB CAMERA (+ Depth Source)
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
                  ▼                ▼                ▼
                SAM-TP         RUGD MODEL      HAZARD AI
                  │                │                │
                  ▼                ▼                ▼
            Traversability A  Traversability B  Hazard/Risk
                  │                │                │
                  └────────────────┼────────────────┘
                                   ▼
                          SAFETY PERCEPTION
                              FUSION OUTPUT
                                   │
                                   ▼
                           IMAGE → BEV PROJECTION
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            BEV Perception Grid          3D Terrain Geometry
            (Trav, Hazards, Risk)        (Slope, Roughness, Step)
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                           MULTI-LAYER COSTMAP
                       (8 Independently Inspectable Layers)
                                   │
                                   ▼
                       TEMPORAL ACCUMULATION ENGINE
                     - SE(2) Motion Compensation
                     - Static vs Dynamic Decay Rates
                     - Dropout & Flicker Smoothing
                                   │
                                   ▼
                           OBSTACLE INFLATION
                     (Vehicle Footprint + Exponential Margin)
                                   │
                                   ▼
                           FINAL PLANNER-READY
                              BEV COSTMAP
                     (0.0 = Safe, 0.5 = Caution/Unk, 1.0 = Lethal)
                                   │
                                   ▼
                           PHASE 6 PLANNER STUB
```

---

## Section B: Coordinate System Conventions

All geometric and raster transformations adhere to the **ISO 8855 / ROS standard**:

```text
Robot Base Frame (base_link):
  +X: Forward along vehicle longitudinal axis (meters)
  +Y: Lateral to vehicle left (meters)
  +Z: Upward perpendicular to ground contact (meters)

Local BEV Grid Frame (local_bev):
  Origin: Robot base center (x = 0.0m, y = 0.0m)
  Grid Array: Top-left cell [0, 0] corresponds to (x_max, y_max)
  Coordinate Mapping:
    x_metric = x_max - (row + 0.5) * resolution_m
    y_metric = y_max - (col + 0.5) * resolution_m
    row = int((x_max - x_metric) / resolution_m)
    col = int((y_max - y_metric) / resolution_m)

Camera Optical Frame (camera_optical):
  +X: Rightward across camera image plane
  +Y: Downward across camera image plane
  +Z: Forward along optical ray direction
```

---

## Section C: Canonical BEV Representation (`BEVTerrainCostmap`)

Implemented in [ai/costmap/bev_costmap.py](file:///c:/Users/shres/Desktop/SIH2026/ai/costmap/bev_costmap.py).
Rather than a slow Python object-per-cell graph, the canonical representation is implemented as a contiguous tensor of synchronized 2D NumPy arrays:

| Layer Property | Array Dimensions | Data Type | Physical Range | Semantics |
| :--- | :---: | :---: | :---: | :--- |
| `cost` | $(H, W)$ | `float32` | $[0.0, 1.0]$ | Continuous traversal cost ($0.0=\text{safe}, 1.0=\text{lethal}$) |
| `traversability` | $(H, W)$ | `float32` | $[0.0, 1.0]$ | Learned visual drivability score ($1.0=\text{drivable}$) |
| `obstacle` | $(H, W)$ | `bool` | `{True, False}` | Hard impassable barrier |
| `hazard` | $(H, W)$ | `float32` | $[0.0, 1.0]$ | Explicit detected hazard severity |
| `unknown` | $(H, W)$ | `bool` | `{True, False}` | Unobserved cell / high perceptual ambiguity |
| `confidence` | $(H, W)$ | `float32` | $[0.0, 1.0]$ | Spatial observation certainty |
| `slope_deg` | $(H, W)$ | `float32` | $[0.0, 90.0^\circ]$ | Physical ground inclination angle |
| `roughness` | $(H, W)$ | `float32` | $[0.0, 1.0]$ | SVD residual coplanarity variance |
| `height_var` | $(H, W)$ | `float32` | $[0.0, \infty)$ | Vertical step height difference ($z_{\max} - z_{\min}$) |
| `observation_count` | $(H, W)$ | `int32` | $[0, \infty)$ | Cumulative temporal observation hits |

---

## Section D: Multi-Layer Cost Formulation & Hard Safety Overrides

Implemented in [ai/costmap/fusion.py](file:///c:/Users/shres/Desktop/SIH2026/ai/costmap/fusion.py) with configuration parameterized in [configs/costmap.yaml](file:///c:/Users/shres/Desktop/SIH2026/configs/costmap.yaml):

### 1. Weighted Cost Formulation:
$$C_{\text{base}} = w_{\text{trav}} \cdot (1 - \tau) + w_{\text{haz}} \cdot H + w_{\text{slope}} \cdot \frac{\theta}{\theta_{\text{safe}}} + w_{\text{rough}} \cdot R + w_{\text{height}} \cdot \frac{\Delta z}{\Delta z_{\text{safe}}} + w_{\text{unk}} \cdot C_{\text{unk}}$$

*Default Configured Weights:*
- $w_{\text{traversability}} = 0.35$
- $w_{\text{hazard}} = 0.25$
- $w_{\text{slope}} = 0.15$ ($\theta_{\text{safe}} = 20.0^\circ$)
- $w_{\text{roughness}} = 0.10$
- $w_{\text{height}} = 0.10$ ($\Delta z_{\text{safe}} = 0.12\text{ m}$)
- $w_{\text{unknown}} = 0.05$

### 2. Hard Safety Overrides (Initial Simulation Safety Thresholds / Engineering Assumptions):

> [!NOTE]
> The thresholds below ($28.0^\circ$ critical slope, $0.20\text{ m}$ step height) are **initial simulation safety thresholds / engineering assumptions** defined to enforce algorithmic safety bounds during software development. They are **NOT** physically validated rover limits, official SIH requirements, or universal UGV limits. Physical chassis and tilt-table validation remains future work.

1. **Critical/High Hazards (Water, Ditch, Rock, Person, Vehicle):** If $H \ge 0.85$ or `obstacle == True`, **cost is forced to $1.00$ (Lethal)**, completely overriding any optimistic learned traversability prediction!
2. **Critical Rollover Slope:** If $\theta_{\text{slope}} \ge 28.0^\circ$, **cost is forced to $1.00$** (initial simulation safety threshold / engineering assumption).
3. **Critical High-Centering Step:** If $\Delta z_{\text{height}} \ge 0.20\text{ m}$, **cost is forced to $1.00$** (initial simulation safety threshold / engineering assumption).
4. **Unknown Policy:** Cells marked `unknown` are assigned cost based on active policy mode (`CONSERVATIVE`: $0.65$, `MODERATE`: $0.50$, `EXPLORATORY`: $0.35$). **Unknown is NEVER converted to SAFE.**

---

## Section E: Temporal Fusion & Motion Compensation

Implemented in [ai/costmap/temporal.py](file:///c:/Users/shres/Desktop/SIH2026/ai/costmap/temporal.py) and [ai/costmap/motion_compensation.py](file:///c:/Users/shres/Desktop/SIH2026/ai/costmap/motion_compensation.py):

### 1. SE(2) Rigid Map Alignment:
Given relative vehicle ego-motion $\Delta \mathbf{T} = [\Delta x, \Delta y, \Delta \theta]$ between frames, the previous local costmap is transformed into the current robot frame via an affine warping matrix:
$$\mathbf{P}_{\text{curr}} = \mathbf{R}(-\Delta \theta) (\mathbf{P}_{\text{prev}} - [\Delta x, \Delta y]^T)$$
Unobserved territory entering the grid boundary is strictly padded with `unknown = True`.

### 2. Differentiated Static vs. Dynamic Persistence:
- **Static Terrain (Ground surface, slopes, ditches):** Decays very slowly ($\alpha_{\text{static}} = 0.98$), enabling the rover to retain knowledge of terrain it has already driven over or scanned.
- **Dynamic Obstacles (Persons, vehicles):** Decays rapidly ($\alpha_{\text{dynamic}} = 0.70$). When clean traversable ground is observed at a dynamic obstacle's prior position, the obstacle is cleared immediately to prevent ghost phantom obstacles.
- **Sensor Dropout Tolerance:** Momentary 1-to-3 frame sensor blackouts or ray-tracing dropouts retain confirmed obstacles without flickering to unknown.

---

## Section F: Obstacle Inflation

Implemented in [ai/costmap/inflation.py](file:///c:/Users/shres/Desktop/SIH2026/ai/costmap/inflation.py):
- **Vehicle Footprint:** $L = 0.65\text{ m}$, $W = 0.45\text{ m}$.
- **Inscribed Collision Radius:** $R_{\text{inscribed}} = 0.35\text{ m}$ (Cells within this radius are forced to lethal $C = 1.00$).
- **Circumscribed Inflation Radius:** $R_{\text{infl}} = 0.55\text{ m}$.
- **Cost Gradient Formulation:**
  For $R_{\text{inscribed}} < d \le R_{\text{infl}}$:
  $$C(d) = \exp\left(-\alpha \cdot (d - R_{\text{inscribed}})\right), \quad \alpha = 3.0$$
  This creates a smooth, continuous gradient that repels the Phase 6 path planner safely away from obstacles while permitting tight maneuvers through narrow gates.

---

## Section G: Empirical Benchmarks & Ablation Studies

### 1. Grid Resolution & Extent Consistency:
The local costmap covers an effective physical extent of **$5.8\text{ m}$ forward** ($x \in [0.2\text{ m}, 6.0\text{ m}]$, ahead of the vehicle bumper) $\times$ **$6.0\text{ m}$ lateral** ($y \in [-3.0\text{ m}, 3.0\text{ m}]$, symmetric across the centerline).

The mathematical relationship between physical extent, resolution, and grid dimensions is exact and verified by unit test `test_physical_extent_resolution_consistency`:
$$\text{Grid Rows } (n_x) = \left\lceil \frac{x_{\max} - x_{\min}}{\text{resolution}} \right\rceil = \frac{5.8\text{ m}}{0.10\text{ m}} = 58 \quad \iff \quad 58 \times 0.10\text{ m} = 5.8\text{ m}$$
$$\text{Grid Cols } (n_y) = \left\lceil \frac{y_{\max} - y_{\min}}{\text{resolution}} \right\rceil = \frac{6.0\text{ m}}{0.10\text{ m}} = 60 \quad \iff \quad 60 \times 0.10\text{ m} = 6.0\text{ m}$$

Measured using [scripts/benchmark_costmap.py](file:///c:/Users/shres/Desktop/SIH2026/scripts/benchmark_costmap.py) on host AMD Ryzen CPU:

| Metric | 5 cm Resolution | 10 cm Resolution (Default) | 20 cm Resolution | Downstream Planning Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Grid Dimensions** | $116 \times 120$ ($13,920$ cells) | **$58 \times 60$ ($3,480$ cells)** | $29 \times 30$ ($870$ cells) | 10cm provides optimal cell count |
| **Physical Extent** | $5.8\text{ m} \times 6.0\text{ m}$ | **$5.8\text{ m} \times 6.0\text{ m}$** | $5.8\text{ m} \times 6.0\text{ m}$ | Matches bumper-to-horizon range |
| **Memory Footprint** | $462.2\text{ KB}$ | **$115.5\text{ KB}$** | $28.9\text{ KB}$ | Extremely lightweight (< 1 MB) |
| **Multi-Layer Fusion** | $0.26\text{ ms}$ | **$0.13\text{ ms}$** | $0.10\text{ ms}$ | Sub-millisecond raster math |
| **Obstacle Inflation** | $0.20\text{ ms}$ | **$0.08\text{ ms}$** | $0.06\text{ ms}$ | Vectorized distance transform |
| **Motion Alignment** | $0.93\text{ ms}$ | **$0.70\text{ ms}$** | $1.05\text{ ms}$ | Real-time SE(2) warping |
| **Temporal Update** | $1.31\text{ ms}$ | **$0.81\text{ ms}$** | $0.73\text{ ms}$ | Sub-millisecond temporal blend |
| **Total Costmap Cycle** | $2.70\text{ ms}$ | **$1.72\text{ ms}$** | $1.94\text{ ms}$ | Runs at $> 500\text{ FPS}$ on costmap logic |

*Resolution Recommendation:* **$10\text{ cm/cell}$** is certified as the default. It captures small $12\text{ cm}$ curbs and protruding rocks while requiring only $115.5\text{ KB}$ memory and $< 2\text{ ms}$ processing time.

---

### 2. Multi-Layer Ablation Study:
Evaluated using [scripts/evaluate_costmap.py](file:///c:/Users/shres/Desktop/SIH2026/scripts/evaluate_costmap.py):

> [!IMPORTANT]
> The blocked-area percentages below represent **scenario-specific observed blocked-area percentages** on the evaluated test frames, NOT universal safety performance across all environments. Crucially, an increase in blocked area is not automatically an improvement—an excessively blocked map degrades path feasibility. The justification for improvement is driven strictly by **false-safe reduction** (eliminating lethal water reflections that Model A marked drivable), **hazard recall** (detecting low rocks), and **safety objectives** (flagging steep terrain), rather than merely blocking more area.

| Ablation Configuration | Safe Area Ratio | Blocked Area Ratio | Mean Cost | Scenario-Specific Safety Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Ablation A: Learned Perception Only** | $68.4\%$ | $8.2\%$ | $0.264$ | Misses thin obstacles, fails on creek reflections |
| **Ablation B: Learned + Explicit Hazards** | $54.1\%$ | $18.9\%$ | $0.382$ | Water and rocks properly classified as blocked |
| **Ablation C: Learned + Terrain Geometry** | $51.6\%$ | $14.3\%$ | $0.341$ | Steep slopes ($>28^\circ$) flagged as blocked |
| **Ablation D: Full Fusion + Hard Overrides** | **$43.2\%$** | **$24.5\%$** | **$0.449$** | **Zero false-safe cells observed on evaluated test cases** |

---

### 3. Temporal Stability & False-Safe Preservation:
- **Temporal Flicker & Jitter Reduction:** **$13.0\%$ reduction** in state transitions compared to instantaneous single frames (evaluated over 20 consecutive frames with simulated ego-motion and noise).
- **BEV False-Safe Preservation:** On the evaluated `creek_water_reflection` stress scenario (based on `rugd_creek_00001.png`), 14 direct hazard cells were evaluated across the 3,480-cell BEV grid. **Zero false-safe cells were observed on this evaluated test case** ($100\%$ of detected water cells were assigned lethal cost $1.00$).
  > [!WARNING]
  > This measurement confirms that the BEV projection and cost fusion logic successfully preserves safety overrides on this specific test case. It does **NOT** establish complete physical safety or general safety across all unseen real-world environments.

---

## Section H: Real-Data Validation Scope

To maintain rigorous engineering integrity, Phase 5 validation claims are explicitly separated into what was tested vs. what remains unverified:

### Validated:
- Phase 5 pipeline execution on available RUGD trail sequences and real rover camera frames (`frame.jpg`, `data/`).
- Multi-layer BEV projection, temporal accumulation, and obstacle inflation on 5 challenging stress scenarios (`outputs/costmap/scenarios/`):
  1. `nominal_safe_trail`
  2. `creek_water_reflection`
  3. `protruding_dark_rock`
  4. `critical_rollover_slope`
  5. `model_disagreement_unknown`

### Not Validated (Future Physical Work):
- **Complete physical rover BEV accuracy:** Real metric distance scaling has not been verified against ground-truth survey markers.
- **Survey-marker calibration:** Optical-to-ground calibration relies on nominal camera height ($0.85\text{ m}$) and pitch ($-15.0^\circ$).
- **Real metric depth accuracy:** Depth sources in Phase 5 rely on analytical planar ground or synthetic disparity; real stereo matching/LiDAR point cloud depth is pending physical integration.
- **Real vehicle footprint validation:** The $0.65\text{ m} \times 0.45\text{ m}$ footprint and $0.35\text{ m}$ inscribed circle have not been experimentally validated on a physical rover chassis navigating tight obstacle gates.

---

## Section I: Planner Input Contract (Phase 6 Handshake)

Verified by `test_planner_input_contract_stub` in [tests/test_costmap_safety.py](file:///c:/Users/shres/Desktop/SIH2026/tests/test_costmap_safety.py).
Phase 6 motion planners consume `BEVTerrainCostmap` via clean, documented methods:
- `costmap.get_cost(x_m, y_m) -> float`: Continuous traversal cost in $[0.0, 1.0]$.
- `costmap.is_passable(x_m, y_m, max_cost=0.65) -> bool`: Binary traversability check.
- `costmap.is_unknown(x_m, y_m) -> bool`: Uncertainty flag for exploratory planners.
- `costmap.meter_to_grid(x_m, y_m) -> (row, col)`: Forward coordinate transform.
- `costmap.grid_to_meter(row, col) -> (x_m, y_m)`: Inverse coordinate transform.

Phase 6 planners require **zero dependencies** on SAM-TP, RUGD, or image-space perception internals.

---

## PHASE 5 STATUS: APPROVED WITH DOCUMENTATION CORRECTIONS

## PHASE 6 READINESS: READY

| Checkpoint Item | Audit Result | Notes |
| :--- | :---: | :--- |
| **Canonical BEV costmap** | **PASS** | `BEVTerrainCostmap` with 10 synchronized NumPy tensor layers |
| **Coordinate convention** | **PASS** | ISO 8855 / ROS standard: $X$ forward, $Y$ left, $Z$ up |
| **Extent & Resolution Consistency** | **PASS** | $5.8\text{ m} \times 6.0\text{ m}$ extent at $10\text{ cm} \iff 58 \times 60$ grid (verified by unit test) |
| **Perception → BEV** | **PASS** | `LayerGenerator` projects fusion, hazards, and risk into BEV raster |
| **Terrain geometry layer** | **PASS** | Slope angle, roughness, and step variation layers active |
| **Hazard layer** | **PASS** | Explicit hazard classes mapped with spatial extent |
| **Unknown handling** | **PASS** | First-class unknown layer; configurable unknown policy: CONSERVATIVE 0.65 |
| **Temporal fusion** | **PASS** | `TemporalCostmapAccumulator` maintains moving temporal persistence ($13.0\%$ flicker reduction) |
| **Motion compensation interface** | **PASS** | `MotionCompensator` SE(2) rigid warp with unobserved boundary padding |
| **Dynamic obstacle persistence** | **PASS** | Rapid decay and active clearing prevents moving phantom obstacles |
| **Cost fusion** | **PASS** | `BEVCostmapFuser` weighted physical cost calculation |
| **Safety overrides** | **PASS** | Lethal cost $1.00$ enforced on water, ditch, rock, slope $\ge 28^\circ$, step $\ge 0.20\text{ m}$ |
| **Inflation** | **PASS** | `ObstacleInflator` distance transform with vehicle footprint $0.65\text{ m} \times 0.45\text{ m}$ |
| **Real-data validation** | **PASS** | Tested on RUGD real trails and rover camera feeds (physical survey calibration pending) |
| **False-safe after BEV conversion** | **PASS** | Zero false-safe cells observed on evaluated creek water reflection test case |
| **Performance benchmark** | **PASS** | $10\text{ cm}$ resolution: $1.72\text{ ms}$ costmap update cycle, $115.5\text{ KB}$ memory |
| **Ablation study** | **PASS** | Ablations A through D executed and documented with scenario-specific findings |
| **Planner input contract** | **PASS** | Phase 6 consumer contract verified by standalone stub test |
| **Regression tests** | **118 PASS** | 100% test pass rate across 118 tests in repository |
| **Documentation** | **PASS** | Fully audited, qualified, and corrected |

---

### Known Limitations & Deferred Work:
1. **Real metric BEV validation pending:** Survey-marker distance verification on physical ground is reserved for physical rover testing.
2. **Physical rover threshold validation pending:** The $28.0^\circ$ critical slope and $0.20\text{ m}$ step height are initial simulation assumptions and must be validated on a physical tilt table.
3. **Shallow muddy-water separation:** Still depends on metric stereo disparity for non-reflective shallow water boundaries.
4. **GPU acceleration not yet benchmarked:** Pipeline benchmarks reflect host CPU execution ($1.72\text{ ms}$ for costmap logic; perception projection takes $503\text{ ms}$ on CPU; GPU execution marked PENDING).

---

*Certified by Antigravity Autonomous Systems Engineering.*
