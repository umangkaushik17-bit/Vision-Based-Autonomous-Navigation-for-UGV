# PHASE 13 — DYNAMIC OBSTACLE TRACKING & PREDICTIVE REPLANNING
# MASTER ENGINEERING CONTEXT & SPECIFICATION

**Project:** SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment
**Document Status:** ACTIVE / SINGLE SOURCE OF TRUTH FOR PHASE 13
**File:** `docs/PHASE_13_MASTER_CONTEXT.md`
**Reference Commit:** `7555164`
**Reference Tag:** `phase-12-conditional-baseline`

---

> [!IMPORTANT]
> **MANDATORY GOVERNANCE RULE:**
> From Milestone 13.1 onward, every milestone MUST read and use this file before making implementation decisions.
> Only the explicitly authorized milestone may be executed.
> The agent MUST NOT automatically continue to the next milestone.
> After completing a milestone: **STOP**, report checkpoint, and wait for explicit instruction.

---

## 1. Phase 13 Objective

The primary objective of Phase 13 is to endow the autonomous outdoor UGV with real-time dynamic obstacle detection, multi-frame tracking, state estimation (Kalman filtering), trajectory prediction, and predictive collision-avoidance replanning capabilities.

Phase 13 enables the UGV to safely navigate semi-structured and unstructured outdoor environments containing moving objects (e.g., pedestrians, other vehicles, wildlife, crossing agents) by:
1. Identifying dynamic obstacles from visual/depth perception without ground-truth cheating.
2. Tracking obstacle bounding representations over time with persistent track identities.
3. Estimating 2D planar velocities $(v_x, v_y)$ and filtering observation noise via linear/extended Kalman tracking.
4. Extrapolating forward motion trajectories over prediction horizons $H \in [1.0, 3.0]\,\text{s}$ with associated spatial uncertainty covariances.
5. Ingesting predicted obstacle footprints into a time-parameterized dynamic occupancy layer separate from the static `WorldTerrainMap`.
6. Evaluating dynamic collision risks inside the local path planner (predictive DWA) and triggering event-driven global replanning ($A^*$) when trajectories are irrevocably blocked.
7. Enforcing an explicit **Wait / Avoid / Stop** behavioral arbitration hierarchy backed by fail-safe Phase 12 safety clamps.

---

## 2. Phase 12 Baseline

Phase 13 builds strictly upon the frozen Phase 12 baseline:

- **Git Commit:** `7555164`
- **Git Tag:** `phase-12-conditional-baseline`
- **Status:** `CONDITIONAL ACCEPTANCE / NOT SEALED`
- **Branch:** `master`

### Core Capabilities Inherited from Phase 12:
- **Perception Pipeline:** Multi-class semantic traversability (5 classes: traversable surface, rough trail, obstacle, hazard, sky/background).
- **Hazard Perception:** Depth/stereo-based geometry reconstruction, slope estimation, step-hazard analysis, drop-off detection.
- **Costmap Representation:** 2D BEV grid raster, multi-layer aggregation (traversability cost, slope cost, step cost, obstacle inflation), configurable cell resolution ($0.10\,\text{m/cell}$).
- **Localization:** Monocular/stereo Visual Odometry (`LocalizationAdapter`), $1.167$ empirical scale compensation factor, covariance reporting, state machine (`UNINITIALIZED`, `INITIALIZING`, `TRACKING`, `LOST`, `RELOCALIZING`).
- **Spatial Fusion:** Recursive Bayesian occupancy updating into a global rolling/static `WorldTerrainMap`.
- **Global Planner:** Grid-based $A^*$ pathfinder with octile heuristics, clearance penalties, and path smoothing (Douglas-Peucker/B-spline).
- **Local Planner:** Dynamic Window Approach (DWA) with kinematic limits ($v_{\max} = 0.6\,\text{m/s}$, $\omega_{\max} = 0.8\,\text{rad/s}$), forward trajectory simulation, exact rectangular UGV footprint collision evaluation ($L = 0.90\,\text{m}$, $W = 0.65\,\text{m}$, $r_{\text{margin}} = 0.15\,\text{m}$).
- **Safety Arbitration:** Hierarchical supervisory controller evaluating hard stops ($d < 0.35\,\text{m}$), speed clamping ($0.35\,\text{m} \le d < 0.60\,\text{m}$), and localization watchdog monitoring.
- **ROS2 Architecture:** Standardized message definitions, publisher/subscriber nodes, lifecycle managers, and SIL simulation runners.

---

## 3. Phase 12 Known Limitations

The following limitations were identified during Phase 12 evaluation and remain active. **They MUST NOT be described as solved unless verified by new empirical evidence in future phases:**

1. **Rock Detour Ground-Truth Terminal Error:**
   In the curved Rock detour SIL mission, the ground-truth terminal position error reached $0.226\,\text{m}$, slightly exceeding the nominal acceptance tolerance of $0.200\,\text{m}$ (planner estimated $0.198\,\text{m}$).
2. **Lateral Visual Odometry Drift:**
   During sustained curved yaw maneuvers without loop closure, the monocular/stereo VO accumulates approximately $5.5\,\text{cm}$ of lateral drift relative to ground-truth simulation coordinates.
3. **Pure Python DWA Latency:**
   The unoptimized Python implementation of DWA evaluates trajectories at approximately $1.36\,\text{s}$ to $2.15\,\text{s}$ per planning cycle, preventing high-frequency ($10\text{--}30\,\text{Hz}$) local reactive control without sub-stepping or vectorization.
4. **Native Linux Gazebo Execution Unverified:**
   All benchmarks were collected under a deterministic Software-In-The-Loop (SIL) simulation framework. Execution under native ROS2 Humble inside Ubuntu 22.04 with hardware GPU acceleration has not been fully regression-tested.
5. **Localization Scale Compensation:**
   The $1.167$ visual odometry scale compensation factor is empirically calibrated to the specific simulated camera mount ($+0.15\,\text{m}$ forward offset, $15^\circ$ downward pitch) and is not a universal metric scale estimator.

---

## 4. Phase 13 Scope

Phase 13 encompasses the following functional subsystems:
- **Obstacle Representation:** Mathematical data models for detected, tracked, and predicted 2D dynamic obstacles with oriented bounding boxes (OBB) or axis-aligned bounding boxes (AABB), velocity vectors, covariance matrices, and tracking confidence.
- **Detection Abstraction:** Clean interface decoupling dynamic obstacle detections from backend sources (e.g., optical flow, moving object segmentation, clustering of persistent map discrepancies, or simulated perception feeds).
- **Multi-Frame Data Association:** Multi-object tracking (MOT) utilizing spatial distance gating (Euclidean/Mahalanobis) and optimal Hungarian bipartite matching.
- **State Estimation & Filtering:** 2D constant-velocity (CV) and coordinated-turn (CT) Kalman Filters (KF/EKF) estimating $[x, y, v_x, v_y]^\top$ with covariance propagation.
- **Trajectory Extrapolation:** Deterministic and probabilistic future state prediction over $1.0\text{--}3.0\,\text{s}$ horizons with time-growing positional covariance ellipses.
- **Dynamic Occupancy Representation:** Ephemeral, time-sliced spatio-temporal reservation grids or spatio-temporal obstacle hulls that do NOT corrupt the static `WorldTerrainMap`.
- **Predictive DWA:** Integration of time-parameterized collision checking where candidate robot trajectories $\gamma(t)$ are evaluated against predicted obstacle distributions $O_i(t)$ at corresponding timesteps $t \in [0, T_h]$.
- **Behavioral Decision Logic:** Explicit finite state machine handling dynamic conflict arbitration: **Wait** (yield right-of-way if clearance is opening), **Avoid** (bias DWA trajectories away from predicted paths), and **Stop** (emergency deceleration if collision is imminent).
- **Event-Driven Replanning:** Global $A^*$ re-triggering logic when a dynamic obstacle permanently occludes the global path or comes to rest within the static corridor.
- **ROS2 & SIL Integration:** ROS2 topic interfaces (`/perception/dynamic_obstacles`, `/planning/predicted_tracks`, `/planning/dynamic_costmap`), visualization markers, and comprehensive benchmark suite.

---

## 5. Phase 13 Non-Goals

To prevent scope creep and maintain architectural boundaries, the following are strictly **OUT OF SCOPE** for Phase 13:
1. **No Deep Learning Object Detectors on Real Hardware:** Training or fine-tuning real-time YOLOv8/v10/3D bounding box networks on raw camera frames is deferred. Phase 13 focuses on the tracking, prediction, and replanning pipeline using synthetic/abstracted dynamic detections.
2. **No Alteration of Static Costmap Engine:** The Phase 5/9 static `WorldTerrainMap` and spatial fusion pipeline must remain unmodified. Dynamic obstacles must never be permanently burned into static terrain costs.
3. **No Redesign of Phase 12 Global A* Core:** The underlying $A^*$ search algorithm is frozen. Phase 13 introduces an event-driven trigger wrapper, not a complete graph-search rewrite.
4. **No Arbitrary Loosening of Phase 12 Safety Limits:** Hard emergency stop distance ($0.35\,\text{m}$), safety clamp distance ($0.60\,\text{m}$), and robot physical bounding footprint ($0.90\,\text{m} \times 0.65\,\text{m}$) cannot be shrunk.
5. **No Full 3D Non-Rigid Physics Simulation:** Obstacles are modeled as 2D rigid kinematic entities (cylinders/OBBs) moving in the plane. Dynamic terrain deformation, splashing, or aerial obstacles are excluded.
6. **No Real-Time 30 Hz Guarantee on Pure Python:** Latency profiling will document incremental overhead honestly. C++/Cython acceleration is reserved for future optimization phases if required.

---

## 6. Mandatory Safety Rules

The Phase 12 safety hierarchy is immutable and supreme. Phase 13 dynamic planning logic operates strictly below the safety supervisory layer:

```
+-------------------------------------------------------------+
|                      EMERGENCY STOP                         |
|   (Distance < 0.35 m OR Localization LOST -> Immediate v=0)  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                       SAFETY CLAMP                          |
|  (0.35 m <= Distance < 0.60 m -> v_cmd <= 0.2 m/s, strict w)|
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 DYNAMIC BEHAVIORAL POLICY                   |
|       (Wait / Yield / Decelerate / Avoid / Stop Policy)     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     PREDICTIVE DWA                          |
|     (Kinodynamic rollout against dynamic predicted hulls)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     PATH FOLLOWING                          |
|             (A* Global Trajectory Tracking)                 |
+-------------------------------------------------------------+
```

### Inviolable Safety Constraints:
1. **Priority Hierarchy:** `Emergency Stop > Safety Clamp > Dynamic Behavior > DWA > Path Following`.
2. **Localization Watchdog:**
   - If localization state enters `LOST`, the robot must execute an immediate **Emergency Stop** ($v=0, \omega=0$).
   - If localization state enters `RELOCALIZING`, the robot must maintain a stationary hold until covariance converges.
   - If localization data becomes stale ($> 0.5\,\text{s}$ latency), velocity commands must decay to zero.
3. **Dynamic Planner Cannot Invalidate Safety:** No predicted clearance, dynamic avoidance maneuver, or behavioral bypass can override a hard safety stop triggered by sensor proximity.
4. **Footprint Rigor:** All predictive collision checks must evaluate the full rectangular UGV footprint ($0.90\,\text{m} \times 0.65\,\text{m} + 0.15\,\text{m}$ safety margin), not a single point mass.

---

## 7. Ground-Truth Isolation Rules

Ground truth (GT) data generated by the simulation environment or benchmark runner is subject to strict isolation:

### Permitted Uses of Ground Truth:
- Generating dynamic obstacle motions in simulation scenarios.
- Offline evaluation of tracking accuracy (RMSE, track purity, MOTA).
- Offline evaluation of trajectory prediction error.
- Post-run benchmark metric calculations.
- Visualizing actual vs. estimated paths in analysis plots and MATLAB exports.

### Prohibited Uses of Ground Truth:
- **RUNTIME CODE ISOLATION:** Ground truth MUST NOT enter:
  - Dynamic detector runtime logic
  - Multi-frame tracker
  - Velocity estimator
  - Kalman filter
  - Trajectory predictor
  - Dynamic occupancy grid generator
  - Dynamic DWA cost evaluator
  - Global $A^*$ replanner
  - Safety supervisor
  - ROS2 planner input messages
  - Command velocity (`/cmd_vel`) generation

All dynamic planning algorithms must operate strictly on perceived sensor measurements, estimated states, and associated covariances.

---

## 8. Existing Phase 12 Architecture Relevant to Phase 13

The existing codebase provides key architectural components that Phase 13 interfaces with:

1. **`ai/planning/dwa.py`:**
   - Generates kinematic velocity pairs $(v, \omega)$.
   - Simulates forward trajectories over horizon $T_{\text{sim}} = 2.0\text{--}3.0\,\text{s}$.
   - Evaluates static obstacle distance via footprint polygons against the static costmap.
2. **`ai/planning/astar.py`:**
   - Computes global discrete 8-connected grid paths on 2D costmaps.
   - Employs octile heuristic and clearance inflation penalties.
3. **`ai/planning/base.py`:**
   - Base classes for planner state, goal definitions, and path structures.
   - Configured goal tolerance: $0.20\,\text{m}$ (corrected in Phase 12 baseline).
4. **`ai/planning/safety.py`:**
   - Safety supervisor enforcing emergency stop and speed clamping thresholds based on laser/depth proximity checks.
5. **`ai/spatial_fusion/terrain_map.py` & `costmap_node.py`:**
   - Multi-layer spatial costmap maintaining static terrain, slope, step, and traversability layers.
6. **`ai/localization/adapter.py`:**
   - Provides estimated robot pose $[x, y, \theta]$ in the world (`odom`) frame with timestamp, linear velocity, and tracking status.
7. **`scripts/evaluate_path_planning.py` & `simulation/sil/`:**
   - Deterministic SIL benchmark runner, metric evaluators, and matplotlib plotting pipelines.

---

## 9. Phase 13 Target Architecture

The target runtime architecture integrates dynamic obstacle tracking cleanly alongside the static map pipeline:

```
                              [ Sensor Perception ]
                             (Stereo Depth / Camera)
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
         [ Static Costmap Engine ]                [ Dynamic Obstacle Detector ]
        (Spatial Fusion / Terrain)             (Clusters / Optical Flow Discrepancy)
                   |                                         |
                   v                                         v
          [ WorldTerrainMap ]                     [ Multi-Frame MOT Tracker ]
        (Static Traversability)                  (Data Association / Hungarian)
                   |                                         |
                   |                                         v
                   |                              [ Kalman State Filter ]
                   |                           (Position + Velocity Estimation)
                   |                                         |
                   |                                         v
                   |                             [ Trajectory Predictor ]
                   |                          (Spatiotemporal Uncertainty Hulls)
                   |                                         |
                   |                                         v
                   |                              [ Dynamic Occupancy Layer ]
                   |                            (Time-Sliced Local Occupancy)
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
                           [ Predictive Collision Check ]
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
        [ Behavioral State Machine ]               [ Dynamic DWA Planner ]
         (Wait / Avoid / Yield Policy)          (Trajectory Rollout vs Time Hulls)
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
                         [ Event-Driven A* Replanner ]
                       (Triggered if Corridor Blocked)
                                        |
                                        v
                          [ Phase 12 Safety Clamp ]
                         (Hard Clearance Supervisory)
                                        |
                                        v
                                    [/cmd_vel]
```

---

## 10. All Milestone Definitions (13.0 through 13.19)

### Milestone 13.0: Architecture Audit
- **Objective:** Perform thorough codebase audit of existing Phase 12 planning, localization, and costmap interfaces; verify extension points for dynamic obstacle integration without breaking baseline guarantees.
- **Inputs:** Phase 12 codebase (`ai/planning/`, `ai/spatial_fusion/`, `ai/localization/`, `scripts/evaluate_path_planning.py`).
- **Expected Outputs:** Architecture audit report documenting existing interfaces, latencies, memory patterns, and extension points.
- **Files Likely Involved:** Read-only audit of existing repositories; `docs/PHASE_13_MASTER_CONTEXT.md`.
- **Dependencies:** Phase 12 baseline tag `phase-12-conditional-baseline`.
- **Tests Required:** Verification of Phase 12 regression test suites.
- **Acceptance Criteria:** Documented audit of data structures, zero code modifications, identification of exact insertion interfaces.
- **What It Must NOT Modify:** Any implementation code.
- **Required Checkpoint:** Audit acceptance report and authorization to proceed to 13.1.

### Milestone 13.1: Dynamic Obstacle Data Model
- **Objective:** Implement the core mathematical and software data structures for representing dynamic obstacles across detection, tracking, estimation, and prediction stages.
- **Inputs:** Mathematical specification of 2D bounding boxes, state vectors $[x, y, v_x, v_y]^\top$, covariance matrices, confidence scores, and trajectory waypoints.
- **Expected Outputs:** Tested data structures (`DynamicObstacle`, `ObstacleTrack`, `PredictedTrajectory`, `DynamicObstacleArray`) with strict serialization and validation.
- **Files Likely Involved:** `ai/planning/dynamic/`, `ai/planning/dynamic/data_model.py`, `tests/test_dynamic_data_model.py`.
- **Dependencies:** Milestone 13.0.
- **Tests Required:** Unit tests for model instantiation, validation checks, bounding geometry methods, serialization, and coordinate transformation utilities.
- **Acceptance Criteria:** 100% test pass on data model geometry, valid covariance handling, zero static costmap modifications.
- **What It Must NOT Modify:** `ai/planning/dwa.py`, `ai/planning/astar.py`, `ai/spatial_fusion/`.
- **Required Checkpoint:** Data model test suite green; checkpoint report submitted.

### Milestone 13.2: Detection Interface
- **Objective:** Create an abstract detection interface that ingests raw perceptual clusters or synthetic simulation detections and outputs standardized `DynamicObstacleDetection` packets.
- **Inputs:** Simulated obstacle bounding clusters, sensor timestamp, coordinate frame metadata.
- **Expected Outputs:** Interface class `DynamicObstacleDetectorBase` and synthetic simulation implementation `SimulatedDynamicObstacleDetector` (operating without ground-truth cheating in runtime logic).
- **Files Likely Involved:** `ai/planning/dynamic/detector_interface.py`, `tests/test_dynamic_detector.py`.
- **Dependencies:** Milestone 13.1.
- **Tests Required:** Detection packaging tests, timestamp verification, noise injection tests, frame validation.
- **Acceptance Criteria:** Standardized detection outputs decoupled from source; zero GT state leaked to runtime logic.
- **What It Must NOT Modify:** Phase 12 perception or spatial fusion logic.
- **Required Checkpoint:** Detection interface green; checkpoint report submitted.

### Milestone 13.3: Multi-Frame Tracking
- **Objective:** Implement multi-object tracking (MOT) associating incoming detections across successive frames to maintain persistent obstacle IDs over time.
- **Inputs:** Sequences of `DynamicObstacleDetection` packets across time $\Delta t$.
- **Expected Outputs:** `MultiFrameTracker` class with track initiation, maintenance, coasting, and pruning; persistent track IDs.
- **Files Likely Involved:** `ai/planning/dynamic/tracker.py`, `ai/planning/dynamic/association.py`, `tests/test_dynamic_tracker.py`.
- **Dependencies:** Milestones 13.1, 13.2.
- **Tests Required:** Hungarian matching validation, track birth/death tests, missed detection coasting tests, ID switch minimization tests.
- **Acceptance Criteria:** Correct association across straight and crossing obstacle paths; robust coasting through $\ge 3$ missed frames.
- **What It Must NOT Modify:** Static costmap or planner files.
- **Required Checkpoint:** Tracking unit and integration tests passing; checkpoint report submitted.

### Milestone 13.4: Motion Estimation
- **Objective:** Implement linear and angular motion estimation utilities deriving empirical velocity vectors $(\hat{v}_x, \hat{v}_y)$ and heading rates from track history.
- **Inputs:** Multi-frame track position histories and timestamps.
- **Expected Outputs:** Velocity estimation module computing finite-difference and regression-based velocities with basic noise rejection.
- **Files Likely Involved:** `ai/planning/dynamic/motion_estimator.py`, `tests/test_motion_estimator.py`.
- **Dependencies:** Milestone 13.3.
- **Tests Required:** Synthetic trajectory velocity recovery tests, noisy observation regression tests, zero-velocity static detection tests.
- **Acceptance Criteria:** Velocity estimation within $10\%$ of true velocity for steady linear motion; stationary obstacles correctly flagged ($v \approx 0$).
- **What It Must NOT Modify:** Localization or Phase 12 safety code.
- **Required Checkpoint:** Motion estimation verified; checkpoint report submitted.

### Milestone 13.5: Filtering / Kalman Tracking
- **Objective:** Implement formal state filtering using a 2D Constant Velocity (CV) Kalman Filter (and optional Coordinated Turn EKF) to output smooth states $[x, y, v_x, v_y]^\top$ and covariance matrices $P_{4\times 4}$.
- **Inputs:** Associated track observations with measurement noise covariance $R$.
- **Expected Outputs:** `KalmanTrackFilter` class executing predict-update cycles with Mahalanobis gating and covariance propagation.
- **Files Likely Involved:** `ai/planning/dynamic/kalman_filter.py`, `tests/test_kalman_filter.py`.
- **Dependencies:** Milestones 13.3, 13.4.
- **Tests Required:** Filter convergence tests, covariance consistency tests, outlier rejection tests, latency benchmarking.
- **Acceptance Criteria:** Filtered state RMSE strictly lower than raw measurement noise; covariance accurately bounds estimation error.
- **What It Must NOT Modify:** Phase 12 core planning files.
- **Required Checkpoint:** Kalman filter test suite passing; checkpoint report submitted.

### Milestone 13.6: Motion Prediction
- **Objective:** Implement trajectory extrapolation predicting future obstacle spatial distributions $O_i(t)$ over horizons $t \in [t_0, t_0 + H]$ ($H \in [1.0, 3.0]\,\text{s}$) with growing uncertainty ellipses.
- **Inputs:** Filtered state $[x, y, v_x, v_y]^\top$ and covariance matrix $P$.
- **Expected Outputs:** `MotionPredictor` generating `PredictedTrajectory` objects containing timestamped waypoints, bounding hulls, and covariance ellipses.
- **Files Likely Involved:** `ai/planning/dynamic/predictor.py`, `tests/test_motion_predictor.py`.
- **Dependencies:** Milestone 13.5.
- **Tests Required:** Multi-horizon prediction tests ($1.0\,\text{s}, 2.0\,\text{s}, 3.0\,\text{s}$), covariance expansion validation, boundary geometry checks.
- **Acceptance Criteria:** Prediction accurately reflects constant-velocity kinematic projection; uncertainty grows monotonically with time.
- **What It Must NOT Modify:** Static costmap or DWA.
- **Required Checkpoint:** Predictor unit tests passing; checkpoint report submitted.

### Milestone 13.7: Dynamic Occupancy Layer
- **Objective:** Implement a dedicated, time-parameterized dynamic occupancy representation that stores predicted obstacle footprints without polluting the static `WorldTerrainMap`.
- **Inputs:** List of `PredictedTrajectory` objects from all active tracks.
- **Expected Outputs:** `DynamicOccupancyLayer` providing fast spatial-temporal query interfaces (e.g., query occupancy at $(x, y, t)$ or query collision along segment $(p_1, p_2)$ at time $t$).
- **Files Likely Involved:** `ai/planning/dynamic/occupancy_layer.py`, `tests/test_dynamic_occupancy.py`.
- **Dependencies:** Milestones 13.1, 13.6.
- **Tests Required:** Point query tests, time-slice evaluation, footprint rasterization tests, reset/clearing validation.
- **Acceptance Criteria:** Spatial queries execute in $< 5\,\text{ms}$; complete separation from static costmap verified (zero persistent leaks).
- **What It Must NOT Modify:** `ai/spatial_fusion/terrain_map.py`.
- **Required Checkpoint:** Dynamic occupancy layer passing tests; checkpoint report submitted.

### Milestone 13.8: Predictive Collision Checking
- **Objective:** Implement time-synchronized collision checking between candidate robot trajectory rollouts and dynamic obstacle predicted distributions.
- **Inputs:** Robot simulated trajectory $\gamma(t) = \{(x_k, y_k, \theta_k, t_k)\}$ and `DynamicOccupancyLayer` / predicted obstacle tracks.
- **Expected Outputs:** Collision checking function returning boolean collision status, time-to-closest-approach (TCA), and minimum spatial clearance.
- **Files Likely Involved:** `ai/planning/dynamic/predictive_collision.py`, `tests/test_predictive_collision.py`.
- **Dependencies:** Milestones 13.6, 13.7.
- **Tests Required:** Frontal collision tests, crossing collision tests, rear overtaking tests, safe clearance tests.
- **Acceptance Criteria:** Reliable detection of future dynamic collisions at correct timestamp $t_c$; exact UGV rectangular footprint evaluated.
- **What It Must NOT Modify:** Phase 12 safety supervisor.
- **Required Checkpoint:** Predictive collision suite passing; checkpoint report submitted.

### Milestone 13.9: Dynamic DWA Integration
- **Objective:** Augment the Phase 12 Local DWA planner to incorporate predictive dynamic obstacle costs into the trajectory scoring objective.
- **Inputs:** Candidate velocity pairs $(v, \omega)$, static costmap, `DynamicOccupancyLayer`, and goal pose.
- **Expected Outputs:** `DynamicDWAPlanner` balancing heading progress, velocity, static clearance, and dynamic obstacle avoidance.
- **Files Likely Involved:** `ai/planning/dynamic_dwa.py`, `tests/test_dynamic_dwa.py`.
- **Dependencies:** Milestones 13.7, 13.8.
- **Tests Required:** Dynamic avoidance detour tests, stationary obstacle equivalence tests, velocity smoothness tests, cycle time profiling.
- **Acceptance Criteria:** Robot successfully steers around moving obstacle while maintaining valid kinematic limits; pure Python latency overhead documented.
- **What It Must NOT Modify:** Static $A^*$ or Phase 12 safety clamps.
- **Required Checkpoint:** Dynamic DWA passing unit/integration tests; checkpoint report submitted.

### Milestone 13.10: Wait / Avoid / Stop Policy
- **Objective:** Implement the high-level behavioral arbitration finite state machine managing interaction modes with dynamic agents: **WAIT** (yield), **AVOID** (steer around), and **STOP** (halt if path completely trapped).
- **Inputs:** Dynamic obstacle tracks, predicted intersection points, path clearance, distance to goal.
- **Expected Outputs:** `DynamicBehavioralController` emitting behavioral mode and speed modifiers to DWA.
- **Files Likely Involved:** `ai/planning/dynamic/behavior_policy.py`, `tests/test_behavior_policy.py`.
- **Dependencies:** Milestones 13.8, 13.9.
- **Tests Required:** Crossing agent yield tests, narrow corridor wait tests, deadlock detection tests.
- **Acceptance Criteria:** Stable mode transitions without rapid chattering; appropriate yielding behavior when right-of-way favors crossing agent.
- **What It Must NOT Modify:** Emergency stop safety thresholds ($0.35\,\text{m}$).
- **Required Checkpoint:** Behavioral policy verified; checkpoint report submitted.

### Milestone 13.11: Event-Driven Global Replanning
- **Objective:** Implement trigger logic that invokes global $A^*$ replanning when a dynamic obstacle permanently stops on the global path or when local avoidance is infeasible.
- **Inputs:** Active global path, dynamic obstacle stationary state, persistent blocking timer.
- **Expected Outputs:** Replanning supervisor that selectively calls $A^*$ with ephemeral dynamic obstacles injected into the search graph.
- **Files Likely Involved:** `ai/planning/dynamic/replan_trigger.py`, `tests/test_replan_trigger.py`.
- **Dependencies:** Milestones 13.9, 13.10.
- **Tests Required:** Persistent blocker replan trigger tests, transient crossing non-trigger tests, replan frequency throttling tests.
- **Acceptance Criteria:** Clean global detour generated around persistent blockers; zero thrashing or continuous re-invocations on transient obstacles.
- **What It Must NOT Modify:** Underlying $A^*$ search algorithm in `ai/planning/astar.py`.
- **Required Checkpoint:** Replanning trigger unit tests passing; checkpoint report submitted.

### Milestone 13.12: ROS2 Integration
- **Objective:** Package the Phase 13 dynamic tracking and predictive planning pipeline into standardized ROS2 nodes, topics, and services.
- **Inputs:** Dynamic tracking modules, ROS2 message specifications.
- **Expected Outputs:** ROS2 node `dynamic_obstacle_tracker_node`, publisher `/planning/predicted_tracks`, subscriber `/perception/dynamic_obstacles`, and RViz visualization marker publishers.
- **Files Likely Involved:** `ai/ros2/dynamic_tracking_node.py`, `ai/ros2/dynamic_planner_node.py`, `tests/test_ros2_dynamic_nodes.py`.
- **Dependencies:** Milestones 13.5, 13.6, 13.9.
- **Tests Required:** Node lifecycle tests, message serialization/deserialization tests, topic throughput tests.
- **Acceptance Criteria:** Clean node execution, compliant ROS2 topic typing, proper lifecycle state management.
- **What It Must NOT Modify:** Existing Phase 10 ROS2 node contracts.
- **Required Checkpoint:** ROS2 integration verified; checkpoint report submitted.

### Milestone 13.13: SIL Dynamic Scenarios (Revalidated)
- **Objective:** Software-In-The-Loop (SIL) simulation benchmark scenarios featuring moving and static obstacles across canonical geometries, revalidated under canonical termination semantics.
- **Inputs:** SIL simulation environment (`simulation/scenarios/dynamic/`, `simulation/runners/dynamic_sil_runner.py`).
- **Expected Outputs:** Authoritative scenario definitions, runners, and evaluator with independent canonical fields (`mission_outcome`, `safety_outcome`, `termination_reason`, `local_planner_cycles`, `global_replan_count`), zero collisions, full-horizon goal arrival validation (Scenario A), and legacy evidence archive in `simulation/outputs/dynamic_sil/legacy/`.
- **Files Involved:** `simulation/scenarios/dynamic/`, `simulation/runners/dynamic_sil_runner.py`, `simulation/metrics/evaluator.py`, `docs/PHASE_13_13_SIL_RESULTS.md`.
- **Dependencies:** Milestones 13.9, 13.10, 13.11, 13.12.
- **Tests Required:** Automated execution of canonical scenarios (A, G1, G2, N3, O) with full termination and safety contract validation; zero collisions; regression suite across 77 dynamic SIL tests.
- **Acceptance Criteria:** Scenarios execute deterministically; ground truth recorded strictly for evaluation; canonical outcomes (`GOAL_REACHED`, `INCOMPLETE`, `SAFE`, `COLLISION`) truthfully reflect measured telemetry; zero production algorithm diff in `ai/planning/dynamic/*`.
- **What It Must NOT Modify:** Production algorithms under `ai/planning/dynamic/*`.
- **Required Checkpoint:** Dynamic scenarios revalidation complete; checkpoint sealed.

### Milestone 13.14: Multi-Obstacle + Failure Testing
- **Objective:** Stress-test the dynamic pipeline under challenging conditions: multiple concurrent obstacles, temporary detection dropouts, false positives, and sudden localization degradation.
- **Inputs:** Multi-obstacle scenarios, synthetic noise and dropout injectors.
- **Expected Outputs:** Robustness test suite and failure resilience report.
- **Files Likely Involved:** `simulation/sil/stress_scenarios.py`, `tests/test_dynamic_failure_modes.py`.
- **Dependencies:** Milestones 13.12, 13.13.
- **Tests Required:** 3+ concurrent obstacles, 5-frame detection blackout, sudden stop, localization `LOST` event injection.
- **Acceptance Criteria:** Zero unhandled exceptions; immediate emergency stop on localization `LOST`; robust recovery from detection blackout.
- **What It Must NOT Modify:** Phase 12 safety supervisor logic.
- **Required Checkpoint:** Multi-obstacle failure test suite passing; checkpoint report submitted.

### Milestone 13.15: MATLAB Validation
- **Objective:** Export tracking, prediction, and trajectory telemetry to structured formats (`.mat` / `.csv` / `.json`) and implement MATLAB validation scripts for offline analysis.
- **Inputs:** Closed-loop SIL benchmark logs from dynamic missions.
- **Expected Outputs:** MATLAB export pipeline and validation scripts plotting GT vs. estimated tracks, velocity errors, prediction bounds, and clearance curves.
- **Files Likely Involved:** `scripts/export_dynamic_matlab.py`, `matlab/validate_dynamic_tracking.m`, `matlab/plot_predictions.m`.
- **Dependencies:** Milestone 13.13.
- **Tests Required:** Export format verification, MATLAB script execution test (or headless parser validation).
- **Acceptance Criteria:** Complete telemetry exported; metrics match Python evaluator outputs within numerical precision.
- **What It Must NOT Modify:** Runtime planning or tracking modules.
- **Required Checkpoint:** Telemetry export verified; checkpoint report submitted.

### Milestone 13.16: Full Closed-Loop Benchmark
- **Objective:** Execute the full formal SIL benchmark across all 9 canonical test categories and record comprehensive navigation and tracking metrics.
- **Inputs:** Complete dynamic planning stack, SIL runner, full scenario suite.
- **Expected Outputs:** Formal benchmark report with quantitative tables across all scenarios (success rate, clearances, latencies, RMSEs).
- **Files Likely Involved:** `scripts/evaluate_dynamic_planning.py`, `outputs/dynamic_benchmarks/`.
- **Dependencies:** Milestones 13.13, 13.14, 13.15.
- **Tests Required:** Execution of all 9 benchmark scenario suites.
- **Acceptance Criteria:** Complete quantitative evaluation with zero unhandled crashes; explicit breakdown of compute latencies.
- **What It Must NOT Modify:** Algorithm implementations (benchmark evaluation only).
- **Required Checkpoint:** Full benchmark data collected; checkpoint report submitted.

### Milestone 13.17: Visualization + Documentation
- **Objective:** Generate visual inspection artifacts (trajectory animations, RViz marker configs, multi-panel debug plots) and compile comprehensive architectural documentation.
- **Inputs:** Benchmark logs, telemetry data, architecture diagrams.
- **Expected Outputs:** Multi-panel diagnostic plots (Trajectory vs Time, Velocity vs Time, Clearance vs Time, Prediction Fan), updated user documentation, and video/GIF recordings.
- **Files Likely Involved:** `scripts/visualize_dynamic_tracks.py`, `docs/DYNAMIC_OBSTACLES.md`, `outputs/dynamic_plots/`.
- **Dependencies:** Milestone 13.16.
- **Tests Required:** Plot generation script validation; image output verification.
- **Acceptance Criteria:** Clear visual evidence demonstrating dynamic obstacle avoidance without collisions or erratic motions.
- **What It Must NOT Modify:** Source code under `ai/`.
- **Required Checkpoint:** Documentation and visual assets verified; checkpoint report submitted.

### Milestone 13.18: Full Regression
- **Objective:** Execute full backward regression testing against all Phase 0 through Phase 12 test suites to prove that zero existing functionality has degraded.
- **Inputs:** Complete test repository (`pytest tests/`).
- **Expected Outputs:** Comprehensive regression test log proving static navigation, traversability, hazard detection, costmaps, and safety are intact.
- **Files Likely Involved:** All test directories.
- **Dependencies:** Milestones 13.0 through 13.17.
- **Tests Required:** Full project `pytest` suite execution; static SIL benchmark re-verification (`scripts/evaluate_path_planning.py`).
- **Acceptance Criteria:** 100% pass on all existing Phase 12 tests; static benchmark scores identical to baseline `7555164`.
- **What It Must NOT Modify:** Any core codebase logic.
- **Required Checkpoint:** Full regression green; checkpoint report submitted.

### Milestone 13.19: Strict Engineering Audit
- **Objective:** Conduct a rigorous final engineering audit verifying data contract adherence, safety isolation, zero GT leakage, test coverage, and documentation completeness.
- **Inputs:** Entire Phase 13 codebase, documentation, logs, and artifacts.
- **Expected Outputs:** Final Phase 13 Audit Report, sign-off on Phase 13 deliverables, and preparation of Phase 14 handoff package.
- **Files Likely Involved:** `docs/PHASE_13_MASTER_CONTEXT.md`, `docs/PHASE_13_AUDIT.md`.
- **Dependencies:** Milestones 13.0 through 13.18.
- **Tests Required:** Codebase static analysis, dependency graph inspection, ground-truth leakage scan.
- **Acceptance Criteria:** Zero violations of safety hierarchy or ground-truth isolation; all acceptance criteria fulfilled.
- **What It Must NOT Modify:** Any implementation code.
- **Required Checkpoint:** Final audit approval; Phase 13 sealed.

---

## 11. Milestone Dependencies Matrix

The following table defines the prerequisite dependency graph across Phase 13 milestones:

| Milestone | Title | Immediate Prerequisites | Core Deliverable |
|---|---|---|---|
| **13.0** | Architecture Audit | Phase 12 Baseline (`7555164`) | Audit Report & Master Context |
| **13.1** | Dynamic Obstacle Data Model | 13.0 | `ai/planning/dynamic/data_model.py` |
| **13.2** | Detection Interface | 13.1 | `ai/planning/dynamic/detector_interface.py` |
| **13.3** | Multi-Frame Tracking | 13.1, 13.2 | `ai/planning/dynamic/tracker.py` |
| **13.4** | Motion Estimation | 13.3 | `ai/planning/dynamic/motion_estimator.py` |
| **13.5** | Filtering / Kalman Tracking | 13.3, 13.4 | `ai/planning/dynamic/kalman_filter.py` |
| **13.6** | Motion Prediction | 13.5 | `ai/planning/dynamic/predictor.py` |
| **13.7** | Dynamic Occupancy Layer | 13.1, 13.6 | `ai/planning/dynamic/occupancy_layer.py` |
| **13.8** | Predictive Collision Checking | 13.6, 13.7 | `ai/planning/dynamic/predictive_collision.py` |
| **13.9** | Dynamic DWA Integration | 13.7, 13.8 | `ai/planning/dynamic_dwa.py` |
| **13.10**| Wait / Avoid / Stop Policy | 13.8, 13.9 | `ai/planning/dynamic/behavior_policy.py` |
| **13.11**| Event-Driven Global Replanning| 13.9, 13.10 | `ai/planning/dynamic/replan_trigger.py` |
| **13.12**| ROS2 Integration | 13.5, 13.6, 13.9 | `ai/ros2/dynamic_tracking_node.py` |
| **13.13**| SIL Dynamic Scenarios | 13.9, 13.10, 13.11 | `simulation/sil/dynamic_scenarios.py` |
| **13.14**| Multi-Obstacle + Failure Testing| 13.12, 13.13 | `tests/test_dynamic_failure_modes.py` |
| **13.15**| MATLAB Validation | 13.13 | `scripts/export_dynamic_matlab.py` |
| **13.16**| Full Closed-Loop Benchmark | 13.13, 13.14, 13.15 | `scripts/evaluate_dynamic_planning.py` |
| **13.17**| Visualization + Documentation | 13.16 | Diagnostic Plots & User Docs |
| **13.18**| Full Regression | 13.0–13.17 | 100% Pytest Pass across repo |
| **13.19**| Strict Engineering Audit | 13.0–13.18 | Final Sign-Off & Phase 14 Handoff |

---

## 12. Data Contracts

To guarantee modularity, clear data contracts must be maintained across pipeline boundaries:

### 12.1. `DynamicObstacleDetection`
```python
@dataclass
class DynamicObstacleDetection:
    detection_id: str
    timestamp: float          # ROS / SIL timestamp in seconds
    frame_id: str             # "odom" or "base_link"
    bbox_center: Tuple[float, float]   # (x, y) in meters
    bbox_size: Tuple[float, float]     # (length, width) in meters
    bbox_yaw: float           # heading angle in radians [-pi, pi]
    confidence: float         # detection confidence [0.0, 1.0]
    raw_cluster_points: Optional[np.ndarray] = None # Optional 2D points
```

### 12.2. `TrackedObstacleState`
```python
@dataclass
class TrackedObstacleState:
    track_id: int
    timestamp: float
    frame_id: str             # Strictly "odom" for tracking
    x: float                  # Estimated position x (m)
    y: float                  # Estimated position y (m)
    vx: float                 # Estimated velocity vx (m/s)
    vy: float                 # Estimated velocity vy (m/s)
    covariance: np.ndarray    # 4x4 covariance matrix
    bbox_size: Tuple[float, float]
    bbox_yaw: float
    track_age: int            # Total frames tracked
    coast_cycles: int         # Consecutive frames missed
    is_confirmed: bool        # Track status (provisional vs confirmed)
```

### 12.3. `PredictedTrajectory`
```python
@dataclass
class TrajectoryWaypoint:
    time_offset: float        # delta t from current time (s)
    timestamp: float          # absolute future timestamp (s)
    x: float                  # predicted x (m)
    y: float                  # predicted y (m)
    vx: float                 # predicted vx (m/s)
    vy: float                 # predicted vy (m/s)
    covariance_2d: np.ndarray # 2x2 position covariance matrix
    uncertainty_radius: float # 3-sigma semi-major axis (m)

@dataclass
class PredictedObstacleTrajectory:
    track_id: int
    current_state: TrackedObstacleState
    horizon_s: float
    dt: float
    waypoints: List[TrajectoryWaypoint]
    bounding_polygon_per_step: List[np.ndarray] # Polygon vertices at each dt
```

### 12.4. `DynamicOccupancyQuery` Interface
- `is_occupied(x: float, y: float, t: float) -> bool`
- `check_trajectory_collision(robot_traj: List[Tuple[float, float, float, float]], robot_footprint: Tuple[float, float]) -> Tuple[bool, float, float]`
  Returns `(has_collision, time_of_collision, min_clearance_m)`.

---

## 13. Coordinate & Frame Conventions

Phase 13 strictly preserves the coordinate system conventions established in Phase 9 and Phase 12:

```
World / Odometry Frame ("odom"):
           +X (Forward / East)
             ^
             |
             |
   +Y <------+ (Origin: Startup / Map Origin, Z = 0)
 (Left / North)
   +Z: Up (Right-Handed System)
   Grid Raster Resolution: 0.10 m/cell

Robot / Body Frame ("base_link"):
           +X (Forward)
             ^
             |
             |
   +Y <------+ (Origin: Center of Rear Axle / Center of Differential Drive Base)
   (Left)
   +Z: Up
```

### Frame Transformations:
- All dynamic obstacle detections received in `base_link` MUST be transformed into `odom` prior to multi-frame tracking:
  $$\begin{bmatrix} x_{\text{odom}} \\ y_{\text{odom}} \end{bmatrix} = \mathbf{R}(\theta_{\text{robot}}) \begin{bmatrix} x_{\text{base}} \\ y_{\text{base}} \end{bmatrix} + \begin{bmatrix} x_{\text{robot}} \\ y_{\text{robot}} \end{bmatrix}$$
- All tracking, state filtering, and forward predictions are maintained strictly in the world (`odom`) frame to eliminate robot egomotion coupling.
- When evaluating DWA forward rollouts generated in the robot's local search frame, trajectories are transformed into `odom` for spatio-temporal collision checking against predicted tracks.

---

## 14. Existing ROS2 Conventions

Phase 13 aligns with Phase 10 ROS2 architecture standards:
- **Middleware:** ROS2 Humble / RCLPY abstractions.
- **Node Naming:** Lowercase snake_case with clear subsystem prefixes (e.g., `dynamic_obstacle_tracker_node`, `predictive_planner_node`).
- **Standard Topics:**
  - Subscriptions:
    - `/localization/odometry` (`nav_msgs/msg/Odometry`)
    - `/perception/dynamic_detections` (`custom_msgs/DynamicObstacleArray`)
    - `/costmap/terrain_costmap` (`nav_msgs/msg/OccupancyGrid`)
  - Publications:
    - `/planning/tracked_obstacles` (`custom_msgs/TrackedObstacleArray`)
    - `/planning/predicted_trajectories` (`custom_msgs/PredictedTrajectoryArray`)
    - `/planning/dynamic_costmap` (`nav_msgs/msg/OccupancyGrid`)
    - `/cmd_vel` (`geometry_msgs/msg/Twist`)
    - `/planning/visualization_markers` (`visualization_msgs/msg/MarkerArray`)
- **Execution Model:** Deterministic single-threaded execution inside the SIL runner; asynchronous executor compatibility for native ROS2 launch.

---

## 15. Existing Simulation Conventions

All SIL simulation benchmarks follow the deterministic conventions defined in `simulation/sil/`:
- **Fixed Timestep:** $\Delta t = 0.05\,\text{s}$ ($20\,\text{Hz}$ simulation rate).
- **Vehicle Kinematics:** 2D differential-drive model with linear acceleration limits ($a_{\max} = 1.0\,\text{m/s}^2$) and angular acceleration limits ($\alpha_{\max} = 2.0\,\text{rad/s}^2$).
- **Wheel Slip / Terrain Friction:** Configurable friction coefficient $\mu \in [0.4, 0.9]$.
- **Deterministic Seeding:** Random seed fixed across benchmark runs for repeatable noise injection.
- **State Logging:** Full cycle telemetry logged at each simulation step to JSON/CSV for post-run evaluation.

---

## 16. Existing Planner Interfaces

The dynamic planner extends the abstract base classes defined in `ai/planning/base.py`:
- `BasePlanner`: Interface requiring `plan(start, goal, costmap) -> PathResult`.
- `BaseLocalPlanner`: Interface requiring `compute_velocity_commands(current_pose, current_vel, global_path, costmap) -> VelocityCommand`.
- **DWA Trajectory Evaluation:**
  Trajectory cost function augmented with dynamic penalty:
  $$G(v, \omega) = \alpha \cdot \text{heading}(v, \omega) + \beta \cdot \text{dist}_{\text{static}}(v, \omega) + \gamma \cdot \text{vel}(v, \omega) + \delta \cdot \text{dist}_{\text{dynamic}}(v, \omega)$$
  where $\text{dist}_{\text{dynamic}}(v, \omega) = 0$ if a collision occurs within prediction horizon $H$.

---

## 17. Required Metrics

The final Phase 13 evaluation must quantitatively report the following metrics:

### 17.1. Tracking Metrics:
- **Position RMSE ($m$):** Root mean squared error between estimated track position $(\hat{x}, \hat{y})$ and ground truth $(x_{\text{gt}}, y_{\text{gt}})$.
- **Velocity RMSE ($m/s$):** Root mean squared error of estimated obstacle velocity $(\hat{v}_x, \hat{v}_y)$.
- **ID Switches ($N_{\text{ids}}$):** Total number of identity swaps during multi-target tracking.
- **Track Availability ($\%$):** Percentage of simulation steps where a visible dynamic obstacle is actively tracked.
- **Track Latency / Age:** Time from obstacle entering sensor range to track confirmation.

### 17.2. Prediction Metrics:
- **Displacement Error @ Horizon (ADE/FDE):** Average and Final Displacement Errors at $t = 1.0\,\text{s}, 2.0\,\text{s}, 3.0\,\text{s}$.
- **Covariance Consistency:** Normalized Innovation Squared (NIS) verifying that predicted covariances correctly capture empirical errors.

### 17.3. Navigation Metrics:
- **Mission Success Rate ($\%$):** Percentage of scenarios reaching goal within $0.20\,\text{m}$ tolerance without collision.
- **Dynamic Collision Count:** Total number of physical collisions with dynamic obstacles.
- **Static Collision Count:** Total number of physical collisions with terrain obstacles.
- **Emergency Stop Count:** Total activations of the Phase 12 safety emergency stop.
- **Terminal Goal Error ($m$):** Ground-truth distance between robot final pose and target goal.
- **Minimum Dynamic Clearance ($m$):** Closest distance between UGV footprint and dynamic obstacle over entire run.
- **Waiting Duration ($s$):** Cumulative time spent in stationary yield/wait states.
- **Global Replanning Count:** Number of $A^*$ replanning invocations triggered.

### 17.4. Performance & Latency Metrics:
- **Tracker Latency ($ms$):** Time to associate and filter tracks per cycle.
- **Predictor Latency ($ms$):** Time to compute multi-horizon predictions per cycle.
- **Occupancy Layer Update Latency ($ms$):** Time to rasterize/update dynamic occupancy.
- **Predictive Collision Checking Latency ($ms$):** Time spent checking candidate trajectories.
- **Planner Cycle Latency ($s$):** Full DWA cycle time.
- **End-to-End Latency ($s$):** Total loop execution duration.

---

## 18. Required Tests

Every milestone must provide comprehensive automated tests under `tests/`:
1. **Unit Tests:**
   - Data structure validation, transformations, bounding box intersections.
   - Hungarian matching correctness and corner cases.
   - Kalman filter numerical stability and convergence.
   - Prediction hull geometric correctness.
2. **Integration Tests:**
   - Tracker-to-predictor pipeline flow.
   - Dynamic DWA cost evaluation against synthetic moving hazards.
   - Safety clamp preemption of DWA dynamic commands.
3. **Scenario Tests (9 Canonical Categories):**
   - Category 1: No dynamic obstacles (static regression equivalence).
   - Category 2: Orthogonal crossing obstacle (yield/avoidance test).
   - Category 3: Head-on approaching obstacle (decelerate/evade test).
   - Category 4: Receding obstacle (smooth follow/pass test).
   - Category 5: Persistent blocker (stops in corridor $\to$ event-driven $A^*$ replan).
   - Category 6: Multiple concurrent dynamic obstacles ($\ge 2$ interacting agents).
   - Category 7: Temporary detection dropout (coasting through $3\text{--}5$ frame blackout).
   - Category 8: No safe trajectory (narrow corridor fully trapped $\to$ controlled stop).
   - Category 9: Localization `LOST` injection (instantaneous emergency stop).
4. **Regression Tests:**
   - Complete Phase 0–12 test suite pass.

---

## 19. Required Visualizations

The evaluation suite must generate high-resolution visual artifacts for engineering validation:
1. **Dynamic Avoidance Trajectory Plot:** UGV trajectory, obstacle ground truth paths, and estimated tracks in the 2D plane.
2. **Spatiotemporal Clearance vs. Time:** Real-time distance between UGV footprint and nearest dynamic obstacle plotted against safety thresholds ($0.35\,\text{m}$, $0.60\,\text{m}$).
3. **Velocity Profile vs. Time:** Commanded $v, \omega$ alongside obstacle velocities and behavioral state mode overlays.
4. **Prediction Fan Visualization:** Forward prediction waypoints and expanding $3\sigma$ covariance ellipses at multiple snapshots.
5. **State Machine Progression:** Timeline plot displaying transitions between `CRUISING`, `WAITING`, `AVOIDING`, `REPLANNING`, and `SAFETY_STOP`.

---

## 20. Required MATLAB Outputs

MATLAB is designated strictly as an offline validation and analysis environment. The Python evaluation runner will export structured telemetry (`outputs/dynamic_matlab/`):
- `gt_obstacle_trajectories`: Matrix of $[t, x, y, v_x, v_y]$ for all simulated agents.
- `estimated_tracks`: Matrix of $[t, \text{track\_id}, \hat{x}, \hat{y}, \hat{v}_x, \hat{v}_y, \sigma_x, \sigma_y]$.
- `predicted_trajectories`: Multi-dimensional tensor of $[t, \text{track\_id}, t_{\text{offset}}, x_p, y_p, \sigma_p]$.
- `ugv_trajectory`: Matrix of $[t, x_{\text{robot}}, y_{\text{robot}}, \theta_{\text{robot}}, v_{\text{cmd}}, \omega_{\text{cmd}}]$.
- `clearance_telemetry`: Vector of minimum dynamic clearance per timestep.
- `event_logs`: Timestamps of behavioral mode shifts, replans, and safety clamp activations.

Companion MATLAB scripts (`matlab/validate_dynamic_tracking.m`) will verify error distributions, generate publication-quality figures, and confirm consistency.

---

## 21. Acceptance Criteria

Phase 13 will be considered complete and validated when:
1. All 20 milestones (13.0 through 13.19) are formally completed and accepted.
2. 100% test pass rate across all new dynamic tracking/prediction unit and integration tests.
3. 100% test pass rate across all Phase 0 through Phase 12 regression tests.
4. Zero collisions with dynamic obstacles across all valid canonical scenarios.
5. Zero unhandled exceptions or deadlocks in multi-obstacle and failure injection tests.
6. The Phase 12 safety supervisor remains unbroken and successfully halts the robot on localization failure or sudden near-field hazards.
7. Terminal goal error on static missions remains within Phase 12 baseline bounds ($0.20\,\text{m}$ nominal tolerance, Rock detour within known $0.226\,\text{m}$ baseline).
8. All required MATLAB, visualization, and benchmark telemetry artifacts are generated and archived.

---

## 22. Known Risks

1. **DWA Latency Compounding:** Evaluating multiple forward trajectories against expanding dynamic obstacle prediction hulls may increase DWA cycle times beyond the Phase 12 baseline ($1.36\text{--}2.15\,\text{s}$), making reactive replanning sluggish.
2. **Track Identity Swapping:** Close proximity crossings of multiple obstacles may induce ID switches if measurement covariances overlap significantly.
3. **Over-Conservative Yield Deadlocks:** In narrow passages, overly cautious yield/wait policies may cause the robot to freeze indefinitely if an obstacle hovers nearby.
4. **Localization Drift Interaction:** Lateral VO drift ($~5.5\,\text{cm}$) observed in Phase 12 could slightly bias world-frame obstacle track positions relative to ground-truth obstacles.

---

## 23. Known Limitations

- Dynamic obstacles are currently modeled as planar 2D kinematic entities with constant-velocity assumptions over short horizons ($1\text{--}3\,\text{s}$). High-maneuverability agile agents (sudden zig-zags) will exhibit higher prediction displacement errors.
- Dynamic obstacle detection in SIL benchmarks relies on clustered simulated perceptions rather than deep neural network inference running on an onboard embedded GPU.
- Execution under native Linux ROS2 Gazebo with hardware camera plugins remains unverified in this phase.

---

## 24. Phase 14 Handoff Requirements

Phase 14 (Multi-Sensor Fusion & Advanced Terrain Traversability) will directly ingest the following validated interfaces from Phase 13:
1. **Tracked Obstacle Streams:** Validated `TrackedObstacleArray` providing persistent IDs, filtered positions, velocities, and covariances.
2. **Prediction Interfaces:** Tested `MotionPredictor` and `DynamicOccupancyLayer` for integration into 3D multi-sensor cost representations.
3. **Behavioral Diagnostics:** Standardized event logging for dynamic conflict resolutions, yielding, and replanning triggers.
4. **Safety Verification Framework:** Reusable SIL stress-testing scenarios with dynamic agents and failure injection modules.
5. **Native ROS 2 & Gazebo Integration Baseline:** Validated native package build and runtime telemetry report produced per [`docs/ROS2_RUNTIME_INTEGRATION_PLAN.md`](ROS2_RUNTIME_INTEGRATION_PLAN.md).

---

## 25. Phase 13 Decision Log

This section records durable architectural and implementation decisions established during Phase 13.

| Milestone | Date | Decision | Reason | Evidence / Reference | Affected Files | Consequences |
|---|---|---|---|---|---|---|
| **13.0** | 2026-09-04 | Separate dynamic obstacles from static `WorldTerrainMap` | Burning moving obstacles into static Bayesian occupancy causes phantom trails and corrupts terrain traversability | Milestone 13.0 Architecture Audit | `ai/planning/dynamic/`, `ai/spatial_fusion/` | Dynamic obstacle layer is ephemeral; static costmap remains unpolluted |
| **13.0** | 2026-09-04 | Enforce strict Ground-Truth Isolation in runtime logic | Prevents algorithmic shortcuts; ensures perception pipeline is authentic and field-deployable | Project Governance Rule | All Phase 13 modules | GT only permitted in offline evaluation and scenario generator |
| **13.0** | 2026-09-04 | Track and predict obstacles strictly in the world (`odom`) frame | Eliminates robot egomotion coupling and simplifies multi-obstacle spatial prediction | Kinematic analysis in Audit 13.0 | `ai/planning/dynamic/tracker.py`, `predictor.py` | Requires robust sensor-to-odom transform; decouples tracking from body rotations |
| **13.1** | 2026-09-04 | Implement strict mathematical data model with SAT collision, covariance PSD validation, and lossless JSON serialization | Standardizes data contracts between detection, tracking, prediction, and dynamic occupancy layers without costmap pollution | Milestone 13.1 Acceptance Tests (25/25 passing) | `ai/planning/dynamic/data_model.py`, `tests/test_dynamic_data_model.py` | Clean decouple between tracking and static costmap; verified base_link <-> odom transformations |
| **13.2** | 2026-09-04 | Establish abstract `BaseDynamicObstacleDetector` and deterministic `SyntheticDynamicObstacleDetector` fixture | Decouples tracker and planning layers from concrete perception algorithms (YOLO, stereo, optical flow) while enforcing strict Ground-Truth Isolation in `detect(frame, timestamp_s)` | Milestone 13.2 Acceptance Tests (15/15 passing, 40 total) | `ai/planning/dynamic/detector_interface.py`, `tests/test_dynamic_detector.py` | Tracker can be built against canonical detection stream; future vision models drop in without changing tracker APIs |
| **13.3** | 2026-09-04 | Implement multi-frame tracking with 5-state lifecycle (NEW, TRACKING, DEGRADED, LOST, EXPIRED) and Hungarian association with spatial gating | Guarantees persistent obstacle IDs across frames, robust coasting across dropouts, and avoids collapsing nearby obstacles without premature Kalman filtering | Milestone 13.3 Acceptance Tests (20/20 passing, 60 total) | `ai/planning/dynamic/tracker.py`, `ai/planning/dynamic/association.py`, `tests/test_dynamic_tracker.py` | Clean tracking stream producing `TrackedObstacleArray` ready for Milestone 13.4 motion estimation |
| **13.4** | 2026-09-04 | Implement MotionEstimator utilizing windowed Ordinary Least Squares (OLS) linear regression over track history | Provides robust velocity (vx, vy, speed, heading) and empirical variance estimation under irregular timestamps and position noise without premature Kalman filtering | Milestone 13.4 Acceptance Tests (20/20 passing, 80 total) | `ai/planning/dynamic/motion_estimator.py`, `tests/test_motion_estimator.py` | Accurately derives 2D velocity vectors and uncertainty in world (odom) frame; verified ego-motion uses estimated localization pose |
| **13.5** | 2026-09-04 | Implement discrete-time Constant-Velocity Kalman Track Filter (`KalmanTrackFilter`) with Joseph stabilized covariance and Mahalanobis innovation gating | Provides authentic recursive probabilistic state estimation for dynamic obstacles [px, py, vx, vy]^T with rigorous covariance propagation, outlier rejection, dropout coasting, and seamless reacquisition without ground-truth leakage | Milestone 13.5 Acceptance Tests (24/24 passing, 104 total across Phase 13) | `ai/planning/dynamic/kalman_filter.py`, `ai/planning/dynamic/tracker.py`, `tests/test_kalman_filter.py` | Filters sensor noise (Euclidean position RMSE 0.0448m, velocity RMSE 0.1594m/s, heading error 0.1019 rad); 10-track update executes in 1.52 ms; ready for Milestone 13.6 trajectory prediction |
| **13.6** | 2026-09-04 | Implement multi-step Constant-Velocity Motion Predictor (`MotionPredictor`) with CWNA covariance propagation and 3-sigma spatial uncertainty envelopes | Consumes filtered state and covariance to project future trajectories across arbitrary horizons [0.0, 2.0] s, with rigorous error analysis, lifecycle state adaptation (DEGRADED inflation, LOST capping, EXPIRED rejection), and zero ground-truth leakage | Milestone 13.6 Acceptance Tests (22/22 passing, 126 total across Phase 13) | `ai/planning/dynamic/predictor.py`, `tests/test_motion_predictor.py`, `scripts/evaluate_motion_prediction.py` | Generates spatiotemporal waypoints and bounding polygons; single-track latency 655 us; 10-track batch 6.30 ms; 100% envelope coverage across synthetic test suite; ready for Milestone 13.7 dynamic occupancy layer |
| **13.7** | 2026-09-04 | Implement spatiotemporally indexed Dynamic Occupancy Layer (`DynamicOccupancyLayer`) with graded risk levels (FREE, LOW, MED, HIGH, LETHAL) | Converts predicted trajectories into 9 discrete time slices [0.0, 2.0]s on 0.10m world grid, preserving physical lethal footprints while representing uncertainty as decaying risk to prevent pathological map blockage | Milestone 13.7 Acceptance Tests (16/16 passing, 142 total across Phase 13) | `ai/planning/dynamic/occupancy_layer.py`, `tests/test_dynamic_occupancy.py`, `scripts/benchmark_dynamic_occupancy.py` | Point query latency 1.4 us; 4-obstacle 9-slice rasterization 15.0 ms; strict lethal space restricted to 0.08% of grid per obstacle; ready for Milestone 13.8 predictive collision checking |
| **13.8** | 2026-09-04 | Implement Predictive Collision Checker (`PredictiveCollisionChecker`) with strict safety semantics (FREE, RISK, LETHAL, UNKNOWN) and full Phase 12 UGV footprint | Evaluates future candidate trajectories against spatiotemporal dynamic layer and predicted obstacle tracks. Strict safety rule: prediction unavailable beyond valid horizon is UNKNOWN, not FREE. Preserves Phase 12 footprint (0.65m x 0.45m x 0.05m margin), exact SAT geometry, and clearance calculation without altering DWA yet | Milestone 13.8 Acceptance Tests (29/29 passing, 171 total across Phase 13) | `ai/planning/dynamic/collision_checker.py`, `tests/test_predictive_collision.py`, `scripts/benchmark_predictive_collision.py` | Standalone contract ready for Milestone 13.9 DWA integration; 20/20 scenarios + 5/5 safety proofs passed; evaluation latency 11.4 ms (1 obs) to 15.1 ms (4 obs) per 20-sample trajectory |
| **13.9** | 2026-09-04 | Integrate standalone predictive collision checker into Phase 12 DWA pipeline via continuous-time relative swept-segment broad-phase pre-filtering | Overcomes the ~15 ms per trajectory bottleneck of naive evaluation; pipeline uses static validity -> continuous relative swept-segment broad-phase filter -> exact SAT checking for relevant trajectories -> dynamic risk/clearance scoring. Strict safety semantics: physical collision strictly rejects; high/medium/low risk heavily penalizes; UNKNOWN_HORIZON is penalized/rejected and never treated as SAFE | Milestone 13.9 Correction Checkpoint (13/13 DWA dynamic tests passing, 215/215 regression suite passing, 0 false negatives across 1,000 + 5,000 adversarial pairs) | `ai/planning/dynamic/dwa_filter.py`, `ai/planning/local_planners/dwa.py`, `tests/test_dwa_dynamic.py`, `scripts/validate_broad_phase_continuous.py`, `scripts/benchmark_dwa_dynamic.py` | Full dynamic obstacle avoidance with zero static costmap pollution; broad-phase clears 100% of non-interacting and 72% of partial-crossing candidates; continuous-time swept segment mathematically guarantees 0 inter-sample tunneling; uncompiled pure Python cycle (4.7s-16.4s) is NOT real-time at 10-20 Hz; ready for Milestone 13.10 state machine |
| **13.10** | 2026-09-04 | Implement deterministic Tactical Behavioral State Machine (`BehavioralStateMachine`) with PROCEED / AVOID / WAIT / STOP states | Provides tactical decision arbitration around DWA evaluations: enables genuine zero-motion waiting for transient crossing obstacles, prevents high-frequency oscillation via dwell time and confirmation hysteresis, and emits DYNAMIC_DEADLOCK upon persistent blockage | Milestone 13.10 Test Suite (26/26 passing, 241/241 regression suite passing) & Benchmark (10 scenarios, 12.1 us mean latency) | `ai/planning/dynamic/behavioral_state.py`, `ai/planning/local_planners/dwa.py`, `tests/test_behavioral_state.py`, `scripts/benchmark_behavioral_state.py` | Seamless tactical layer; authoritative Phase 12 emergency stop preserved; isolated evaluation latency ~12 us; clean architectural handoff to Milestone 13.11 global replanning via DYNAMIC_DEADLOCK event |
| **13.11** | 2026-09-04 | Implement Event-Driven Global Replanning (`EventDrivenGlobalReplanner`) with ephemeral `DynamicPlanningMapOverlay` | Regenerates global A* route only when explicit persistent dynamic blockage (> 3.0s), lack of progress, path invalidation, or DYNAMIC_DEADLOCK renders current route unusable. Ephemeral planning overlay prevents canonical WorldTerrainMap mutation; debouncing and configurable cooldown (4.0s) prevent replan thrashing; stale path rule guarantees invalid paths are never silently executed | Milestone 13.11 Test Suite (29/29 passing: 20 operational + 9 safety proofs A-I) & Benchmark Suite (10 scenarios A-J, ~89.8 ms total replanning latency) | `ai/planning/dynamic/global_replanner.py`, `tests/test_global_replanning.py`, `scripts/benchmark_global_replanning.py` | Event-driven replanning fully verified; detection latency 0.127 ms; validation latency 0.015 ms; A* search over overlay 89.6 ms; entrance safety verified; 0 false free dynamic conversions; clean architectural boundary with M13.10 / M13.9 |
| **13.12** | 2026-09-05 | Complete ROS2 interface package split (`ugv_navigation_msgs`), bidirectional converter boundary, and end-to-end node migration | Connects Phase 13 dynamic perception, tracking, prediction, dynamic occupancy, tactical behavior, planning, and global replanning to canonical ROS2 interfaces while keeping algorithmic core 100% ROS-independent. Pure Phase 13 algorithms isolated behind boundary converters; 13 canonical .msg interfaces declared; 11 canonical topics wired; publisher ownership audited | Milestone 13.12 Test Suite (180/180 ROS2 integration passing; 239/239 Phase 13 regression passing; 42/42 Phase 12 regression passing; 9/9 end-to-end structural scenarios passing) | `ros2_ws/src/ugv_navigation_msgs/`, `ros2_ws/src/ugv_navigation/`, `tests/test_*_ros_boundary.py`, `tests/test_phase13_ros2_end_to_end.py` | Full ROSIDL boundary established; zero ROS imports in ai/planning/dynamic; native build unverified on Windows host; real-time performance limitations of pure Python DWA documented |

---

## 26. Living Status Tracking

### Proven:
- Phase 12 baseline frozen at commit `7555164` (tag: `phase-12-conditional-baseline`).
- Existing A*, DWA, footprint collision checking, and safety supervisor operate cleanly under SIL simulation.
- Goal tolerance contract is established at $0.20\,\text{m}$.
- Architectural insertion points for dynamic tracking identified in Milestone 13.0.
- Dynamic obstacle data structures (`DynamicObstacleDetection`, `TrackedObstacleState`, `TrajectoryWaypoint`, `PredictedObstacleTrajectory`, and container arrays) fully implemented and validated with 25 unit tests.
- Separating Axis Theorem (SAT) 2D convex polygon intersection and 3-sigma spatial uncertainty calculation verified against analytical geometry.
- Base_link <-> odom coordinate transformations verified with translation, rotation, and point clusters.
- Zero static costmap or Phase 12 planning files modified (clean architectural separation).
- Abstract perception interface `BaseDynamicObstacleDetector` and test fixture `SyntheticDynamicObstacleDetector` implemented and validated (15 unit tests, 40/40 combined dynamic suite passing).
- Ground-truth isolation verified: runtime `detect()` signature does not consume GT pose, GT registry, future velocity, or GT track IDs.
- Ephemeral detection ID generation decoupled from ground-truth obstacle identities.
- Compatibility with Phase 13.1 `DynamicObstacleDetection` data model confirmed across bounding geometry, serialization, and coordinate transforms.
- Multi-frame tracking layer (`MultiFrameTracker`, `HungarianAssociation`) implemented and validated (20 unit tests, 60/60 combined suite passing).
- 5-state lifecycle verified: `NEW` $\to$ `TRACKING` (after 3 confirmations), `TRACKING` $\to$ `DEGRADED` (on missed frames), `DEGRADED` $\to$ `TRACKING` (reacquisition), `LOST` $\to$ `EXPIRED` (timeout/max misses pruning).
- Track ID persistence verified across linear motion, crossing trajectories, and temporary dropouts.
- Spatial gating verified: observations outside association gate (default 1.5 m) spawn separate tracks.
- Zero ground-truth leakage verified: tracker receives no GT poses, future velocities, or simulation object IDs.
- `MotionEstimator` module implemented and validated (20 unit tests, 80/80 total suite passing).
- Explicit distinction between measured position, raw finite-difference velocity, and windowed OLS regression estimate verified.
- Stationary target thresholding accurately zeros speed and prevents heading jitter.
- Irregular timestamps ($\Delta t \in [0.03, 0.15]\text{ s}$) and duplicate timestamps ($\Delta t \le 0$) handled without numeric failure or distortion.
- Quantitative metrics on $1.0\text{ m/s}$ motion with $2\text{ cm}$ noise: Velocity RMSE $= 0.0987\text{ m/s}$, MAE $= 0.0806\text{ m/s}$, Bias $= -0.0055\text{ m/s}$, Heading error $= 0.0759\text{ rad}$ ($4.35^\circ$).
- Ego-motion audit verified: tracker and motion pipelines operate strictly in world (`odom`) frame using estimated localization poses, with zero simulator ground-truth coupling.
- Discrete-time Constant-Velocity Kalman Track Filter (`KalmanTrackFilter`) implemented and verified (24 unit tests, 104/104 total suite passing across Milestones 13.1–13.5).
- Recursive state vector $\mathbf{x} = [p_x, p_y, v_x, v_y]^\top$ and $4\times 4$ covariance $P$ strictly validated for symmetry and positive semi-definiteness via Joseph stabilized update: $P = (I - KH)P(I - KH)^\top + KRK^\top$.
- Continuous white-noise acceleration process noise $Q(\Delta t)$ discretized over variable $\Delta t$; handles irregular intervals ($\Delta t \in [0.03, 0.15]\,\text{s}$), duplicate/zero intervals ($\Delta t < 10^{-5}\,\text{s}$ treated as identity), and non-monotonic timestamp guards.
- Measurement gating via Mahalanobis innovation distance $d_M^2 = \mathbf{y}^\top S^{-1} \mathbf{y}$ against $\chi^2_2(0.99) = 9.21$ rejects extreme outliers while preserving predicted track state.
- Coasting through observation dropouts verified: filter executes predict-only steps with monotonic covariance growth ($\operatorname{Tr}(P): 0.0799 \to 0.3236$ over 5 missed cycles), tracker state degrades smoothly (`TRACKING` $\to$ `DEGRADED` $\to$ `LOST`), and reacquisition immediately contracts uncertainty ($\operatorname{Tr}(P) \to 0.0961$) with zero identity switch.
- 3-Way Estimator Benchmark on identical noisy trajectory ($\sigma_p = 0.04\,\text{m}$):
  - Position: $x\text{-RMSE} = 0.0306\,\text{m}$, $y\text{-RMSE} = 0.0327\,\text{m}$, $\text{Euclidean RMSE} = 0.0448\,\text{m}$
  - Velocity RMSE: Raw Finite Difference $= 0.6732\,\text{m/s}$, Windowed OLS $= 0.1856\,\text{m/s}$, Kalman Filter $= 0.1594\,\text{m/s}$
  - Velocity MAE: Raw Finite Difference $= 0.6241\,\text{m/s}$, Windowed OLS $= 0.1720\,\text{m/s}$, Kalman Filter $= 0.1435\,\text{m/s}$
  - Velocity Heading Error: Raw Finite Difference $= 0.4476\,\text{rad}$ ($25.6^\circ$), Windowed OLS $= 0.1336\,\text{rad}$ ($7.65^\circ$), Kalman Filter $= 0.1019\,\text{rad}$ ($5.84^\circ$)
- Filter runtime latency: single predict $= 42.15\,\mu\text{s}$, single update $= 57.14\,\mu\text{s}$, batch 10-track update $= 1.52\,\text{ms}$.
- Milestone 13.6 Motion Predictor (`MotionPredictor`, `PredictorConfig`) implemented and verified (22 unit tests A-V, 126/126 total suite passing across Milestones 13.1–13.6).
- Motion prediction model: Constant-Velocity (CV) forward extrapolation consuming filtered obstacle state:
  $$\mathbf{x}_{\text{pred}}(\Delta t) = F(\Delta t) \mathbf{x}_0 = \begin{bmatrix} p_x + v_x \Delta t \\ p_y + v_y \Delta t \\ v_x \\ v_y \end{bmatrix}$$
- Covariance propagation: Continuous White-Noise Acceleration (CWNA) matching Milestone 13.5:
  $$P_{\text{future}}(\Delta t) = F(\Delta t) P_0 F(\Delta t)^\top + Q_{\text{eff}}(\Delta t)$$
  with 2D positional uncertainty radius derived from semi-major axis:
  $$r_{\text{unc}}(\Delta t) = \sigma_{\text{scale}} \cdot \sqrt{\max(0, \lambda_{\max}(P_{\text{pos}}(\Delta t)))}$$
- Configurable prediction horizons validated: $0.0\,\text{s}, 0.25\,\text{s}, 0.50\,\text{s}, 0.75\,\text{s}, 1.00\,\text{s}, 1.25\,\text{s}, 1.50\,\text{s}, 2.00\,\text{s}$.
- Input offset sequence validation: strict rejection of negative offsets, non-finite values, duplicate timestamps, and non-monotonic offsets.
- Lifecycle state handling validated:
  - `TRACKING`: nominal prediction up to requested horizon with nominal process noise.
  - `DEGRADED`: uncertainty inflated by `degraded_noise_multiplier = 2.0`.
  - `LOST`: horizon capped at `lost_max_horizon_s = 1.0 s` and uncertainty inflated by `lost_noise_multiplier = 4.0`.
  - `EXPIRED`: strictly rejected in `predict_track`, safely excluded from `predict_tracks`.
- Multi-track prediction: 1, 2, 4, 10 tracks evaluated with zero cross-track contamination.
- Prediction accuracy metrics across horizons (aggregated across 10 heterogeneous evaluation scenarios: 3 matched linear/stationary CV, 3 model mismatch maneuvers [acceleration, braking, turning], and 4 covariance/lifecycle scenarios):
  | Horizon | Position RMSE | Mean Error | Max Error | Uncertainty Radius | Coverage | Trace(P_pos) |
  |---------|---------------|------------|-----------|--------------------|----------|--------------|
  | 0.25s   | 0.0379 m      | 0.0194 m   | 0.0999 m  | 0.8038 m           | 100.0%   | 0.1585 m^2   |
  | 0.50s   | 0.1513 m      | 0.0773 m   | 0.3982 m  | 1.1584 m           | 100.0%   | 0.3174 m^2   |
  | 0.75s   | 0.3391 m      | 0.1735 m   | 0.8910 m  | 1.7162 m           | 100.0%   | 0.6883 m^2   |
  | 1.00s   | 0.5996 m      | 0.3072 m   | 1.5718 m  | 2.4131 m           | 100.0%   | 1.3586 m^2   |
  | 1.25s   | 0.9808 m      | 0.5306 m   | 2.4313 m  | 3.0172 m           | 100.0%   | 2.0655 m^2   |
  | 1.50s   | 1.4006 m      | 0.7593 m   | 3.4583 m  | 3.8383 m           | 100.0%   | 3.3323 m^2   |
  | 2.00s   | 2.4380 m      | 1.3286 m   | 5.9577 m  | 5.6897 m           | 100.0%   | 7.3010 m^2   |

- Per-scenario breakdown at 2.0 s (1.0 s for LOST):
  | Scenario Class | Scenario Name | Track Status | Horizon | Error (m) | r_unc (m) | Covered |
  |----------------|---------------|--------------|---------|-----------|-----------|---------|
  | Matched Model (Stationary) | `A_Stationary` | TRACKING | 2.0s | 0.0000 | 5.2536 | True |
  | Matched Model (Linear CV) | `B_ConstantVelocity` | TRACKING | 2.0s | 0.0000 | 5.2536 | True |
  | Matched Model (Diagonal CV) | `C_DiagonalConstantVelocity` | TRACKING | 2.0s | 0.0000 | 5.2536 | True |
  | Model Mismatch (Acceleration) | `D_PositiveAcceleration` | TRACKING | 2.0s | 3.0000 | 5.2536 | True |
  | Model Mismatch (Braking) | `E_Deceleration` | TRACKING | 2.0s | 3.0000 | 5.2536 | True |
  | Model Mismatch (Turning) | `F_DirectionChange` | TRACKING | 2.0s | 5.9577 | 5.2536 | True |
  | Noisy Filtered Covariance | `G_NoisyFilteredState` | TRACKING | 2.0s | 0.0000 | 5.4816 | True |
  | High Initial Uncertainty | `H_HighUncertaintyState` | TRACKING | 2.0s | 0.0000 | 7.0207 | True |
  | Lifecycle Adaptation (Degraded) | `I_LowConfidenceDegraded` | DEGRADED | 2.0s | 0.0000 | 7.1833 | True |
  | Lifecycle Adaptation (Lost) | `J_LostTrack` | LOST | 1.0s | 0.0000 | 3.6290 | True |

- Uncertainty calibration & containment analysis:
  67/67 spatiotemporal test points covered (100.0%), confirming uncertainty envelopes conservatively bound target trajectories.
  > [!NOTE]
  > 100% envelope coverage in the evaluated synthetic set demonstrates containment only; it does not by itself demonstrate probabilistic calibration or planning usefulness. The large 2 s uncertainty envelope is a known design constraint for Milestone 13.7 dynamic occupancy.
- Model mismatch behavior documented: Constant-velocity prediction is a baseline model; prediction error increases for unmodelled acceleration (+1.5 m/s^2 -> 3.00m error at 2.0s), braking (-1.5 m/s^2 -> 3.00m error at 2.0s), and turning (R=5m circle -> 5.96m error at 2.0s).
- Execution latency benchmarks:
  - Single-track prediction: 655 us
  - Batch 1 track: Predictor 618 us | Kalman 125 us | Combined 698 us
  - Batch 2 tracks: Predictor 1.30 ms | Kalman 238 us | Combined 1.44 ms
  - Batch 4 tracks: Predictor 2.50 ms | Kalman 502 us | Combined 3.12 ms
  - Batch 10 tracks: Predictor 6.30 ms | Kalman 1.09 ms | Combined 7.51 ms
- Ground-truth isolation: zero GT pose, velocity, or simulation object references in predictor runtime code.
- Zero Phase 12 planner, costmap, or safety files modified.
- Milestone 13.7 Dynamic Occupancy Layer (`DynamicOccupancyLayer`, `DynamicOccupancyConfig`, `DynamicOccupancyGridSlice`, `DynamicRiskLevel`) implemented and verified (16 unit tests, 142/142 total Phase 13 tests passing).
- Time-bin discretization: 9 discrete future time slices at $[0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]\,\text{s}$ on a $0.10\,\text{m/cell}$ metric grid ($400 \times 400 = 160,000$ cells, bounds $[-20, 20]\,\text{m}$ in odom frame).
- Occupancy semantics & graded risk representation:
  - `LETHAL (1.00)`: Strictly inside physical obstacle footprint boundary ($1.5\,\text{m} \times 0.8\,\text{m} \approx 120\text{--}128$ cells, $1.28\,\text{m}^2 = 0.080\%$ of grid).
  - `HIGH_RISK ([0.70, 0.95))`: Immediate uncertainty envelope ($0\text{--}1\sigma_{\text{pos}}$, $17.76\,\text{m}^2$ at $2.0\,\text{s}$).
  - `MEDIUM_RISK ([0.40, 0.70))`: Intermediate uncertainty envelope ($1\text{--}2\sigma_{\text{pos}}$, $36.88\,\text{m}^2$ at $2.0\,\text{s}$).
  - `LOW_RISK ([0.15, 0.40))`: Outer uncertainty envelope ($2\text{--}3\sigma_{\text{pos}}$, $56.20\,\text{m}^2$ at $2.0\,\text{s}$).
  - `FREE (0.00)`: Outside obstacle influence.
- Critical Pathological Blockage Finding:
  - Naive binary inflation (treating full 3-sigma radius as lethal) would occupy $113.23\,\text{m}^2$ ($11,323$ cells = $7.08\%$ of the local grid per obstacle), causing pathological blockage (~$28.3\%$ of the entire map blocked for 4 obstacles).
  - The implemented graded dynamic occupancy restricts true lethal cells strictly to the physical vehicle footprint ($128$ cells, $0.080\%$). Uncertainty is represented as decaying continuous cost, steering planners away while preserving narrow navigable corridors.
- Multi-obstacle safety fusion: Overlapping obstacle occupancy resolves via conservative element-wise maximum (`np.maximum`) across cost, risk level, and footprint flags.
- Sub-bin querying & conservative union: Supports exact lookup, nearest-bin lookup, and conservative union ($\max(\text{slice}_0, \text{slice}_1)$) for sub-bin queries.
- Lifecycle state handling: DEGRADED tracks retain lethal nominal footprints with expanded uncertainty; LOST tracks cap occupancy rasterization at $1.0\,\text{s}$ and expire beyond; EXPIRED tracks are strictly rejected.
- Layer runtime performance:
  - Point query latency (`get_cost`): $1.4\,\mu\text{s}$
  - Slice lookup latency (`get_slice`): $428\,\text{ns}$
  - Ingestion and rasterization across all 9 time slices: 1 obstacle $= 7.41\,\text{ms}$, 2 obstacles $= 9.72\,\text{ms}$, 4 obstacles $= 15.01\,\text{ms}$, 10 obstacles $= 24.44\,\text{ms}$.
  - Memory footprint: $1.09\,\text{MB}$ per slice, $9.61\,\text{MB}$ total for all 9 time slices.
- Architectural Boundary: Milestone 13.7 is a dynamic spatial representation layer only; predictive collision checking, DWA trajectory evaluation, and wait/avoid/stop policy are deferred to Milestone 13.8.
- Milestone 13.8 Predictive Collision Checker (`PredictiveCollisionChecker`, `PredictiveCollisionConfig`, `PredictiveCollisionResult`, `CollisionStatus`, `TrajectorySample`) implemented and verified (29 unit tests, 171 total Phase 13 tests passing; 182 total across Phase 12/13 regression suite).
- Strict Safety Semantics:
  - Categorical classification: `SAFE`, `PREDICTED_COLLISION`, `PREDICTED_HIGH_RISK`, `PREDICTED_MEDIUM_RISK`, `PREDICTED_LOW_RISK`, `UNKNOWN_HORIZON`, `INVALID_INPUT`.
  - **"Prediction unavailable beyond the valid horizon is UNKNOWN, not FREE."** Queries exceeding valid prediction horizon ($t > \text{max\_horizon\_s}$) are explicitly flagged (`unknown_horizon_encountered=True`) and reported as `UNKNOWN_HORIZON`, never silently defaulting to `SAFE`.
  - LOST/expired tracks beyond their valid horizon (e.g. $1.0\,\text{s}$) do NOT fabricate future certainty; queries beyond track horizon are strictly reported as `UNKNOWN_HORIZON`.
  - EXPIRED tracks are rejected on input as `INVALID_INPUT`.
  - Uncertainty vs Physical Collision distinction: A large 3-sigma envelope alone does NOT automatically become a physical collision. Physical footprint overlap $\implies$ `collision = True` (`PREDICTED_COLLISION`); uncertainty penetration $\implies$ `collision = False` with graded risk status (`PREDICTED_HIGH_RISK` [cost $\ge 0.70$], `PREDICTED_MEDIUM_RISK` [cost $\ge 0.40$], `PREDICTED_LOW_RISK` [cost $\ge 0.15$]).
- UGV Footprint Model:
  - Reuses the established Phase 12 `FootprintCollisionChecker` model: $L = 0.65\,\text{m}$, $W = 0.45\,\text{m}$, $\text{margin} = 0.05\,\text{m}$.
  - Samples: 4 oriented bounding corners, perimeter edge samples ($\le 0.05\,\text{m}$ step), and $7 \times 5 = 35$ interior grid points.
  - Never assumes a point robot.
- Exact Geometric SAT & Clearance:
  - Continuous Separating Axis Theorem (SAT) 2D polygon intersection evaluates physical collision between UGV OBB and obstacle OBB.
  - Exact Euclidean distance calculation across 32 vertex-to-edge pairs computes positive metric clearance ($d > 0\,\text{m}$) for near misses.
- Time Alignment & Interpolation:
  - Handles arbitrary continuous trajectory sample times $t \in [0.0, H]$.
  - Dynamic layer queries support exact bin lookup, nearest bin, and conservative element-wise maximum union between enclosing bins.
- Multi-Obstacle Arbitration:
  - Evaluates all active predicted obstacles across trajectory samples.
  - Deterministic severity ranking (`PREDICTED_COLLISION` > `UNKNOWN_HORIZON` > `HIGH_RISK` > `MED_RISK` > `LOW_RISK` > `SAFE`) prevents less critical events from overwriting collisions.
  - Earliest collision timestamp and obstacle track ID are strictly preserved.
- Validation Benchmark Results (Scenarios A through E):
  | Scenario | Physical Collision | Risk Level | Unknown Horizon | Collision Time | Minimum Clearance | Status | Latency |
  |---|---|---|---|---|---|---|---|
  | **Scenario A (Clear Avoidance)** | False | `FREE` | False | None | $2.3250\,\text{m}$ | `SAFE` | $9.2\,\text{ms}$ |
  | **Scenario B (Physical Intersection)** | True | `LETHAL` | False | $0.42\,\text{s}$ | $0.0000\,\text{m}$ | `PREDICTED_COLLISION` | $14.6\,\text{ms}$ |
  | **Scenario C (Uncertainty Penetration)** | False | `HIGH_RISK` | False | None | $0.1750\,\text{m}$ | `PREDICTED_HIGH_RISK` | $10.0\,\text{ms}$ |
  | **Scenario D (Beyond Horizon)** | False | `FREE` | True | $2.08\,\text{s}$ | $5.0000\,\text{m}$ | `UNKNOWN_HORIZON` | $8.0\,\text{ms}$ |
  | **Scenario E (Multi-Obstacle Collision)** | True | `LETHAL` | False | $0.74\,\text{s}$ | $0.0000\,\text{m}$ | `PREDICTED_COLLISION` | $16.7\,\text{ms}$ |
- Performance & Scalability Latency Profiling (100 runs per condition):
  | Configuration | Mean | Median | P95 | Max |
  |---|---|---|---|---|
  | **1 traj / 1 obs (20 samples)** | $11.4\,\text{ms}$ | $10.8\,\text{ms}$ | $16.3\,\text{ms}$ | $18.2\,\text{ms}$ |
  | **1 traj / 4 obs (20 samples)** | $15.1\,\text{ms}$ | $15.4\,\text{ms}$ | $20.2\,\text{ms}$ | $21.5\,\text{ms}$ |
  | **Traj Length: 5 samples (4 obs)** | $4.8\,\text{ms}$ | $4.8\,\text{ms}$ | $6.4\,\text{ms}$ | $7.8\,\text{ms}$ |
  | **Traj Length: 10 samples (4 obs)** | $8.6\,\text{ms}$ | $9.0\,\text{ms}$ | $11.2\,\text{ms}$ | $12.9\,\text{ms}$ |
  | **Traj Length: 20 samples (4 obs)** | $12.5\,\text{ms}$ | $11.9\,\text{ms}$ | $16.9\,\text{ms}$ | $20.8\,\text{ms}$ |
  | **Traj Length: 30 samples (4 obs)** | $23.1\,\text{ms}$ | $23.2\,\text{ms}$ | $29.3\,\text{ms}$ | $31.6\,\text{ms}$ |
  | **Obstacle Scalability: 1 obs** | $12.7\,\text{ms}$ | $12.8\,\text{ms}$ | $16.8\,\text{ms}$ | $18.9\,\text{ms}$ |
  | **Obstacle Scalability: 2 obs** | $11.5\,\text{ms}$ | $10.4\,\text{ms}$ | $16.8\,\text{ms}$ | $18.3\,\text{ms}$ |
  | **Obstacle Scalability: 4 obs** | $13.3\,\text{ms}$ | $12.5\,\text{ms}$ | $17.8\,\text{ms}$ | $25.4\,\text{ms}$ |
  | **Obstacle Scalability: 8 obs** | $16.0\,\text{ms}$ | $16.2\,\text{ms}$ | $20.0\,\text{ms}$ | $22.7\,\text{ms}$ |
- Milestone 13.9 Dynamic Predictive Collision Integration into DWA (`DWAPlanner`, `DynamicBroadPhaseFilter`, `DynamicPlanningConfig`, `ObstacleSpatiotemporalHull`) implemented and verified (13 unit tests, 182 Phase 12+13 tests, 215 total tests passing across entire regression suite).
- **Hierarchical DWA Architecture:**
  $$\text{Candidate Generation (375)} \to \text{Static Phase 12 Footprint Validity} \to \text{Spatiotemporal Broad-Phase Filter} \to \text{Exact Predictive Collision Check (Relevant Only)} \to \text{Dynamic Risk/Clearance Scoring} \to \text{Command Selection}$$
- **Hierarchical Broad-Phase Design:**
  - Level 1: 2D Spatial AABB intersection test against obstacle spatiotemporal bounding box ($O(1)$ box rejection in $<0.1\,\mu\text{s}$).
  - Level 2: Spatiotemporal proximity test between trajectory points $(x_i, y_i, t_i)$ and obstacle waypoint circles $(x_{\text{obs}}(t_i), y_{\text{obs}}(t_i), r_{\text{obs}}(t_i))$. Candidate is cleared if distance strictly exceeds $r_{\text{ugv}} + r_{\text{obs}}(t)$ for all $t$ ($\sim 1\,\mu\text{s}$ per candidate).
  - Guarantees 100% conservative pruning with zero false negatives.
  - When obstacles are absent or outside the candidate corridor (Scenarios A & B), broad-phase clears 100% of candidates (0 exact checks), adding 0.0 ms overhead over Phase 12 static baseline.
- **Safety Semantics & Risk Policy:**
  - `PHYSICAL COLLISION`: Candidate trajectory is strictly rejected/invalidated.
  - `PREDICTED_HIGH_RISK`: Configurable penalty ($\text{penalty} = 0.60 + w_{\text{dyn\_risk}} \cdot \text{cost}_{\max}$) or hard rejection (`reject_high_risk=True`). Does NOT automatically become physical collision.
  - `PREDICTED_MEDIUM_RISK`: Substantial penalty ($\text{penalty} = 0.35 + w_{\text{dyn\_risk}} \cdot \text{cost}_{\max}$).
  - `PREDICTED_LOW_RISK`: Mild clearance penalty ($\text{penalty} = 0.15 + 0.5 \cdot w_{\text{dyn\_risk}} \cdot \text{cost}_{\max}$).
  - Dynamic clearance reward: $w_{\text{clear}} \cdot \min(d_{\text{clear}}, 1.50) / 1.50$.
- **UNKNOWN_HORIZON Policy:**
  - **"UNKNOWN_HORIZON is NEVER treated as SAFE."**
  - Trajectories extending past prediction horizon ($t > H$) or querying expired/lost tracks are explicitly intercepted.
  - Configurable policy: `"PENALIZE"` (default, score penalty 0.50), `"REJECT"` (discard candidate), `"ALLOW"` (explicitly diagnosed).
  - Fully observable in candidate diagnostics (`unknown_horizon = True`, `unknown_horizon_candidates > 0`).
- **Phase 12 Preservation:**
  - Static obstacle footprint collision checking, kinematic limits, goal tolerances, and emergency stop behavior completely preserved.
  - When all forward/turning trajectories are blocked dynamically, DWA immediately commands emergency stop: $(v=0.0\,\text{m/s}, \omega=0.0\,\text{rad/s})$, `status=LOCAL_NO_SAFE_TRAJECTORY`, `emergency_stop_triggered=True`.
- **Extended Dynamic Diagnostics:**
  - Reports: `dynamic_collision`, `dynamic_risk`, `unknown_horizon`, `blocking_obstacle_id`, `earliest_collision_time`, `minimum_dynamic_clearance`, `maximum_dynamic_cost`, `number_of_expensive_collision_checker_invocations`, `number_of_broad_phase_rejections`, `number_of_candidates_evaluated`, `dynamically_rejected_candidates`, `dynamically_penalized_candidates`, `unknown_horizon_candidates`.
- **Milestone 13.9 Correction Checkpoint: Continuous-Time Broad-Phase Safety Proof & Validation:**
  - **Continuous-Time Temporal Safety (Option A Proven):**
    - The broad-phase filter was audited and upgraded from discrete-sample proximity to **continuous-time relative swept-segment distance**.
    - For each trajectory interval $[t_k, t_{k+1}]$, relative displacement $\delta_k = p_{\text{ugv}}(t_k) - p_{\text{obs}}(t_k)$ and $\delta_{k+1} = p_{\text{ugv}}(t_{k+1}) - p_{\text{obs}}(t_{k+1})$ form a relative swept segment with vector $\mathbf{v} = \delta_{k+1} - \delta_k$.
    - The exact minimum Euclidean distance to the relative origin over the continuous interval is solved analytically:
      $$\alpha^* = \operatorname{clamp}\left(\frac{-\delta_k \cdot \mathbf{v}}{\|\mathbf{v}\|^2}, 0, 1\right), \quad d_{\min} = \|\delta_k + \alpha^* \mathbf{v}\|$$
    - Evaluated against conservative reach: $R_{\max} = r_{\text{ugv}} + \max(r_{\text{obs},0}, r_{\text{obs},1}) + \text{margin} + \epsilon_{\text{arc}}$, where $\epsilon_{\text{arc}} = 0.005\,\text{m}$ bounds the circular arc sagitta for UGV motion.
    - **Waypoint Coverage Integrity:** Fixed a boundary truncation defect where obstacle samples previously broke before covering `max_horizon_s`. Waypoint collection now strictly includes bounding samples up to $t_{\max}$ (`t_max = samples[-1].t`). Any temporal query beyond $t_{\max}$ returns `None`, forcing immediate exact collision checking.
  - **Temporal Safety Audit: 5 Adversarial Deterministic Corner Cases (PASS - SAFE):**
    1. *Inter-Sample Crossing ($t = 0.15\,\text{s}$ between $0.10\,\text{s}$ samples):* Swept-segment detects closest approach at $\alpha^* = 0.50$ ($d_{\min} = 0.0\,\text{m}$); cleared = False $\to$ **PASS (SAFE)**.
    2. *Crossing Between 0.25 s Occupancy Bins ($t = 0.375\,\text{s}$):* Swept segment captures inter-bin crossing; cleared = False $\to$ **PASS (SAFE)**.
    3. *Obstacle Accelerating and Turning:* Piecewise linear segment bounds curvature; cleared = False $\to$ **PASS (SAFE)**.
    4. *Heterogeneous Sample Densities (UGV $\Delta t = 0.08\,\text{s}$ vs Obstacle $\Delta t = 0.30\,\text{s}$):* Dynamic waypoint interpolation evaluates true relative vectors across mismatched intervals; cleared = False $\to$ **PASS (SAFE)**.
    5. *Closest Approach Strictly Between Samples ($t = 0.55\,\text{s}$):* Discrete points show clearance $> 0.58\,\text{m}$, but continuous closest approach drops to $0.42\,\text{m} < R_{\max}(0.48\,\text{m})$; cleared = False $\to$ **PASS (SAFE)**.
  - **1,000-Case Adversarial Randomized Validation (vs 200 Hz Continuous Ground Truth):**
    - Evaluated 1,000 randomized deterministic pairs ($v \in [0, 0.8]\,\text{m/s}$, $\omega \in [-1, 1]\,\text{rad/s}$, obstacle speeds up to $2.5\,\text{m/s}$, variable angles, random dimensions, uncertainty growth, heterogeneous $\Delta t \in [0.05, 0.25]\,\text{s}$).
    - **Broad-Phase Cleared:** 570 (57.0%)
    - **Exact Checks Invoked:** 430 (43.0%)
    - **True Positives:** 265
    - **True Negatives:** 570
    - **Conservative Over-Approximations (Safe Margin):** 165
    - **FALSE NEGATIVES: 0 (ZERO)**
    - Multi-seed confirmation across 5 additional seeds (1, 99, 123, 777, 2026; 5,000 total cases): **0 FALSE NEGATIVES**.
  - **UNKNOWN_HORIZON Policy Audit:**
    - `PENALIZE` (default): candidate is evaluated, receives `unknown_horizon_penalty = 0.50`, selectable only as fallback if all alternatives are blocked.
    - `REJECT`: candidate extending beyond prediction horizon is strictly invalidated (`is_valid = False`).
    - `ALLOW`: candidate is not penalized, but diagnostics **strictly expose `dynamic_risk = "UNKNOWN_HORIZON"` and `unknown_horizon = True`**; it can NEVER silently claim `SAFE` or `FREE`. Clearly designated as research/non-safety configuration.
    - Broad-phase unknown horizon guard: any candidate extending beyond obstacle's valid horizon is forced to exact evaluation.
- **Scenario Benchmark Results (Corrected Swept Broad-Phase Filter, Windows x86_64, Python 3.10):**
  | Scenario | Mean (ms) | P95 (ms) | Max (ms) | Exact Checks | Broad-Phase Cleared | Dyn Rejected | Selected Cmd $(v, \omega)$ |
  |---|---|---|---|---|---|---|---|
  | **Phase 12 Static Baseline** | $4778.6\,\text{ms}$ | $4895.1\,\text{ms}$ | $4923.1\,\text{ms}$ | $0$ | $0$ | $0$ | $(0.45, 0.00)$ |
  | **Scenario A: No Dynamic Obstacles** | $4712.2\,\text{ms}$ | $4830.1\,\text{ms}$ | $4833.9\,\text{ms}$ | $0$ | $375$ | $0$ | $(0.45, 0.00)$ |
  | **Scenario B: Moving Outside Corridor** | $4786.3\,\text{ms}$ | $4947.2\,\text{ms}$ | $5012.5\,\text{ms}$ | $0$ | $375$ | $0$ | $(0.45, 0.00)$ |
  | **Scenario C: Crossing Selected Trajectory** | $13497.1\,\text{ms}$ | $13801.0\,\text{ms}$ | $13842.6\,\text{ms}$ | $375$ | $0$ | $322$ | $(0.15, -0.10)$ |
  | **Scenario D: Crossing Subset of Trajectories** | $7154.7\,\text{ms}$ | $7390.3\,\text{ms}$ | $7392.0\,\text{ms}$ | $105$ | $270$ | $0$ | $(0.36, 0.00)$ |
  | **Scenario E: Four Moving Obstacles** | $16355.3\,\text{ms}$ | $16994.2\,\text{ms}$ | $17020.8\,\text{ms}$ | $340$ | $35$ | $5$ | $(0.15, 0.00)$ |
  | **Scenario F: Uncertain Obstacle (Risk Penalty)**| $12831.1\,\text{ms}$ | $13258.0\,\text{ms}$ | $13306.2\,\text{ms}$ | $375$ | $0$ | $0$ | $(0.21, -0.25)$ |
  | **Scenario G: Prediction Horizon Exceeded** | $9326.4\,\text{ms}$ | $11969.6\,\text{ms}$ | $11971.3\,\text{ms}$ | $375$ | $0$ | $140$ | $(0.36, 0.00)$ |
  | **Scenario H: Static + Dynamic Combination** | $9823.7\,\text{ms}$ | $10300.1\,\text{ms}$ | $10366.0\,\text{ms}$ | $260$ | $0$ | $98$ | $(0.17, 0.00)$ |
- **Real-Time Performance Verdict & Scientific Finding:**
  - **The current uncompiled pure Python implementation is NOT real-time at 10 Hz or 20 Hz.**
  - Phase 12 static DWA baseline alone requires $\sim 4.78\,\text{s}$ per cycle in Python due to forward kinematic integration of 375 candidates $\times$ 15 steps $\times$ 80 footprint polygon sample points ($450,000$ lookups/cycle).
  - Continuous swept broad-phase filtering effectively eliminates $100\%$ of exact checks in non-interacting scenes (Scenarios A & B) and $72\%$ of exact checks in subset-crossing scenes (Scenario D: 270 cleared / 105 checked).
  - When obstacles directly cross candidate corridors (Scenarios C, E, F), safety demands exact predictive checks, resulting in cycle times of $7.1\text{--}16.4\,\text{s}$ ($0.06\text{--}0.14\,\text{Hz}$).
  - Broad-phase integration reduces unnecessary workload, but does NOT solve the underlying pure Python computational throughput bottleneck. Hardware real-time deployment ($10\text{--}20\,\text{Hz}$) strictly requires compilation (C++/Cython/Numba), GPU/tensor parallelization, or candidate pruning (e.g. evaluating 25 trajectories instead of 375).
- **13 Verified Safety Validation Proofs (13/13 Passing):**
  1. Physically colliding candidates are strictly rejected (`test_1_physically_colliding_candidate_rejected`).
  2. Safe candidates remain selectable without penalty (`test_2_safe_candidate_remains_selectable`).
  3. Near-miss candidates receive intended risk penalties (`test_3_near_miss_candidate_receives_risk_penalty`).
  4. HIGH_RISK does not automatically become physical collision (`test_4_high_risk_does_not_automatically_become_collision`).
  5. UNKNOWN_HORIZON is never treated as SAFE (`test_5_unknown_horizon_is_never_treated_as_safe`).
  6. Multiple dynamic obstacles are evaluated simultaneously and conservatively (`test_6_multiple_dynamic_obstacles_handled_correctly`).
  7. Static Phase 12 obstacles still reject candidates before dynamic phase (`test_7_static_phase_12_obstacles_still_reject_candidates`).
  8. Dynamic and static obstacles compose conservatively (`test_8_dynamic_and_static_obstacles_compose_conservatively`).
  9. Deterministic repeated runs produce bitwise identical decisions and scores (`test_9_deterministic_repeated_runs`).
  10. Emergency stop triggers immediately when all trajectories are blocked (`test_10_existing_emergency_stop_behavior`).
  11. Dynamic obstacles behind the UGV do not cause false forward rejection (`test_11_obstacles_behind_ugv_do_not_cause_false_forward_rejection`).
  12. Crossing dynamic obstacles are detected at the correct future timestamp (`test_12_crossing_obstacle_detected_at_correct_time`).
  13. Continuous relative swept-segment filter eliminates inter-sample tunneling (`test_13_continuous_relative_swept_segment_eliminates_tunneling`).
- **Architectural Boundary & Responsibility Split:**
  - **Milestone 13.9 Scope:** Dynamic obstacle-aware DWA candidate scoring, rejection, and conservative continuous-time broad-phase pre-filtering only.
  - **Milestone 13.10 Responsibility:** Behavioral State Machine (`WAIT`, `AVOID`, `STOP`, `PROCEED`) to handle persistent blockages, dynamic yield/wait-in-place semantics, deadlock resolution, and oscillation suppression.
  - **Milestone 13.11 Responsibility:** Dynamic obstacle feedback into global replanning / A* path updates when local corridors remain blocked over extended horizons.
- **Production Readiness Statement:**
  - Milestone 13.9 is **NOT production ready for real-time 10-20 Hz deployment** in pure Python.
  - Milestone 13.9 is **APPROVED** as mathematically, algorithmically, and behaviorally safe: continuous-time relative swept-segment checking proves zero inter-sample tunneling, 0 false negatives across 6,005 adversarial pairs (5 deterministic corner cases + 1,000 seeded pairs + 5,000 multi-seed pairs), and 215/215 regression suite passing. Note: Broad-phase clearance is mathematically conservative under the modeled piecewise-linear trajectory and interpolation assumptions; it does not claim physical-world sensor noise or non-holonomic wheel slip proof.
- **Milestone 13.10: Tactical Behavioral State Machine (WAIT / AVOID / STOP / PROCEED):**
  - **Core States:**
    - `PROCEED`: Nominal forward travel along the global path when the preferred route is collision-free with acceptable dynamic risk (FREE or LOW).
    - `AVOID`: Executing an active collision-free local detour when the preferred route is obstructed or penalized by dynamic obstacles.
    - `WAIT`: Genuine zero-motion holding state ($v=0.0\,\text{m/s}, \omega=0.0\,\text{rad/s}$) commanded when an obstacle is transiently crossing and predicted to clear the corridor within a bounded horizon ($T_{\text{clear}} \le 1.50\,\text{s}$), or when waiting is safer than a risky near-miss detour.
    - `STOP`: Genuine zero-motion safety intervention ($v=0.0\,\text{m/s}, \omega=0.0\,\text{rad/s}$) when no safe trajectory exists, physical collision is imminent, wait has timed out, or safety policies demand halt.
  - **Deterministic State Transition Rules (11 Transitions Covered):**
    | Transition | Trigger Condition | Rationale / Mechanism |
    |---|---|---|
    | `PROCEED -> AVOID` | Preferred trajectory dynamically blocked/penalized, safe alternative detour exists | Deviates locally while maintaining progress |
    | `PROCEED -> WAIT` | Preferred trajectory blocked, obstacle predicted to clear corridor within horizon | Waiting in place is safer than forcing tight avoidance |
    | `PROCEED -> STOP` | No safe trajectory exists or physical collision imminent | Authoritative safety intervention |
    | `AVOID -> PROCEED` | Preferred straight path revalidated as collision-free | Returns to nominal path following |
    | `AVOID -> WAIT` | Detour blocked, but obstacle predicted to clear | Holds position until corridor opens |
    | `AVOID -> STOP` | All detour options blocked, obstacle not clearing | Immediate safety stop |
    | `WAIT -> PROCEED` | Obstacle exits corridor, confirmed over recovery cycles | Resumes nominal travel |
    | `WAIT -> AVOID` | Obstacle still blocking preferred path, safe alternative detour opens | Takes detour if obstacle stalls |
    | `WAIT -> STOP` | Elapsed wait time exceeds `wait_timeout_s` ($2.5\text{--}3.0\,\text{s}$) or emergency condition | Prevents infinite waiting in front of stalled obstacle |
    | `STOP -> PROCEED` | Safe preferred path confirmed over `recovery_confirmation_cycles` | Safe re-engagement |
    | `STOP -> AVOID` | Valid safe detour explicitly validated over recovery cycles | Safe maneuvering out of stop |
  - **Hysteresis, Dwell Time, and Churn Prevention:**
    - `min_dwell_time_s = 0.20\,\text{s}`: State cannot transition between non-emergency states before minimum dwell time, dampening sub-cycle oscillation.
    - `recovery_confirmation_cycles = 2`: Transitioning out of STOP/WAIT back to PROCEED/AVOID strictly requires 2 consecutive clean cycles.
    - `wait_timeout_s = 2.5\,\text{s}`: Bound on maximum waiting duration before transitioning to STOP.
    - **Safety Invariant Priority:** Emergency transitions to STOP bypass dwell delays and execute instantaneously ($0.0\,\text{ms}$).
  - **Genuine Zero-Motion Semantics:**
    - Both `WAIT` and `STOP` strictly command $v = 0.0\,\text{m/s}$ and $\omega = 0.0\,\text{rad/s}$ (zero creep).
    - Clear diagnostic distinction: `WAIT` indicates temporary positional hold awaiting clearance; `STOP` indicates an active safety intervention.
  - **UNKNOWN_HORIZON Handling:**
    - UNKNOWN is never treated as SAFE or FREE. Under default safety policies, candidate extending into unknown space is rejected, forcing WAIT (if clearance predicted) or STOP.
  - **Deadlock Detection & Milestone 13.11 Interface:**
    - Monitors: (1) consecutive STOP cycles ($\ge 5$ default in `BehavioralConfig`; fully configurable, e.g. verified with 4-cycle override in test 15), (2) repeated wait timeouts ($\ge 2$), and (3) net physical displacement ($< 0.10\,\text{m}$ over $5.0\,\text{s}$).
    - When triggered: sets `deadlock = True` and emits `deadlock_event = "DYNAMIC_DEADLOCK"`.
    - Establishes clean architectural handoff: Milestone 13.11 will consume `DYNAMIC_DEADLOCK` to trigger global A* replanning.
  - **26 Verified Safety Validation Tests (26/26 Passing):**
    - 20 focused operational tests + 6 invariant proofs A through H passing in `tests/test_behavioral_state.py`.
    - Total regression suite: **241/241 tests passing in 95.3s** across Phase 12 and Phase 13.
  - **Scenario Benchmark Results (N=200 Cycles across Scenarios A–J, Isolated SM Latency):**
    | Scenario | Final State | Commanded $(v, \omega)$ | Max Wait (s) | Stop? | Deadlock? | Mean Latency ($\mu\text{s}$) |
    |---|---|---|---|---|---|---|
    | **Scenario A: Unobstructed Travel** | `PROCEED` | $(0.40, 0.00)$ | $0.00$ | False | False | $9.64\,\mu\text{s}$ |
    | **Scenario B: Obstacle Crossing with Detour** | `AVOID` | $(0.30, -0.12)$ | $0.00$ | False | False | $34.00\,\mu\text{s}$ |
    | **Scenario C: Obstacle Stopped in Corridor** | `STOP` | $(0.00, 0.00)$ | $0.00$ | True | True | $16.59\,\mu\text{s}$ |
    | **Scenario D: Temporary Obstacle that Clears** | `PROCEED` | $(0.35, 0.00)$ | $0.80$ | False | False | $21.97\,\mu\text{s}$ |
    | **Scenario E: All Paths Blocked** | `STOP` | $(0.00, 0.00)$ | $0.00$ | True | True | $17.07\,\mu\text{s}$ |
    | **Scenario F: Unknown Prediction Horizon** | `AVOID` | $(0.20, 0.00)$ | $0.00$ | False | False | $18.85\,\mu\text{s}$ |
    | **Scenario G: Obstacle Disappears While Waiting**| `PROCEED` | $(0.35, 0.00)$ | $0.60$ | False | False | $21.76\,\mu\text{s}$ |
    | **Scenario H: Repeated Blockage / Deadlock** | `STOP` | $(0.00, 0.00)$ | $0.00$ | True | True | $11.59\,\mu\text{s}$ |
    | **Scenario I: Noisy Obstacle State (Hysteresis)**| `PROCEED` | $(0.30, 0.00)$ | $0.00$ | False | False | $14.10\,\mu\text{s}$ |
    | **Scenario J: Multiple Dynamic Obstacles** | `AVOID` | $(0.25, 0.08)$ | $0.00$ | False | False | $42.14\,\mu\text{s}$ |
  - **Behavioral State Machine Isolated Execution Latency:**
    - Mean: $12.123\,\mu\text{s}$ ($0.0121\,\text{ms}$)
    - Median: $11.750\,\mu\text{s}$ ($0.0118\,\text{ms}$)
    - P95: $14.700\,\mu\text{s}$ ($0.0147\,\text{ms}$)
    - Max: $24.300\,\mu\text{s}$ ($0.0243\,\text{ms}$)
    - Highly lightweight: adds $< 0.025\,\text{ms}$ per control cycle.
  - **Architectural Boundary & Responsibility Split:**
    - **Milestone 13.10 Scope:** Tactical behavioral state machine (`PROCEED`, `AVOID`, `WAIT`, `STOP`), hysteresis, dwell protection, and deadlock detection only.
    - **Milestone 13.11 Scope:** Global replanning (A* invocation upon `DYNAMIC_DEADLOCK` event to route around persistent obstacles).

- **Milestone 13.11: Event-Driven Global Replanning & Dynamic Planning Overlay:**
  - **Core Architectural Role & Responsibility:**
    - Milestone 13.11 owns `EVENT -> GLOBAL PATH REGENERATION`. It operates strictly above Milestone 13.10 tactical decisions (`PROCEED`, `AVOID`, `WAIT`, `STOP`), Milestone 13.9 DWA dynamic scoring, and Milestone 13.8 predictive collision checking.
    - **Primary Principle:** Do NOT replan globally whenever any dynamic obstacle appears. Global replanning is triggered ONLY by explicit persistent blockage, deadlock, path invalidation, or lack of progress.
    - **Operational Distinctions Enforced:**
      1. *Temporary dynamic obstruction:* Handled locally by M13.10 `WAIT` or `AVOID` -> NO global replan.
      2. *Unknown prediction:* Handled by M13.10 safety policies -> Does NOT automatically trigger global replanning.
      3. *Persistent local blockage:* Vehicle held in `STOP`/`WAIT` for $\ge 3.0\,\text{s}$ (`persistent_blockage_duration_s`) -> candidate for global replan.
      4. *DYNAMIC_DEADLOCK:* Explicit M13.10 deadlock signal -> immediately triggers global replan.
      5. *Goal already reached:* Distance to goal $\le 0.20\,\text{m}$ (`goal_tolerance_m`) -> global replanning strictly suppressed.
  - **Structured Replan Event (`DynamicReplanEvent`):**
    - Fields: `trigger_type` (Enum: `DYNAMIC_DEADLOCK`, `PERSISTENT_BLOCKAGE`, `INSUFFICIENT_PROGRESS`, `PATH_INVALIDATED`, `MANUAL`), `timestamp`, `current_pose` $(x, y, \theta)$, `current_goal` $(x, y)$, `current_global_path`, `blocking_obstacle_ids`, `blocked_duration`, `deadlock_duration`, `current_progress`, `reason`, `confidence`, `event_id`.
  - **Hysteresis, Debouncing, and Cooldown:**
    - `cooldown_s = 4.0\,\text{s}`: Prevents oscillatory replanning churn across successive cycles.
    - `debounce_cycles = 2`: Single-frame noise or spurious blips do NOT trigger global replanning.
    - **Cooldown Override Condition:** If the active path suffers a direct physical collision with an obstacle or lethal terrain, cooldown is overridden as an emergency condition to restore safety.
  - **Ephemeral Dynamic Planning Map Overlay (`DynamicPlanningMapOverlay`):**
    - **Zero Mutation Invariant:** Canonical static `WorldTerrainMap` is NEVER mutated in-place.
    - Ephemeral layer copies created strictly for the duration of the search.
    - Physical dynamic footprints rasterized as lethal obstacles (`obstacle[r, c] = True`).
    - Surrounding safety inflation halos rasterized as cost/hazard penalties.
    - **Unknown Space Rule:** Static unknown regions remain unknown; unknown dynamic obstacles flagged unknown; unknown dynamic space is NEVER falsely cleared to free.
  - **Path Validation & Entrance Safety Guard:**
    - Pre-replan and post-replan path validation checks every waypoint against map bounds, static lethal obstacles, terrain costs, and dynamic obstacles.
    - **Entrance Safety Check:** First `entrance_clearance_steps = 5` waypoints of any newly generated path are evaluated against active dynamic obstacles. If entrance clearance $< 0.35\,\text{m}$ (`entrance_min_clearance_m`), the path is rejected as unsafe.
  - **Stale Path Rule & Failure Semantics:**
    - Upon replan triggering, the active path is explicitly marked stale (`is_path_stale = True`).
    - If A* finds no feasible route, emits `GLOBAL_REPLAN_FAILED`.
    - System holds in `STOP` / `WAIT`.
    - **A stale or invalid global path is NEVER silently executed or commanded.**
  - **Quantitative Path Change Metrics (`PathChangeMetrics`):**
    - Computes `old_path_length`, `new_path_length`, `old_path_cost`, `new_path_cost`, `path_deviation` (directed Hausdorff distance), `goal_reachable`, `num_dynamic_obstacle_intersections`, and `min_dynamic_clearance`.
  - **29 Verified Safety Validation Tests (29/29 Passing):**
    - 20 operational tests covering all trigger types, guards, hysteresis, debounce, and failure modes.
    - 9 safety invariant proofs (A through I) fully passing in `tests/test_global_replanning.py`.
    - Total Phase 13 + Phase 12 planning suite: **269/270 tests passing in 98.5s** (single historical pre-existing interpolation length test in `test_astar.py` untouched).
  - **9 Proven Safety Invariants:**
    - **Invariant A:** Temporary local obstruction alone does not trigger global replan.
    - **Invariant B:** DYNAMIC_DEADLOCK can trigger global replan.
    - **Invariant C:** GLOBAL_REPLAN_FAILED never produces motion (path is None, stale flag is True).
    - **Invariant D:** A stale/invalid global path cannot be silently executed.
    - **Invariant E:** New path cannot override static or dynamic safety constraints.
    - **Invariant F:** Unknown dynamic space is never treated as confirmed free.
    - **Invariant G:** Replanning cannot create a path that immediately collides with a known dynamic obstacle.
    - **Invariant H:** Replanning is deterministic for identical inputs.
    - **Invariant I:** Replan cooldown prevents oscillatory replanning.
  - **Scenario Benchmark Results (Deterministic Scenarios A–J):**
    | Scenario | Description | Trigger | Replans | Success | Goal Reachable | Details |
    |---|---|---|---|---|---|---|
    | **Scenario A** | Obstacle crosses local path and clears quickly | `NONE` | 0 | True | True | Debounced (no replan) |
    | **Scenario B** | Obstacle blocks local path but side-step exists | `NONE` | 0 | True | True | AVOID handles locally |
    | **Scenario C** | Persistent obstacle blocks corridor | `PERSISTENT_BLOCKAGE` | 1 | True | True | dev=0.95m, len=4.97m |
    | **Scenario D** | DYNAMIC_DEADLOCK triggers immediate replan | `DYNAMIC_DEADLOCK` | 1 | True | True | plan=94.8ms |
    | **Scenario E** | New A* route exists around obstacle | `DYNAMIC_DEADLOCK` | 1 | True | True | dev=1.05m |
    | **Scenario F** | All global alternatives blocked | `DYNAMIC_DEADLOCK` | 1 | False | False | fail='GLOBAL_REPLAN_FAILED' |
    | **Scenario G** | Multiple dynamic obstacles | `DYNAMIC_DEADLOCK` | 1 | True | True | 0 intersections |
    | **Scenario H** | Unknown prediction region | `DYNAMIC_DEADLOCK` | 1 | True | True | Unknown preserved |
    | **Scenario I** | Path invalidated again shortly after replan | `NONE` | 2 | True | True | Cooldown override on collision |
    | **Scenario J** | Goal reached before replan | `NONE` | 0 | True | True | Suppressed at goal |
  - **Measured Performance & Latency Profile (Windows x86_64, Python 3.10):**
    | Stage | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |
    |---|---|---|---|---|
    | **Event Detection Latency** | $0.127\,\text{ms}$ | $0.117\,\text{ms}$ | $0.244\,\text{ms}$ | $0.314\,\text{ms}$ |
    | **Path Validation Latency** | $0.015\,\text{ms}$ | $0.014\,\text{ms}$ | $0.023\,\text{ms}$ | $0.026\,\text{ms}$ |
    | **A* Planning Search Latency** | $89.558\,\text{ms}$ | $86.539\,\text{ms}$ | $127.669\,\text{ms}$ | $145.390\,\text{ms}$ |
    | **Total Replanning Latency** | $89.760\,\text{ms}$ | $86.707\,\text{ms}$ | $127.903\,\text{ms}$ | $145.567\,\text{ms}$ |
  - **Performance Finding:**
    - Trigger condition evaluation and path validation are extremely fast ($< 0.35\,\text{ms}$), adding negligible overhead ($< 0.01\%$) to the periodic control loop.
    - Global A* search over the ephemeral overlay executes in $\sim 89.6\,\text{ms}$ on a $120 \times 120$ grid ($0.10\,\text{m/cell}$ resolution), providing fast event-driven route recovery without stalling the UGV.

### Not Proven:
- Real-time 10 Hz / 20 Hz performance of uncompiled pure Python DWA loop under active dynamic obstacles.
- Long-term behavioral stability in complex multi-agent deadlock scenarios beyond single-agent corridor yielding.
- High-speed emergency avoidance maneuvers beyond vehicle kinematic limits.

### Current Known Limitations:
- Rock detour GT terminal error remains at $0.226\,\text{m}$ (exceeds $0.20\,\text{m}$ tolerance due to lateral VO drift).
- Pure Python DWA cycle latency is $4.7\text{--}16.4\,\text{s}$ per cycle ($4.78\,\text{s}$ static baseline; $+2.3\text{--}11.6\,\text{s}$ dynamic evaluation under active corridor interactions). NOT real-time at 10-20 Hz in pure Python.
- Real-time deployment requires compiling DWA trajectory generation and footprint collision checking (e.g. C++, Cython, Numba) or pruning candidate density.
- Dynamic obstacle detection is synthetic/scripted for deterministic validation; vision-based detector integration is a future milestone.
- Fixed 5-frame regression window exhibits estimation lag during high-maneuverability sharp accelerations/turns.
- Constant-velocity prediction is a baseline model; prediction error increases for unmodelled acceleration, braking, or turning maneuvers over horizons > 1.0 s.
### Open Risks:
- High cycle latency in pure Python simulation during complex multi-obstacle crossings.
- Unintentional emergency stops in narrow cluttered terrain when static and dynamic corridors overlap.

---

## 27. Phase 13.12 Completion — ROS2 Integration and End-to-End Node Migration

### Summary of Completed Work
Phase 13.12 completed the ROS2 integration milestone for the dynamic navigation stack. It established the dedicated ROS2 interface package (`ugv_navigation_msgs`), defined the 13 canonical `.msg` interfaces, implemented the bidirectional converter layer (`dynamic_converters.py`), aligned canonical topic and QoS contracts, and migrated all 7 application nodes to the ROSIDL boundary while keeping the mathematical/algorithmic core (`ai/planning/dynamic/*`) 100% pure and ROS-independent.

### NATIVE ROS2 EXECUTION STATUS:
UNVERIFIED ON WINDOWS HOST

The development and validation host is Windows, where native ROS2 toolchains (colcon, native rclpy runtime, C-extension ROSIDL code generators, native TF2, and DDS middleware) are unavailable.
Accordingly:
- No native `colcon build` was executed.
- No native DDS inter-process communication was executed.
- No native Gazebo ROS2 daemon was executed.
- All ROS2 boundary validation was executed deterministically via structural test doubles matching the exact ROSIDL .msg schemas and validated Python adapter pipelines.
- For the complete, gated native Linux / ROS 2 Jazzy execution, Gazebo Harmonic integration, and validation roadmap, see [`docs/ROS2_RUNTIME_INTEGRATION_PLAN.md`](ROS2_RUNTIME_INTEGRATION_PLAN.md).

### Package Architecture
1. **Interface Package (`ugv_navigation_msgs`):**
   - Type: `ament_cmake` pure interface package.
   - Declares `rosidl_default_generators` and `rosidl_default_runtime`.
   - Defines and exports the 13 canonical ROSIDL `.msg` specifications:
     - `DynamicObstacleDetection.msg`, `DynamicObstacleDetectionArray.msg`
     - `TrackedObstacle.msg`, `TrackedObstacleArray.msg`
     - `TrajectoryWaypoint.msg`, `PredictedObstacleTrajectory.msg`, `PredictedTrajectoryArray.msg`
     - `DynamicOccupancyGridSlice.msg`, `DynamicOccupancySliceArray.msg`
     - `BehavioralDecision.msg`, `DynamicReplanEvent.msg`
     - `WorldTerrainMap.msg`, `LocalizationStatus.msg`
2. **Application Package (`ugv_navigation`):**
   - Type: `ament_python` application package.
   - Explicitly declares package dependency `<depend>ugv_navigation_msgs</depend>` in `package.xml`.
   - Houses nodes, converters, adapters, configuration, and launch files.

### Converter Boundary Architecture
- Dedicated conversion module: `ros2_ws/src/ugv_navigation/ugv_navigation/converters/dynamic_converters.py`.
- Strict Boundary Flow:
  $$\text{ROSIDL Message} \longrightarrow \text{Converter} \longrightarrow \text{Pure Phase 13 Model} \longrightarrow \text{Phase 13 Algorithm} \longrightarrow \text{Pure Output Model} \longrightarrow \text{Converter} \longrightarrow \text{ROSIDL Message}$$
- Zero ROS imports (`rclpy`, `rosidl`, `std_msgs`, `geometry_msgs`) inside `ai/planning/dynamic/*`.
- Field and state preservation: 100% field, coordinate frame, covariance, confidence, and lifecycle counter preservation across all 13 message types.
- Timestamp fidelity and precision semantics:
  - ROS timestamps are represented using discrete `sec` and `nanosec` integer fields at the ROS interface (`builtin_interfaces/Time`).
  - The converter reconstructs float seconds using those fields (`float(sec) + float(nanosec) * 1e-9`) and serializes back via integer floor and rounding.
  - Internal absolute timestamps represented as IEEE-754 double-precision float (`binary64`, 53-bit significand) have finite precision, so exact arbitrary nanosecond identity is not mathematically guaranteed at Unix-epoch-scale absolute times ($t \sim 10^9\,\text{s}$, where float resolution is $\approx 120\text{--}240\,\text{ns}$).
  - Relative mission/simulation elapsed times have much better practical floating-point granularity ($< 1\,\text{ns}$ for $t < 10^5\,\text{s}$), without overclaiming exactness at arbitrary epoch magnitudes.
  - The current implementation and test suite preserve required timestamp semantics and temporal ordering within validated numerical tolerances ($1\times 10^{-6}\,\text{s}$ for SIL simulation and $1\times 10^{-4}\,\text{s}$ for adapter tolerances).

### Canonical Topic and Node Graph
| Node | Executable | Subscriptions | Publications | Canonical QoS |
|---|---|---|---|---|
| `DynamicDetectorNode` | `dynamic_detector_node` | Sensor / frame inputs | `/perception/dynamic_obstacles` | Best Effort, Volatile |
| `DynamicTrackerNode` | `dynamic_tracker_node` | `/perception/dynamic_obstacles` | `/tracking/dynamic_obstacles` | Best Effort, Volatile |
| `DynamicPredictorNode` | `dynamic_predictor_node` | `/tracking/dynamic_obstacles` | `/prediction/obstacle_trajectories` | Best Effort, Volatile |
| `DynamicOccupancyNode` | `dynamic_occupancy_node` | `/prediction/obstacle_trajectories` | `/occupancy/dynamic_slices` | Best Effort, Volatile |
| `BehavioralStateNode` | `behavioral_state_node` | `/tracking/dynamic_obstacles`, `/prediction/obstacle_trajectories`, `/localization/status` | `/behavior/tactical_decision` | Reliable, Volatile |
| `DynamicPlannerNode` | `dynamic_planner_node` | `/terrain/world_map`, `/localization/status`, `/goal_pose`, `/tracking/dynamic_obstacles`, `/prediction/obstacle_trajectories` | `/cmd_vel`, `/planning/dynamic_trajectory`, `/planning/dynamic_replan_event` | Reliable, Volatile (`cmd_vel`: Volatile) |
| `GlobalReplannerNode` | `global_replanner_node` | `/terrain/world_map`, `/localization/status`, `/goal_pose`, `/behavior/tactical_decision`, `/tracking/dynamic_obstacles`, `/prediction/obstacle_trajectories` | `/planning/global_path`, `/planning/dynamic_replan_event` | Reliable, Volatile |

### Publisher Ownership Audit
- `/cmd_vel`: Sole and exclusive owner is `DynamicPlannerNode`.
- `/behavior/tactical_decision`: Sole and exclusive owner in standalone distributed mode is `BehavioralStateNode`; emitted by `DynamicPlannerNode` in the active integrated launch pipeline.
- `/planning/dynamic_replan_event`:
  - In the active `dynamic_navigation.launch.py` deployment, `DynamicPlannerNode` is the runtime publisher of `/planning/dynamic_replan_event` (wrapping the integrated replanning pipeline).
  - `GlobalReplannerNode` contains an independent publisher for the same canonical interface as part of the alternative distributed/modular deployment mode.
  - The active launch file does NOT instantiate `GlobalReplannerNode`.
  - Therefore, the active launch configuration has a single runtime publisher for `/planning/dynamic_replan_event`, eliminating any runtime publisher collision.

### Structural End-to-End Validation (`tests/test_phase13_ros2_end_to_end.py`)
All 9 deterministic scenario tests pass using real Phase 13 algorithmic cores with ROSIDL boundaries:
- **Scenario A (Static / No Hazard):** 0 tracks, 0 predictions, free occupancy slices, safe forward proceed command ($v_x > 0$).
- **Scenario B (Moving Obstacle):** Multi-frame detection, track confirmation ($hits \ge 3$), forward trajectory prediction, dynamic occupancy cost generation.
- **Scenario C (Collision Path):** Head-on obstacle triggers tactical avoidance or stop policy (`STOP`/`WAIT`/`AVOID`).
- **Scenario D (Temporary Blockage):** Imminent collision clamps $v_x = 0.0$ and enters `STOP` state.
- **Scenario E (Persistent Blockage):** Sustained blockage exceeding `persistent_blockage_duration_s` (2.0s) triggers `PERSISTENT_BLOCKAGE` global replan.
- **Scenario F (Unknown Horizon):** High-uncertainty predictions maintain non-zero uncertainty radius and are never treated as confirmed free.
- **Scenario G (Stale Message):** Inputs with timestamps moving backward in time are safely rejected; internal clock protected against regression.
- **Scenario H (Malformed Message):** Corrupted input objects without headers or expected fields are safely handled without exceptions, producing no spurious replans.
- **Scenario I (Multiple Moving Obstacles):** Multiple moving obstacles maintain distinct track IDs, independent trajectory predictions, and simultaneous cost footprint generation.

### Comprehensive Test Suite Totals
- Dedicated ROS2 Integration Suites (12 suites): **180/180 PASS**
  - `tests/test_dynamic_detector_ros_boundary.py`: 12/12 PASS
  - `tests/test_dynamic_tracker_ros_boundary.py`: 10/10 PASS
  - `tests/test_dynamic_predictor_ros_boundary.py`: 12/12 PASS
  - `tests/test_dynamic_occupancy_ros_boundary.py`: 12/12 PASS
  - `tests/test_dynamic_planner_ros_boundary.py`: 15/15 PASS
  - `tests/test_behavioral_state_ros_boundary.py`: 12/12 PASS
  - `tests/test_global_replanner_ros_boundary.py`: 13/13 PASS
  - `tests/test_phase13_ros2_end_to_end.py`: 9/9 PASS
  - `tests/test_ros2_topic_contract.py`: 13/13 PASS
  - `tests/test_ros2_dynamic_integration.py`: 29/29 PASS
  - `tests/test_dynamic_converters.py`: 22/22 PASS
  - `tests/test_ros2_node_graph.py`: 15/15 PASS
- Pure Phase 13 Algorithmic Core Regression (11 suites, M13.1–M13.11): **239/239 PASS**
- Targeted Phase 12 Navigation Regression (9 suites): **42/42 PASS**
- **Combined Test Total Across Regression: 461/461 PASS (100%)**

### Known Unresolved Limitations
1. Host limitation: Windows OS lacks native ROS2/colcon toolchains; native execution must be verified on a Linux/ROS2 target.
2. Pure Python DWA loop latency ($4.7\text{--}16.4\,\text{s}$) is not real-time at 10-20 Hz; production deployment requires compiled evaluation or candidate pruning.
3. Monocular visual odometry scale drift and terminal goal error ($0.226\,\text{m}$) from Phase 12 remains unaddressed until future multi-sensor SLAM integration.
4. Native Gazebo simulation with physical camera streams is deferred to future SIL integration phases.

---

## 28. Phase 13.13 Completion — Dynamic SIL Scenario Validation

### Summary of Completed Work
Phase 13.13 successfully executed comprehensive Software-in-the-Loop (SIL) scenario validation across the entire Phase 13 dynamic navigation stack. All 13 sub-milestones (M13.13.1 through M13.13.13) were executed and validated. The validation verified closed-loop navigation, ground-truth isolation, dynamic obstacle tracking, trajectory prediction, spatiotemporal occupancy grid evaluation, dynamic collision avoidance, behavioral arbitration, and global replanning without modifying the frozen production algorithm implementations (`ai/planning/dynamic/*`).

### NATIVE ROS2 EXECUTION STATUS:
UNVERIFIED ON WINDOWS HOST / VALIDATION TRACKED SEPARATELY

The development and validation host is Windows, where native ROS2 toolchains (colcon, native rclpy runtime, C-extension ROSIDL code generators, native TF2, and DDS middleware) are unavailable.
Accordingly:
- Native ROS2 execution status remains unverified on the Windows host and is tracked as a separate validation workstream.
- No native `colcon build`, native DDS, or Gazebo ROS2 daemon was executed during this milestone.
- All Phase 13.13 validation was executed deterministically via the dedicated closed-loop SIL simulation framework (`simulation/runners/dynamic_sil_runner.py`).

### SIL Framework Architecture & Ground-Truth Isolation
- **Harness Separation**: Closed-loop simulation with strictly decoupled ground-truth obstacle physics (`BaseObstacleMotionModel`), observation modeling (`DynamicObservationModel`), unicycle vehicle kinematics, and post-hoc scenario metrics evaluation (`ScenarioEvaluator`).
- **Isolation Invariant**: Production algorithms (`ai/planning/dynamic/*`) receive only synthetic observations (`DynamicObstacleObservation`). Ground truth is strictly inaccessible to the detector, tracker, predictor, occupancy layer, collision checker, DWA planner, and behavioral state machine.
- **Invariant K (Staleness Protection)**: Observation packets with backward or lagged timestamps ($|t_{obs} - t_{now}| > 10^{-4}\,\text{s}$) are rejected at the harness boundary, strictly protecting Kalman filter state covariance against temporal regression.
- **Unknown Horizon Semantics**: Slices beyond an obstacle's valid prediction horizon are classified as `UNKNOWN` and never treated as `FREE`.

### Scenario Coverage & Inventory (24 Scenarios)
The validation suite exercises 24 deterministic, closed-loop scenarios:
- **Core Scenarios A–E (7 scenarios)**:
  - `scenario_a_static`: Static baseline navigation (Seed 42)
  - `scenario_b_parallel`: Parallel moving obstacle (Seed 42, min clearance $0.77\,\text{m}$)
  - `scenario_c_head_on`: Head-on collision course (Seed 42, min clearance $1.20\,\text{m}$)
  - `scenario_d1_crossing_left_to_right`: Lateral 90° crossing left-to-right (Seed 42, min clearance $0.59\,\text{m}$)
  - `scenario_d2_crossing_right_to_left`: Lateral 90° crossing right-to-left (Seed 42, min clearance $0.60\,\text{m}$)
  - `scenario_e1_accelerating`: Forward obstacle accelerating at $+0.15\,\text{m/s}^2$ (Seed 42, min clearance $0.63\,\text{m}$)
  - `scenario_e2_decelerating`: Forward obstacle decelerating at $-0.15\,\text{m/s}^2$ (Seed 42, min clearance $0.74\,\text{m}$)
- **Advanced Scenarios F–O (17 scenarios)**:
  - `scenario_f_temporary_blockage`: Obstacle blocks corridor then vacates (Seed 101, min clearance $0.87\,\text{m}$)
  - `scenario_g1_persistent_blockage_alternate`: Persistent blockage with bypass route (Seed 102, min clearance $0.77\,\text{m}$)
  - `scenario_g2_persistent_blockage_no_route`: Persistent blockage without bypass; safe stop executed (Seed 103, min clearance $0.83\,\text{m}$)
  - `scenario_h_multiple_obstacles`: 4 concurrent crossing obstacles (Seed 104, min clearance $0.79\,\text{m}$)
  - `scenario_i1_noise_low`: Sensor noise $\sigma=0.05\,\text{m}$ (Seed 105, tracking RMSE $0.012\,\text{m}$)
  - `scenario_i2_noise_med`: Sensor noise $\sigma=0.15\,\text{m}$ (Seed 106, tracking RMSE $0.048\,\text{m}$)
  - `scenario_i3_noise_high`: Sensor noise $\sigma=0.30\,\text{m}$ (Seed 107, tracking RMSE $0.101\,\text{m}$)
  - `scenario_j1_dropout_short`: Sensor blind window $0.4\,\text{s}$ (Seed 108, tracking RMSE $0.041\,\text{m}$)
  - `scenario_j2_dropout_medium`: Sensor blind window $0.8\,\text{s}$ (Seed 109, tracking RMSE $0.028\,\text{m}$)
  - `scenario_j3_dropout_intermittent`: Intermittent sensor dropouts (Seed 110, tracking RMSE $0.089\,\text{m}$)
  - `scenario_k_stale_data`: Delayed timestamp packets rejected by Invariant K (Seed 111, min clearance $0.89\,\text{m}$)
  - `scenario_l_unknown_horizon`: Truncated prediction horizon (Seed 112, min clearance $1.09\,\text{m}$)
  - `scenario_m_mixed_quality_tracks`: Mixed high/low covariance and tentative tracks (Seed 113, min clearance $0.98\,\text{m}$)
  - `scenario_n1_narrow_clear`: Corridor width $1.2\,\text{m}$ traversed safely (Seed 114)
  - `scenario_n2_narrow_marginal`: Corridor width $0.9\,\text{m}$ traversed in AVOID mode (Seed 115)
  - `scenario_n3_narrow_blocked`: Corridor width $0.5\,\text{m} < 0.6\,\text{m}$ footprint; safe stop executed (Seed 116)
  - `scenario_o_deadlock`: Concave trap; safe stop executed without collision (Seed 117)

### Measured Algorithmic Performance
1. **Safety**:
   - Total Simulated Collisions: **0** across all 24 scenarios.
   - Unsafe Commands Emitted: **0**.
   - Minimum clearance strictly exceeded safety threshold ($>0.4\,\text{m}$) in all navigable scenarios.
2. **Tracking State Estimation**:
   - Position RMSE: $0.012\,\text{m}$ to $0.026\,\text{m}$ under nominal sensing; bounded at $0.101\,\text{m}$ under high noise ($\sigma=0.30\,\text{m}$).
   - Velocity RMSE: $\le 0.087\,\text{m/s}$ steady state.
   - Multi-obstacle data association: Zero track ID swaps across 4 concurrent obstacles in Scenario H.
3. **Motion Prediction**:
   - Timestamp-aligned evaluation ($t + h$ prediction vs $t + h$ ground truth).
   - Constant-velocity trajectories achieve $0.035\,\text{m}$ RMSE at $0.25\,\text{s}$ horizon and $0.185\,\text{m}$ RMSE at $2.00\,\text{s}$ horizon.
   - Under obstacle acceleration ($\pm 0.15\,\text{m/s}^2$), CV model exhibits theoretical quadratic error growth ($\approx 0.45\,\text{m} - 0.51\,\text{m}$ at $2.0\,\text{s}$), fully enveloped by spatial covariance expansion.
4. **Planning & Behavioral Performance**:
   - Dynamic Window Approach (DWA) prunes all trajectory rollouts that intersect predicted dynamic occupancy.
   - Behavioral arbitration reliably transitions across `PROCEED`, `AVOID`, `WAIT`, and `STOP`.
   - Authoritative STOP is enforced in physically impassable corridors (N3) and deadlocks (O).
5. **Runtime Performance & Characterization**:
   - Planner Mean Latency: $\approx 1.38\,\text{s} - 1.73\,\text{s}$ per cycle.
   - Planner P95 Latency: $\approx 1.47\,\text{s} - 1.97\,\text{s}$ per cycle.
   - Real-Time Factor (RTF): $\approx 0.057\times - 0.072\times$.
   - **Verdict**: Functional correctness validated in pure Python; **NOT real-time**. Production real-time operation ($10-20\,\text{Hz}$) requires C++ vectorized rollouts and spatial grid hashing.
6. **Reproducibility**:
   - Bit-for-bit metric reproducibility confirmed across repeated runs with identical seeds. Trajectory coordinates match within numerical tolerance ($\le 10^{-6}$).

### Regression Test Totals
- SIL Scenario Harness Suite (`tests/test_dynamic_sil_scenarios.py`): **14/14 PASS**
- Core Dynamic SIL Scenarios Suite (`tests/test_dynamic_sil_core_scenarios.py`): **12/12 PASS**
- Advanced Dynamic SIL Scenarios Suite (`tests/test_dynamic_sil_advanced_scenarios.py`): **18/18 PASS**
- Full Phase 13.13 24-Scenario Regression (`tests/test_dynamic_sil_phase13_regression.py`): **26/26 PASS**
- Pure Phase 13 Dynamic Algorithmic Core Regression (11 suites, M13.1–M13.11): **239/239 PASS**
- Phase 12 DWA and Simulation Core Regression: **15/15 PASS**
- **Combined Test Total: 324/324 PASS (100%)**
- Detailed formal report: [`docs/PHASE_13_13_SIL_RESULTS.md`](PHASE_13_13_SIL_RESULTS.md).
