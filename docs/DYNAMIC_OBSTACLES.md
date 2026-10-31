# Dynamic Obstacle Tracking & Predictive Replanning Architecture

**Authoritative Technical Reference — Phase 13**  
**Project:** SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Repository Branch:** `demo-01-interactive-ugv`  
**Milestone:** 13.17 (Visualization + Architectural Documentation)  

---

## 1. System Architecture Overview

The Phase 13 dynamic obstacle avoidance stack endows the autonomous outdoor UGV with real-time detection, multi-target tracking (MOT), Kalman-filtered state estimation, forward motion prediction, time-parameterized dynamic occupancy evaluation, predictive local trajectory optimization (Dynamic Window Approach / DWA), and behavioral state arbitration.

```
+-----------------------------------------------------------------------------------+
|                        PERCEPTION / OBSERVATION LAYER                             |
|  [Simulated Depth / Optical Flow / Moving Object Detector / Camera Observation]   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v  /perception/dynamic_obstacles (DynamicObstacleList)
+-----------------------------------------------------------------------------------+
|                     MULTI-OBJECT TRACKING (MOT) & FILTERING                       |
|  * Spatial Gating (Mahalanobis / Euclidean distance threshold)                    |
|  * Hungarian Bipartite Data Association                                           |
|  * Constant Velocity (CV) Kalman Filters (State: [x, y, vx, vy]^T)                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v  /planning/tracked_obstacles (TrackedObstacleList)
+-----------------------------------------------------------------------------------+
|                     SPATIOTEMPORAL MOTION PREDICTION                              |
|  * Continuous White Noise Acceleration (CWNA) Extrapolation                       |
|  * Multi-Horizon Waypoints (t + 0.5s, 1.0s, 1.5s, 2.0s, 3.0s)                      |
|  * Growing 3-sigma Spatial Uncertainty Envelopes: r(tau) = r_0 + 3*sigma_pos(tau) |
+-----------------------------------------------------------------------------------+
                                         |
                                         v  /planning/predicted_tracks (PredictedTrajectoryList)
+-----------------------------------------------------------------------------------+
|                     TIME-PARAMETERIZED DYNAMIC OCCUPANCY                          |
|  * Ephemeral Spatiotemporal Reservation Hulls (Static Map Protected)              |
|  * Spatiotemporal Footprint Collision Oracle: C(gamma(t), O_i(t))                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|              PREDICTIVE DWA LOCAL PLANNER (CPU Reference & GPU Accelerated)       |
|  * Dynamic Window Sampling: (v, w) in [v_min, v_max] x [w_min, w_max]             |
|  * Forward Spatiotemporal Footprint Simulation (tau in [0, T_h])                   |
|  * Multi-Objective Scoring: Clearance, Heading, Velocity, Smoothness              |
|  * High-Throughput T4 GPU Tensor Core Batch Evaluation (Hist. 13.17.1-13.17.4)    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                      BEHAVIORAL ARBITRATION & SAFETY HIERARCHY                    |
|  * State Machine: PROCEED -> AVOID -> WAIT -> STOP                                |
|  * Event-Driven Global Replanning Trigger (A* pathfinder invoked on blockage)     |
|  * Strict Fail-Safe Safety Clamps: E-Stop (<0.35m), Caution Clamp (<0.60m)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v  /cmd_vel (Twist)
+-----------------------------------------------------------------------------------+
|                           UGV MOTOR CONTROLLER / ACTUATORS                        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Detection Interface & Data Contracts

The dynamic obstacle subsystem is decoupled from raw sensory feeds via strongly-typed data contracts defined in `ai/planning/dynamic/dynamic_data_model.py`:

- `DynamicObstacleDetection`:
  - `detection_id: str`: Unique sensor-frame observation identifier.
  - `timestamp: float`: Sensor observation timestamp $[s]$.
  - `center_x, center_y: float`: Obstacle centroid in metric world/odometry frame $[m]$.
  - `length, width: float`: Bounding extent dimensions $[m]$.
  - `yaw: float`: Estimated geometric orientation $[rad]$.
  - `confidence: float`: Detection probability $[0.0, 1.0]$.
  - `covariance_xy: Tuple[float, float, float, float]`: Observation error covariance $\begin{bmatrix}\sigma_{xx} & \sigma_{xy} \\ \sigma_{yx} & \sigma_{yy}\end{bmatrix}$.

---

## 3. Multi-Object Tracking (MOT) & Data Association

The Multi-Object Tracker (`DynamicObstacleTracker`) manages track lifecycles across discrete time frames:
1. **Spatial Gating:** Candidate detections within distance threshold $d_{\text{gate}} = 1.5\,\text{m}$ of existing predicted track states are evaluated.
2. **Cost Matrix Construction:** The cost between track $i$ and detection $j$ combines Euclidean distance, Mahalanobis distance, and bounding box overlap:
   $$C_{i,j} = \| \mathbf{p}_{\text{track}, i} - \mathbf{p}_{\text{det}, j} \|_2 + \lambda_{\text{conf}} (1.0 - \text{conf}_j)$$
3. **Bipartite Matching:** Optimal global pairing is resolved using the Munkres/Hungarian algorithm in $\mathcal{O}(N^3)$ time.
4. **Track Lifecycle State Machine:**
   - `TENTATIVE`: Initialized upon new detection; requires $M_{\text{init}} = 3$ consecutive hits within $N_{\text{window}} = 5$ frames.
   - `CONFIRMED`: Active track actively fed into prediction and planning.
   - `COASTING`: Retained without measurement update for up to $N_{\text{max\_miss}} = 5$ frames ($0.5\,\text{s}$).
   - `TERMINATED`: Deleted from memory to prevent memory leaks and track duplication.

---

## 4. State Estimation & Kalman Filtering

Dynamic obstacle motion is estimated using a 4-state Constant Velocity (CV) linear Kalman Filter (`DynamicKalmanFilter`):
$$\mathbf{x}_k = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^\top$$

### 4.1 State Propagation
$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}, \quad \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^\top + \mathbf{Q}$$
Where:
$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{Q} = q_{\text{cwna}} \begin{bmatrix} \frac{\Delta t^3}{3} & 0 & \frac{\Delta t^2}{2} & 0 \\ 0 & \frac{\Delta t^3}{3} & 0 & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & 0 & \Delta t & 0 \\ 0 & \frac{\Delta t^2}{2} & 0 & \Delta t \end{bmatrix}$$
with Continuous White Noise Acceleration spectral density $q_{\text{cwna}} = 0.25\,\text{m}^2/\text{s}^3$.

### 4.2 Measurement Update
$$\mathbf{z}_k = \begin{bmatrix} z_x & z_y \end{bmatrix}^\top, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
$$\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1}$$
$$\mathbf{S}_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^\top \mathbf{S}_k^{-1}$$
$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k, \quad \mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

---

## 5. Spatiotemporal Trajectory Prediction

The trajectory predictor (`MotionPredictor`) projects the future spatiotemporal path of each confirmed track over horizon $T_h = 3.0\,\text{s}$ with discretization $\delta t = 0.1\,\text{s}$:
$$\mu_x(\tau) = x_0 + v_x \tau, \quad \mu_y(\tau) = y_0 + v_y \tau$$
Positional covariance grows with extrapolation lookahead time $\tau$:
$$\sigma_{\text{pos}}(\tau) = \sqrt{\sigma_0^2 + \frac{1}{3} q_{\text{cwna}} \tau^3}$$
The effective obstacle collision boundary expands as a $3\sigma$ spatial uncertainty radius:
$$r_{\text{eff}}(\tau) = r_{\text{geom}} + 3 \sigma_{\text{pos}}(\tau)$$
This generates the characteristic **Prediction Fan** that inflates potential collision volumes at distant time horizons to ensure conservative avoidance.

---

## 6. Dynamic Occupancy Representation

Unlike static obstacles which are fused into the persistent `WorldTerrainMap` (Phase 5/9), dynamic obstacles are maintained in an ephemeral, time-parameterized data structure (`DynamicOccupancyGrid`):
- Dynamic obstacles are never permanently burned into static terrain costs.
- The occupancy structure evaluates queries at discrete time horizons $t \in [0, T_h]$.
- Prevents "ghost obstacles" from corrupting the static costmap when a moving agent leaves a previously occupied corridor.

---

## 7. Predictive Collision Checking

Collision checking evaluates candidate UGV trajectories $\gamma(t) = (x_{\text{ugv}}(t), y_{\text{ugv}}(t))$ against predicted obstacle distributions $O_i(t) = (\mu_{x,i}(t), \mu_{y,i}(t), r_{\text{eff}, i}(t))$:
$$d_{\text{min}}(t) = \min_{i} \left( \|\mathbf{p}_{\text{ugv}}(t) - \boldsymbol{\mu}_i(t)\|_2 - r_{\text{eff}, i}(t) - r_{\text{ugv\_footprint}} \right)$$
If $d_{\text{min}}(t) \le 0$ for any $t \in [0, T_h]$, the candidate trajectory is marked as `COLLISION_LETHAL` and discarded.

---

## 8. Dynamic Window Approach (DWA) Local Planner

The local planner samples linear and angular velocity pairs $(v, \omega)$ within the robot's dynamic capability window:
$$V_d = \left\{ (v, \omega) \mid v \in [v_c - a_{\max} \Delta t, v_c + a_{\max} \Delta t], \omega \in [\omega_c - \alpha_{\max} \Delta t, \omega_c + \alpha_{\max} \Delta t] \right\} \cap [0, v_{\max}] \times [-\omega_{\max}, \omega_{\max}]$$
Trajectories are simulated forward in time and scored via:
$$G(v, \omega) = \alpha \cdot \text{HeadingScore}(v, \omega) + \beta \cdot \text{ClearanceScore}(v, \omega) + \gamma \cdot \text{VelocityScore}(v, \omega) - \delta \cdot \text{DynamicPenalty}(v, \omega)$$

### GPU Acceleration Workstream (Hist. 13.17.1–13.17.4)
While the pure Python CPU reference implementation operates at $1.36\text{--}2.15\,\text{s}$ per cycle, the GPU-accelerated planner (`GPUDynamicDWA`) evaluates 1,200 candidate trajectories across 25 dynamic time horizons concurrently on NVIDIA Tesla T4 GPU Tensor Cores, reducing planning latency to $\approx 3.7\,\text{ms}$ while preserving 100% behavioral and safety parity.

---

## 9. Behavioral Arbitration State Machine

High-level navigation modes are governed by a deterministic finite state machine (`BehavioralStateMachine`):

```
                     +---------------------------------------+
                     |                PROCEED                |
                     |  Clear corridor, nominal DWA tracking |
                     +---------------------------------------+
                                    |         ^
          Clearance < 1.5m          |         | Clearance > 2.0m
          Time-to-Collision < 3.0s  v         |
                     +---------------------------------------+
                     |                 AVOID                 |
                     |  Active lateral deviation & bias DWA  |
                     +---------------------------------------+
                                    |         ^
          Corridor pinched          |         | Crossing clears
          Clearance < 0.8m          v         |
                     +---------------------------------------+
                     |                 WAIT                  |
                     |  Zero forward speed, hold heading     |
                     +---------------------------------------+
                                    |
          Emergency trigger         |
          Clearance < 0.35m         v
                     +---------------------------------------+
                     |                 STOP                  |
                     |  Hard E-Stop clamp, disengage motors  |
                     +---------------------------------------+
```

---

## 10. Event-Driven Global Replanning

When a dynamic obstacle stops and becomes stationary within the planned corridor for longer than $T_{\text{persist}} = 3.0\,\text{s}$, or when all local DWA trajectories in `AVOID` and `WAIT` remain blocked, an event-driven replanning trigger invokes the global $A^*$ planner to compute a brand-new global detour path around the blockage.

---

## 11. Safety Hierarchy & Hardware Clamps

Safety is guaranteed by non-overridable low-level monitors that supersede all planner suggestions:
1. **Emergency Stop (Lethal Barrier):** If any object penetrates $d < 0.35\,\text{m}$, commanded linear and angular velocities are clamped to $(0.0\,\text{m/s}, 0.0\,\text{rad/s})$ immediately.
2. **Caution Speed Clamp:** If $0.35\,\text{m} \le d < 0.60\,\text{m}$, linear speed is strictly clamped to $v_{\text{cmd}} \le 0.20\,\text{m/s}$.
3. **Watchdog Monitor:** If localization quality drops or dynamic tracking fails for $> 0.5\,\text{s}$, the vehicle transitions to `WAIT`.

---

## 12. Ground-Truth Isolation

To maintain strict scientific integrity:
- **Planner/Perception Isolation:** The tracker, predictor, and DWA planner operate solely on noisy observations.
- **Collision Oracle / Evaluator:** Ground-truth (GT) simulation positions are restricted exclusively to the offline evaluation harness (`evaluate_dynamic_sil.py`) to measure true clearance and detect collisions independently.

---

## 13. Visualization & Operator Inspection Tools

Phase 13.17 formalizes a complete inspection suite:
1. **Engineering Visualization Script:** `scripts/visualize_dynamic_tracks.py`
2. **Standard Output Directory:** `simulation/outputs/dynamic_visualization/`
3. **Canonical Diagnostic Artifacts:**
   - `01_trajectory_xy.png`: XY trajectory with start, end, goal, and obstacle footprints.
   - `02_velocity_profiles.png`: Commanded vs actual linear/angular velocities with behavioral state bands.
   - `03_clearance_vs_time.png`: Continuous dynamic clearance and distance to goal relative to 0.35m/0.60m safety thresholds.
   - `04_prediction_fan.png`: Forward extrapolation waypoints with expanding 3-sigma spatial uncertainty circles.
   - `05_dynamic_tracking.png`: Multi-frame MOT track positions, velocity vectors, and estimated speeds.
   - `06_engineering_dashboard.png`: Unified 4-panel diagnostic dashboard.
   - `07_avoidance_animation.gif`: Frame-by-frame animated visual playback of avoidance maneuver.
4. **RViz2 Configuration:** `simulation/rviz/judge_demo.rviz` provides real-time 3D inspection of `/planning/global_path`, `/planning/dynamic_trajectory`, `/visualization_marker_array`, `/odom`, and `/dynamic_obstacle/odom`.

---

## 14. Simulation Limitations & Deployment Boundary

- **Software-In-The-Loop (SIL):** Current validation benchmarks were executed in a deterministic SIL environment.
- **Sensor Observation Adapter:** Simulated sensor feeds provide 2D metric bounding detections; deployment on physical UGV requires integration of the hardware RGB-D / Stereo-Vision YOLO/Segmentation pipeline.
- **Monocular Scale:** Real camera visual odometry requires active stereo or IMU fusion to eliminate the empirical 1.167 scale compensation factor.
- **Planar Kinematics:** Obstacles and vehicle are modeled in 2D ($x, y, \theta$); steep 3D terrain pitch and roll require 3D spatial occupancy extensions in Phase 14+.
