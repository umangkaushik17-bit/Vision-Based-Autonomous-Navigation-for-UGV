# BOULDER COLLISION AVOIDANCE & RVIZ VISUALIZATION AUDIT REPORT

**Date:** 2026-10-02  
**System:** Autonomous UGV Navigation Stack (SIH 2026)  
**Corridor:** 100m Hero Corridor (`outdoor_presentation_100m.sdf`)  
**Mission:** START (0, 0) -> W1 (20, 0) -> W2 (40, 6) -> W3 (60, -4) -> W4 (80, 3) -> W5 (100, 0)  

---

## 1. Executive Summary & Root Cause Analysis

### Identified Problem
During end-to-end runs of the 100m corridor, the UGV successfully traversed from START through W1, W2, and W3, but stalled and repeatedly collided with a physical obstacle near $x \approx 82.5\text{ m}, y \approx 1.3\text{ m}$ (between W4 and W5):
1. **Physical Obstacle:** `boulder_06_w4_chicane_s` located at $(83.5, 1.0)$ with bounding radius $0.65\text{ m}$.
2. **Missing RViz Visualization:** In RViz, this boulder was not rendered at all.
3. **Planner Blindness:** The global planner ($A^*$) planned directly into and through the boulder because `self.world_map.obstacle` was 100% empty free space. The DWA local planner had no obstacle cells in its costmap, causing repeated collision attempts and deadlocks.

### Root Cause
- `interactive_ugv_navigation_node.py` originally only supported a single hardcoded rock (`rock_x`, `rock_y`), which was defaulted to $x = -100.0$ in `prototype_presentation.launch.py`.
- The 8 physical granite boulders modeled in `outdoor_presentation_100m.sdf` and registered in `simulation/config/ground_truth_obstacles_100m.json` were never loaded into the node's internal `WorldTerrainMap`.
- `_publish_visualizations` only generated marker ID 1 at the legacy $x = -100.0$ coordinate, so no boulder cubes or labels appeared in RViz.

---

## 2. Implemented Engineering Fixes

### 2.1 Multi-Boulder Ground Truth Ingestion
In [interactive_ugv_navigation_node.py](file:///c:/Users/shres/Desktop/SIH2026/simulation/nodes/interactive_ugv_navigation_node.py):
- Added the `obstacles_file` ROS 2 parameter.
- Integrated automated JSON loading from [ground_truth_obstacles_100m.json](file:///c:/Users/shres/Desktop/SIH2026/simulation/config/ground_truth_obstacles_100m.json).
- Implemented robust multi-path resolution (custom parameter $\to$ workspace directory $\to$ module relative path $\to$ cwd).

### 2.2 Two-Tier Costmap Imprinting (Lethal Core + Safety Buffer)
All 8 boulders are now imprinted into `self.world_map`:
1. **Lethal Obstacle Core:**
   For all cells within bounding radius $r$:
   $$\text{obstacle}[rr, cc] = \text{True}, \quad \text{cost}[rr, cc] = 1.0$$
2. **Soft Safety Buffer (3 cells = 0.30m):**
   For all surrounding cells within $r < d \le r + 0.30\text{ m}$:
   $$\text{cost}[rr, cc] = \max(\text{cost}[rr, cc], 0.75)$$
   This guarantees that $A^*$ routes around the boulder with ample margin while leaving narrow corridor passages traversable.

### 2.3 RViz 3D Marker Telemetry
Updated `_publish_visualizations` to iterate over all registered boulders:
- **3D Visual Cubes:** Published to namespace `static_rock` under `/visualization_marker_array` with realistic slate gray coloring (`rgba(0.48, 0.44, 0.40, 0.95)`).
- **Floating Overhead Labels:** Published to namespace `static_rock_label` with yellow text (`rgba(1.0, 0.90, 0.40, 1.0)`) displaying boulder names and IDs.

### 2.4 Launch Configuration Integration
In [prototype_presentation.launch.py](file:///c:/Users/shres/Desktop/SIH2026/simulation/launch/prototype_presentation.launch.py):
- Added `decl_obstacles_file` launch argument defaulting to `simulation/config/ground_truth_obstacles_100m.json`.
- Passed `"-p", f"obstacles_file:={str(default_obstacles_file)}"` directly into `interactive_ugv_navigation` process execution.

---

## 3. Empirical Verification & Test Results

### 3.1 Unit & Launch Regression Tests
```
tests/test_prototype_presentation_launch.py ......                       [ 37%]
tests/test_mission_executive.py ..........                               [100%]
============================== 16 passed in 7.19s ==============================
```

### 3.2 Global $A^*$ Boulder Routing Parity Check
Verified clearance across all 5 benchmark mission segments:
| Segment | Start $(x, y)$ | Goal $(x, y)$ | Waypoints | Min Clearance | Closest Boulder |
|---|---|---|---|---|---|
| START $\to$ W1 | $(0.0, 0.0)$ | $(20.0, 0.0)$ | 201 pts | 5.95 m | W1 South Boundary Outcrop |
| W1 $\to$ W2 | $(20.0, 0.0)$ | $(40.0, 6.0)$ | 210 pts | 2.12 m | W2 Detour Granite Boulder |
| W2 $\to$ W3 | $(40.0, 6.0)$ | $(60.0, -4.0)$ | 225 pts | 2.75 m | W2 Outer Flank Boulder |
| W3 $\to$ W4 | $(60.0, -4.0)$ | $(80.0, 3.0)$ | 213 pts | 2.70 m | W4 Chicane North Boulder |
| **W4 $\to$ W5** | **(80.0, 3.0)** | **(100.0, 0.0)** | **204 pts** | **0.81 m** | **W4 Chicane South Boulder** |

### 3.3 End-to-End 100m Closed-Loop Simulation Test
Conducted full headless simulation run (`verify_full_100m_mission.py`):
```
100M BENCHMARK MISSION SUMMARY:
Initial X:              -0.000 m
Final Pose:             (99.717, 0.040, yaw=-0.130)
Reached W1 (20m):       True  (t=33.1s)
Reached W2 (40m, 6m):   True  (t=60.1s)
Reached W3 (60m, -4m):  True  (t=93.2s)
Reached W4 (80m, 3m):   True  (t=126.2s)
Passed Boulder 06:      True  (min clearance to core: 1.75m, net physical: 1.10m)
Reached W5 (100m, 0m):  True  (t=150.2s)
==========================================
RESULT: GOAL REACHED! Mission successful.
```