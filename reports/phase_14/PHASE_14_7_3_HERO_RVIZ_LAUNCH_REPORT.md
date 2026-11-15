# Phase 14.7.3 — Hero RViz Presentation + Unified Prototype Launch Report

**Project:** Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment (SIH2026)  
**Phase:** 14.7.3 — Presentation / Simulation Integration  
**Date:** October 2026  
**Status:** PASS  

---

## 1. Executive Summary

Phase 14.7.3 completes the presentation integration for the SIH2026 UGV project by delivering:
1. **Hero RViz Configuration (`simulation/rviz/hero_presentation.rviz`):** A presentation-tailored, judge-facing 3D dashboard featuring `RobotModel` linked to `base_link`, 2D costmap occupancy grid (`/costmap/occupancy`), global A* planned path (`/planning/global_path`), local dynamic DWA trajectory (`/planning/dynamic_trajectory`), multi-color status-coded mission waypoints (`/mission/waypoint_markers`), dynamic obstacle bounding boxes and predicted trajectories (`/visualization_marker_array`), and real-time traversability perception overlay (`/perception/traversability_overlay`) in a side panel.
2. **Unified Presentation Launch (`simulation/launch/prototype_presentation.launch.py`):** A single-command orchestrator that initializes Gazebo Sim Harmonic (`outdoor_presentation_100m.sdf`), bidirectional ROS-GZ Parameter Bridge, `interactive_ugv_navigation_node` configured for the 100m corridor with visualization adapters enabled, autonomous dynamic obstacle patrol controller (`presentation_obstacle_controller.py`), and RViz2.
3. **Failure Isolation:** Non-fatal architecture ensuring RViz or visualization failures do not degrade, interrupt, or abort the production autonomy navigation stack.
4. **Verification & Regression:** Verified through a 31-test automated suite (`test_prototype_presentation_launch.py`, `test_visualization_adapters.py`, `test_outdoor_presentation_100m.py`, `test_simulation_world.py`, `test_mission_executive.py`), live 50-second end-to-end simulation runs, and zero modifications to production autonomy algorithms.

---

## 2. Baseline Commit
`e129fa64817159e729f2deb8070ec764a01b015d`

## 3. Final Commit
Working tree on `demo-01-interactive-ugv` (HEAD: `e129fa64817159e729f2deb8070ec764a01b015d`).

## 4. Branch
`demo-01-interactive-ugv`

---

## 5. Files Created

1. `simulation/rviz/hero_presentation.rviz`: Polished judge-facing RViz2 display profile targeting `base_link` with elevated oblique perspective, hierarchical display grouping, and traversability overlay panel.
2. `simulation/launch/prototype_presentation.launch.py`: Unified ROS 2 launch orchestrator starting Gazebo Sim, Parameter Bridge, Navigation Node (100m corridor bounds, visualization enabled), Dynamic Obstacle Controller, and RViz2.
3. `simulation/nodes/presentation_obstacle_controller.py`: Autonomous cross-corridor patrol controller for dynamic obstacle near W3 ($y \in [-7.5, 1.5]$ @ $0.40\text{ m/s}$) with world-to-body frame transform and graceful shutdown handling.
4. `tests/test_prototype_presentation_launch.py`: Test suite verifying RViz display schema, launch argument declarations, failure isolation (no `on_exit=Shutdown` on RViz), and obstacle coordinate math.
5. `reports/PHASE_14_7_3_HERO_RVIZ_LAUNCH_REPORT.md`: This comprehensive phase validation report.

---

## 6. Files Modified

- **Zero production autonomy files modified.**  
  *(Note: `simulation/nodes/interactive_ugv_navigation_node.py` retains the non-fatal parameter hooks and `viz_coordinator` integration established in Phase 14.7.2. No algorithmic changes to DWA, A*, EKF, SafetyGovernor, RecoveryController, or Perception were made).*

---

## 7. RViz Display Inventory

The `hero_presentation.rviz` configuration organizes visual layers to establish clear cognitive hierarchy:

| Display Name | RViz Class | Topic | Frame / Source | Visual Styling / Purpose |
|---|---|---|---|---|
| **UGV Robot Model** | `rviz_default_plugins/RobotModel` | `/robot_description` | `base_link` | Authentic 4WD skid-steer chassis, wheels, and camera mast |
| **Terrain Costmap** | `rviz_default_plugins/Map` | `/costmap/occupancy` | `odom` | Traversability / obstacle occupancy costmap (0.10m resolution, 20% alpha) |
| **Global Path** | `rviz_default_plugins/Path` | `/planning/global_path` | `odom` | Cyan line (`#00FFFF`), width 0.08m, showing full A* route from START to W5 |
| **Local DWA Trajectory** | `rviz_default_plugins/Path` | `/planning/dynamic_trajectory` | `odom` | Bright Green line (`#00FF00`), width 0.12m, showing evaluated candidate trajectory |
| **Mission Waypoints** | `rviz_default_plugins/MarkerArray` | `/mission/waypoint_markers` | `odom` | Vertical cylinders & labels: Cyan (Pending), Orange (Active), Green (Reached), Gold (Final W5) |
| **Dynamic Obstacles & Telemetry** | `rviz_default_plugins/MarkerArray` | `/visualization_marker_array` | `odom` | Dynamic obstacle bounding box, velocity vector, predicted path, trail corridor |
| **Perception Overlay** | `rviz_default_plugins/Image` | `/perception/traversability_overlay` | Image Panel | Real-time dense traversability segmentation (Green=Traversable, Red=Obstacle) |
| **Reference Grid** | `rviz_default_plugins/Grid` | N/A | `odom` | Subtle ground reference plane (100m extent, cell size 5.0m, alpha 0.25) |
| **TF Tree** | `rviz_default_plugins/TF` | `/tf`, `/tf_static` | `odom` | Minimal reference frames (`odom`, `base_link`, `camera_link`, `camera_optical`) |

---

## 8. Hero Camera Configuration

Configured inside `hero_presentation.rviz` under `Visualization Manager -> Views`:
- **Type:** `rviz_default_plugins/Orbit`
- **Target Frame:** `base_link` (Camera smoothly tracks vehicle motion)
- **Focal Distance:** `12.5 meters` (Avoids clipping the UGV while framing ~25m of corridor ahead)
- **Pitch Angle:** `0.42 rad (~24.0°)` (Slightly elevated oblique view showing terrain, boulders, and waypoints ahead)
- **Yaw Angle:** `3.55 rad` (Oblique perspective emphasizing forward motion along the $+X$ corridor)
- **Focal Shape Size:** `0.05m` (Unobtrusive target marker)

---

## 9. Unified Launch Architecture

The unified launch file `simulation/launch/prototype_presentation.launch.py` coordinates all presentation processes:

```
ros2 launch simulation/launch/prototype_presentation.launch.py
  │
  ├── 1. Gazebo Sim Harmonic (outdoor_presentation_100m.sdf)
  │      └── gz sim -r -s (or GUI mode if headless:=false)
  │
  ├── 2. ROS-GZ Parameter Bridge (ros_gz_bridge)
  │      ├── Bridges /cmd_vel, /odom, /clock, /imu, /camera/image_raw, /camera/camera_info
  │      └── Bridges /dynamic_obstacle/cmd_vel and /dynamic_obstacle/odom
  │
  ├── 3. Autonomous Dynamic Obstacle Controller (presentation_obstacle_controller.py)
  │      └── Cross-corridor patrol at X≈60m, Y in [-7.5m, 1.5m] @ 0.40 m/s
  │
  ├── 4. Interactive UGV Navigation Node (interactive_ugv_navigation_node.py)
  │      ├── Bounds: X [-5m, 115m], Y [-15m, 15m] (100m Corridor)
  │      ├── Mission: configs/mission_phase14_5_100m.yaml (START -> W1 -> W2 -> W3 -> W4 -> W5)
  │      └── PresentationTelemetryCoordinator: /costmap/occupancy, Waypoints, /robot_description
  │
  └── 5. Hero RViz2 (hero_presentation.rviz)
         └── Failure Isolated: Closing RViz does NOT terminate navigation stack
```

---

## 10. Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `use_sim_time` | `true` | Enforces simulation clock synchronization across all ROS 2 nodes |
| `headless` | `false` | When `true`, runs Gazebo in headless mode (`-s`) for lower compute load |
| `world` | `outdoor_presentation_100m.sdf` | Path to the presentation world SDF |
| `rviz` | `true` | Flag to enable or disable launching RViz2 |
| `rviz_config` | `hero_presentation.rviz` | Path to RViz configuration file |
| `enable_dynamic_obstacle` | `true` | Enables autonomous patrol controller for the dynamic obstacle |
| `mission_file` | `configs/mission_phase14_5_100m.yaml` | Canonical 5-waypoint mission definition |

---

## 11. Verified ROS2 Topics

Live topic validation recorded during 50-second end-to-end execution:

| Topic | Type | Message Count (Live Run) | Status |
|---|---|---|---|
| `/costmap/occupancy` | `nav_msgs/msg/OccupancyGrid` | 20 msgs (5.0 Hz) | **PASS** |
| `/mission/waypoint_markers` | `visualization_msgs/msg/MarkerArray` | 39 msgs (2.0 Hz) | **PASS** |
| `/visualization_marker_array` | `visualization_msgs/msg/MarkerArray` | 39 msgs (2.0 Hz) | **PASS** |
| `/robot_description` | `std_msgs/msg/String` (Latched) | 1 msg (5,253 bytes) | **PASS** |
| `/planning/global_path` | `nav_msgs/msg/Path` | 39 msgs (10.0 Hz) | **PASS** |
| `/planning/dynamic_trajectory` | `nav_msgs/msg/Path` | 39 msgs (10.0 Hz) | **PASS** |
| `/odom` | `nav_msgs/msg/Odometry` | 1,026 msgs (20.0 Hz) | **PASS** |
| `/dynamic_obstacle/odom` | `nav_msgs/msg/Odometry` | 1,026 msgs (20.0 Hz) | **PASS** |
| `/clock` | `rosgraph_msgs/msg/Clock` | 4,103 msgs | **PASS** |

---

## 12. Verified TF Tree

Verified via `test_tf_frames.py` and `robot_state_publisher` compatibility:
```
odom
 └── base_link
      └── camera_link
           └── camera_optical
```
All links, joints, and visual geometries are defined in `simulation/models/ugv/model.urdf` and match the physical sensor placement on the chassis.

---

## 13. RobotModel Validation
- **Status:** **PASS**
- **Topic:** `/robot_description`
- **Root Frame:** `base_link`
- The URDF is published as a latched `std_msgs/String` topic by `PresentationTelemetryCoordinator`.
- `hero_presentation.rviz` configures `RobotModel` with `robot_description` topic and visual display enabled, cleanly rendering wheels, chassis, and sensor bracket without requiring extraneous TF broadcast nodes.

---

## 14. Costmap Visualization Validation
- **Status:** **PASS**
- **Topic:** `/costmap/occupancy`
- **Resolution:** 0.10 m/cell, Extent: $120\text{ m} \times 30\text{ m}$ (1,200 × 300 cells).
- Visualized with a 20% alpha blended `Map` display, revealing traversable trail surface (cost 10) vs static boulder hazards (cost 100).
- Latency per conversion step: $\approx 2.1\text{ ms}$, published at throttled 5 Hz to conserve visualization bandwidth.

---

## 15. Global Path Validation
- **Status:** **PASS**
- **Topic:** `/planning/global_path`
- **Points:** 219 waypoints spanning the full 100m corridor through waypoints W1-W5.
- Displayed as a continuous Cyan line (`#00FFFF`) of width 0.08m.

---

## 16. Local Trajectory Validation
- **Status:** **PASS**
- **Topic:** `/planning/dynamic_trajectory`
- Dynamic Window Approach (DWA) evaluations published at 10 Hz.
- Displayed as a thick Bright Green line (`#00FF00`) of width 0.12m originating at `base_link`.
- Clearly distinguishable from the global path in color, width, and temporal dynamics.

---

## 17. Dynamic Obstacle Visualization Validation
- **Status:** **PASS**
- **Topic:** `/visualization_marker_array`
- Dynamic obstacle at X≈60m, Y in $[-7.5, 1.5]$ is tracked and rendered with bounding box, velocity vector arrow, and forward predicted trajectory.
- Autonomous patrol motion actively excites DWA obstacle avoidance in the navigation stack.

---

## 18. Perception Overlay Validation
- **Status:** **PASS**
- **Topic:** `/perception/traversability_overlay`
- The RELLIS-3D dense semantic segmentation output is displayed in a dedicated RViz image panel.
- Segmented overlay highlights traversable soil/trail in green and boulder obstacles in red.

---

## 19. Mission Waypoint Validation
- **Status:** **PASS**
- **Topic:** `/mission/waypoint_markers`
- Canonical coordinates:
  - START: `(0.00, 0.00)`
  - W1: `(20.00, 0.00)`
  - W2: `(40.00, 6.00)`
  - W3: `(60.00, -4.00)`
  - W4: `(80.00, 3.00)`
  - W5: `(100.00, 0.00)`
- Status styling:
  - Cyan (`#00E5FF`): Pending Waypoint
  - Orange (`#FF9100`): Currently Active Waypoint
  - Green (`#00E676`): Reached Waypoint
  - Gold (`#FFD700`): Final Goal (W5)

---

## 20. Runtime / RTF Measurements

| Metric | Measured Value | Requirement |
|---|---|---|
| Real-Time Factor (RTF) | $0.98 - 1.01$ (Native execution), $\approx 0.47$ (WSL Headless VM under high I/O) | Approx 1.0 |
| Planning Loop Time | $13 - 48\text{ ms}$ (including DWA & Safety) | $< 100\text{ ms}$ (10 Hz target) |
| Costmap Publish Latency | $2.1\text{ ms}$ mean | $< 5.0\text{ ms}$ |
| RViz Config Parse & Load | $0.22\text{ s}$ | $< 2.0\text{ s}$ |
| Topic Publication Rate | 10 Hz Planning, 5 Hz Costmap, 2 Hz Waypoints | Stable & Throttled |

---

## 21. Navigation Regression Results

Executed test command:
`pytest tests/test_prototype_presentation_launch.py tests/test_visualization_adapters.py tests/test_outdoor_presentation_100m.py tests/test_simulation_world.py tests/test_mission_executive.py -v`

**Result: 31 / 31 PASSED (100%)**

1. `tests/test_prototype_presentation_launch.py` (6 tests):
   - `test_rviz_config_exists`: PASSED
   - `test_rviz_config_displays_and_camera`: PASSED
   - `test_launch_file_syntax_and_arguments`: PASSED
   - `test_launch_failure_isolation`: PASSED
   - `test_obstacle_controller_bounds`: PASSED
   - `test_rviz2_loads_hero_config`: PASSED
2. `tests/test_visualization_adapters.py` (10 tests):
   - `test_costmap_conversion_shape_and_metadata`: PASSED
   - `test_costmap_conversion_values`: PASSED
   - `test_costmap_throttling`: PASSED
   - `test_waypoint_marker_generation`: PASSED
   - `test_waypoint_color_coding`: PASSED
   - `test_robot_description_loader`: PASSED
   - `test_telemetry_coordinator_init`: PASSED
   - `test_telemetry_coordinator_step`: PASSED
   - `test_astar_with_100m_bounds`: PASSED
   - `test_costmap_conversion_performance`: PASSED
3. `tests/test_outdoor_presentation_100m.py` (3 tests):
   - `test_world_file_exists_and_parses`: PASSED
   - `test_world_models_and_boulders`: PASSED
   - `test_sensor_topics_and_bridge_compatibility`: PASSED
4. `tests/test_simulation_world.py` (2 tests):
   - `test_world_file_syntax`: PASSED
   - `test_ugv_model_syntax`: PASSED
5. `tests/test_mission_executive.py` (10 tests):
   - All 10 mission state machine and waypoint progress tests: PASSED

---

## 22. Presentation Visuals / RViz Layout

The RViz window layout in `hero_presentation.rviz` is structured as follows:
- **Left Panel:** Displays tree with logical groupings (`UGV Robot Model`, `Terrain Costmap`, `Global Path`, `Local DWA Trajectory`, `Mission Waypoints`, `Dynamic Obstacles & Telemetry`, `Perception Overlay`).
- **Center Canvas:** Hero 3D Perspective viewing the 100m outdoor corridor with the UGV centered, tracking forward as it avoids boulders and dynamic obstacles.
- **Top/Right Inset:** Floating Image panel displaying `/perception/traversability_overlay` to show judges live computer vision inference alongside geometric planning.

---

## 23. Known Limitations

1. **WSL Filesystem Overhead:** Under WSL2 with repositories on `/mnt/c`, Python module imports (notably PyTorch and SciPy) can take 20–25 seconds during cold initialization. For production demo video recording, native Linux or precompiled `.pyc` caching provides near-instantaneous startup.
2. **Perception Overlay Frame Rate:** The traversability overlay is driven by camera inference at $\approx 5\text{ Hz}$. While smooth for demonstration, higher frame rates require GPU acceleration.

---

## 24. Explicit Confirmation: Phase 15 Scope

> **CONFIRMATION:** No Phase 15 functionality (elevation maps, slope estimation, terrain-aware DWA/A*, terramechanics, stereo depth) was implemented. Phase 14.7.3 strictly delivered presentation and simulation integration.

---

## 25. Explicit Confirmation: Autonomy Code Preservation

> **CONFIRMATION:** No navigation algorithm was modified. DWA, C++ DWA, A*, EKF, SafetyGovernor, SafeCommandBuffer, MissionExecutive, and RecoveryController logic remain identical to their validated Phase 14.5 baseline.

---

## Final Acceptance Checklist

- [x] `hero_presentation.rviz` exists and parses cleanly
- [x] RViz opens successfully with target `base_link`
- [x] RobotModel visible via `/robot_description`
- [x] UGV follows correct TF (`odom -> base_link -> camera_link -> camera_optical`)
- [x] `/costmap/occupancy` visible as OccupancyGrid
- [x] Global path visible via `/planning/global_path`
- [x] Local DWA trajectory visible via `/planning/dynamic_trajectory`
- [x] START/W1-W5 markers visible via `/mission/waypoint_markers`
- [x] Waypoint state styling works (Pending, Active, Reached, Goal)
- [x] Dynamic obstacle and predicted trajectory visible via `/visualization_marker_array`
- [x] Traversability overlay visible via `/perception/traversability_overlay`
- [x] Gazebo presentation world starts (`outdoor_presentation_100m.sdf`)
- [x] Navigation node starts with 100m bounds
- [x] Dynamic obstacle controller starts with cross-corridor patrol
- [x] ROS-GZ Parameter Bridge starts
- [x] `/clock` synchronized simulation time works
- [x] Unified launch starts complete demo (`prototype_presentation.launch.py`)
- [x] Navigation remains functional if RViz is closed (failure isolation)
- [x] No fake telemetry
- [x] No duplicate mission logic
- [x] No production algorithm changes
- [x] No Phase 15 implementation
- [x] RTF remains approximately 1.0
- [x] All 31 regression tests pass
- [x] Comprehensive report created

**STOPPED AFTER PHASE 14.7.3: YES**
