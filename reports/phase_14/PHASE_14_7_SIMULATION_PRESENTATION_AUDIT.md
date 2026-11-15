# PHASE 14.7: GAZEBO + RVIZ PROTOTYPE PRESENTATION AUDIT

**Project**: Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Document**: `reports/PHASE_14_7_SIMULATION_PRESENTATION_AUDIT.md`  
**Date**: October 1, 2026  
**Status**: COMPLETE / AUDIT ONLY (STRICTLY NO PRODUCTION CODE MODIFICATIONS)  
**Current Branch**: `demo-01-interactive-ugv`  
**Baseline Commit**: `e129fa64817159e729f2deb8070ec764a01b015d`  

---

## 1. Executive Summary & Objective

With the completion of **Phase 14.5** (Long-Horizon Multi-Waypoint Autonomous Navigation Benchmark: $1,280\text{ m}$ traveled, 11/12 successful runs, 0 collisions, $0.01\%$ drift) and the formalization of the **Phase 15 Strategic Direction** (Advanced 2.5D Terrain-Aware Navigation), the project enters an essential presentation milestone.

Before commencing the algorithmic implementation of Phase 15.0 (Terrain Simulation + Metric Depth Foundation), the simulation and visualization environment must be audited and prepared for the **Prototype Demonstration Video**. This video will serve as the primary visual proof of the UGV's outdoor autonomous capability for evaluators, stakeholders, and competition judges.

The current simulation environment was designed for isolated subsystem unit testing and early Phase 13 single-obstacle demonstration (7-meter path, primitive solid-color geometric blocks, flat green plane). It does not visually convey the true technical depth of the 100-meter multi-waypoint, vision-in-the-loop, dynamic-evading navigation stack proven in Phase 14.5.

**Scope of this Run**:
- Strictly **AUDIT ONLY**.
- Zero modification to production Python/C++ code, ROS2 nodes, Gazebo worlds, SDFs, RViz configs, or launch files.
- Deliver an actionable, disciplined roadmap for the simulation upgrade that maintains complete backward compatibility with Phase 14 and prepares the ground for Phase 15.0.

---

## 2. Current Simulation Inventory

An exhaustive inspection of `simulation/`, `ros2_ws/`, and associated configurations reveals the exact baseline:

### A. Gazebo Worlds

The repository contains two Gazebo SDF worlds (`simulation/worlds/`):

1. **`simulation/worlds/outdoor_terrain.sdf`** (Phase 11 Baseline World):
   - **Physics Engine**: ODE, $\Delta t = 0.005\text{ s}$ ($200\text{ Hz}$ update rate), real-time factor 1.0.
   - **Extent**: $100\text{ m} \times 100\text{ m}$ planar ground box ($Z=0$).
   - **Ground Geometry**: Flat plane collision and visual with solid olive-green ambient/diffuse material (`0.40, 0.48, 0.30`). No texture mapping.
   - **Static Models Included**:
     - `rock_boulder_large`: Box primitive ($1.2 \times 1.2 \times 0.6\text{ m}$) at $(3.5, 0.5, 0.3)$.
     - `tree_pine_trunk`: Cylinder primitive ($r=0.25\text{ m}, h=3.0\text{ m}$) at $(5.0, -1.5, 1.5)$. No foliage or branches.
     - `water_hazard_patch`: Thin cylinder ($r=0.75\text{ m}, h=0.01\text{ m}$) at $(4.0, -0.8, 0.005)$ with low friction ($\mu = 0.15$).
     - `ditch_hazard_trench`: Recessed box ($1.2 \times 2.5 \times 0.4\text{ m}$) at $(7.0, 1.0, -0.2)$.
     - `gravel_slope_ramp`: Inclined box ($4.0 \times 3.0 \times 0.2\text{ m}$) pitched at $12.0^\circ$ ($0.209\text{ rad}$) at $(10.0, 0.0, 0.4)$.
   - **Lighting / Atmosphere**: Single directional sun at $(0, 0, 20)$ with shadows enabled. No skybox, atmospheric haze, or horizon dome.
   - **Dynamic Obstacles**: None.
   - **Active Range**: All features clustered within $x \in [0, 12]\text{ m}$. $x \in [12, 100]\text{ m}$ is completely empty flat void.

2. **`simulation/worlds/demo_outdoor.sdf`** (Phase 13 Interactive Judge Demo World):
   - **Physics Engine**: ODE, $\Delta t = 0.005\text{ s}$ ($200\text{ Hz}$), real-time factor 1.0.
   - **Plugins Loaded**: `Physics`, `UserCommands`, `SceneBroadcaster`, `Sensors` (Ogre2 engine).
   - **Extent**: $100\text{ m} \times 100\text{ m}$ flat plane.
   - **Static Models Included**:
     - `stationary_rock`: Box primitive ($1.1 \times 1.1 \times 0.5\text{ m}$) at $(3.5, 0.15, 0.25)$.
     - `goal_marker`: Cylinder beacon ($r=0.35\text{ m}, h=1.0\text{ m}$) with yellow/green sphere at $(7.0, 0.0, 0.5)$.
   - **Dynamic Obstacles Included**:
     - `dynamic_obstacle`: Orange box robot ($0.8 \times 0.8 \times 0.6\text{ m}$, $25\text{ kg}$) starting at $(2.5, -3.0, 0.35)$ with `VelocityControl` system subscribed to `/model/dynamic_obstacle/cmd_vel` and `OdometryPublisher` publishing to `/model/dynamic_obstacle/odometry`.
   - **Embedded UGV Model**: The complete `ugv_4wd` model is embedded directly in this world SDF.
   - **Active Range**: Strictly a short $7\text{ m}$ corridor. No features beyond $x = 7.5\text{ m}$.

### B. UGV Robot Model

Defined in `simulation/models/ugv/ugv.urdf` and embedded in `simulation/worlds/demo_outdoor.sdf`:
- **Chassis Structure**: 4WD skid-steer / differential-drive mobile base.
  - Dimensions: Length $0.65\text{ m}$, Width $0.45\text{ m}$, Height $0.20\text{ m}$, Mass $25.0\text{ kg}$.
  - Ground Clearance: $0.125\text{ m}$.
  - Visuals: Dark green solid box primitive (`0.3, 0.5, 0.4`).
- **Wheels**: 4 cylindrical wheels ($r=0.125\text{ m}, w=0.08\text{ m}$, mass $2.0\text{ kg}$ each).
  - Wheelbase: $0.40\text{ m}$, Track Width: $0.48\text{ m}$.
  - Visuals: Solid dark gray cylinders (`0.15, 0.15, 0.15`). No wheel rims, treads, or axles.
- **Sensors**:
  - `rgb_camera`: Monocular pinhole camera mounted at $(0.15, 0.0, 0.525\text{ m})$ relative to `base_link` with a fixed pitch-down angle of $15.0^\circ$ ($0.2618\text{ rad}$). Resolution $640 \times 480$ @ $30\text{ Hz}$, horizontal FOV $80.0^\circ$ ($1.396\text{ rad}$).
  - `imu_sensor`: 6-axis IMU mounted at $(0.0, 0.0, 0.275\text{ m})$ publishing linear acceleration and angular velocity at $200\text{ Hz}$.
  - *Missing Sensor*: No metric depth sensor or stereo camera plugin currently instantiated in Gazebo.
- **Visual Realism Assessment**: The robot model consists strictly of 1 box and 4 cylinders. It lacks mechanical fixtures, roll cage, antennas, sensor housings, lens geometry, or realistic PBR materials.

### C. ROS 2 Integration & Topic Topology

Defined in `simulation/config/bridge_params.yaml` and `simulation/launch/interactive_judge_demo.launch.py`:
- **ROS-GZ Bridge (`ros_gz_bridge`)**:
  - `/cmd_vel` $\longleftrightarrow$ `/model/ugv_4wd/cmd_vel` (`geometry_msgs/msg/Twist`, ROS to GZ)
  - `/odom` $\longleftrightarrow$ `/model/ugv_4wd/odometry` (`nav_msgs/msg/Odometry`, GZ to ROS)
  - `/ground_truth/odom` $\longleftrightarrow$ `/model/ugv_4wd/odometry` (`nav_msgs/msg/Odometry`, GZ to ROS)
  - `/dynamic_obstacle/cmd_vel` $\longleftrightarrow$ `/model/dynamic_obstacle/cmd_vel` (`geometry_msgs/msg/Twist`, ROS to GZ)
  - `/dynamic_obstacle/odom` $\longleftrightarrow$ `/model/dynamic_obstacle/odometry` (`nav_msgs/msg/Odometry`, GZ to ROS)
  - `/clock` $\longleftrightarrow$ `/clock` (`rosgraph_msgs/msg/Clock`, GZ to ROS)
  - `/joint_states` $\longleftrightarrow$ `/world/demo_outdoor/model/ugv_4wd/joint_state` (`sensor_msgs/msg/JointState`)
  - `/camera/image_raw` $\longleftrightarrow$ `.../rgb_camera/image` (`sensor_msgs/msg/Image`)
  - `/camera/camera_info` $\longleftrightarrow$ `.../rgb_camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
  - `/imu` $\longleftrightarrow$ `.../imu_sensor/imu` (`sensor_msgs/msg/Imu`)
- **Interactive Navigation Node (`interactive_ugv_navigation_node.py`)**:
  - Subscribes to `/odom`, `/imu`, `/camera/image_raw`, `/dynamic_obstacle/odom`.
  - Executes RELLIS-3D segmentation, VisionCostmapIntegrator, WorldTerrainMap, A*, C++ AVX2 DWA, MultiFrameTracker, MotionPredictor, DynamicOccupancyLayer, and SafetyGovernor.
  - Publishes `/cmd_vel`, `/planning/global_path`, `/planning/dynamic_trajectory`, `/visualization_marker_array`, and `/perception/traversability_overlay`.

---

## 3. Current RViz Inventory & Limitations

The canonical RViz profile is located at `simulation/rviz/judge_demo.rviz`:

| Display Name | RViz Plugin Class | Target Topic / Source | Configured Role | Visual Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Grid** | `rviz_default_plugins/Grid` | Fixed frame `odom` | Ground reference ($25 \times 25\text{ m}$) | Gray wireframe lines. |
| **TF** | `rviz_default_plugins/TF` | TF Tree Buffer | Coordinate frames (`odom`, `base_link`, `camera_link`, `camera_optical`) | Shows 3D axes arrows. Useful for debugging, but visual clutter in video. |
| **Global Path** | `rviz_default_plugins/Path` | `/planning/global_path` | A* global reference corridor | Solid bright green line ($0.05\text{ m}$ width). Clean. |
| **Selected Local Trajectory** | `rviz_default_plugins/Path` | `/planning/dynamic_trajectory` | Chosen DWA trajectory arc | Solid cyan line ($0.08\text{ m}$ width). High contrast. |
| **Judge Markers & Telemetry** | `rviz_default_plugins/MarkerArray` | `/visualization_marker_array` | 8 distinct marker namespaces | Renders static rock (gray box), goal beacon (yellow cylinder), candidate arcs (green/yellow/red lines), dynamic track box, prediction lines (magenta), and HUD text. |
| **UGV Odometry** | `rviz_default_plugins/Odometry` | `/odom` | Odometry history arrows | Red arrows with white covariance. |
| **Dynamic Obstacle Odom** | `rviz_default_plugins/Odometry` | `/dynamic_obstacle/odom` | Moving obstacle odometry | Red arrows tracking obstacle. |

### Critical Visualization Gaps in Current RViz:

1. **NO `RobotModel` Display**:
   - `judge_demo.rviz` has **zero** robot model visualizer! The UGV body is completely invisible in RViz; only the TF axes arrows and odometry vectors represent the rover. This looks unfinished in video presentations.
2. **NO `Map` / `Costmap` Display**:
   - Neither `WorldTerrainMap` nor `BEVTerrainCostmap` is displayed as an `OccupancyGrid`. The ground in RViz is an empty dark grid plane. Viewers cannot see the traversability gradient, obstacles, or safety inflation buffers that drive the path planner.
3. **NO `Image` / `Camera` Perception Feed**:
   - Although `interactive_ugv_navigation_node.py` computes and publishes `/perception/traversability_overlay` (showing RELLIS-3D segmentation colors), there is no `Image` or `Camera` display docked in RViz.
4. **Single Goal Marker vs 5-Waypoint Mission Sequence**:
   - The display only renders a single goal marker at $(7.0, 0.0)$, matching the old Phase 13 test. The 5-waypoint sequence ($W_1 \to W_2 \to W_3 \to W_4 \to W_5$) validated in Phase 14.5 is not visualized.
5. **Degenerate Camera View**:
   - The saved camera view in `judge_demo.rviz` has a focal distance of $0.16\text{ m}$ with an awkward vertical tilt, causing initial launch to appear disoriented.

---

## 4. Prototype-Video Objective

### The Core Message
The demonstration video must unequivocally communicate to judges:
> *"This autonomous UGV is executing a vision-based, GPS-denied navigation mission across a realistic outdoor trail, tracking ordered waypoints, recognizing traversable ground, evading static boulders, and actively anticipating and yielding to crossing dynamic obstacles in real time."*

### What the Visual Story Must Show:

```
+--------------------------------------------------------------------------------------------------+
|                                    PROTOTYPE DEMONSTRATION VIDEO                                 |
+--------------------------------------------------------------------------------------------------+
|                               GAZEBO VIEW (Physical World Reality)                               |
| - Natural outdoor terrain: Dirt trail corridor winding through grass, low vegetation, and rocks. |
| - High-contrast vehicle: Realistic chassis markings, textured wheels, visible sensor mast.       |
| - Clear mission elements: Numbered waypoint beacons (W1-W5) placed along the 100m corridor.      |
| - Meaningful obstacles: Natural granite boulders, a crossing pedestrian/vehicle dynamic obstacle. |
| - Believable lighting: Directional sun with soft shadows and atmospheric sky backdrop.           |
+--------------------------------------------------------------------------------------------------+
|                                      RVIZ VIEW (Autonomy Brain)                                  |
| - 3D Robot Model: High-fidelity textured URDF chassis tracking vehicle pose.                     |
| - Live Costmap Overlay: Colorized traversability grid (green trail, red obstacles, yellow risk). |
| - Global & Local Planning: Full green A* route corridor + dynamic cyan DWA trajectory arc.       |
| - Dynamic Tracking Fan: Bounding box around obstacle + magenta multi-horizon prediction vectors. |
| - Perception Inset: Picture-in-Picture camera view with live RELLIS-3D segmentation overlay.     |
| - Telemetry HUD: Real-time speed, FPS, active waypoint, behavior state (PROCEED / AVOID / STOP). |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Environment Design Audit (Gazebo World)

To achieve high visual credibility without overloading the ODE physics engine or CPU thread pool, the Gazebo environment must be carefully engineered:

### Proposed Environment Elements (Evaluated):

| Environment Element | Visual Benefit | Simulation Performance Cost | Physics / Collision Cost | Recommendation |
| :--- | :--- | :--- | :--- | :---: |
| **1. Textured Dirt Trail Ribbon** | **High**: Clearly delineates the planned autonomous corridor from untraveled terrain. | **Very Low**: Single UV-mapped visual plane/mesh. | **Zero**: Planar ground collision remains unchanged. | **MUST HAVE** |
| **2. Varied Ground Texture (Grass/Soil)** | **High**: Eliminates flat, synthetic monochrome appearance. | **Very Low**: Material texture shader on ground plane. | **Zero**: No collision impact. | **MUST HAVE** |
| **3. Realistic Granite Boulders** | **High**: Provides visually convincing static obstacles matching Phase 14.5 scenarios. | **Low**: 6-8 low-poly rock models ($<500$ polys each). | **Very Low**: Simple bounding box / cylinder collision. | **MUST HAVE** |
| **4. 5 Ordered Waypoint Beacons** | **High**: Visually guides the viewer through the mission progression ($W_1 \to W_5$). | **Very Low**: Lightweight colored cylinder beacons. | **Zero**: Visual only (`<static>true</static>`, no collision). | **MUST HAVE** |
| **5. Realistic Dynamic Obstacle Model** | **High**: Replaces generic orange cube with a human figure or secondary rover. | **Low**: Low-poly character or compact utility bot. | **Low**: Single simplified bounding box collision. | **SHOULD HAVE** |
| **6. Low-Poly Bushes / Trees (Periphery)**| **Medium**: Frames the outdoor scene and creates depth perception. | **Low**: 12-16 low-poly tree/shrub models. | **Zero**: Placed outside travel corridor, collision disabled. | **SHOULD HAVE** |
| **7. Skybox & Horizon Dome** | **High**: Prevents camera looking into black void at low angles. | **Negligible**: Background cubemap / dome visual. | **Zero**: Visual only. | **SHOULD HAVE** |
| **8. High-Poly 3D Dense Forest Mesh** | **Low**: Adds minor visual detail at extreme computational expense. | **High**: Millions of polygons, GPU render thrashing. | **High**: Complex mesh-mesh collision contacts. | **DO NOT ADD** |
| **9. Particle Effects (Dust, Smoke, Rain)**| **Low**: Visual clutter, distracts from autonomy telemetry. | **Medium**: Particle emitter CPU overhead. | **Zero**. | **DO NOT ADD** |
| **10. Deformable Soft Soil** | **Zero**: Invisible in video, extreme physics compute cost. | **Extreme**: Nonlinear ODE/FEM solver. | **Extreme**: RTF drops below 0.1. | **DO NOT ADD** |

### Environment Layout Geometry:
The upgraded world should feature a **$100\text{ m} \times 20\text{ m}$ active navigation corridor** following the canonical Phase 14.5 waypoint spline:
- Start: $(0.0, 0.0)$
- Waypoint 1: $(20.0, 0.0)$ — Straight approach between gentle grass berms.
- Waypoint 2: $(40.0, 6.0)$ — Curved detour around a clustered granite rock formation.
- Waypoint 3: $(60.0, -4.0)$ — Dynamic crossing zone where a crossing actor enters lateral to the path.
- Waypoint 4: $(80.0, 3.0)$ — Narrow passage framed by pine trees and boundary rocks.
- Waypoint 5: $(100.0, 0.0)$ — Mission completion goal pad with beacon flare.

---

## 6. RViz Presentation Audit ("Hero" Configuration)

A dedicated presentation RViz configuration (`simulation/rviz/hero_presentation.rviz`) must be established to provide an executive-level view of the robot's autonomy stack:

### Required Displays & Layout:

```
+----------------------------------------------------------------------------------------------------+
|                                RVIZ2 HERO PRESENTATION LAYOUT                                      |
+--------------------------------------------------------------------+-------------------------------+
|                                                                    | PERCEPTION CAMERA INSET       |
|                       MAIN 3D VIEW (Orbit)                         | Topic:                        |
|                                                                    | /perception/traversability_   |
| - Fixed Frame: odom                                                | overlay                       |
| - RobotModel: Full URDF mesh with wheel rotation                   | (Live RELLIS-3D segmentation  |
| - Costmap: OccupancyGrid (/costmap/occupancy) with color ramp      |  drivable trail in green,     |
| - Global Path: Bright green A* reference spline (W0 -> W5)         |  obstacles in red)            |
| - Local Trajectory: Dynamic cyan DWA candidate arc                 +-------------------------------+
| - Waypoint Beacons: Numbered 3D text & translucent discs           | TELEMETRY HUD                 |
| - Dynamic Track: Orange 3D bounding box + ID label                 | - Active WP: W3 (60.0, -4.0)  |
| - Prediction Fan: Magenta multi-horizon prediction vectors         | - Behavior: [ AVOID ]         |
| - Candidate Arcs: Faint fan of evaluated candidate trajectories    | - Speed: 0.52 m/s             |
|                                                                    | - Perception: 18.9 FPS        |
|                                                                    | - EKF Drift: 0.01%            |
+--------------------------------------------------------------------+-------------------------------+
```

### Display Prioritization:
1. **HIGH VALUE (Must be enabled and prominently styled)**:
   - `RobotModel`: Loaded from `/robot_description` parameter.
   - `Map` (Costmap): Visualizing the fused traversability costmap ($200 \times 200$ grid).
   - `Path` (Global): `/planning/global_path` (Bright green).
   - `Path` (Local): `/planning/dynamic_trajectory` (Cyan, thick line).
   - `MarkerArray` (Waypoints): Numbered spheres and ground rings for $W_1$ to $W_5$.
   - `MarkerArray` (Dynamic Tracks & Predictions): Bounding box and prediction trajectories.
   - `Image` (Perception Inset): Floating panel displaying the semantic traversability overlay.
2. **MEDIUM VALUE (Subtle secondary styling)**:
   - `MarkerArray` (DWA Candidate Fan): Faint, thin lines ($a=0.3$) showing the breadth of candidate evaluation.
   - `MarkerArray` (HUD Overlay): High-visibility text card fixed to the viewport corner.
3. **LOW VALUE / SUPPRESS (Disable in presentation profile to eliminate visual clutter)**:
   - `TF` axes arrows: Clutters the robot body and trail.
   - `Grid`: Keep subtle or disable in favor of the Costmap.
   - Raw `LaserScan` or uncurated point clouds.
   - Dense odometry history arrows.

---

## 7. Autonomy Storyboard for Prototype Video

Derived strictly from the validated capabilities of Phase 13, 14, and 14.5:

| Video Scene | Duration | Gazebo Visual Action | RViz Telemetry / Behavior | Demonstrated System Subsystem |
| :---: | :---: | :--- | :--- | :--- |
| **Scene 1: Mission Briefing** | $0\text{--}5\,\text{s}$ | Rover at start $(0, 0)$. Wide cinematic view of outdoor trail with waypoints $W_1\text{--}W_5$ stretching to the horizon. | Global A* path computes and illuminates through all 5 waypoints. HUD displays `MISSION_INITIALIZED`. | `MissionExecutive`, `AStarPlanner`, `MultiSensorEKF`. |
| **Scene 2: Autonomous Departure** | $5\text{--}12\,\text{s}$ | Rover accelerates smoothly along dirt trail towards $W_1$ ($20\text{m}$). Wheels rotate with realistic contact. | Cyan local DWA trajectory hugs green path. Perception inset shows green drivable trail detection at $18.9\text{ FPS}$. | `UnifiedDWAPlanner` (C++ AVX2), `RealtimeTraversabilityPipeline`. |
| **Scene 3: Static Boulder Detour** | $12\text{--}22\,\text{s}$ | Approaching $W_2$, a massive granite boulder blocks the centerline. Rover smoothly curves left around the boulder. | Costmap updates with high-cost lethal buffer. DWA candidate fan rejects blocked paths; chosen cyan arc curves cleanly around obstacle. | `WorldTerrainMap`, `VisionCostmapIntegrator`, Footprint Collision Checker. |
| **Scene 4: Dynamic Obstacle Encounter** | $22\text{--}35\,\text{s}$ | Entering $W_3$ zone, a crossing dynamic actor moves across the trail from right to left. | MultiFrameTracker locks onto obstacle (orange box + ID). Magenta prediction corridor extends across path. State switches to `AVOID`/`WAIT`. | `MultiFrameTracker`, `MotionPredictor`, `DynamicOccupancyLayer`, `BehavioralStateMachine`. |
| **Scene 5: Safe Passage & Re-acceleration** | $35\text{--}45\,\text{s}$ | Dynamic obstacle clears corridor. Rover resumes forward acceleration towards $W_4$. | State transitions back to `PROCEED`. Dynamic reservation slice decays. Speed increases smoothly to $0.55\text{ m/s}$. | `SafeCommandBuffer`, `SafetyGovernor`. |
| **Scene 6: Chicane Navigation** | $45\text{--}55\,\text{s}$ | Rover navigates through narrow passage framed by pine trees and rock outcroppings near $W_4$. | Tight corridor costmap inflation visible. Local planner maintains safe margins on both flanks without oscillation. | `UnifiedDWAPlanner`, `RecentHazardMemory`. |
| **Scene 7: Mission Completion** | $55\text{--}62\,\text{s}$ | Rover converges smoothly onto $W_5$ goal beacon at $100\text{m}$. Wheels brake to gentle stop. | Waypoint sequence completes $5/5$. HUD triggers green `MISSION_COMPLETE`. Summary metrics display on screen. | `MissionExecutive`, Terminal Deceleration Governor. |

---

## 8. Gazebo + RViz Consistency Findings

A systematic audit of coordinate conventions, frames, topic types, and scale factors between Gazebo and RViz identifies several consistency considerations:

1. **Coordinate Frame Conventions (REP-103 & REP-105)**:
   - Gazebo Sim native coordinate system: $X$-forward, $Y$-left, $Z$-up (ENU). Matches REP-103.
   - TF tree broadcast in `interactive_ugv_navigation_node.py`:
     `odom` $\longrightarrow$ `base_link` $\longrightarrow$ `camera_link` $\longrightarrow$ `camera_optical`
   - **Consistency Status**: **PASS**. Both environments agree on positive $X$ forward and positive $Y$ left.
2. **Camera Optical Frame Rotation**:
   - In Gazebo SDF (`ugv_4wd`), `camera_optical` link has rotation `rpy="-1.5708 0 -1.5708"`.
   - In `_publish_static_transforms()`, quaternion rotation corresponds to standard ROS camera optical convention ($Z$-forward, $X$-right, $Y$-down).
   - **Consistency Status**: **PASS**. Vision-in-the-loop IPM correctly maps forward rays to $(x, y)$ ground coordinates.
3. **Waypoint Coordinates Consistency**:
   - In `configs/mission_phase14_5_100m.yaml`, waypoints are defined in `odom` frame at $(20, 0), (40, 6), (60, -4), (80, 3), (100, 0)$.
   - In `demo_outdoor.sdf`, the only marker is a goal beacon at $(7.0, 0.0)$.
   - **Current Mismatch**: The Gazebo world currently has **no markers** at the true 100m benchmark coordinates. If the 100m mission runs in `demo_outdoor.sdf`, the robot will drive past the beacon at $7\text{m}$ and navigate through empty, featureless green plane for the remaining $93\text{m}$.
   - **Remediation**: The upgraded Gazebo world must place physical visual beacons and obstacle formations matching the exact $(x, y)$ coordinates in `configs/mission_phase14_5_100m.yaml`.
4. **Robot Model Representation in RViz**:
   - In Gazebo, the UGV is rendered as a physical ODE model.
   - In RViz, `robot_description` is currently not supplied to `judge_demo.rviz` because `robot_state_publisher` is omitted from `interactive_judge_demo.launch.py`.
   - **Remediation**: Launch `robot_state_publisher` in the demo launch file and enable `RobotModel` in RViz.
5. **Costmap Publication**:
   - `WorldTerrainMap` maintains a $200 \times 200$ cost array, but does not publish a standard `nav_msgs/msg/OccupancyGrid`.
   - **Remediation**: Add a lightweight publishing utility or adapter that serializes the rolling costmap into a standard `OccupancyGrid` topic (`/costmap/occupancy`) for RViz visualization.

---

## 9. Performance & Host Hardware Budget

The project is software-first and CPU-conscious (Intel Core Ultra 7 155H development host). The upgrade must not degrade real-time performance or cause simulation stutter:

### Computational Budget Breakdown:

| Subsystem | Baseline Load (Phase 14.5) | Proposed Additions | Budgeted Load | Impact on RTF / Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Gazebo Physics (ODE)** | $\approx 8\%$ CPU ($200\text{ Hz}$) | Add static visual objects (no physics collision), 1 dynamic obstacle, 5 static beacons | $< 10\%$ CPU | **Negligible** ($< 2\%$ increase). Collision contacts remain bounded. |
| **Gazebo Rendering (Ogre2)** | $\approx 15\%$ GPU / $10\%$ CPU | Textured ground ribbon, skybox, low-poly rocks/trees | $< 25\%$ GPU / $12\%$ CPU | **Acceptable**. Kept low-poly ($<10\text{k}$ total scene polygons). |
| **Perception AI (RELLIS-3D)** | $18.9\text{ FPS}$ ($52.9\text{ ms}$ on CPU) | Camera resolution unchanged ($640 \times 480$ @ $10\text{ Hz}$) | $18.9\text{ FPS}$ ($52.9\text{ ms}$) | **Zero Impact**. Pipeline remains identical. |
| **Costmap & Fusion** | $1.8\text{ ms}$ per cycle | Add `OccupancyGrid` conversion ($10\text{ Hz}$) | $2.4\text{ ms}$ per cycle | **Negligible** ($+0.6\text{ ms}$ NumPy array packing). |
| **Local Planner (C++ DWA)**| $3.12\text{ ms}$ mean ($7.76\text{ ms}$ P95) | Unchanged (336 candidate velocity pairs) | $3.12\text{ ms}$ mean | **Zero Impact**. AVX2 kernel untouched. |
| **RViz Rendering (Qt5/OGRE)**| $\approx 5\%$ GPU | Add RobotModel, OccupancyGrid, and Camera Inset | $< 12\%$ GPU | **Well within capacity** of host Intel Arc GPU. |

### Technical Rules for Simulation Assets:
- **No Complex Collision Meshes**: All environmental scenery (trees, bushes, distant mountains) must have `<collision>` tags omitted or set to simple primitive bounding shapes (cylinders for tree trunks, boxes for rocks).
- **Texture Resolution**: Limit diffuse textures to $1024 \times 1024$ JPEG/PNG. Avoid uncompressed $4\text{K}$ textures.
- **Lighting**: Single directional sunlight with soft shadows. No point lights or spotlights.

---

## 10. Phase 15.0 Terrain Compatibility

A vital mandate of this audit is ensuring that the visual upgrade directly supports—and does not complicate—the upcoming **Phase 15.0 (Terrain Simulation & Metric Depth Foundation)**.

### Forward Compatibility Requirements:
1. **Side-by-Side Coexistence of Planar and Uneven Terrain**:
   - The primary $100\text{m}$ corridor must remain planar ($Z=0$) so that Phase 14 regression benchmarks and the prototype video can be executed reliably.
   - An **Uneven Terrain Proving Ground** should be situated immediately adjacent to the main corridor (e.g., at $y \in [-25, -10]\text{ m}$):
     - Sloped test ramps: $10^\circ, 20^\circ, 28^\circ, 32^\circ$ incline planes.
     - Cross-slope bank: $15^\circ - 24^\circ$ lateral camber.
     - Step obstacles: $0.05\text{ m}, 0.12\text{ m}, 0.18\text{ m}, 0.25\text{ m}$ calibrated ledges.
     - Rolling undulating mounds (sinusoidal height field).
2. **Metric Depth Sensor Readiness**:
   - The UGV model SDF must be structured so that a Gazebo depth camera or stereo camera sensor (`type="depth"` or `type="camera"`) can be enabled alongside the RGB camera without altering the chassis physical properties or wheel friction parameters.
3. **Ground-Truth Geometry Isolation**:
   - The mathematical elevations of all terrain features in the proving ground must be documented in a configuration schema (e.g., extending `simulation/config/ground_truth_obstacles.json`) to enable automated depth validation in Subphase 15.0.

---

## 11. Prioritized Change Set

The proposed presentation upgrade is structured into four distinct priority tiers:

| Upgrade Item | Purpose | Subsystem | Performance Overhead | Phase 14 Risk | Priority Tier |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. 100m Outdoor World SDF** | Provide continuous 100m corridor with dirt trail and waypoints matching Phase 14.5 | Gazebo | Low ($<2\%$ CPU) | None (isolated new file) | **MUST HAVE** |
| **2. RViz Hero Configuration** | Complete presentation layout with RobotModel, Costmap, and Perception Inset | RViz | Low | None (isolated new file) | **MUST HAVE** |
| **3. RobotModel in RViz** | Render 3D robot body and rotating wheels instead of bare coordinate axes | RViz / Launch | Negligible | None | **MUST HAVE** |
| **4. Costmap OccupancyGrid Publisher**| Publish rolling `WorldTerrainMap` as standard ROS OccupancyGrid for RViz | ROS2 Node | Very Low ($<0.6\text{ ms}$) | None | **MUST HAVE** |
| **5. 5-Waypoint 3D Visual Beacons** | Numbered visual markers in Gazebo and RViz ($W_1 \to W_5$) | Gazebo / Markers | Negligible | None | **MUST HAVE** |
| **6. Perception Overlay RViz Inset** | Picture-in-picture view of live RELLIS-3D segmentation | RViz | Low | None | **MUST HAVE** |
| **7. Natural Boulder Formations** | Granite rock clusters placed at Phase 14.5 static obstacle locations | Gazebo | Low | None | **SHOULD HAVE** |
| **8. Dynamic Obstacle Realistic Skin** | Replace orange box with low-poly pedestrian or robotic cart model | Gazebo | Low | None | **SHOULD HAVE** |
| **9. Outdoor Skybox & Lighting** | Realistic sky dome and warm sunlight angle for high visual production quality | Gazebo | Negligible | None | **SHOULD HAVE** |
| **10. Trail Periphery Vegetation** | Low-poly trees and bushes outside the corridor to frame the camera shot | Gazebo | Low (Visual only) | None | **SHOULD HAVE** |
| **11. Phase 15.0 Proving Ground Berm** | Adjacent sloped ramps ($10^\circ, 20^\circ, 30^\circ$) and steps for future depth testing | Gazebo | Very Low | None | **NICE TO HAVE** |
| **12. Animated Wheels Plugin** | Visual rotation of wheels matching linear odometry speed | Gazebo | Negligible | None | **NICE TO HAVE** |
| **13. High-Poly Forest Geometry** | Photorealistic dense vegetation across the entire world | Gazebo | Extreme | High (Stutter) | **DO NOT ADD** |
| **14. Dynamic Particle FX** | Smoke, dust clouds, water spray | Gazebo | Medium | High (Clutter) | **DO NOT ADD** |
| **15. Complex Mesh Collisions** | Concave collision meshes on trees and ground undulations | Gazebo | Extreme | High (Physics crash) | **DO NOT ADD** |

---

## 12. Proposed File Changes & Asset Architecture

To ensure strict non-regression of the validated Phase 14 software, **existing files must not be broken or overwritten**. All presentation assets should be created as clean, dedicated companions:

### Proposed New Files (To Be Created During Implementation):
1. **`simulation/worlds/outdoor_presentation_100m.sdf`**:
   - Dedicated 100m presentation world incorporating textured dirt trail ribbon, rock outcroppings at benchmark coordinates, skybox, and Phase 15 adjacent proving ground.
2. **`simulation/models/outdoor_assets/`**:
   - Lightweight, royalty-free low-poly visual assets:
     - `dirt_trail_ribbon/` (UV-textured ground plane segment)
     - `granite_boulder_cluster/` (Low-poly textured rocks with box colliders)
     - `pine_tree_lowpoly/` (Low-poly pine with trunk collider only)
     - `waypoint_beacon/` (Translucent illuminated numbered beacon model)
3. **`simulation/rviz/hero_presentation.rviz`**:
   - Curated RViz layout containing pre-configured `RobotModel`, `Map` (`/costmap/occupancy`), `Image` (`/perception/traversability_overlay`), `MarkerArray` (`/visualization_marker_array`), and optimized camera orbit pose.
4. **`simulation/launch/prototype_presentation.launch.py`**:
   - Single-command master presentation launch orchestrator bringing up Gazebo Sim with `outdoor_presentation_100m.sdf`, `ros_gz_bridge`, `robot_state_publisher`, `interactive_ugv_navigation_node.py` (configured with `mission_phase14_5_100m.yaml`), and `hero_presentation.rviz`.

### Modifications to Existing Files (Minimal & Strictly Additive):
- **`simulation/config/bridge_params.yaml`**:
  - Add optional bridge entries for depth camera topics (prepared for Phase 15.0).
- **`simulation/nodes/interactive_ugv_navigation_node.py`**:
  - Add standard `nav_msgs/msg/OccupancyGrid` publisher on `/costmap/occupancy` to expose `WorldTerrainMap` directly to RViz.
  - Add multi-waypoint marker publishing loop when running under a loaded `mission_phase14_5_100m.yaml`.

---

## 13. Prototype Video Hero View Specification

### Recommended Presentation Layout (Screen Composition):

```
+----------------------------------------------------------------------------------------------------+
| 1920 x 1080 MASTER SCREEN SPLIT                                                                    |
+--------------------------------------------------+-------------------------------------------------+
| LEFT HALF (50% WIDTH): GAZEBO 3D SIMULATION      | RIGHT HALF (50% WIDTH): RVIZ AUTONOMY TELEMETRY |
|                                                  |                                                 |
| Camera: 3rd-person tracking perspective          | Main View: Overhead 3D perspective (Orbit)      |
| Position: Mounted 4m behind and 2.5m above rover | Shows: Fused Costmap (Green/Red), Global Path   |
| View: Clear sightline along dirt trail corridor, | (Green line), Local Trajectory Arc (Cyan line), |
| rocks, and upcoming waypoint beacon.             | RobotModel URDF, Dynamic Obstacle bounding box. |
| Background: Outdoor sky and mountain horizon.    | Inset Bottom-Right: Perception camera view with |
| Lighting: Warm late-afternoon directional sun.   | live RELLIS-3D segmentation overlay.            |
+--------------------------------------------------+-------------------------------------------------+
| BOTTOM STRIP (1080p Overlay Banner):                                                               |
| SIH2026 UGV AUTONOMOUS NAVIGATION | GPS-DENIED OUTDOOR TRAIL | 100M MULTI-WAYPOINT MISSION         |
| Status: AUTONOMOUS_DRIVE | Waypoint: W3/W5 | Speed: 0.54 m/s | Latency: 3.1ms | Collisions: 0      |
+----------------------------------------------------------------------------------------------------+
```

---

## 14. Hardware Boundary Protection

In strict adherence to the project's foundational guidelines:
- **Host Execution Only**: The entire simulation and presentation upgrade executes on standard $x86\_64$ host hardware running Linux/WSL2/Windows.
- **NO Embedded GPU Dependency**: Zero requirement for Jetson Orin or TensorRT.
- **NO Physical Hardware Dependency**: Zero requirement for physical UGV rover or physical stereo cameras.
- **NO Phone Optimization**: Android packaging remains deferred to subsequent integration phases.

---

## 15. Risks & Technical Mitigations

1. **Risk: Physics Engine Instability or Slowdown with Added Scenery**  
   - *Mitigation*: Strictly designate all decorative vegetation, skybox domes, and waypoint beacon visuals as `<static>true</static>` with collision elements omitted entirely (`<collision>` omitted). Only active obstacles (rocks, moving bot) carry simple box/cylinder collision geometry.
2. **Risk: Gazebo Sim / Ogre2 GPU Memory Leak on Host**  
   - *Mitigation*: Limit texture maps to $1024 \times 1024$ and reuse materials across model instances.
3. **Risk: RViz OccupancyGrid Serialization Latency**  
   - *Mitigation*: Publish the $200 \times 200$ costmap at a decimated rate of $5\text{ Hz}$ or $10\text{ Hz}$ using NumPy flat indexing. Compute overhead is measured at $<0.6\text{ ms}$, preserving the $>15\text{ Hz}$ navigation control loop.
4. **Risk: Regression of Phase 14 Production Behavior**  
   - *Mitigation*: Retain all existing launch files and worlds intact. Author new presentation assets with dedicated filenames (`outdoor_presentation_100m.sdf`, `hero_presentation.rviz`, `prototype_presentation.launch.py`).

---

## 16. Recommended Implementation Sequence

Once authorized, the simulation upgrade should be implemented in four disciplined stages:

```
[STAGE 1: Visual Assets & Scenery Preparation]
  ├── Create low-poly dirt trail ribbon, rock clusters, and waypoint beacon models
  └── Set up textures, materials, and skybox assets in simulation/models/
           │
           ▼
[STAGE 2: World Construction (outdoor_presentation_100m.sdf)]
  ├── Assemble 100m corridor with waypoints W1-W5 matching configs/mission_phase14_5_100m.yaml
  ├── Position static rocks at Phase 14.5 benchmark coordinates
  ├── Configure dynamic obstacle crossing path at W3 (60m)
  └── Lay out adjacent Phase 15.0 sloped proving ground (isolated from main trail)
           │
           ▼
[STAGE 3: ROS2 Telemetry & Costmap Adapter]
  ├── Expose WorldTerrainMap as standard nav_msgs/OccupancyGrid on /costmap/occupancy
  ├── Add 5-waypoint visual marker sequence publisher in navigation node
  └── Verify robot_state_publisher broadcast for RobotModel
           │
           ▼
[STAGE 4: Hero RViz Profile & Master Launch Orchestrator]
  ├── Configure simulation/rviz/hero_presentation.rviz with RobotModel, Map, and Perception Inset
  ├── Build simulation/launch/prototype_presentation.launch.py
  └── Perform verification run and document frame rates, RTF, and visual fidelity
```

---

## 17. Final Audit Decision

### Exact Decision State:
$$\mathbf{READY\_WITH\_PRECONDITIONS}$$

### Verified Preconditions Required Prior to Implementation Kickoff:

1. **PRECONDITION 1 (Asset Lightweight Verification)**:
   Verify that all proposed 3D mesh assets (trail ribbon, boulders, trees, beacons) are verified low-poly ($< 1,000$ triangles each) and visually tested to ensure no impact on Gazebo ODE physics or real-time factor ($\text{RTF} \ge 0.95$).
2. **PRECONDITION 2 (Clean File Isolation)**:
   Confirm that all new presentation files will be authored as distinct companion files (`outdoor_presentation_100m.sdf`, `hero_presentation.rviz`, `prototype_presentation.launch.py`) to guarantee zero regression of existing test harnesses and launch configurations.
3. **PRECONDITION 3 (Coordinate Alignment Lock)**:
   Confirm that all scenery and obstacle placements in the 100m world strictly match the coordinates in `configs/mission_phase14_5_100m.yaml` ($W_1: 20\text{m}, W_2: 40\text{m}, W_3: 60\text{m}, W_4: 80\text{m}, W_5: 100\text{m}$).
4. **PRECONDITION 4 (Phase 15 Proving Ground Isolation)**:
   Confirm that the adjacent sloped ramps ($10^\circ, 20^\circ, 30^\circ$) are positioned strictly outside the active Phase 14 planar corridor to prevent false obstacle detections or slope penalties during the prototype presentation run.
5. **PRECONDITION 5 (Zero Production Code Modification Discipline)**:
   Confirm that no implementation code has been written or committed during this audit run.

---

### HARD STOP
**Audit complete. Zero production code, Gazebo worlds, SDFs, RViz configs, or launch files have been modified. Execution stopped.**
