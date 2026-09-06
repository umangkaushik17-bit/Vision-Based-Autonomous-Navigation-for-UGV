# DEMO-01 MASTER CONTEXT

## Objective
Build a single interactive outdoor UGV simulation demonstration that runs LOCALLY on the user's laptop inside WSL2 (Ubuntu 24.04, ROS2 Jazzy, Gazebo Sim Harmonic, RViz2) on CPU ONLY (no GPU, no Kaggle, no cloud compute).
The final judge demonstration executes a complete end-to-end interactive story:
1. UGV navigates autonomously along a global path towards a designated goal.
2. UGV encounters a stationary rock and safely executes local avoidance around it.
3. Operator manually maneuvers a dynamic obstacle into the UGV's forward path using keyboard teleoperation (WASD).
4. Full Phase 13 perception/tracking/prediction/occupancy/DWA safety pipeline engages: tracking, Kalman filter velocity estimation, future trajectory prediction, dynamic occupancy computation, DWA candidate evaluation, and behavioral risk assessment (AVOID / WAIT / STOP).
5. UGV adapts trajectory or halts safely.
6. Operator moves dynamic obstacle out of the path -> risk clears -> state transitions to PROCEED -> UGV continues and reaches GOAL.
7. System supports single-key repeatable reset ('R') for at least 5 repeatable demonstrations.

## Current Status
M1 (Create Minimal Gazebo World) COMPLETE. World `simulation/worlds/demo_outdoor.sdf` created, validated with Gazebo Sim Harmonic 8.11.0. UGV and Dynamic Obstacle physics and topic motions verified. Proceeding to M2 (ROS2 Simulation Bridge).

## Current Milestone
M1 Complete -> Active: M2 (ROS2 Simulation Bridge).

## Environment
- OS: Microsoft Windows 11 / WSL2 Linux subsystem
- WSL version: WSL2 (WSLg enabled with native Wayland/X11 GUI pass-through)
- Ubuntu version: Ubuntu 24.04.3 LTS (Noble Numbat)
- ROS2 distribution: ROS 2 Jazzy Jalisco (`/opt/ros/jazzy`)
- Gazebo version: Gazebo Sim Harmonic (available via `ros-jazzy-ros-gz` / `gz-sim8`)
- RViz2 version: RViz2 14.1.7 (`/opt/ros/jazzy/bin/rviz2`)
- Python version: Python 3.12.3 (in WSL2 Ubuntu)
- GPU status: CPU-only execution active. Host GPU ignored; zero CUDA/PyTorch CUDA/CuPy dependency.
- CPU-only requirement: Strictly enforced. All planning, simulation, and visualization must run locally on CPU.

## Repository State
- branch: `demo-01-interactive-ugv`
- commit: `1c1943e` (feat(perf): accelerate Phase 13 DWA with vectorized tensor kernels)
- working tree: Clean (untracked `notebooks/`)
- relevant packages:
  - `ros2_ws/src/ugv_navigation`: Python ROS2 navigation package with nodes, launch scripts, adapters, and parameter configurations.
  - `ros2_ws/src/ugv_navigation_msgs`: ROS2 message definitions for dynamic obstacle detection, tracking, trajectory prediction, occupancy slices, and behavioral decisions.
  - `simulation/`: Gazebo models (`simulation/models/ugv/model.sdf`, `simulation/models/ugv/ugv.urdf`), worlds (`simulation/worlds/outdoor_terrain.sdf`), and launch files.
  - `ai/planning/dynamic/`: Frozen reference core implementing Kalman tracker, polynomial predictor, dynamic occupancy slicer, trajectory evaluator, DWA local planner, behavioral state machine, and global replanner.

## Existing Architecture
- **Perception/Tracking/Prediction**:
  - `DynamicDetectorNode`: processes detections into `/perception/dynamic_obstacles`.
  - `DynamicTrackerNode`: associates tracks, runs Kalman filter, emits `/tracking/dynamic_obstacles`.
  - `DynamicPredictorNode`: forecasts future trajectories (e.g. +0.5s, +1.0s, +1.5s, +2.0s), emits `/prediction/obstacle_trajectories`.
  - `DynamicOccupancyNode`: builds dynamic spatial-temporal occupancy slices, emits `/occupancy/dynamic_slices`.
- **Planning & Control**:
  - `DynamicPlannerNode`: Evaluates DWA candidate trajectories against dynamic occupancy & static terrain, evaluates risk, determines behavioral state (`PROCEED`, `AVOID`, `WAIT`, `STOP`), and commands `/cmd_vel`.
  - `BehavioralStateNode`: Publishes `/behavior/tactical_decision`.
  - `GlobalReplannerNode`: Handles macroscopic path replanning on persistent blockage.
- **Simulation Interface**:
  - `ros_gz_bridge`: Bridges Gazebo topics (`/model/ugv_4wd/cmd_vel`, `/model/ugv_4wd/odometry`, clock, IMU, cameras) to ROS 2 topics.

## Existing ROS2 Topics
- `/perception/dynamic_obstacles` (`ugv_navigation_msgs/DynamicObstacleDetectionArray`)
- `/tracking/dynamic_obstacles` (`ugv_navigation_msgs/TrackedObstacleArray`)
- `/prediction/obstacle_trajectories` (`ugv_navigation_msgs/PredictedTrajectoryArray`)
- `/occupancy/dynamic_slices` (`ugv_navigation_msgs/DynamicOccupancySliceArray`)
- `/behavior/tactical_decision` (`ugv_navigation_msgs/BehavioralDecision`)
- `/planning/dynamic_replan_event` (`ugv_navigation_msgs/DynamicReplanEvent`)
- `/planning/global_path` (`nav_msgs/Path`)
- `/planning/dynamic_trajectory` (`nav_msgs/Path`)
- `/cmd_vel` (`geometry_msgs/Twist`)
- `/odom` or `/ground_truth/odom` (`nav_msgs/Odometry`)
- `/localization/status` (`ugv_navigation_msgs/LocalizationStatus`)
- `/terrain/world_map` (`ugv_navigation_msgs/WorldTerrainMap`)

## Existing ROS2 Messages
- `DynamicObstacleDetection.msg` & `DynamicObstacleDetectionArray.msg`
- `TrackedObstacle.msg` & `TrackedObstacleArray.msg`
- `PredictedObstacleTrajectory.msg` & `PredictedTrajectoryArray.msg`
- `DynamicOccupancyGridSlice.msg` & `DynamicOccupancySliceArray.msg`
- `BehavioralDecision.msg`
- `DynamicReplanEvent.msg`
- `TrajectoryWaypoint.msg`
- `WorldTerrainMap.msg`
- `LocalizationStatus.msg`

## Existing Simulation Components
- UGV Model: 4WD skid-steer/differential chassis in `simulation/models/ugv/model.sdf` with `gz::sim::systems::DiffDrive`, IMU, wheel joints, and camera.
- Outdoor Terrain: `simulation/worlds/outdoor_terrain.sdf` with ground plane, lighting, and static boulders.
- Bridge configuration: `simulation/config/bridge_params.yaml`.

## Existing Navigation Components
- Frozen dynamic planning algorithms in `ai/planning/dynamic/`.
- ROS 2 node implementations in `ros2_ws/src/ugv_navigation/ugv_navigation/nodes/`.

## Demo World
- Planned single unified world: `simulation/worlds/demo_outdoor.sdf`.
- World components:
  - Flat outdoor ground plane with grassy/dirt texture.
  - Directional sunlight.
  - UGV spawn position at origin $(0.0, 0.0, 0.15)$.
  - Goal target beacon at $(8.0, 0.0, 0.0)$.
  - Stationary rock obstacle at $(3.5, 0.2, 0.25)$ directly challenging the initial straight-line path.
  - Dynamic obstacle (box/cylinder vehicle model) spawned at neutral initial location $(2.5, -3.0, 0.25)$ ready for operator keyboard maneuvering.

## UGV
- 4WD differential-drive rover model (`ugv_4wd`) controlled via `/cmd_vel` and reporting state via odometry.

## Static Obstacle
- Physical boulder rock model positioned at $(3.5, 0.2)$ with ODE collision geometry.
- Visible in Gazebo and mapped into local costmap / spatial grid for DWA avoidance.

## Dynamic Obstacle
- Controllable entity in Gazebo Sim (`dynamic_obstacle`) driven by differential or velocity plugin.
- Controlled via `/dynamic_obstacle/cmd_vel` ROS2 topic.

## Dynamic Obstacle Controls
- Interactive keyboard control node (`scripts/teleop_obstacle.py` or ROS2 node):
  - W: Move Forward (+x)
  - S: Move Backward (-x)
  - A: Move Left (+y)
  - D: Move Right (-y)
  - SPACE: Stop
  - R: Reset demo (UGV, obstacle, planner state)

## Navigation Integration
- Real closed-loop connection:
  - Bridge transmits `/cmd_vel` to UGV.
  - Odometry feedback feeds localization and TF (`odom` -> `base_link`).
  - Dynamic planner computes collision-free trajectories avoiding static rock and dynamic obstacle.

## Visualization
- RViz2 configured with custom display layout (`simulation/rviz/judge_demo.rviz`):
  - Robot model / footprint.
  - Global path (green).
  - Selected local DWA trajectory (cyan).
  - DWA candidate trajectory bundle (color-coded: green=safe, yellow=risk, red=collision).
  - Static rock marker.
  - Dynamic obstacle track bounding box and velocity arrow.
  - Predicted obstacle trajectory with timestamp markers (+0.5s, +1.0s, +1.5s, +2.0s).
  - Risk occupancy slice / polygon.
  - Real-time text overlay: Behavioral state (`PROCEED` / `AVOID` / `WAIT` / `STOP`), linear velocity, distance to goal, and safety clearance.

## Actual Data vs Simulated Data
- Real production algorithms: Tracker, Kalman filter, Predictor, Dynamic Occupancy, DWA local planner, and Behavioral State Machine run real code without simulation shortcuts.
- Dynamic obstacle detection: A dedicated simulated sensor observation adapter feeds real dynamic obstacle detections into `/perception/dynamic_obstacles`, accurately labeled as SIMULATED SENSOR INPUT without claiming raw vision bounding box inference.

## Ground Truth Isolation
- Ground truth positions from Gazebo physics are strictly isolated.
- The navigation stack only receives observation messages (`/perception/dynamic_obstacles`) with realistic noise/latency, preventing future ground-truth leakage into the Kalman tracker and predictor.

## Current Known Problems
- None identified. Gazebo integration package (`ros-jazzy-ros-gz`) and `colcon` must be installed in WSL2.

## Decisions Made
- Use Gazebo Sim Harmonic (`ros-jazzy-ros-gz`) native to Ubuntu 24.04 and ROS 2 Jazzy.
- Standardize all coordinates: UGV starts at $(0,0)$, static obstacle at $(3.5, 0.2)$, dynamic obstacle standby at $(2.5, -3.0)$, goal at $(8.0, 0.0)$.
- Ensure single-command launch file `simulation/launch/interactive_judge_demo.launch.py` launches Gazebo, RViz2, bridges, and navigation nodes.

## Files Created
- `docs/DEMO_01_MASTER_CONTEXT.md`: Persistent demo memory.
- `simulation/worlds/demo_outdoor.sdf`: Unified demo world containing outdoor ground, lighting, goal marker, static rock boulder, dynamic obstacle with velocity control, and 4WD UGV.
- `simulation/scripts/test_m1_gazebo.sh`: Headless Gazebo validation script.

## Files Modified
- None yet.

## Tests
- M0: Environment audit verified WSL2 Ubuntu 24.04, ROS2 Jazzy, and WSLg display.
- M1: `simulation/scripts/test_m1_gazebo.sh` verified Gazebo world loading, physics, topic generation (`/model/ugv_4wd/cmd_vel`, `/model/dynamic_obstacle/cmd_vel`), and odometry velocity responses.

## Milestone History
- M0: Complete environment audit and repository inspection. Master Context created.
- M1: Minimal Gazebo demo world created (`demo_outdoor.sdf`) and validated.

## Remaining Work
- M2: Configure ROS2-Gazebo bridge for UGV and dynamic obstacle control/sensing.
- M3: Connect navigation stack to UGV closed-loop.
- M4: Validate static rock avoidance.
- M5: Integrate dynamic obstacle pipeline with simulated sensor adapter.
- M6: Implement keyboard teleoperation for dynamic obstacle (WASD + Space + R).
- M7: Build comprehensive RViz2 judge-facing visualization.
- M8: Polish demo interface, diagnostics, and overlays.
- M9: Validate 5 repeatable end-to-end demo runs.
- Final Audit: Comprehensive verification and operator documentation.

## Next Action
Implement and validate M2: ROS2 Simulation Bridge (`bridge_params.yaml`), mapping `/cmd_vel` to UGV, `/odom` to ROS2, and `/dynamic_obstacle/cmd_vel` / `/dynamic_obstacle/odom`.
