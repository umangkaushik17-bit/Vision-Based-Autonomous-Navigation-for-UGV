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
M3 (Connect Navigation Stack) & M4 (Static Rock Avoidance) COMPLETE.
- Decoupled torch/CUDA imports across `ai/costmap` and `ai/obstacles` for pure CPU execution.
- Created `simulation/nodes/interactive_ugv_navigation_node.py` integrating closed-loop Gazebo `/odom` subscriber, TF `odom` -> `base_link` broadcasting, static rock mapping, A* global routing, DWA local planning (with real-time CPU sampling grid 5x9 at 4Hz), Kalman tracking, motion prediction, dynamic occupancy, and behavioral arbitration.
- Validated with `simulation/scripts/test_m3_m4_navigation.sh`: UGV navigated autonomously from (0.0, 0.0) forward to X=4.86m, safely bypassing the stationary rock boulder at (3.5, 0.15) with 1.10m clearance (passing required >0.45m margin).
- Proceeding autonomously to M5 (Dynamic Obstacle Pipeline Integration) & M6 (Dynamic Obstacle Manual Keyboard Control).

## Current Milestone
M4 Complete -> Active: M5 (Dynamic Obstacle Pipeline) & M6 (WASD Keyboard Control).

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
- commit: `3205b7d` (feat(demo): add ROS2 simulation bridge)
- working tree: Clean (with tracked test/verification scripts)
- relevant packages:
  - `ros2_ws/src/ugv_navigation`: Python ROS2 navigation package with nodes, launch scripts, adapters, and parameter configurations.
  - `ros2_ws/src/ugv_navigation_msgs`: ROS2 message definitions for dynamic obstacle detection, tracking, trajectory prediction, occupancy slices, and behavioral decisions.
  - `simulation/`: Gazebo models (`simulation/models/ugv/model.sdf`, `simulation/models/ugv/ugv.urdf`), worlds (`simulation/worlds/demo_outdoor.sdf`), nodes (`simulation/nodes/interactive_ugv_navigation_node.py`), and bridge config (`simulation/config/bridge_params.yaml`).
  - `ai/planning/dynamic/`: Frozen reference core implementing Kalman tracker, polynomial predictor, dynamic occupancy slicer, trajectory evaluator, DWA local planner, behavioral state machine, and global replanner.

## Existing Architecture
- **Perception/Tracking/Prediction**:
  - `DynamicDetectorNode` / Sensor Adapter: processes detections into `/perception/dynamic_obstacles`.
  - `DynamicTrackerNode` / `MultiFrameTracker`: associates tracks, runs Kalman filter, emits `/tracking/dynamic_obstacles`.
  - `DynamicPredictorNode` / `MotionPredictor`: forecasts future trajectories (e.g. +0.5s, +1.0s, +1.5s, +2.0s), emits `/prediction/obstacle_trajectories`.
  - `DynamicOccupancyNode` / `DynamicOccupancyLayer`: builds dynamic spatial-temporal occupancy slices, emits `/occupancy/dynamic_slices`.
- **Planning & Control**:
  - `InteractiveUGVNavigationNode`: Orchestrates A* global routing around static rock, evaluates DWA candidate trajectories against dynamic occupancy & static terrain, evaluates risk, determines behavioral state (`PROCEED`, `AVOID`, `WAIT`, `STOP`), and commands `/cmd_vel`.
  - `BehavioralStateMachine`: Publishes `/behavior/tactical_decision`.
  - `EventDrivenGlobalReplanner`: Handles macroscopic path replanning.
- **Simulation Interface**:
  - `ros_gz_bridge`: Bridges Gazebo topics (`/model/ugv_4wd/cmd_vel`, `/model/ugv_4wd/odometry`, `/model/dynamic_obstacle/cmd_vel`, `/model/dynamic_obstacle/odometry`, clock, IMU, cameras) to ROS 2 topics.

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
- `/dynamic_obstacle/cmd_vel` (`geometry_msgs/Twist`)
- `/dynamic_obstacle/odom` (`nav_msgs/Odometry`)
- `/demo/reset` (`std_msgs/Empty`)
- `/visualization_marker_array` (`visualization_msgs/MarkerArray`)

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
- UGV Model: 4WD skid-steer/differential chassis in `simulation/worlds/demo_outdoor.sdf` with `gz::sim::systems::DiffDrive`, IMU, wheel joints, and camera.
- Outdoor Terrain: `simulation/worlds/demo_outdoor.sdf` with ground plane, directional lighting, goal marker at (7.0, 0.0), static rock boulder at (3.5, 0.15), and dynamic obstacle at (2.5, -3.0).
- Bridge configuration: `simulation/config/bridge_params.yaml`.

## Existing Navigation Components
- Frozen dynamic planning algorithms in `ai/planning/dynamic/`.
- Interactive UGV navigation node in `simulation/nodes/interactive_ugv_navigation_node.py`.

## Demo World
- Unified single world: `simulation/worlds/demo_outdoor.sdf`.
- Ground: flat outdoor terrain with sun illumination.
- UGV: (0.0, 0.0, 0.0).
- Goal: (7.0, 0.0, 0.0).
- Static Rock Boulder: (3.5, 0.15) with collision geometry.
- Dynamic Obstacle: (2.5, -3.0, 0.25) with `VelocityControl` plugin.

## UGV
- 4WD differential-drive rover model (`ugv_4wd`) controlled via `/cmd_vel` and reporting state via `/odom`.

## Static Obstacle
- Physical boulder rock model positioned at (3.5, 0.15) with ODE collision geometry.
- Mapped into `WorldTerrainMap` raster costmap with safety clearance buffer. UGV autonomously detours around it.

## Dynamic Obstacle
- Controllable entity in Gazebo Sim (`dynamic_obstacle`) driven by `gz::sim::systems::VelocityControl`.
- Controlled via `/dynamic_obstacle/cmd_vel` ROS2 topic.

## Dynamic Obstacle Controls
- Interactive keyboard control node (`simulation/nodes/teleop_obstacle_keyboard.py`):
  - W: Move Forward (+x)
  - S: Move Backward (-x)
  - A: Move Left (+y)
  - D: Move Right (-y)
  - SPACE: Stop
  - R: Reset demo (UGV, obstacle, planner state)

## Navigation Integration
- Closed-loop connection verified:
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
  - Real-time text overlay: Behavioral state (`PROCEED` / `AVOID` / `WAIT` / `STOP`), linear velocity, distance to goal, and safety clearance.

## Actual Data vs Simulated Data
- Real production algorithms: Tracker, Kalman filter, Predictor, Dynamic Occupancy, DWA local planner, and Behavioral State Machine run real code without simulation shortcuts.
- Dynamic obstacle detection: A dedicated simulated sensor observation adapter feeds real dynamic obstacle detections into `/perception/dynamic_obstacles`, accurately labeled as SIMULATED SENSOR INPUT without claiming raw vision bounding box inference.

## Ground Truth Isolation
- Ground truth positions from Gazebo physics are strictly isolated.
- The navigation stack only receives observation messages (`/perception/dynamic_obstacles`) with realistic noise/latency, preventing future ground-truth leakage into the Kalman tracker and predictor.

## Current Known Problems
- None. CPU latency resolved via candidate sample tuning (5x9) running at ~500ms cycle time.

## Decisions Made
- Use Gazebo Sim Harmonic (`ros-jazzy-ros-gz`) native to Ubuntu 24.04 and ROS 2 Jazzy.
- Standardize all coordinates: UGV starts at (0,0), static obstacle at (3.5, 0.15), dynamic obstacle standby at (2.5, -3.0), goal at (7.0, 0.0).
- Forward FOV gating (6m range, +-45 deg bearing) ensures dynamic obstacle is only tracked when entering front sensor view.
- Single-command launch file `simulation/launch/interactive_judge_demo.launch.py` to launch Gazebo, RViz2, bridge, and navigation node.

## Files Created
- `docs/DEMO_01_MASTER_CONTEXT.md`: Persistent demo memory.
- `simulation/worlds/demo_outdoor.sdf`: Unified demo world containing outdoor ground, lighting, goal marker, static rock boulder, dynamic obstacle with velocity control, and 4WD UGV.
- `simulation/nodes/interactive_ugv_navigation_node.py`: Integrated closed-loop navigation, tracking, DWA, and RViz telemetry node.
- `simulation/scripts/test_m1_gazebo.sh`: Headless Gazebo validation script.
- `simulation/scripts/test_m2_bridge.sh`: ROS2 simulation bridge validation script.
- `simulation/scripts/test_m3_m4_navigation.sh`: Automated M3 & M4 validation test script.
- `simulation/scripts/verify_navigation_run.py`: Closed-loop odometry trajectory and clearance verifier.

## Files Modified
- `simulation/config/bridge_params.yaml`: Configured mappings for `/cmd_vel`, `/odom`, `/dynamic_obstacle/cmd_vel`, `/dynamic_obstacle/odom`, clock, and sensors.
- `ai/costmap/__init__.py`: Added optional import fallback for `LayerGenerator` to decouple torch from CPU navigation.
- `ai/obstacles/model.py`: Made `torch` optional for CPU-only execution.
- `ai/planning/local_planners/dwa.py`: Added read-only `latest_candidates` exposure for judge visualization.
- `.gitignore`: Added `ros2_ws` build/install/log artifacts.

## Tests
- M0: Environment audit verified WSL2 Ubuntu 24.04, ROS2 Jazzy, and WSLg display.
- M1: `simulation/scripts/test_m1_gazebo.sh` verified Gazebo world loading, physics, topic generation, and odometry velocity responses.
- M2: `simulation/scripts/test_m2_bridge.sh` verified `ros_gz_bridge` bidirectional topic routing, ROS2 `/cmd_vel` command actuation, and ROS2 `/odom` / `/dynamic_obstacle/odom` telemetry.
- M3 & M4: `simulation/scripts/test_m3_m4_navigation.sh` verified full closed-loop navigation from (0.0, 0.0) forward past the static rock boulder (X=4.86m), maintaining 1.10m safe clearance.

## Milestone History
- M0: Complete environment audit and repository inspection. Master Context created.
- M1: Minimal Gazebo demo world created (`demo_outdoor.sdf`) and validated.
- M2: ROS2 Simulation Bridge configured and validated with active message routing.
- M3: Existing navigation stack connected closed-loop with Gazebo UGV.
- M4: Static rock boulder avoidance validated with 1.10m clearance.

## Remaining Work
- M5: Integrate dynamic obstacle into Phase 13 pipeline with simulated sensor adapter (active and verified in node, now connecting operator interaction).
- M6: Implement keyboard teleoperation node for dynamic obstacle (`teleop_obstacle_keyboard.py` with WASD + Space + R).
- M7: Build comprehensive RViz2 judge-facing visualization configuration (`judge_demo.rviz`).
- M8: Build single-command demo launch file `interactive_judge_demo.launch.py`.
- M9: Execute 5 repeatable end-to-end interactive demo runs with reset ('R').
- Final Audit: Comprehensive verification and operator documentation.

## Next Action
Implement Milestone 5 & Milestone 6: create `simulation/nodes/teleop_obstacle_keyboard.py` to enable keyboard teleoperation of `/dynamic_obstacle/cmd_vel` (W/A/S/D to drive, Space to stop, R to publish `/demo/reset`) and validate dynamic obstacle interaction triggering DWA risk assessment.
