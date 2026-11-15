# Phase 14 Consolidated Master Reference: GPS-Denied 100m Mission Executive, Vision-in-the-Loop & 15D EKF Localization

**Project**: SIH26126 — Vision-Based Autonomous Navigation for UGV in Outdoor Environments  
**Document**: `docs/PHASE_14_CONSOLIDATED_MASTER.md`  
**Primary Benchmark Report**: [`reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md`](../reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md)  
**Archived Subphase Reports**: [`reports/phase_14/`](../reports/phase_14/)  
**Status**: **COMPLETE & VERIFIED (`100/100` Core Tests PASS + `12/12` 100m Corridor Validation Runs PASS)**  

---

## 1. Phase 14 Subphase Architecture & Deliverables (14.1 to 14.7.4)

| Subphase | Title | Primary Files | Key Metrics & Verification | Archived Report |
| :---: | :--- | :--- | :--- | :--- |
| **14.1** | Multi-Waypoint Mission Executive & Recovery Orchestrator | `src/ugv_mission/mission_executive.py`, `waypoint_manager.py`, `recovery_orchestrator.py` | Ordered progression `START -> W1 -> W2 -> W3 -> W4 -> W5`, stuck detection, `15/15` tests PASS | [`reports/phase_14/PHASE_14_1_MISSION_EXECUTIVE_IMPLEMENTATION_REPORT.md`](../reports/phase_14/PHASE_14_1_MISSION_EXECUTIVE_IMPLEMENTATION_REPORT.md) |
| **14.2** | Realtime Production Perception Pipeline | `production/perception/realtime_pipeline.py`, `adaptive_predictor.py`, `temporal_filter.py` | ONNX Runtime INT8/FP32 inference (`12.5 ms`), blackout protection, `12/12` tests PASS | [`reports/phase_14/PHASE_14_2_REALTIME_PERCEPTION_REPORT.md`](../reports/phase_14/PHASE_14_2_REALTIME_PERCEPTION_REPORT.md) |
| **14.3** | Closed-Loop Vision-in-the-Loop BEV Costmap Integrator | `production/perception/vision_costmap_integrator.py` | Precomputed ray LUT + SE(3) attitude compensation (`1.61 ms`), 5-frame clear hysteresis, `21/21` PASS | [`reports/phase_14/PHASE_14_3_VISION_IN_THE_LOOP_REPORT.md`](../reports/phase_14/PHASE_14_3_VISION_IN_THE_LOOP_REPORT.md) |
| **14.4** | 15-State GPS-Denied Multi-Sensor EKF Localization | `production/localization/ekf_fusion.py`, `stereo_vio_adapter.py` | Fuses Wheel Odom + 200 Hz 6-Axis IMU + Stereo VIO; **`0.01%` drift over $1,280\,\text{m}$**, `23/23` PASS | [`reports/phase_14/PHASE_14_4_LOCALIZATION_IMPLEMENTATION_REPORT.md`](../reports/phase_14/PHASE_14_4_LOCALIZATION_IMPLEMENTATION_REPORT.md) |
| **14.5** | Long-Horizon 100m Outdoor Benchmark | `benchmarks/run_phase14_5_benchmark.py`, `configs/mission_phase14_5_100m.yaml` | 10-run 100m evaluation (`1,280 m` total distance, `0` collisions, `3.12 ms` mean control step) | [`reports/phase_14/PHASE_14_5_LONG_HORIZON_BENCHMARK_REPORT.md`](../reports/phase_14/PHASE_14_5_LONG_HORIZON_BENCHMARK_REPORT.md) |
| **14.7.1** | 100m Outdoor Presentation World & Low-Poly Assets | `simulation/worlds/outdoor_presentation_100m.sdf`, `simulation/models/outdoor_assets/` | 200 Hz ODE physics, 5 luminous waypoint beacons, 8 granite boulders, 5 dynamic rovers | [`reports/phase_14/PHASE_14_7_1_WORLD_IMPLEMENTATION_REPORT.md`](../reports/phase_14/PHASE_14_7_1_WORLD_IMPLEMENTATION_REPORT.md) |
| **14.7.2** | ROS 2 Presentation Visualization & Telemetry Adapters | `simulation/nodes/presentation_visualization_adapter.py` | Vectorized `/costmap/occupancy` (`< 1.5 ms`), `/mission/waypoint_markers`, latched `/robot_description` | [`reports/phase_14/PHASE_14_7_2_ROS2_VISUALIZATION_REPORT.md`](../reports/phase_14/PHASE_14_7_2_ROS2_VISUALIZATION_REPORT.md) |
| **14.7.3** | Hero RViz2 Profile & Unified 1-Command Launch | `simulation/rviz/hero_presentation.rviz`, `simulation/launch/prototype_presentation.launch.py` | Oblique chase camera tracking `base_link`, non-fatal RViz failure isolation | [`reports/phase_14/PHASE_14_7_3_HERO_RVIZ_LAUNCH_REPORT.md`](../reports/phase_14/PHASE_14_7_3_HERO_RVIZ_LAUNCH_REPORT.md) |
| **14.7.4** | Full 100m Multi-Obstacle Closed-Loop Video Validation | `benchmarks/run_phase14_7_4_validation.py`, `simulation/nodes/interactive_ugv_navigation_node.py` | **12/12 runs PASS (`100%` completion)** across `W1 -> W5` (`0` collisions, `0.01%` EKF drift) | [`reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md`](../reports/PHASE_14_7_4_FULL_VIDEO_VALIDATION_REPORT.md) |

---

## 2. Canonical 100m 5-Waypoint Course Layout (`configs/mission_phase14_5_100m.yaml`)

- **`START` `(0.0, 0.0)` $\to$ `W1` `(20.0, 0.0)`:** Straight opening corridor with oncoming patrol rover `dynamic_obs_w1`.
- **`W1` `(20.0, 0.0)` $\to$ `W2` `(40.0, 6.0)`:** Lateral left switchback detouring around granite boulder cluster and crossing rover `dynamic_obs_w2`.
- **`W2` `(40.0, 6.0)` $\to$ `W3` `(60.0, -4.0)`:** Valley traversal avoiding lateral crossing rover `dynamic_obs_w3`.
- **`W3` `(60.0, -4.0)` $\to$ `W4` `(80.0, 3.0)`:** Forest chicane passage with diagonal intercept rover `dynamic_obs_w4`.
- **`W4` `(80.0, 3.0)` $\to$ `W5` `(100.0, 0.0)`:** Final gateway approach past sentinel rover `dynamic_obs_w5` and north/south gateway boulders.
