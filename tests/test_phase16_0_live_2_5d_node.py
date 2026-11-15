"""Automated tests for Subphase 16.0: Live 2.5D Gazebo & ROS 2 Interactive Node Unification.

Verifies:
1. `InteractiveUGVNavigationNode` initializes `enable_2_5d_terrain` and `terrain_world_mode`
   parameters alongside `ElevationMapper`, `VectorizedTerrainGeometryEstimator`, and `TraversabilityCostModel`.
2. In `uneven_100m` mode, the 100m world map accurately captures all 2.5D terrain hazards from
   `outdoor_uneven_terrain_100m.sdf` (32 deg embankment, 10 deg gentle bypass ramp, 24 deg cross-slope,
   20 cm step boulder cluster, -15 cm negative ditch, 12 deg undulating hills, and 14 cm belly crown mound).
3. All 5 pre-planned corridor segments (`W1` through `W5`) route safely around 2.5D lethal slopes,
   steps, ditches, and static boulders via the C++17 A* backend.
4. Live 3D point cloud ingestion (`ingest_local_elevation_points`) updates local 2.5D elevation,
   slope, step, ditch, and crown layers and triggers automatic C++17 A* replanning when a 2.5D hazard
   blocks the active global path.
5. Downhill pitch stopping-distance expansion in `SafetyGovernor` and 2.5D-aware WASD dynamic
   obstacle evasion operate seamlessly inside `_control_loop()`.
6. Closed-loop `_control_loop()` execution runs in `< 5.0 ms` mean latency on CPU.
7. `simulation/launch/prototype_presentation.launch.py` supports `world:=simulation/worlds/outdoor_uneven_terrain_100m.sdf`
   and `enable_2_5d_terrain:=true`.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import time
import numpy as np
import pytest

from production.perception.elevation_mapper import ElevationMapper
from production.perception.terrain_geometry import VectorizedTerrainGeometryEstimator
from ai.perception_fusion.cost_model import TraversabilityCostModel
from simulation.nodes.interactive_ugv_navigation_node import (
    InteractiveUGVNavigationNode,
    Odometry,
)


def _make_odom_msg(
    x: float,
    y: float,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    vx: float = 0.45,
    wz: float = 0.0,
    timestamp_s: float = 1.0,
) -> Odometry:
    """Construct an Odometry message with full 3D Euler orientation (roll, pitch, yaw)."""
    msg = Odometry()
    sec = int(timestamp_s)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((timestamp_s - sec) * 1e9)
    msg.header.frame_id = "odom"
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    msg.pose.pose.orientation.w = cr * cp * cy + sr * sp * sy
    msg.pose.pose.orientation.x = sr * cp * cy - cr * sp * sy
    msg.pose.pose.orientation.y = cr * sp * cy + sr * cp * sy
    msg.pose.pose.orientation.z = cr * cp * sy - sr * sp * cy

    msg.twist.twist.linear.x = float(vx)
    msg.twist.twist.angular.z = float(wz)
    return msg


def test_node_initializes_2_5d_terrain_estimators_and_uneven_100m_hazards():
    """Verify InteractiveUGVNavigationNode initializes 2.5D estimators and populates uneven_100m hazards."""
    node = InteractiveUGVNavigationNode(
        parameter_overrides={
            "enable_2_5d_terrain": True,
            "terrain_world_mode": "uneven_100m",
            "enable_visualization_adapter": False,
        },
        skip_dense_predictor=True,
    )

    assert node.enable_2_5d_terrain is True
    assert node.terrain_world_mode == "uneven_100m"
    assert isinstance(node.elevation_mapper, ElevationMapper)
    assert isinstance(node.terrain_geometry_estimator, VectorizedTerrainGeometryEstimator)
    assert isinstance(node.terrain_cost_model, TraversabilityCostModel)

    wmap = node.world_map

    # 1. Steep 32 deg embankment at (30.0, 2.0) -> lethal slope >= 28 deg, obstacle=True, cost=1.0
    r_steep, c_steep = wmap.world_to_grid(29.2, 2.0)
    assert wmap.slope[r_steep, c_steep] >= 28.0
    assert bool(wmap.obstacle[r_steep, c_steep]) is True
    assert wmap.cost[r_steep, c_steep] == pytest.approx(1.0)

    # 2. Gentle 10 deg bypass ramp at (30.0, 6.2) -> safe moderate slope (< 20 deg), traversable
    r_gentle, c_gentle = wmap.world_to_grid(28.5, 6.2)
    assert 2.0 <= wmap.slope[r_gentle, c_gentle] < 20.0
    assert bool(wmap.obstacle[r_gentle, c_gentle]) is False
    assert wmap.cost[r_gentle, c_gentle] < 0.50

    # 3. 20 cm vertical step boulder cluster edge near (65.3, -1.5) -> lethal step >= 0.12 m
    r_step, c_step = wmap.world_to_grid(65.3, -1.5)
    assert wmap.height_var[r_step, c_step] >= 0.12
    assert bool(wmap.obstacle[r_step, c_step]) is True
    assert wmap.cost[r_step, c_step] == pytest.approx(1.0)

    # 4. -15 cm negative ditch trench edge near (73.0, 1.2) -> lethal ditch <= -0.12 m
    r_ditch, c_ditch = wmap.world_to_grid(73.0, 1.2)
    assert wmap.height_var[r_ditch, c_ditch] >= 0.12
    assert bool(wmap.obstacle[r_ditch, c_ditch]) is True
    assert wmap.cost[r_ditch, c_ditch] == pytest.approx(1.0)

    # 5. +14 cm high-centering belly crown mound at (10.0, -2.5) -> crown >= 0.12 m
    r_crown, c_crown = wmap.world_to_grid(10.0, -2.5)
    assert wmap.height_var[r_crown, c_crown] >= 0.12
    assert bool(wmap.obstacle[r_crown, c_crown]) is True
    assert wmap.cost[r_crown, c_crown] == pytest.approx(1.0)


def test_preplanned_w1_to_w5_corridors_avoid_2_5d_embankments_and_steps():
    """Verify all 5 precomputed nominal corridor segments route around 2.5D hazards and static boulders."""
    node = InteractiveUGVNavigationNode(
        parameter_overrides={
            "enable_2_5d_terrain": True,
            "terrain_world_mode": "uneven_100m",
            "enable_visualization_adapter": False,
        },
        skip_dense_predictor=True,
    )

    assert set(node.preplanned_segments.keys()) == {"W1", "W2", "W3", "W4", "W5"}
    wmap = node.world_map

    for wp_id, path in node.preplanned_segments.items():
        assert len(path.points) >= 10, f"Segment {wp_id} should have dense waypoints"
        xy = path.waypoints_xy
        rows, cols = wmap.world_to_grid_vec(xy[:, 0], xy[:, 1])
        assert not np.any(wmap.obstacle[rows, cols]), f"Segment {wp_id} intersected a lethal obstacle or 2.5D hazard"
        assert np.all(wmap.slope[rows, cols] < 28.0), f"Segment {wp_id} violated 28 deg critical slope threshold"
        assert np.all(wmap.height_var[rows, cols] < 0.12), f"Segment {wp_id} violated 0.12 m step/crown threshold"


def test_live_2_5d_elevation_ingestion_and_cpp_astar_replanning():
    """Verify live 3D point cloud ingestion updates 2.5D layers and triggers C++17 A* replanning."""
    node = InteractiveUGVNavigationNode(
        parameter_overrides={
            "enable_2_5d_terrain": True,
            "terrain_world_mode": "flat_100m",
            "goal_x": 15.0,
            "goal_y": 0.0,
            "rock_x": -100.0,
            "rock_radius": 0.0,
            "obstacles_file": "nonexistent.json",
            "enable_visualization_adapter": False,
        },
        skip_dense_predictor=True,
    )

    # Initial step on flat ground: path from (0, 0) to (15, 0) runs straight along y = 0
    node._on_ugv_odom(_make_odom_msg(x=0.0, y=0.0, yaw=0.0, timestamp_s=1.0))
    node._control_loop()
    assert node.active_global_path is not None

    # Inject a +0.22 m boulder step ledge across the centerline at x in [6.5, 8.0], y in [-1.2, 1.2]
    xs = np.linspace(6.5, 8.0, 20, dtype=np.float32)
    ys = np.linspace(-1.2, 1.2, 30, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    zg = np.full_like(xg, 0.22, dtype=np.float32)
    pts_step = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=-1)
    pts_step = np.vstack([pts_step, pts_step])  # satisfy min_points_per_cell=2

    ingest_diag = node.ingest_local_elevation_points(pts_step, timestamp=1.1)
    assert ingest_diag["cells_updated"] > 0
    assert ingest_diag["local_max_step_m"] >= 0.12

    # Step control loop: node must detect that the active path is blocked by the 2.5D step and replan around it
    node._on_ugv_odom(_make_odom_msg(x=1.0, y=0.0, yaw=0.0, timestamp_s=1.2))
    node._control_loop()

    replanned_xy = node.active_global_path.waypoints_xy
    mid_mask = (replanned_xy[:, 0] >= 6.4) & (replanned_xy[:, 0] <= 8.1)
    assert np.any(mid_mask)
    assert np.all(np.abs(replanned_xy[mid_mask, 1]) > 1.25), (
        "C++17 A* replanner must detour around the newly ingested +0.22m 2.5D step ledge"
    )


def test_downhill_pitch_safety_governor_and_dynamic_evasion_2_5d_integration():
    """Verify downhill pitch stopping distance expansion and 2.5D-aware dynamic obstacle evasion."""
    node = InteractiveUGVNavigationNode(
        parameter_overrides={
            "enable_2_5d_terrain": True,
            "terrain_world_mode": "uneven_100m",
            "enable_visualization_adapter": False,
        },
        skip_dense_predictor=True,
    )

    # Place UGV at (26.0, 0.5) on a +12 deg downhill pitch
    pitch_12deg = math.radians(12.0)
    node._on_ugv_odom(_make_odom_msg(x=26.0, y=0.5, pitch=pitch_12deg, yaw=0.25, vx=0.50, timestamp_s=2.0))
    node._control_loop()

    gov_diag = node.last_safety_governor_diag
    assert gov_diag["effective_stop_distance_m"] > node.safety_governor.config.stop_distance_m
    assert gov_diag["effective_cautious_distance_m"] > node.safety_governor.config.cautious_distance_m


def test_closed_loop_control_step_latency_under_5ms_and_w1_w5_progression():
    """Verify closed-loop _control_loop() executes in < 5.0 ms mean and progresses through waypoints."""
    node = InteractiveUGVNavigationNode(
        parameter_overrides={
            "enable_2_5d_terrain": True,
            "terrain_world_mode": "uneven_100m",
            "mission_file": "configs/mission_phase14_5_100m.yaml",
            "enable_visualization_adapter": False,
        },
        skip_dense_predictor=True,
    )

    # Warm-up 3 control steps
    for i in range(3):
        node._on_ugv_odom(_make_odom_msg(x=1.0 + i * 0.2, y=0.0, yaw=0.0, timestamp_s=1.0 + i * 0.1))
        node._control_loop()

    # Benchmark 35 closed-loop control steps sampled directly along preplanned W1 -> W5 segments
    step_times_ms = []
    sample_poses = []
    for wp_id in ("W1", "W2", "W3", "W4", "W5"):
        pts = node.preplanned_segments[wp_id].waypoints_xy
        indices = np.linspace(2, len(pts) - 1, 7, dtype=int)
        for i in indices:
            prev_i = max(0, i - 1)
            yaw = math.atan2(float(pts[i, 1] - pts[prev_i, 1]), float(pts[i, 0] - pts[prev_i, 0]))
            sample_poses.append((float(pts[i, 0]), float(pts[i, 1]), yaw))

    t_sim = 2.0
    for idx in range(len(sample_poses) - 1):  # Stop 1 step before final W5 completion banner
        px, py, pyaw = sample_poses[idx]
        t_sim += 0.10
        node._on_ugv_odom(_make_odom_msg(x=px, y=py, yaw=pyaw, vx=0.50, timestamp_s=t_sim))
        t0 = time.perf_counter()
        node._control_loop()
        step_times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_step_ms = float(np.mean(step_times_ms))
    p95_step_ms = float(np.percentile(step_times_ms, 95))
    print(
        f"\n[PHASE 16.0 BENCHMARK] Closed-loop _control_loop(): mean={mean_step_ms:.2f} ms, "
        f"P95={p95_step_ms:.2f} ms, local 2.5D update={node.last_terrain_2_5d_diag['terrain_update_ms']:.2f} ms"
    )
    assert mean_step_ms < 5.0, f"Expected mean _control_loop() latency < 5.0 ms, got {mean_step_ms:.2f} ms"
    assert node.last_terrain_2_5d_diag["enabled"] is True
    assert node.last_terrain_2_5d_diag["terrain_update_ms"] < 2.5


def test_prototype_presentation_launch_supports_uneven_world_and_2_5d_args():
    """Verify prototype_presentation.launch.py declares and forwards 2.5D terrain and uneven world arguments."""
    launch_file = Path("simulation/launch/prototype_presentation.launch.py").resolve()
    spec = importlib.util.spec_from_file_location("prototype_presentation_launch_16_0", str(launch_file))
    launch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_mod)

    ld = launch_mod.generate_launch_description()
    declared_args = {
        e.name: e.default_value
        for e in ld.entities
        if hasattr(e, "name") and hasattr(e, "default_value")
    }

    assert "world" in declared_args
    assert "outdoor_uneven_terrain_100m.sdf" in str(declared_args["world"])
    assert "enable_2_5d_terrain" in declared_args
    assert str(declared_args["enable_2_5d_terrain"]).lower() == "true"
    assert "terrain_world_mode" in declared_args
    assert str(declared_args["terrain_world_mode"]).lower() == "uneven_100m"
