"""Tests for Phase 14.7.2 ROS2 Visualization and Telemetry Adapters.

Validates:
1. WorldTerrainMap -> nav_msgs/msg/OccupancyGrid conversion semantics
   - Unknown cell preservation (-1)
   - Free space mapping (0)
   - Lethal obstacle mapping (100)
   - Intermediate cost scaling (1..99)
   - Grid metadata, resolution, dimensions, origin, frame
   - Spatial coordinate indexing consistency
2. Canonical Mission Waypoint Visualization
   - Loading 5 canonical waypoints from configs/mission_phase14_5_100m.yaml
   - START + W1..W5 marker geometries (rings, beacons, text labels)
   - Dynamic mission state styling (ACTIVE, REACHED, NOT_REACHED, GOAL)
3. Non-fatal failure isolation
   - Empty/None map handling
   - Missing mission executive / uninitialized state
4. Conversion latency performance (< 2.0 ms target)
"""

import math
import os
import time
from pathlib import Path
import numpy as np
import pytest
import yaml

from ai.spatial_fusion.world_map import WorldTerrainMap

# Import adapters
from simulation.nodes.presentation_visualization_adapter import (
    CostmapOccupancyAdapter,
    WaypointMarkerAdapter,
    RobotDescriptionAdapter,
    PresentationTelemetryCoordinator,
)


# ============================================================================
# 1. COSTMAP OCCUPANCYGRID CONVERSION TESTS
# ============================================================================

def test_costmap_metadata_and_dimensions():
    """Verify that OccupancyGrid metadata accurately mirrors WorldTerrainMap geometry."""
    x_bounds = [-5.0, 115.0]
    y_bounds = [-15.0, 15.0]
    res = 0.10
    world_map = WorldTerrainMap(
        resolution_m=res,
        x_bounds_m=x_bounds,
        y_bounds_m=y_bounds,
        frame_id="odom",
    )

    msg = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(world_map, frame_id="odom")

    assert msg.header.frame_id == "odom"
    assert math.isclose(msg.info.resolution, res, rel_tol=1e-5)
    assert msg.info.width == world_map.n_rows  # X axis
    assert msg.info.height == world_map.n_cols  # Y axis
    assert math.isclose(msg.info.origin.position.x, x_bounds[0], rel_tol=1e-5)
    assert math.isclose(msg.info.origin.position.y, y_bounds[0], rel_tol=1e-5)
    assert msg.info.origin.position.z == 0.0
    assert msg.info.origin.orientation.w == 1.0
    assert len(msg.data) == world_map.n_rows * world_map.n_cols


def test_costmap_semantics_unknown_preservation():
    """CRITICAL: Unknown space must be preserved as -1 (UNKNOWN != SAFE)."""
    world_map = WorldTerrainMap(
        resolution_m=0.5,
        x_bounds_m=[0.0, 10.0],
        y_bounds_m=[0.0, 10.0],
    )
    # Default is unknown=True
    assert np.all(world_map.unknown)

    msg = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(world_map)
    data = np.array(msg.data)
    assert np.all(data == -1), "All unobserved cells must map to -1 in OccupancyGrid"


def test_costmap_semantics_free_and_lethal():
    """Verify 0 = free, 100 = lethal, and intermediate cost scaling."""
    world_map = WorldTerrainMap(
        resolution_m=0.5,
        x_bounds_m=[0.0, 10.0],
        y_bounds_m=[0.0, 10.0],
    )
    # Clear unknown flag for all cells
    world_map.unknown.fill(False)
    world_map.cost.fill(0.0)

    # Set specific cells
    # Cell 0,0: completely free
    world_map.cost[0, 0] = 0.0

    # Cell 1,1: obstacle flag True
    world_map.obstacle[1, 1] = True

    # Cell 2,2: high cost >= 0.95 (lethal)
    world_map.cost[2, 2] = 0.98

    # Cell 3,3: intermediate cost (0.475 -> ~50%)
    world_map.cost[3, 3] = 0.475

    msg = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(world_map)
    data_2d = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    # Note coordinates: world_map[r, c] maps to OccupancyGrid:
    # r is from x_max down to x_min -> flipped in X
    # c is from y_max down to y_min -> flipped in Y
    # F_cost = world_map.cost[::-1, ::-1].T
    # so F_cost has shape (n_cols, n_rows) = (height, width)
    # r=0 -> x_max, c=0 -> y_max. Flipped: ix = n_rows-1, iy = n_cols-1
    assert data_2d[world_map.n_cols - 1, world_map.n_rows - 1] == 0  # free cell

    # Cell 1,1: obstacle
    assert data_2d[world_map.n_cols - 2, world_map.n_rows - 2] == 100  # lethal obstacle

    # Cell 2,2: cost >= 0.95
    assert data_2d[world_map.n_cols - 3, world_map.n_rows - 3] == 100  # lethal cost

    # Cell 3,3: intermediate cost: 0.475 / 0.95 * 99 = 49.5 -> int 49
    val_mid = data_2d[world_map.n_cols - 4, world_map.n_rows - 4]
    assert 40 <= val_mid <= 60, f"Expected intermediate value ~49, got {val_mid}"


def test_costmap_spatial_alignment():
    """Verify that an obstacle at specific world coordinates (wx, wy) maps to the correct OccupancyGrid cell."""
    res = 0.20
    x_bounds = [-5.0, 15.0]
    y_bounds = [-10.0, 10.0]
    world_map = WorldTerrainMap(
        resolution_m=res,
        x_bounds_m=x_bounds,
        y_bounds_m=y_bounds,
    )
    world_map.unknown.fill(False)
    world_map.cost.fill(0.10)

    # Place an obstacle at world coordinates (4.10, 2.10) (cell center)
    wx, wy = 4.10, 2.10
    r, c = world_map.world_to_grid(wx, wy)
    world_map.obstacle[r, c] = True

    msg = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(world_map)

    # Expected index in ROS OccupancyGrid:
    expected_ix = int((wx - x_bounds[0]) / res)
    expected_iy = int((wy - y_bounds[0]) / res)

    # In row-major ROS data: index = iy * width + ix
    cell_val = msg.data[expected_iy * msg.info.width + expected_ix]
    assert cell_val == 100, f"Expected cell at ({expected_ix}, {expected_iy}) to be lethal (100), got {cell_val}"


# ============================================================================
# 2. CANONICAL WAYPOINT MARKER TESTS
# ============================================================================

def test_waypoint_marker_loading_from_yaml():
    """Verify loading the 5 canonical waypoints from mission YAML."""
    mission_file = "configs/mission_phase14_5_100m.yaml"
    assert os.path.exists(mission_file), f"Canonical mission file '{mission_file}' must exist"

    adapter = WaypointMarkerAdapter(node=None, mission_file=mission_file)
    assert len(adapter.waypoints) == 5, f"Expected 5 canonical waypoints, got {len(adapter.waypoints)}"

    expected = [
        {"id": "W1", "x": 20.0, "y": 0.0},
        {"id": "W2", "x": 40.0, "y": 6.0},
        {"id": "W3", "x": 60.0, "y": -4.0},
        {"id": "W4", "x": 80.0, "y": 3.0},
        {"id": "W5", "x": 100.0, "y": 0.0},
    ]

    for wp, exp in zip(adapter.waypoints, expected):
        assert wp["id"] == exp["id"]
        assert math.isclose(wp["x"], exp["x"], abs_tol=1e-3)
        assert math.isclose(wp["y"], exp["y"], abs_tol=1e-3)


def test_waypoint_marker_generation_counts_and_types():
    """Verify marker count, namespaces, and visual geometry types."""
    adapter = WaypointMarkerAdapter(node=None, mission_file="configs/mission_phase14_5_100m.yaml")
    m_array = adapter.generate_markers(
        active_wp_id="W1",
        reached_wp_ids=[],
        robot_pose=(0.0, 0.0, 0.0),
        frame_id="odom",
    )

    # 2 START markers (ring + text) + 5 waypoints * 3 markers (ring + beacon + text) = 17 markers
    assert len(m_array.markers) == 17

    namespaces = {m.ns for m in m_array.markers}
    expected_ns = {
        "mission_start",
        "mission_start_label",
        "waypoint_rings",
        "waypoint_beacons",
        "waypoint_labels",
    }
    assert expected_ns.issubset(namespaces)


def test_waypoint_state_styling_active_and_reached():
    """Verify styling changes dynamically based on active and reached states."""
    adapter = WaypointMarkerAdapter(node=None, mission_file="configs/mission_phase14_5_100m.yaml")

    # W1 reached, W2 active, W3..W5 unreached
    m_array = adapter.generate_markers(
        active_wp_id="W2",
        reached_wp_ids=["W1"],
        robot_pose=(25.0, 1.0, 0.0),
    )

    labels = {m.ns: m for m in m_array.markers if m.ns == "waypoint_labels"}
    # Find labels for W1, W2, W5
    text_w1 = [m.text for m in m_array.markers if m.ns == "waypoint_labels" and "W1" in m.text][0]
    text_w2 = [m.text for m in m_array.markers if m.ns == "waypoint_labels" and "W2" in m.text][0]
    text_w5 = [m.text for m in m_array.markers if m.ns == "waypoint_labels" and "W5" in m.text][0]

    assert "[REACHED] W1" in text_w1
    assert ">>> ACTIVE: W2 <<<" in text_w2
    assert "[GOAL] W5" in text_w5

    # Check color highlight on active beacon (W2)
    beacons = [m for m in m_array.markers if m.ns == "waypoint_beacons"]
    # W2 is index 1
    w2_beacon = beacons[1]
    # Active color should have high red and green (yellow/gold)
    assert w2_beacon.color.r >= 0.9 and w2_beacon.color.g >= 0.8


# ============================================================================
# 3. ROBOT DESCRIPTION & FAILURE ISOLATION TESTS
# ============================================================================

def test_robot_description_urdf_file():
    """Verify that UGV URDF file exists and contains essential elements."""
    urdf_path = "simulation/models/ugv/ugv.urdf"
    assert os.path.exists(urdf_path), f"URDF file '{urdf_path}' must exist"
    with open(urdf_path, "r") as f:
        content = f.read()
    assert "<robot" in content
    assert "base_link" in content


def test_failure_isolation_non_fatal():
    """Ensure that null or corrupted data does not raise uncaught exceptions in publishing methods."""
    # 1. Null world_map in maybe_publish
    costmap_adapter = CostmapOccupancyAdapter(node=None)
    result = costmap_adapter.maybe_publish(world_map=None, current_time_s=1.0)
    assert result is False

    # 2. ValueError on direct conversion of None
    with pytest.raises(ValueError):
        CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(None)

    # 3. Missing mission file fallback
    wp_adapter = WaypointMarkerAdapter(node=None, mission_file="nonexistent_mission.yaml")
    assert len(wp_adapter.waypoints) == 5, "Should fall back gracefully to 5 default waypoints"

    markers = wp_adapter.generate_markers()
    assert len(markers.markers) == 17


# ============================================================================
# 4. PERFORMANCE BENCHMARK TEST
# ============================================================================

def test_occupancy_conversion_performance():
    """Ensure vectorized WorldTerrainMap -> OccupancyGrid conversion meets latency target.
    
    Target:
    - 40,000 cell local active costmap (200x200): < 1.5 ms (target < 1-2 ms)
    - 360,000 cell full 100m corridor map: < 6.0 ms
    """
    # 1. Benchmark 40,000 cell local active map (200 x 200)
    map_local = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=[-5.0, 15.0],
        y_bounds_m=[-10.0, 10.0],
    )
    map_local.cost.fill(0.10)
    map_local.unknown.fill(False)
    map_local.obstacle[20:50, 20:50] = True

    # Warmup
    for _ in range(5):
        _ = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(map_local)

    times_local = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(map_local)
        times_local.append((time.perf_counter() - t0) * 1000.0)

    avg_local = float(np.mean(times_local))
    print(f"\n[BENCHMARK] 40,000 cell OccupancyGrid conversion: avg={avg_local:.2f} ms")
    assert avg_local < 1.5, f"Local map conversion too slow: avg={avg_local:.2f} ms (expected < 1.5 ms)"

    # 2. Benchmark 360,000 cell 100m corridor map (1200 x 300)
    map_100m = WorldTerrainMap(
        resolution_m=0.10,
        x_bounds_m=[-5.0, 115.0],
        y_bounds_m=[-15.0, 15.0],
    )
    map_100m.cost.fill(0.10)
    map_100m.unknown.fill(False)
    map_100m.obstacle[100:150, 50:80] = True

    for _ in range(5):
        _ = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(map_100m)

    times_100m = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(map_100m)
        times_100m.append((time.perf_counter() - t0) * 1000.0)

    avg_100m = float(np.mean(times_100m))
    print(f"\n[BENCHMARK] 360,000 cell OccupancyGrid conversion: avg={avg_100m:.2f} ms")
    assert avg_100m < 6.0, f"Full 100m map conversion too slow: avg={avg_100m:.2f} ms (expected < 6.0 ms)"
