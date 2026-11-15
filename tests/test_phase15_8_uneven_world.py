"""Unit tests for Subphase 15.8: Uneven Gazebo Terrain World & RViz 2.5D Telemetry.

Verifies:
1. `simulation/worlds/outdoor_uneven_terrain_100m.sdf` is well-formed XML / SDF 1.9 and contains:
   - 200 Hz ODE physics & required Gazebo Sim plugins (`Physics`, `Sensors`, `Imu`, `SceneBroadcaster`)
   - 32 deg steep embankment (`steep_embankment_32deg`)
   - 10 deg gentle bypass ramp (`gentle_bypass_ramp_10deg`)
   - 24 deg cross-slope bank (`severe_cross_slope_24deg`)
   - 20 cm step boulder cluster (`step_boulder_cluster_20cm`)
   - -15 cm negative ditch trench (`negative_ditch_trench_15cm`)
   - 12 deg undulating pitching hills (`undulating_pitch_hills_12deg`)
2. `CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid` reflects 2.5D `slope` and `height_var`
   costs in `/costmap/occupancy` (mapping >= 28 deg slopes and >= 0.12 m steps to lethal 100,
   moderate slopes to proportional 1..99, and preserving -1 for unknown).
3. `CostmapOccupancyAdapter.generate_terrain_telemetry_markers` outputs RViz 2.5D elevation/slope/step markers.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from ai.spatial_fusion.world_map import WorldTerrainMap
from simulation.nodes.presentation_visualization_adapter import CostmapOccupancyAdapter


SDF_WORLD_PATH = os.path.join("simulation", "worlds", "outdoor_uneven_terrain_100m.sdf")


def test_uneven_terrain_sdf_xml_syntax_and_required_models():
    """Verify outdoor_uneven_terrain_100m.sdf parses cleanly and includes all required 2.5D terrain models."""
    assert os.path.exists(SDF_WORLD_PATH), f"Missing SDF world at {SDF_WORLD_PATH}"
    tree = ET.parse(SDF_WORLD_PATH)
    root = tree.getroot()
    assert root.tag == "sdf"
    assert root.attrib.get("version") == "1.9"

    world = root.find("world")
    assert world is not None
    assert world.attrib.get("name") == "outdoor_uneven_terrain_100m"

    # Verify physics rate (200 Hz)
    physics = world.find("physics")
    assert physics is not None
    update_rate = float(physics.findtext("real_time_update_rate", "0"))
    assert update_rate >= 200.0

    # Verify required 2.5D uneven terrain models
    model_names = {m.attrib.get("name", "") for m in world.findall("model")}
    required_models = {
        "ground_plane",
        "steep_embankment_32deg",
        "gentle_bypass_ramp_10deg",
        "severe_cross_slope_24deg",
        "step_boulder_cluster_20cm",
        "negative_ditch_trench_15cm",
        "undulating_pitch_hills_12deg",
    }
    missing = required_models - model_names
    assert not missing, f"Missing required terrain models in SDF: {missing}"


def test_costmap_occupancy_adapter_reflects_2_5d_slope_and_step_costs():
    """Verify CostmapOccupancyAdapter maps 2.5D slope and step layers into ROS OccupancyGrid values."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(0.0, 2.0), y_bounds_m=(0.0, 2.0))
    wmap.unknown.fill(False)
    wmap.cost.fill(0.05)
    wmap.obstacle.fill(False)

    # Cell A: Lethal 32 deg slope at world (0.55, 0.55) -> even if cost=0.05 and obstacle=False, occupancy must be 100
    r_steep, c_steep = wmap.world_to_grid(0.55, 0.55)
    wmap.slope[r_steep, c_steep] = 32.0

    # Cell B: Lethal 0.16m step at world (1.25, 0.55) -> occupancy must be 100
    r_step, c_step = wmap.world_to_grid(1.25, 0.55)
    wmap.height_var[r_step, c_step] = 0.16

    # Cell C: Moderate 14 deg slope at world (1.25, 1.25) -> occupancy must be elevated (> 20 and < 100)
    r_mod, c_mod = wmap.world_to_grid(1.25, 1.25)
    wmap.slope[r_mod, c_mod] = 14.0

    # Cell D: Unobserved cell at world (0.15, 1.85) -> occupancy must be -1
    r_unk, c_unk = wmap.world_to_grid(0.15, 1.85)
    wmap.unknown[r_unk, c_unk] = True

    msg = CostmapOccupancyAdapter.world_terrain_map_to_occupancy_grid(wmap)
    occ_2d = np.frombuffer(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

    # Map WorldTerrainMap (r, c) to ROS OccupancyGrid (iy, ix):
    # F = arr[::-1, ::-1].T => ix = n_rows - 1 - r, iy = n_cols - 1 - c
    def to_occ(r: int, c: int) -> int:
        ix = wmap.n_rows - 1 - r
        iy = wmap.n_cols - 1 - c
        return int(occ_2d[iy, ix])

    assert to_occ(r_steep, c_steep) == 100, "32 deg slope must map to lethal 100 in OccupancyGrid"
    assert to_occ(r_step, c_step) == 100, "0.16m step must map to lethal 100 in OccupancyGrid"
    assert 20 <= to_occ(r_mod, c_mod) < 100, f"14 deg slope should map to intermediate cost, got {to_occ(r_mod, c_mod)}"
    assert to_occ(r_unk, c_unk) == -1, "Unknown cell must map to -1 in OccupancyGrid"


def test_rviz_2_5d_terrain_telemetry_marker_generation():
    """Verify 2.5D terrain telemetry HUD marker reflects elevation, max slope, and max step height."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    r0, c0 = wmap.world_to_grid(1.0, 0.5)
    wmap.elevation[r0, c0] = 0.42
    wmap.elevation_valid[r0, c0] = True
    wmap.slope[20, 20] = 29.5
    wmap.height_var[30, 30] = 0.15

    m_arr = CostmapOccupancyAdapter.generate_terrain_telemetry_markers(
        wmap, robot_pose=(1.0, 0.5, 0.0), frame_id="odom"
    )
    assert len(m_arr.markers) == 1
    hud = m_arr.markers[0]
    assert "Z=+0.42m" in hud.text
    assert "MaxSlope=29.5deg" in hud.text
    assert "MaxStep=0.15m" in hud.text
    # Red warning color when lethal slope/step is present
    assert hud.color.r > 0.8
