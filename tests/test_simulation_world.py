"""Unit tests for Phase 11 Gazebo Outdoor Terrain World and Ground Truth Registry."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest


def test_outdoor_terrain_sdf_syntax_and_elements():
    """Verify simulation/worlds/outdoor_terrain.sdf is syntactically valid and has

    physics, sun, terrain, obstacles, and hazards.
    """
    world_path = Path(__file__).resolve().parent.parent / "simulation" / "worlds" / "outdoor_terrain.sdf"
    assert world_path.exists(), f"Missing world file at {world_path}"

    tree = ET.parse(world_path)
    root = tree.getroot()
    assert root.tag == "sdf"

    world = root.find("world")
    assert world is not None
    assert world.attrib.get("name") == "outdoor_terrain"

    # Physics check
    physics = world.find("physics")
    assert physics is not None
    assert float(physics.find("max_step_size").text) == 0.005
    assert float(physics.find("real_time_update_rate").text) == 200.0

    # Models check
    models = {m.attrib["name"]: m for m in world.findall("model")}
    expected_models = [
        "ground_plane",
        "rock_boulder_large",
        "tree_pine_trunk",
        "water_hazard_patch",
        "ditch_hazard_trench",
        "gravel_slope_ramp",
    ]
    for em in expected_models:
        assert em in models, f"Required world element {em} missing from outdoor_terrain.sdf"


def test_ground_truth_obstacle_registry_consistency():
    """Verify simulation/config/ground_truth_obstacles.json matches world geometry

    and defines valid metric bounding boxes and classifications.
    """
    gt_path = Path(__file__).resolve().parent.parent / "simulation" / "config" / "ground_truth_obstacles.json"
    assert gt_path.exists(), f"Missing ground truth file at {gt_path}"

    with open(gt_path, "r") as f:
        data = json.load(f)

    assert data["world_name"] == "outdoor_terrain"
    assert "obstacles" in data
    assert len(data["obstacles"]) >= 4

    obs_map = {o["id"]: o for o in data["obstacles"]}
    assert "obs_rock_large" in obs_map
    assert "obs_tree_pine" in obs_map
    assert "haz_water_puddle" in obs_map
    assert "haz_ditch_trench" in obs_map

    # Check Rock metrics
    rock = obs_map["obs_rock_large"]
    assert rock["position"] == [3.5, 0.5, 0.3]
    assert rock["bounding_radius_m"] == pytest.approx(0.60)
    assert rock["traversable"] is False
    assert rock["ground_truth_class"] == "obstacle"

    # Check Water hazard metrics
    water = obs_map["haz_water_puddle"]
    assert water["position"][0] == 4.0
    assert water["ground_truth_class"] == "hazard"
    assert water["expected_hazard"] >= 0.70

    # Check Safe region
    assert "safe_regions" in data
    assert len(data["safe_regions"]) >= 1
    safe = data["safe_regions"][0]
    assert safe["ground_truth_class"] == "safe_flat"
    assert safe["expected_cost"] <= 0.10
