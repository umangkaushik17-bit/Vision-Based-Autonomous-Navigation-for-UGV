"""Unit and integration tests for Phase 14.7.1 100m Outdoor Presentation World & Lightweight Assets."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import pytest


def test_outdoor_presentation_100m_sdf_syntax_and_elements():
    """Verify outdoor_presentation_100m.sdf is syntactically valid and contains all

    required Phase 14.7.1 world elements, plugins, and models.
    """
    repo_root = Path(__file__).resolve().parent.parent
    world_path = repo_root / "simulation" / "worlds" / "outdoor_presentation_100m.sdf"
    mission_path = repo_root / "configs" / "mission_phase14_5_100m.yaml"

    assert world_path.exists(), f"Missing presentation world file at {world_path}"
    assert mission_path.exists(), f"Missing canonical mission configuration at {mission_path}"

    with open(mission_path, "r") as f:
        mission_cfg = yaml.safe_load(f)["mission"]

    canonical_wps = {wp["id"]: (float(wp["x"]), float(wp["y"])) for wp in mission_cfg["waypoints"]}

    tree = ET.parse(world_path)
    root = tree.getroot()
    assert root.tag == "sdf", "Root element must be <sdf>"

    world = root.find("world")
    assert world is not None, "Missing <world> tag"
    assert world.attrib.get("name") == "outdoor_presentation_100m"

    # 1. Physics Engine Check
    physics = world.find("physics")
    assert physics is not None, "Missing <physics> configuration"
    assert physics.attrib.get("type") == "ode"
    assert float(physics.find("max_step_size").text) == 0.005
    assert float(physics.find("real_time_update_rate").text) == 200.0

    # 2. Scene Atmosphere Check
    scene = world.find("scene")
    assert scene is not None, "Missing <scene> tag"
    assert scene.find("ambient") is not None
    assert scene.find("background") is not None
    assert scene.find("sky") is not None
    assert scene.find("shadows").text.lower() == "true"

    # 3. Sunlight Check
    sun = world.find("light")
    assert sun is not None
    assert sun.attrib.get("type") == "directional"
    assert sun.find("cast_shadows").text.lower() == "true"

    # 4. Extract all models
    models = {m.attrib["name"]: m for m in world.findall("model")}

    # Base Meadow Plane
    assert "ground_plane" in models, "Missing ground_plane model"
    ground_link = models["ground_plane"].find("link")
    assert ground_link.find("collision") is not None, "Ground plane must have collision"

    # Waypoint Beacons (W1 through W5)
    for wid, (expected_x, expected_y) in canonical_wps.items():
        beacon_name = f"waypoint_beacon_{wid.lower()}"
        assert beacon_name in models, f"Missing beacon {beacon_name} in world"
        b_model = models[beacon_name]
        assert b_model.find("static").text.lower() == "true"
        pose_parts = b_model.find("pose").text.strip().split()
        actual_x = float(pose_parts[0])
        actual_y = float(pose_parts[1])
        assert actual_x == pytest.approx(expected_x, abs=0.50), f"{wid} X mismatch"
        assert actual_y == pytest.approx(expected_y, abs=2.00), f"{wid} Y mismatch"

        # Check collision-free
        link = b_model.find("link")
        assert link.find("collision") is None, f"{wid} beacon must be collision-free"

    # Continuous Trail Ribbon
    trail_segs = [
        "trail_seg_start_w1",
        "trail_seg_w1_w2",
        "trail_seg_w2_w3",
        "trail_seg_w3_w4",
        "trail_seg_w4_w5",
    ]
    for seg in trail_segs:
        assert seg in models, f"Missing trail segment {seg}"
        s_link = models[seg].find("link")
        assert s_link.find("collision") is None, f"Trail segment {seg} must be visual-only (no collision)"

    # Granite Boulders (Active Obstacles)
    boulders = [m for m in models if m.startswith("boulder_")]
    assert len(boulders) >= 6, f"Expected at least 6 boulders, found {len(boulders)}"
    for b_name in boulders:
        b_link = models[b_name].find("link")
        assert b_link.find("collision") is not None, f"Boulder {b_name} must have collision"
        col = b_link.find("collision")
        # Ensure primitive geometry (box or cylinder)
        geom = col.find("geometry")
        assert geom.find("box") is not None or geom.find("cylinder") is not None

    # Peripheral Vegetation (Collision-Free)
    trees = [m for m in models if m.startswith("tree_")]
    shrubs = [m for m in models if m.startswith("shrub_")]
    assert len(trees) + len(shrubs) >= 12, "Expected at least 12 vegetation items"
    for v_name in trees + shrubs:
        v_link = models[v_name].find("link")
        assert v_link.find("collision") is None, f"Vegetation {v_name} must be collision-free"

    # Dynamic Obstacles (W1 through W5 Multi-Obstacle Architecture)
    dyn_models = [m for m in models if m.startswith("dynamic_obs_") or m == "dynamic_obstacle"]
    assert len(dyn_models) >= 1, "Missing dynamic obstacles"
    for d_name in dyn_models:
        dyn_model = models[d_name]
        dyn_plugins = {p.attrib["name"]: p for p in dyn_model.findall("plugin")}
        assert "gz::sim::systems::VelocityControl" in dyn_plugins, f"Missing VelocityControl on {d_name}"
        assert "gz::sim::systems::OdometryPublisher" in dyn_plugins, f"Missing OdometryPublisher on {d_name}"

    # UGV 4WD
    assert "ugv_4wd" in models, "Missing ugv_4wd model"
    ugv_model = models["ugv_4wd"]
    ugv_plugins = {p.attrib["name"]: p for p in ugv_model.findall("plugin")}
    assert "gz::sim::systems::DiffDrive" in ugv_plugins, "Missing DiffDrive plugin on UGV"
    assert "gz::sim::systems::OdometryPublisher" in ugv_plugins, "Missing OdometryPublisher on UGV"
    assert "gz::sim::systems::JointStatePublisher" in ugv_plugins, "Missing JointStatePublisher on UGV"


def test_outdoor_assets_structure_and_triangle_budgets():
    """Verify simulation/models/outdoor_assets contains valid models with <1000 triangles."""
    assets_dir = Path(__file__).resolve().parent.parent / "simulation" / "models" / "outdoor_assets"
    assert assets_dir.exists(), f"Missing outdoor_assets directory at {assets_dir}"

    expected_assets = ["waypoint_beacon", "granite_boulder", "pine_tree", "bush_shrub"]
    for asset_name in expected_assets:
        asset_path = assets_dir / asset_name
        assert asset_path.exists(), f"Missing asset directory {asset_path}"
        assert (asset_path / "model.config").exists(), f"Missing model.config in {asset_name}"
        assert (asset_path / "model.sdf").exists(), f"Missing model.sdf in {asset_name}"

    # Check mesh triangle count for granite_boulder
    boulder_obj = assets_dir / "granite_boulder" / "meshes" / "boulder.obj"
    assert boulder_obj.exists(), f"Missing mesh {boulder_obj}"
    with open(boulder_obj, "r") as f:
        faces = [line for line in f if line.startswith("f ")]
    assert 0 < len(faces) < 1000, f"Boulder triangle count must be < 1000, got {len(faces)}"

    # Check mesh triangle count for pine_tree
    pine_obj = assets_dir / "pine_tree" / "meshes" / "pine_foliage.obj"
    assert pine_obj.exists(), f"Missing mesh {pine_obj}"
    with open(pine_obj, "r") as f:
        faces = [line for line in f if line.startswith("f ")]
    assert 0 < len(faces) < 1000, f"Pine foliage triangle count must be < 1000, got {len(faces)}"


def test_ground_truth_obstacles_100m_consistency():
    """Verify ground_truth_obstacles_100m.json matches world geometry and canonical mission."""
    repo_root = Path(__file__).resolve().parent.parent
    gt_path = repo_root / "simulation" / "config" / "ground_truth_obstacles_100m.json"
    mission_path = repo_root / "configs" / "mission_phase14_5_100m.yaml"

    assert gt_path.exists(), f"Missing ground truth file at {gt_path}"
    with open(gt_path, "r") as f:
        data = json.load(f)

    with open(mission_path, "r") as f:
        mission_cfg = yaml.safe_load(f)["mission"]

    assert data["world_name"] == "outdoor_presentation_100m"
    assert len(data["obstacles"]) >= 6
    assert len(data["waypoints"]) == 5

    # Check waypoint alignment
    canonical_wps = {wp["id"]: (float(wp["x"]), float(wp["y"])) for wp in mission_cfg["waypoints"]}
    for wp_item in data["waypoints"]:
        wid = wp_item["id"]
        assert wid in canonical_wps
        assert wp_item["position"][0] == pytest.approx(canonical_wps[wid][0], abs=0.01)
        assert wp_item["position"][1] == pytest.approx(canonical_wps[wid][1], abs=0.01)
