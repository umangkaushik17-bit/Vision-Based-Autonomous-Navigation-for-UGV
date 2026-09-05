"""Unit tests for Phase 11 Gazebo UGV model descriptions (SDF and URDF)."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest


def test_ugv_model_sdf_syntax_and_structure():
    """Verify simulation/models/ugv/model.sdf is syntactically valid and contains

    all required links, joints, sensors, and plugins.
    """
    sdf_path = Path(__file__).resolve().parent.parent / "simulation" / "models" / "ugv" / "model.sdf"
    assert sdf_path.exists(), f"Missing SDF file at {sdf_path}"

    tree = ET.parse(sdf_path)
    root = tree.getroot()
    assert root.tag == "sdf"
    assert root.attrib.get("version") == "1.9"

    model = root.find("model")
    assert model is not None
    assert model.attrib.get("name") == "ugv_4wd"

    # Verify Links
    links = {link.attrib["name"]: link for link in model.findall("link")}
    expected_links = [
        "base_link",
        "wheel_fl_link",
        "wheel_fr_link",
        "wheel_rl_link",
        "wheel_rr_link",
        "camera_link",
        "camera_optical",
        "imu_link",
    ]
    for el in expected_links:
        assert el in links, f"Required link {el} missing from model.sdf"

    # Verify Joints
    joints = {joint.attrib["name"]: joint for joint in model.findall("joint")}
    expected_joints = [
        "wheel_fl_joint",
        "wheel_fr_joint",
        "wheel_rl_joint",
        "wheel_rr_joint",
        "camera_joint",
        "camera_optical_joint",
        "imu_joint",
    ]
    for ej in expected_joints:
        assert ej in joints, f"Required joint {ej} missing from model.sdf"

    # Verify Camera Sensor in camera_optical
    cam_link = links["camera_optical"]
    cam_sensor = cam_link.find("sensor")
    assert cam_sensor is not None
    assert cam_sensor.attrib["type"] == "camera"
    cam = cam_sensor.find("camera")
    assert cam is not None
    img = cam.find("image")
    assert int(img.find("width").text) == 640
    assert int(img.find("height").text) == 480
    assert float(cam_sensor.find("update_rate").text) == 30.0

    # Verify IMU Sensor in imu_link
    imu_link = links["imu_link"]
    imu_sensor = imu_link.find("sensor")
    assert imu_sensor is not None
    assert imu_sensor.attrib["type"] == "imu"
    assert float(imu_sensor.find("update_rate").text) == 200.0

    # Verify Plugins
    plugins = [p.attrib.get("name") for p in model.findall("plugin")]
    assert "gz::sim::systems::DiffDrive" in plugins
    assert "gz::sim::systems::JointStatePublisher" in plugins
    assert "gz::sim::systems::Sensors" in plugins


def test_ugv_urdf_syntax_and_kinematics():
    """Verify simulation/models/ugv/ugv.urdf parses correctly and has valid

    mass, wheel parameters, and transforms.
    """
    urdf_path = Path(__file__).resolve().parent.parent / "simulation" / "models" / "ugv" / "ugv.urdf"
    assert urdf_path.exists(), f"Missing URDF file at {urdf_path}"

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib.get("name") == "ugv_4wd"

    links = {link.attrib["name"]: link for link in root.findall("link")}
    assert "base_link" in links
    assert "camera_link" in links
    assert "camera_optical" in links
    assert "imu_link" in links

    # Check Total Mass calculation
    total_mass = 0.0
    for link in links.values():
        inertial = link.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                total_mass += float(mass_el.attrib.get("value", 0.0))

    assert total_mass == pytest.approx(33.3, abs=0.5)  # 25 chassis + 4*2 wheels + mounts

    # Check Camera Offset and Pitch
    joints = {j.attrib["name"]: j for j in root.findall("joint")}
    cam_joint = joints["camera_joint"]
    origin = cam_joint.find("origin")
    xyz = [float(v) for v in origin.attrib["xyz"].split()]
    rpy = [float(v) for v in origin.attrib["rpy"].split()]

    assert xyz == pytest.approx([0.15, 0.0, 0.40], abs=1e-4)
    assert rpy[1] == pytest.approx(math.radians(15.0), abs=1e-4)
