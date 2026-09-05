"""Unit tests for UGV core types and data structures."""

import numpy as np
import pytest

from ugv_core.types import (
    TraversabilityClass,
    RiskLevel,
    MissionState,
    Pose2D,
    ObstacleBBox,
    VelocityCommand,
    CostmapGrid
)
from ugv_core.logger import get_logger

def test_enums():
    assert TraversabilityClass.SAFE == 0
    assert TraversabilityClass.UNKNOWN == 1
    assert TraversabilityClass.UNSAFE == 2

    assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL
    assert MissionState.NAVIGATING.value == "NAVIGATING"

def test_pose2d():
    pose = Pose2D(x=1.5, y=-2.0, yaw=0.785)
    arr = pose.to_array()
    assert len(arr) == 3
    assert np.isclose(arr[0], 1.5)
    assert np.isclose(arr[1], -2.0)
    assert np.isclose(arr[2], 0.785)

def test_obstacle_bbox():
    bbox = ObstacleBBox(xmin=100, ymin=50, xmax=200, ymax=150, label="rock", confidence=0.92)
    assert bbox.width == 100
    assert bbox.height == 100
    assert bbox.center == (150.0, 100.0)

def test_velocity_command_clamp():
    cmd = VelocityCommand(linear_x=2.5, angular_z=-3.0)
    clamped = cmd.clamp(max_linear=1.0, max_angular=1.5)
    assert clamped.linear_x == 1.0
    assert clamped.angular_z == -1.5

def test_costmap_grid_transforms():
    grid_data = np.zeros((100, 100), dtype=np.uint8)
    costmap = CostmapGrid(
        grid=grid_data,
        resolution=0.1,  # 10cm / cell
        origin_x=-5.0,
        origin_y=-5.0,
        width_cells=100,
        height_cells=100
    )

    # Origin check (-5.0, -5.0) -> cell (0, 0)
    gx, gy = costmap.world_to_grid(-5.0, -5.0)
    assert (gx, gy) == (0, 0)

    # Center check (0.0, 0.0) -> cell (50, 50)
    gx, gy = costmap.world_to_grid(0.0, 0.0)
    assert (gx, gy) == (50, 50)

    # Inverse check: grid_to_world
    wx, wy = costmap.grid_to_world(50, 50)
    assert np.isclose(wx, 0.05)
    assert np.isclose(wy, 0.05)

    # Out of bounds check
    assert costmap.world_to_grid(10.0, 10.0) is None

def test_logger(tmp_path):
    log_file = tmp_path / "test_run.log"
    logger = get_logger("test_subsystem", log_file=str(log_file))
    logger.info("Test log message")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message" in content
