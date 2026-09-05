"""Unit tests for UGV ROS2 custom message data structures and serializers."""

from __future__ import annotations

import numpy as np
import pytest

from ai.costmap.base import CostmapMetadata
from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg, Point3D, Quaternion
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.messages.serialization import MessageSerializer
from ugv_navigation.utils.time_utils import RosHeader, RosTime


def test_world_terrain_map_msg_construction_and_arrays():
    """Test WorldTerrainMapMsg creation from Phase 9 WorldTerrainMap and array conversion."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0))
    # Inject an obstacle and hazard
    r_obs, c_obs = wmap.world_to_grid(2.0, 1.0)
    wmap.obstacle[r_obs, c_obs] = True
    wmap.cost[r_obs, c_obs] = 1.0
    r_haz, c_haz = wmap.world_to_grid(-1.0, 3.0)
    wmap.hazard[r_haz, c_haz] = 0.85
    wmap.cost[r_haz, c_haz] = 1.0

    msg = WorldTerrainMapMsg.from_world_terrain_map(wmap, frame_id="odom", timestamp_s=123.456)
    assert msg.header.frame_id == "odom"
    assert abs(msg.header.stamp.to_sec() - 123.456) < 1e-6
    assert msg.resolution == 0.10
    assert msg.width == 100
    assert msg.height == 100
    assert len(msg.cost) == 10000
    assert len(msg.obstacle) == 10000
    assert len(msg.hazard) == 10000

    arrays = msg.to_arrays()
    assert arrays["cost"].shape == (100, 100)
    assert arrays["obstacle"][r_obs, c_obs] is np.True_ or arrays["obstacle"][r_obs, c_obs] == 1
    assert arrays["hazard"][r_haz, c_haz] == pytest.approx(0.85, abs=1e-4)


def test_world_terrain_map_serialization_json():
    """Test JSON serialization round-trip for WorldTerrainMapMsg."""
    wmap = WorldTerrainMap(resolution_m=0.20, x_bounds_m=(-2.0, 2.0), y_bounds_m=(-2.0, 2.0))
    msg = WorldTerrainMapMsg.from_world_terrain_map(wmap, frame_id="odom", timestamp_s=10.0)

    payload = MessageSerializer.serialize_terrain_map_json(msg)
    assert isinstance(payload, str)

    deserialized = MessageSerializer.deserialize_terrain_map_json(payload)
    assert deserialized.header.frame_id == "odom"
    assert deserialized.header.stamp.sec == 10
    assert deserialized.resolution == pytest.approx(0.20)
    assert deserialized.width == msg.width
    assert deserialized.height == msg.height
    assert len(deserialized.cost) == len(msg.cost)


def test_world_terrain_map_serialization_binary():
    """Test compressed binary serialization round-trip for WorldTerrainMapMsg."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-2.0, 2.0), y_bounds_m=(-2.0, 2.0))
    msg = WorldTerrainMapMsg.from_world_terrain_map(wmap, frame_id="odom", timestamp_s=42.125)

    bin_data = MessageSerializer.serialize_terrain_map_binary(msg, compress=True)
    assert bin_data.startswith(b"ZLIB")

    recovered = MessageSerializer.deserialize_terrain_map_binary(bin_data)
    assert recovered.header.frame_id == "odom"
    assert recovered.header.stamp.to_sec() == pytest.approx(42.125, abs=1e-6)
    assert recovered.width == 40
    assert recovered.height == 40
    assert len(recovered.cost) == 1600
    assert recovered.cost == msg.cost


def test_localization_status_msg_conversion():
    """Test LocalizationStatusMsg conversion from Phase 8 LocalizationEstimate."""
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[0, 3] = 3.5
    pose_mat[1, 3] = -1.2
    pose_mat[2, 3] = 0.4

    est = LocalizationEstimate(
        pose=pose_mat,
        confidence=0.88,
        tracking_state=TrackingState.TRACKING_OK,
        timestamp=105.5,
        linear_velocity=np.array([0.5, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.1]),
        tracked_features=85,
        inliers=45,
        keyframe_id=12,
        is_keyframe=True,
        relocalized=False,
    )

    msg = LocalizationStatusMsg.from_localization_estimate(est, frame_id="odom", child_frame_id="base_link")
    assert msg.header.frame_id == "odom"
    assert msg.child_frame_id == "base_link"
    assert msg.tracking_state == LocalizationStatusMsg.TRACKING_OK
    assert msg.confidence == pytest.approx(0.88)
    assert msg.is_pose_valid is True
    assert msg.allows_spatial_fusion is True
    assert msg.pose.pose.position.x == pytest.approx(3.5)
    assert msg.pose.pose.position.y == pytest.approx(-1.2)
    assert msg.twist.twist.linear.x == pytest.approx(0.5)

    # Convert back to LocalizationEstimate
    recovered_est = msg.to_localization_estimate()
    assert np.allclose(recovered_est.pose, est.pose, atol=1e-5)
    assert recovered_est.confidence == pytest.approx(est.confidence)
    assert recovered_est.tracking_state == TrackingState.TRACKING_OK
    assert recovered_est.tracked_features == 85
    assert recovered_est.inliers == 45


def test_localization_status_serialization_json():
    """Test JSON serialization round-trip for LocalizationStatusMsg."""
    est = LocalizationEstimate(
        pose=np.eye(4),
        confidence=0.60,
        tracking_state=TrackingState.DEGRADED,
        timestamp=5.0,
    )
    msg = LocalizationStatusMsg.from_localization_estimate(est)
    payload = MessageSerializer.serialize_localization_status_json(msg)

    recovered = MessageSerializer.deserialize_localization_status_json(payload)
    assert recovered.tracking_state == LocalizationStatusMsg.DEGRADED
    assert recovered.confidence == pytest.approx(0.60)
    assert recovered.allows_spatial_fusion is True
    assert recovered.header.stamp.sec == 5


def test_world_terrain_map_full_roundtrip():
    """CRITICAL (Section 7): Full message round-trip test:

    Phase 9 WorldTerrainMap -> ROS2 message -> Binary serialization
    -> Deserialization -> Phase 9 WorldTerrainMap.
    Verifies numerical equivalence across all layers and metadata.
    """
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0), frame_id="odom")

    # Populate multiple cells across all layers
    r1, c1 = wmap.world_to_grid(1.5, 0.5)
    wmap.cost[r1, c1] = 0.82
    wmap.traversability[r1, c1] = 0.91
    wmap.obstacle[r1, c1] = True
    wmap.hazard[r1, c1] = 0.77
    wmap.unknown[r1, c1] = False
    wmap.confidence[r1, c1] = 0.95
    wmap.slope[r1, c1] = 12.5
    wmap.roughness[r1, c1] = 0.08
    wmap.height_var[r1, c1] = 0.03
    wmap.observation_count[r1, c1] = 4

    r2, c2 = wmap.world_to_grid(-2.0, -1.0)
    wmap.cost[r2, c2] = 0.45
    wmap.traversability[r2, c2] = 0.65
    wmap.obstacle[r2, c2] = False
    wmap.hazard[r2, c2] = 0.30
    wmap.unknown[r2, c2] = False
    wmap.confidence[r2, c2] = 0.80
    wmap.slope[r2, c2] = 5.2
    wmap.roughness[r2, c2] = 0.02
    wmap.height_var[r2, c2] = 0.01
    wmap.observation_count[r2, c2] = 2

    # Step 1: Conversion to WorldTerrainMapMsg
    msg = WorldTerrainMapMsg.from_world_terrain_map(wmap, frame_id="odom", timestamp_s=250.75)
    assert msg.header.frame_id == "odom"
    assert abs(msg.header.stamp.to_sec() - 250.75) < 1e-6

    # Step 2: Serialized Representation (ZLIB Binary)
    payload_bytes = MessageSerializer.serialize_terrain_map_binary(msg, compress=True)

    # Step 3: Deserialization
    deserialized_msg = MessageSerializer.deserialize_terrain_map_binary(payload_bytes)

    # Step 4: Reconstruct back to WorldTerrainMap
    recovered_wmap = deserialized_msg.to_world_terrain_map()

    # Step 5: Verify Numerical Equivalence
    assert recovered_wmap.resolution_m == pytest.approx(wmap.resolution_m)
    assert recovered_wmap.shape == wmap.shape
    assert recovered_wmap.frame_id == "odom"

    assert np.allclose(recovered_wmap.cost, wmap.cost, atol=1e-6)
    assert np.allclose(recovered_wmap.traversability, wmap.traversability, atol=1e-6)
    assert np.array_equal(recovered_wmap.obstacle, wmap.obstacle)
    assert np.allclose(recovered_wmap.hazard, wmap.hazard, atol=1e-6)
    assert np.array_equal(recovered_wmap.unknown, wmap.unknown)
    assert np.allclose(recovered_wmap.confidence, wmap.confidence, atol=1e-6)
    assert np.allclose(recovered_wmap.slope, wmap.slope, atol=1e-4)
    assert np.allclose(recovered_wmap.roughness, wmap.roughness, atol=1e-6)
    assert np.allclose(recovered_wmap.height_var, wmap.height_var, atol=1e-6)
    assert np.array_equal(recovered_wmap.observation_count, wmap.observation_count)

    # Specific cell checks
    assert recovered_wmap.obstacle[r1, c1] is np.True_ or recovered_wmap.obstacle[r1, c1] == 1
    assert recovered_wmap.hazard[r1, c1] == pytest.approx(0.77, abs=1e-4)
    assert recovered_wmap.slope[r1, c1] == pytest.approx(12.5, abs=1e-4)
    assert recovered_wmap.observation_count[r1, c1] == 4

