"""Unit tests for Perception, Localization, and Spatial Fusion Adapters."""

from __future__ import annotations

import numpy as np
import pytest

from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.result import LocalizationEstimate
from ugv_navigation.adapters.perception_adapter import PerceptionAdapter
from ugv_navigation.adapters.localization_adapter import LocalizationAdapter
from ugv_navigation.adapters.spatial_fusion_adapter import SpatialFusionAdapter
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.tf.tf_broadcaster import DynamicOdomBroadcaster, StaticCameraExtrinsicsPublisher
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def test_perception_adapter_costmap_generation():
    """Test PerceptionAdapter generates standard BEVTerrainCostmap with obstacles and hazards."""
    adapter = PerceptionAdapter(grid_resolution_m=0.10, x_range_m=(0.2, 6.0), y_range_m=(-3.0, 3.0))
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # In local BEV grid: shape is (58, 60)
    obst_mask = np.zeros((58, 60), dtype=bool)
    obst_mask[20:25, 28:32] = True
    haz_mask = np.zeros((58, 60), dtype=np.float32)
    haz_mask[35:40, 15:20] = 0.90  # Critical hazard

    costmap = adapter.process_image(dummy_img, timestamp_s=1.0, obstacle_mask=obst_mask, hazard_mask=haz_mask)

    assert isinstance(costmap, BEVTerrainCostmap)
    assert costmap.shape == (58, 60)
    assert np.all(costmap.obstacle[20:25, 28:32] == True)
    assert np.all(costmap.cost[20:25, 28:32] == 1.0)
    assert np.all(costmap.hazard[35:40, 15:20] == pytest.approx(0.90))
    assert np.all(costmap.cost[35:40, 15:20] == 1.0)


def test_localization_adapter_adaptation():
    """Test LocalizationAdapter converts LocalizationEstimate to LocalizationStatusMsg and TransformStamped."""
    adapter = LocalizationAdapter(frame_id="odom", child_frame_id="base_link")
    pose = np.eye(4)
    pose[0, 3] = 2.0
    est = LocalizationEstimate(
        pose=pose,
        confidence=0.92,
        tracking_state=TrackingState.TRACKING_OK,
        timestamp=2.5,
    )

    msg, tf_stamped = adapter.adapt_estimate(est)
    assert msg.tracking_state == LocalizationStatusMsg.TRACKING_OK
    assert msg.confidence == pytest.approx(0.92)
    assert msg.is_pose_valid is True
    assert tf_stamped is not None
    assert tf_stamped.header.frame_id == "odom"
    assert tf_stamped.child_frame_id == "base_link"
    assert tf_stamped.transform.translation.x == pytest.approx(2.0)


def test_spatial_fusion_adapter_pipeline():
    """Test SpatialFusionAdapter fuses BEV costmap and publishes WorldTerrainMapMsg."""
    spatial_adapter = SpatialFusionAdapter(resolution_m=0.10, x_bounds_m=(-10.0, 10.0), y_bounds_m=(-10.0, 10.0))
    tf_buf = TfTreeBuffer()
    StaticCameraExtrinsicsPublisher.configure_extrinsics(tf_buf)
    broadcaster = DynamicOdomBroadcaster(tf_buf)

    # Broadcast pose at t=1.0s
    broadcaster.broadcast_pose(np.eye(4), timestamp_s=1.0)

    # Generate local costmap at t=1.0s
    perception = PerceptionAdapter()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    costmap = perception.process_image(dummy_img, timestamp_s=1.0)

    # Fuse with TF lookup
    success, map_msg, reason = spatial_adapter.fuse_costmap_with_tf(
        bev=costmap,
        tf_buffer=tf_buf,
        tracking_state=TrackingState.TRACKING_OK,
        confidence=0.95,
    )

    assert success is True
    assert reason == "ok"
    assert isinstance(map_msg, WorldTerrainMapMsg)
    assert map_msg.header.frame_id == "odom"
    assert map_msg.width == 200
    assert map_msg.height == 200
