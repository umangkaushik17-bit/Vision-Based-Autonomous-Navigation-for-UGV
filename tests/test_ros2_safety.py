"""Integration tests for LOST, DEGRADED, and stale-pose safety behaviors in ROS2 layer."""

from __future__ import annotations

import numpy as np
import pytest

from ai.costmap.bev_costmap import BEVTerrainCostmap
from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate
from ugv_navigation.adapters.localization_adapter import LocalizationAdapter
from ugv_navigation.adapters.perception_adapter import PerceptionAdapter
from ugv_navigation.adapters.spatial_fusion_adapter import SpatialFusionAdapter
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.tf.tf_broadcaster import DynamicOdomBroadcaster, StaticCameraExtrinsicsPublisher
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def test_lost_state_blocks_tf_and_fusion():
    """CRITICAL (Section 36): Force LOST state and verify zero-trust behavior.

    1. TF transform must NOT be broadcast for untrusted poses.
    2. Spatial fusion must be strictly rejected.
    3. World map must remain uncorrupted by stale poses.
    """
    loc_adapter = LocalizationAdapter()
    spatial_adapter = SpatialFusionAdapter()
    tf_buf = TfTreeBuffer()
    StaticCameraExtrinsicsPublisher.configure_extrinsics(tf_buf)

    # Pose marked LOST
    est_lost = LocalizationEstimate(
        pose=np.eye(4),
        confidence=0.10,
        tracking_state=TrackingState.LOST,
        timestamp=4.0,
    )

    msg, tf_stamped = loc_adapter.adapt_estimate(est_lost)
    assert msg.tracking_state == LocalizationStatusMsg.LOST
    assert msg.is_pose_valid is False
    assert msg.allows_spatial_fusion is False
    assert tf_stamped is None  # Zero TF broadcast on LOST!

    # Attempt spatial fusion with LOST state
    perception = PerceptionAdapter()
    dummy_bev = perception.process_image(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=4.0)
    success, map_msg, reason = spatial_adapter.fuse_costmap_with_tf(
        bev=dummy_bev,
        tf_buffer=tf_buf,
        tracking_state=TrackingState.LOST,
        confidence=0.10,
    )

    assert success is False
    assert "fusion_blocked" in reason
    assert map_msg is None


def test_lost_to_recovered_transition():
    """Test full cycle: TRACKING_OK -> LOST -> RECOVERED."""
    loc_adapter = LocalizationAdapter()
    spatial_adapter = SpatialFusionAdapter()
    tf_buf = TfTreeBuffer()
    StaticCameraExtrinsicsPublisher.configure_extrinsics(tf_buf)
    broadcaster = DynamicOdomBroadcaster(tf_buf)
    perception = PerceptionAdapter()

    # Step 1: OK at t=1.0s
    broadcaster.broadcast_pose(np.eye(4), timestamp_s=1.0)
    bev1 = perception.process_image(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=1.0)
    s1, m1, _ = spatial_adapter.fuse_costmap_with_tf(bev1, tf_buf, TrackingState.TRACKING_OK, 0.95)
    assert s1 is True

    # Step 2: LOST at t=2.0s
    bev2 = perception.process_image(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=2.0)
    s2, m2, r2 = spatial_adapter.fuse_costmap_with_tf(bev2, tf_buf, TrackingState.LOST, 0.10)
    assert s2 is False

    # Step 3: RECOVERED at t=3.0s
    broadcaster.broadcast_pose(np.eye(4), timestamp_s=3.0)
    bev3 = perception.process_image(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=3.0)
    s3, m3, _ = spatial_adapter.fuse_costmap_with_tf(bev3, tf_buf, TrackingState.RECOVERED, 0.90)
    assert s3 is True
    assert m3 is not None


def test_degraded_state_conservative_weighting():
    """CRITICAL (Section 37): Verify DEGRADED state retains pose validity but

    applies conservative confidence down-weighting.
    """
    loc_adapter = LocalizationAdapter()
    est_degraded = LocalizationEstimate(
        pose=np.eye(4),
        confidence=0.50,
        tracking_state=TrackingState.DEGRADED,
        timestamp=10.0,
    )

    msg, tf_stamped = loc_adapter.adapt_estimate(est_degraded)
    assert msg.tracking_state == LocalizationStatusMsg.DEGRADED
    assert msg.is_pose_valid is True
    assert msg.allows_spatial_fusion is True
    assert tf_stamped is not None
    assert msg.confidence == pytest.approx(0.50)


def test_stale_pose_lookup_rejection():
    """Verify spatial fusion rejects updates when TF buffer exceeds sync tolerance (gap > 0.10s)."""
    spatial_adapter = SpatialFusionAdapter(max_sync_delta_s=0.10)
    tf_buf = TfTreeBuffer(max_extrapolation_s=0.10)
    broadcaster = DynamicOdomBroadcaster(tf_buf)

    # Pose at t=1.0s
    broadcaster.broadcast_pose(np.eye(4), timestamp_s=1.0)

    # Costmap arriving at t=1.5s (gap 0.50s > 0.10s)
    perception = PerceptionAdapter()
    bev_stale = perception.process_image(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=1.5)

    success, map_msg, reason = spatial_adapter.fuse_costmap_with_tf(
        bev=bev_stale,
        tf_buffer=tf_buf,
        tracking_state=TrackingState.TRACKING_OK,
        confidence=0.90,
    )

    assert success is False
    assert "tf_lookup_failed" in reason
    assert map_msg is None
