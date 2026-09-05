"""End-to-end integration test running the full ROS2 architecture in offline replay mode."""

from __future__ import annotations

import numpy as np
import pytest

from ugv_navigation.adapters.localization_adapter import LocalizationAdapter
from ugv_navigation.adapters.perception_adapter import PerceptionAdapter
from ugv_navigation.adapters.spatial_fusion_adapter import SpatialFusionAdapter
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.perception_node import PerceptionNode
from ugv_navigation.nodes.replay_node import ReplayNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.nodes.static_tf_node import StaticTfNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def test_offline_replay_pipeline_end_to_end():
    """PRINCIPAL INTEGRATION TEST (Section 35):

    Replay Camera Frame -> Perception Node -> Localization Node -> TF2
    -> Spatial Fusion Node -> WorldTerrainMapMsg.
    """
    # 1. Initialize Shared TF Buffer and Static Broadcaster
    tf_buf = TfTreeBuffer()
    static_node = StaticTfNode(tf_buffer=tf_buf)

    # 2. Initialize Nodes
    replay_node = ReplayNode(mode="synthetic", num_frames=10)
    perception_node = PerceptionNode()
    localization_node = LocalizationNode(tf_buffer=tf_buf)
    spatial_node = SpatialFusionNode(tf_buffer=tf_buf)

    fused_maps_count = 0
    total_frames = 0

    while True:
        frame = replay_node.get_next_frame()
        if frame is None:
            break
        total_frames += 1

        # A. Run Localization
        loc_msg, tf_stamped = localization_node.process_frame(
            img_left=frame.image_left,
            timestamp_s=frame.timestamp,
            img_right=frame.image_right,
        )
        spatial_node.on_localization_status(loc_msg)

        # B. Run Perception
        costmap = perception_node.process_frame(
            image_rgb=frame.image_left,
            timestamp_s=frame.timestamp,
        )

        # C. Run Spatial Fusion
        success, map_msg, reason = spatial_node.on_costmap_received(costmap)
        if success:
            fused_maps_count += 1
            assert isinstance(map_msg, WorldTerrainMapMsg)
            assert map_msg.header.frame_id == "odom"
            assert abs(map_msg.header.stamp.to_sec() - frame.timestamp) < 1e-4

    assert total_frames == 10
    assert fused_maps_count > 0, "Expected at least one successful map fusion during replay"

    # Verify Diagnostics
    status = spatial_node.diagnostics.get_status()
    assert status.level in (0, 1, 2)
    assert status.values["total_updates"] == total_frames
    assert int(status.values["rejected_poses"]) > 0
