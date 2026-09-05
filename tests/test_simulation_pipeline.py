"""Integration test verifying end-to-end sensor data flow from Gazebo simulation

through Perception, Localization, TF2, and Spatial Fusion to WorldTerrainMapMsg.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.sim_engine import GazeboSimEngine
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.perception_node import PerceptionNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.nodes.static_tf_node import StaticTfNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def test_simulated_pipeline_end_to_end():
    """PRINCIPAL INTEGRATION TEST (Section 32):

    Simulated Camera Frame -> Perception Node -> Localization Node -> TF2
    -> Spatial Fusion Node -> WorldTerrainMapMsg.
    """
    # 1. Initialize Simulation Engine
    sim = GazeboSimEngine()

    # 2. Initialize Shared TF Buffer and Static Broadcaster
    tf_buf = TfTreeBuffer()
    static_node = StaticTfNode(tf_buffer=tf_buf)

    # 3. Initialize Production ROS2 Navigation Nodes
    perception_node = PerceptionNode()
    localization_node = LocalizationNode(tf_buffer=tf_buf)
    spatial_node = SpatialFusionNode(tf_buffer=tf_buf)

    # Command forward motion (0.4 m/s)
    sim.set_command(linear_x=0.4, angular_z=0.0)

    fused_maps = 0
    total_frames = 15

    for frame_idx in range(total_frames):
        sensor_stream = sim.get_next_sensor_cycle()

        # Step A: Run Production Localization Node (No ground truth!)
        loc_msg, tf_stamped = localization_node.process_frame(
            img_left=sensor_stream.image_left,
            timestamp_s=sensor_stream.timestamp,
            img_right=sensor_stream.image_right,
        )
        spatial_node.on_localization_status(loc_msg)

        # Step B: Run Production Perception Node
        costmap = perception_node.process_frame(
            image_rgb=sensor_stream.image_left,
            timestamp_s=sensor_stream.timestamp,
        )

        # Step C: Run Production Spatial Fusion Node
        success, map_msg, reason = spatial_node.on_costmap_received(costmap)
        if success:
            fused_maps += 1
            assert isinstance(map_msg, WorldTerrainMapMsg)
            assert map_msg.header.frame_id == "odom"
            assert map_msg.width > 0
            assert map_msg.height > 0

    assert fused_maps > 0, "Expected at least one valid world terrain map fusion from simulated stream"

    # Verify TF Chain is populated
    T_odom_cam = tf_buf.lookup_transform_matrix("odom", "camera_optical")
    assert T_odom_cam.shape == (4, 4)

    # Verify Diagnostics
    diag = spatial_node.diagnostics.get_status()
    assert diag.values["total_updates"] == total_frames
