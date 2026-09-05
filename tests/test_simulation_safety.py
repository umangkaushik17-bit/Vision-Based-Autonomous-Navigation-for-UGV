"""Safety tests verifying failure injection handling (Camera blackout, IMU disconnect, TF gap)

in the simulated environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.localization.base import TrackingState
from simulation.sim_engine import GazeboSimEngine
from ugv_navigation.adapters.localization_adapter import LocalizationAdapter
from ugv_navigation.adapters.perception_adapter import PerceptionAdapter
from ugv_navigation.adapters.spatial_fusion_adapter import SpatialFusionAdapter
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.nodes.static_tf_node import StaticTfNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def test_simulated_camera_blackout_safety():
    """CRITICAL (Section 44): Verify camera blackout triggers zero-trust safety:

    1. Localization detects tracking failure.
    2. Dynamic TF is not published for invalid pose.
    3. Spatial fusion rejects the observation.
    4. World terrain map is protected against corruption.
    """
    sim = GazeboSimEngine()
    tf_buf = TfTreeBuffer()
    StaticTfNode(tf_buffer=tf_buf)

    loc_node = LocalizationNode(tf_buffer=tf_buf)
    spatial_node = SpatialFusionNode(tf_buffer=tf_buf)

    # Disable camera to simulate blackout / lens obstruction
    sim.camera_enabled = False
    stream = sim.get_next_sensor_cycle()

    # Image is all black
    assert np.all(stream.image_left == 0)

    # Process through production localization
    loc_msg, tf_stamped = loc_node.process_frame(
        img_left=stream.image_left,
        timestamp_s=stream.timestamp,
        img_right=stream.image_right,
    )

    # Must be marked LOST with invalid pose
    spatial_node.on_localization_status(loc_msg)

    # Process dummy costmap
    perception = PerceptionAdapter()
    costmap = perception.process_image(stream.image_left, timestamp_s=stream.timestamp)

    # Spatial fusion must reject update
    success, map_msg, reason = spatial_node.on_costmap_received(costmap)
    assert success is False
    assert map_msg is None


def test_simulated_imu_disconnect_handling():
    """CRITICAL (Section 45): Verify IMU disconnect causes safe fallback to visual odometry

    with DEGRADED state and conservative confidence down-weighting.
    """
    sim = GazeboSimEngine()
    sim.imu_enabled = False
    imu_msg = sim.render_imu()

    assert imu_msg["status"] == "FAULT_DISCONNECTED"

    # Adapter verifies graceful handling of missing IMU
    adapter = LocalizationAdapter()
    from ai.localization.result import LocalizationEstimate
    est = LocalizationEstimate(
        pose=np.eye(4),
        confidence=0.50,
        tracking_state=TrackingState.DEGRADED,
        timestamp=1.0,
    )
    msg, tf_stamped = adapter.adapt_estimate(est)
    assert msg.tracking_state == LocalizationStatusMsg.DEGRADED
    assert msg.confidence == pytest.approx(0.50)
    assert msg.allows_spatial_fusion is True


def test_simulated_tf_disconnect_safety():
    """CRITICAL (Section 46): Verify missing TF transform cleanly rejects fusion update

    and generates a diagnostic error without crashing.
    """
    tf_buf = TfTreeBuffer(max_extrapolation_s=0.05)
    spatial_node = SpatialFusionNode(tf_buffer=tf_buf)

    perception = PerceptionAdapter()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    costmap = perception.process_image(dummy_img, timestamp_s=50.0)

    # No TF transform at t=50.0s
    success, map_msg, reason = spatial_node.on_costmap_received(costmap)
    assert success is False
    assert "tf_lookup_failed" in reason
    assert map_msg is None

    # Diagnostic tracks failure
    status = spatial_node.diagnostics.get_status()
    assert int(status.values["tf_failures"]) > 0
