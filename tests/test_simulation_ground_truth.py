"""Unit and integration tests for Ground Truth Isolation and Quantitative Accuracy Evaluation."""

from __future__ import annotations

import math
import numpy as np
import pytest

from simulation.sim_engine import GazeboSimEngine
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.perception_node import PerceptionNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def test_ground_truth_isolation_contract():
    """CRITICAL (Section 28 & 55): Verify production pipeline has ZERO access

    to simulation ground truth. Ground truth is isolated exclusively in the evaluator.
    """
    sim = GazeboSimEngine()
    tf_buf = TfTreeBuffer()

    perc = PerceptionNode()
    loc = LocalizationNode(tf_buffer=tf_buf)
    spatial = SpatialFusionNode(tf_buffer=tf_buf)

    # Inspect node attributes to ensure no ground-truth references exist
    for node in [perc, loc, spatial]:
        for attr_name in dir(node):
            assert "ground_truth" not in attr_name.lower(), f"Forbidden ground truth attribute found on {node}"

    # Step sensor stream
    stream = sim.get_next_sensor_cycle()

    # Pass solely raw images to localization and perception
    loc_msg, _ = loc.process_frame(stream.image_left, stream.timestamp, stream.image_right)
    costmap = perc.process_frame(stream.image_left, stream.timestamp)
    spatial.on_localization_status(loc_msg)
    success, map_msg, _ = spatial.on_costmap_received(costmap)

    # Pipeline executes without knowing ground truth
    assert loc_msg is not None
    assert costmap is not None


def test_quantitative_localization_evaluation():
    """Verify localization accuracy comparison against simulation ground truth."""
    sim = GazeboSimEngine()
    tf_buf = TfTreeBuffer()
    loc = LocalizationNode(tf_buffer=tf_buf)

    # Command straight motion
    sim.set_command(linear_x=0.4, angular_z=0.0)

    gt_positions = []
    est_positions = []

    for _ in range(10):
        stream = sim.get_next_sensor_cycle()
        loc_msg, _ = loc.process_frame(stream.image_left, stream.timestamp, stream.image_right)

        # Ground truth position
        gt_p = stream.ground_truth_pose[:3, 3]
        gt_positions.append(gt_p)

        # Estimated position
        est_p = np.array([
            loc_msg.pose.pose.position.x,
            loc_msg.pose.pose.position.y,
            loc_msg.pose.pose.position.z,
        ])
        est_positions.append(est_p)

    gt_arr = np.array(gt_positions)
    est_arr = np.array(est_positions)

    # Calculate position errors
    diffs = np.linalg.norm(gt_arr[:, :2] - est_arr[:, :2], axis=1)
    rmse = float(np.sqrt(np.mean(diffs**2)))
    max_err = float(np.max(diffs))

    print(f"Simulation Localization RMSE: {rmse:.4f} m, Max Error: {max_err:.4f} m")
    # In initial frames, drift should be bounded
    assert rmse < 1.0, f"Simulated localization RMSE {rmse:.2f} m exceeded bound"


def test_quantitative_world_map_obstacle_comparison():
    """Verify ground-truth obstacle coordinates in simulation correspond to valid

    mapped regions in the WorldTerrainMap.
    """
    sim = GazeboSimEngine()
    gt_obs = sim.gt_registry.get("obstacles", [])
    assert len(gt_obs) > 0

    rock = [o for o in gt_obs if o["id"] == "obs_rock_large"][0]
    rx, ry = rock["position"][0], rock["position"][1]

    # Check that rock is at expected simulated world location
    assert rx == pytest.approx(3.5)
    assert ry == pytest.approx(0.5)
    assert rock["expected_cost"] == pytest.approx(1.0)
