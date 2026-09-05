"""Unit tests for PlannerAdapter and PlannerNode ROS2 middleware integration."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.adapters.planner_adapter import PathMsg, PlannerAdapter, PoseStampedMsg, TwistMsg
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.planner_node import PlannerNode
from ugv_navigation.utils.time_utils import RosHeader


def create_sample_world_map_msg(timestamp_s: float = 1.0) -> WorldTerrainMapMsg:
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0), created_timestamp=timestamp_s)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    wmap.observation_count.fill(1)
    wmap.last_observed_timestamp.fill(timestamp_s)
    return WorldTerrainMapMsg.from_world_terrain_map(wmap, frame_id="odom", timestamp_s=timestamp_s)


def create_sample_loc_msg(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, timestamp_s: float = 1.0, state: int = 0) -> LocalizationStatusMsg:
    header = RosHeader.create(frame_id="odom", timestamp_s=timestamp_s)
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    from ugv_navigation.messages.localization_status_msg import (
        Point3D,
        Pose,
        PoseWithCovariance,
        Quaternion,
        Twist,
        TwistWithCovariance,
    )

    pose = PoseWithCovariance(
        pose=Pose(
            position=Point3D(x=float(x), y=float(y), z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw)),
        )
    )
    twist = TwistWithCovariance(
        twist=Twist(
            linear=Point3D(x=0.0, y=0.0, z=0.0),
            angular=Point3D(x=0.0, y=0.0, z=0.0),
        )
    )

    return LocalizationStatusMsg(
        header=header,
        child_frame_id="base_link",
        tracking_state=state,
        confidence=1.0,
        is_pose_valid=True,
        allows_spatial_fusion=True,
        pose=pose,
        twist=twist,

        tracked_features=100,
        inliers=80,
    )


def test_planner_adapter_message_conversion():
    """Verify conversion between ROS2 messages and Python planner state."""
    adapter = PlannerAdapter()

    loc_msg = create_sample_loc_msg(x=1.5, y=-2.0, yaw=0.5, timestamp_s=2.0)
    state = adapter.loc_msg_to_robot_state(loc_msg)
    assert math.isclose(state.x, 1.5, abs_tol=1e-3)
    assert math.isclose(state.y, -2.0, abs_tol=1e-3)
    assert math.isclose(state.yaw, 0.5, abs_tol=1e-3)
    assert math.isclose(state.timestamp, 2.0, abs_tol=1e-3)

    goal_msg = PoseStampedMsg.from_xy_yaw(3.0, 1.0, 0.0, frame_id="odom", timestamp_s=2.0)
    goal = adapter.goal_msg_to_goal_pose(goal_msg)
    assert goal.x == 3.0
    assert goal.y == 1.0


def test_planner_adapter_plan_step():
    """Verify complete plan step across ROS2 message structures."""
    adapter = PlannerAdapter()
    map_msg = create_sample_world_map_msg(timestamp_s=5.0)
    loc_msg = create_sample_loc_msg(x=0.0, y=0.0, yaw=0.0, timestamp_s=5.0)
    goal_msg = PoseStampedMsg.from_xy_yaw(2.0, 0.0, 0.0, timestamp_s=5.0)

    twist, path, status, diag = adapter.plan_step_ros(
        loc_msg=loc_msg,
        map_msg=map_msg,
        goal_msg=goal_msg,
        sim_time_s=5.0,
    )

    assert status == PlannerStatus.FOLLOWING
    assert isinstance(twist, TwistMsg)
    assert isinstance(path, PathMsg)
    assert twist.linear.x > 0.0
    assert len(path.poses) >= 2


def test_planner_node_cycle():
    """Verify PlannerNode topic caching and cycle execution."""
    node = PlannerNode()
    map_msg = create_sample_world_map_msg(timestamp_s=1.0)
    loc_msg = create_sample_loc_msg(x=0.0, y=0.0, yaw=0.0, timestamp_s=1.0)
    goal_msg = PoseStampedMsg.from_xy_yaw(1.5, 0.0, 0.0, timestamp_s=1.0)

    node.on_map_received(map_msg)
    node.on_localization_status(loc_msg)
    node.on_goal_received(goal_msg)

    cmd, path, status, diag = node.process_cycle(sim_time_s=1.0)
    assert status == PlannerStatus.FOLLOWING
    assert cmd.linear.x > 0.0
    assert len(path.poses) >= 2
