"""Targeted unit tests for Phase 12 goal-tolerance contract (0.20 m)."""

import math
import pytest
import numpy as np

from ai.planning.base import GoalPose, RobotState, PlannerStatus, PlanningConfig
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.adapters.planner_adapter import PoseStampedMsg, WorldTerrainMapMsg
from ugv_navigation.messages.localization_status_msg import (
    LocalizationStatusMsg, Point3D, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance
)
from ugv_navigation.nodes.planner_node import PlannerNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer
from ugv_navigation.utils.time_utils import RosHeader


def make_loc_msg(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, timestamp_s: float = 1.0, state: int = 0) -> LocalizationStatusMsg:
    header = RosHeader.create(frame_id="odom", timestamp_s=timestamp_s)
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
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
    )


def test_goal_pose_default_tolerance():
    """Verify GoalPose and planning.yaml use exact 0.20 m tolerance."""
    cfg = PlanningConfig.load()
    assert cfg.tolerances.get("goal_xy_tolerance_m") == 0.20

    goal = GoalPose(x=2.0, y=0.0)
    assert goal.xy_tolerance_m == 0.20


def test_goal_pose_boundary_conditions():
    """Verify boundary conditions: 0.199m, 0.200m, 0.201m, and 0.500m."""
    goal = GoalPose(x=2.0, y=0.0)

    # Pose at 0.199 m -> GOAL_REACHED
    st_199 = RobotState(x=2.0 - 0.199, y=0.0)
    assert goal.is_reached(st_199) is True

    # Pose at 0.200 m -> GOAL_REACHED (implementation boundary <= 0.200)
    st_200 = RobotState(x=2.0 - 0.200, y=0.0)
    assert goal.is_reached(st_200) is True

    # Pose at 0.201 m -> NOT GOAL_REACHED
    st_201 = RobotState(x=2.0 - 0.201, y=0.0)
    assert goal.is_reached(st_201) is False

    # Pose clearly outside tolerance (0.500 m) -> NOT GOAL_REACHED
    st_far = RobotState(x=2.0 - 0.500, y=0.0)
    assert goal.is_reached(st_far) is False


def test_planner_node_goal_check_ros2():
    """Verify PlannerNode triggers GOAL_REACHED at 0.199m and 0.200m, and blocks on blocked goal."""
    tf = TfTreeBuffer()
    pn = PlannerNode(tf_buffer=tf)
    pn.on_goal_received(PoseStampedMsg.from_xy_yaw(2.0, 0.0, None))

    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0), created_timestamp=1.0)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    pn.on_map_received(WorldTerrainMapMsg.from_world_terrain_map(wmap))

    # Pose at 0.199 m error
    loc_199 = make_loc_msg(x=1.801, y=0.0, yaw=0.0, timestamp_s=1.0)
    pn.on_localization_status(loc_199)
    cmd, path, status, diag = pn.process_cycle(sim_time_s=1.0, dt_s=0.10)
    assert status == PlannerStatus.GOAL_REACHED
    assert cmd.linear.x == 0.0 and cmd.angular.z == 0.0

    # Pose at 0.200 m error
    loc_200 = make_loc_msg(x=1.800, y=0.0, yaw=0.0, timestamp_s=1.1)
    pn.on_localization_status(loc_200)
    cmd, path, status, diag = pn.process_cycle(sim_time_s=1.1, dt_s=0.10)
    assert status == PlannerStatus.GOAL_REACHED
    assert cmd.linear.x == 0.0 and cmd.angular.z == 0.0

    # Pose clearly outside tolerance (0.35 m error)
    loc_outside = make_loc_msg(x=1.650, y=0.0, yaw=0.0, timestamp_s=1.2)
    pn.on_localization_status(loc_outside)
    cmd, path, status, diag = pn.process_cycle(sim_time_s=1.2, dt_s=0.10)
    assert status != PlannerStatus.GOAL_REACHED

    # Blocked goal: obstacle centered on goal
    pn_blocked = PlannerNode(tf_buffer=tf)
    pn_blocked.on_goal_received(PoseStampedMsg.from_xy_yaw(2.0, 0.0, None))
    wmap_b = WorldTerrainMap(resolution_m=0.10, x_bounds_m=(-5.0, 5.0), y_bounds_m=(-5.0, 5.0), created_timestamp=1.0)
    wmap_b.cost.fill(0.10)
    r_g, c_g = wmap_b.world_to_grid(2.0, 0.0)
    wmap_b.obstacle[r_g, c_g] = True
    pn_blocked.on_map_received(WorldTerrainMapMsg.from_world_terrain_map(wmap_b))

    loc_start = make_loc_msg(x=0.0, y=0.0, yaw=0.0, timestamp_s=1.0)
    pn_blocked.on_localization_status(loc_start)
    cmd_b, path_b, status_b, diag_b = pn_blocked.process_cycle(sim_time_s=1.0, dt_s=0.10)
    assert status_b in (PlannerStatus.GOAL_BLOCKED, PlannerStatus.NO_PATH)
    assert cmd_b.linear.x == 0.0 and cmd_b.angular.z == 0.0
