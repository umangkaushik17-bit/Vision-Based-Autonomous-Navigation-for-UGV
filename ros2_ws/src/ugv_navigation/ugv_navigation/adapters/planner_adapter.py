"""ROS2 Planner Adapter converting between ROS2 middleware messages and pure-Python UGV planning classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.navigator import UGVNavigationPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.utils.time_utils import RosHeader, RosTime


@dataclass
class Vector3Msg:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class TwistMsg:
    """In-memory representation of geometry_msgs/Twist."""
    linear: Vector3Msg = field(default_factory=Vector3Msg)
    angular: Vector3Msg = field(default_factory=Vector3Msg)

    @classmethod
    def create(cls, linear_x: float, angular_z: float) -> TwistMsg:
        return cls(
            linear=Vector3Msg(x=float(linear_x), y=0.0, z=0.0),
            angular=Vector3Msg(x=0.0, y=0.0, z=float(angular_z)),
        )


@dataclass
class PointMsg:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class QuaternionMsg:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class PoseMsg:
    position: PointMsg = field(default_factory=PointMsg)
    orientation: QuaternionMsg = field(default_factory=QuaternionMsg)


@dataclass
class PoseStampedMsg:
    """In-memory representation of geometry_msgs/PoseStamped."""
    header: RosHeader
    pose: PoseMsg = field(default_factory=PoseMsg)
    has_orientation: bool = True

    @classmethod
    def from_xy_yaw(cls, x: float, y: float, yaw: Optional[float] = None, frame_id: str = "odom", timestamp_s: float = 0.0) -> PoseStampedMsg:
        if yaw is not None:
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            has_ori = True
        else:
            qz = 0.0
            qw = 1.0
            has_ori = False
        header = RosHeader.create(frame_id=frame_id, timestamp_s=timestamp_s)
        pose = PoseMsg(
            position=PointMsg(x=float(x), y=float(y), z=0.0),
            orientation=QuaternionMsg(x=0.0, y=0.0, z=float(qz), w=float(qw)),
        )
        return cls(header=header, pose=pose, has_orientation=has_ori)


@dataclass
class PathMsg:
    """In-memory representation of nav_msgs/Path."""
    header: RosHeader
    poses: List[PoseStampedMsg] = field(default_factory=list)

    @classmethod
    def from_planner_path(cls, path: Optional[Path], frame_id: str = "odom", timestamp_s: float = 0.0) -> PathMsg:
        header = RosHeader.create(frame_id=frame_id, timestamp_s=timestamp_s)
        if path is None or len(path) == 0:
            return cls(header=header, poses=[])

        poses = []
        for pt in path.points:
            poses.append(PoseStampedMsg.from_xy_yaw(pt.x, pt.y, pt.yaw, frame_id=frame_id, timestamp_s=timestamp_s))
        return cls(header=header, poses=poses)


class PlannerAdapter:
    """Bridges ROS2 messaging and the Phase 12 UGVNavigationPlanner core engine."""

    def __init__(self, planner: Optional[UGVNavigationPlanner] = None) -> None:
        self.planner = planner or UGVNavigationPlanner()
        self._last_cmd_v: float = 0.0
        self._last_cmd_w: float = 0.0

    def loc_msg_to_robot_state(self, loc_msg: LocalizationStatusMsg) -> RobotState:
        """Extract kinematic RobotState from LocalizationStatusMsg."""
        p = loc_msg.pose.pose.position
        o = loc_msg.pose.pose.orientation
        t = loc_msg.twist.twist.linear
        w = loc_msg.twist.twist.angular

        # Extract 2D yaw from quaternion (yaw around Z)
        siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        yaw = float(math.atan2(siny_cosp, cosy_cosp))

        t_sec = float(loc_msg.header.stamp.to_sec())
        v_linear = float(t.x)
        w_angular = float(w.z)

        # Odometry velocity feedback: when VO only estimates pose (twist is 0),
        # feed back last commanded velocity to maintain acceleration in DWA dynamic window
        if abs(v_linear) < 1e-4 and abs(self._last_cmd_v) > 1e-4:
            v_linear = self._last_cmd_v
        if abs(w_angular) < 1e-4 and abs(self._last_cmd_w) > 1e-4:
            w_angular = self._last_cmd_w

        return RobotState(
            x=float(p.x),
            y=float(p.y),
            yaw=yaw,
            v=v_linear,
            omega=w_angular,
            timestamp=t_sec,
        )

    def goal_msg_to_goal_pose(self, goal_msg: PoseStampedMsg) -> GoalPose:
        """Extract GoalPose from PoseStampedMsg."""
        p = goal_msg.pose.position
        o = goal_msg.pose.orientation

        if getattr(goal_msg, "has_orientation", True):
            siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
            cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
            yaw = float(math.atan2(siny_cosp, cosy_cosp))
        else:
            yaw = None

        tol_xy = float(self.planner.config.tolerances.get("goal_xy_tolerance_m", 0.20))
        return GoalPose(
            x=float(p.x),
            y=float(p.y),
            yaw=yaw,
            frame_id=goal_msg.header.frame_id,
            timestamp=float(goal_msg.header.stamp.to_sec()),
            xy_tolerance_m=tol_xy,
        )

    def terrain_msg_to_world_map(self, map_msg: Union[WorldTerrainMapMsg, WorldTerrainMap]) -> WorldTerrainMap:
        """Convert WorldTerrainMapMsg into Phase 9 WorldTerrainMap instance."""
        if isinstance(map_msg, WorldTerrainMap):
            return map_msg
        return map_msg.to_world_terrain_map()

    def plan_step_ros(
        self,
        loc_msg: LocalizationStatusMsg,
        map_msg: Union[WorldTerrainMapMsg, WorldTerrainMap],
        goal_msg: Optional[PoseStampedMsg] = None,
        sim_time_s: Optional[float] = None,
        dt_s: float = 0.10,
        tf_available: bool = True,
    ) -> Tuple[TwistMsg, PathMsg, PlannerStatus, Dict[str, Any]]:
        """Run complete planning cycle from ROS2 message inputs and return ROS2 outputs."""
        robot_state = self.loc_msg_to_robot_state(loc_msg)
        world_map = self.terrain_msg_to_world_map(map_msg)
        goal_pose = self.goal_msg_to_goal_pose(goal_msg) if goal_msg is not None else None

        t_now = sim_time_s if sim_time_s is not None else robot_state.timestamp
        t_map = float(map_msg.header.stamp.to_sec()) if hasattr(map_msg, "header") else world_map.created_timestamp
        t_loc = float(loc_msg.header.stamp.to_sec())

        v_x, w_z, global_path, local_traj, status, diag = self.planner.plan_step(
            current_state=robot_state,
            terrain_map=world_map,
            goal=goal_pose,
            tracking_state=loc_msg.tracking_state,
            map_timestamp_s=t_map,
            loc_timestamp_s=t_loc,
            tf_available=tf_available,
            sim_time_s=t_now,
            dt_s=dt_s,
        )

        self._last_cmd_v = float(v_x)
        self._last_cmd_w = float(w_z)

        twist_cmd = TwistMsg.create(v_x, w_z)
        path_ros = PathMsg.from_planner_path(global_path, frame_id="odom", timestamp_s=t_now)

        return twist_cmd, path_ros, status, diag
