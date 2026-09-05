"""Deterministic Motion Controller and Kinematics for Simulated 4WD UGV."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class UGVCommand:
    """Velocity command for differential/skid-steer drive."""
    linear_x: float = 0.0      # m/s forward
    angular_z: float = 0.0     # rad/s yaw rate


@dataclass
class RobotState2D:
    """2D ground truth pose and velocity state."""
    x: float = 0.0             # meters in odom frame
    y: float = 0.0             # meters in odom frame
    yaw: float = 0.0           # radians in odom frame
    linear_x: float = 0.0      # m/s
    angular_z: float = 0.0     # rad/s
    timestamp: float = 0.0     # simulation seconds


class UGVMotionController:
    """Controls simulated 4WD robot motion using differential-drive kinematics

    and deterministic waypoint/segment trajectories.
    """

    def __init__(
        self,
        track_width_m: float = 0.48,
        wheel_radius_m: float = 0.125,
        max_linear_vel_m_s: float = 1.5,
        max_angular_vel_rad_s: float = 1.0,
    ):
        self.track_width = float(track_width_m)
        self.wheel_radius = float(wheel_radius_m)
        self.max_linear_vel = float(max_linear_vel_m_s)
        self.max_angular_vel = float(max_angular_vel_rad_s)

        # Current kinematic state
        self.state = RobotState2D()
        self.current_cmd = UGVCommand()

    def set_command(self, linear_x: float, angular_z: float) -> None:
        """Set immediate manual /cmd_vel velocity command."""
        v = float(np.clip(linear_x, -self.max_linear_vel, self.max_linear_vel))
        w = float(np.clip(angular_z, -self.max_angular_vel, self.max_angular_vel))
        self.current_cmd = UGVCommand(linear_x=v, angular_z=w)

    def compute_wheel_speeds(self) -> Tuple[float, float]:
        """Compute left and right wheel angular velocities (rad/s)."""
        v = self.current_cmd.linear_x
        w = self.current_cmd.angular_z
        v_left = v - (w * self.track_width / 2.0)
        v_right = v + (w * self.track_width / 2.0)
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        return omega_left, omega_right

    def step(self, dt_s: float) -> RobotState2D:
        """Integrate differential drive kinematics forward by dt_s seconds."""
        v = self.current_cmd.linear_x
        w = self.current_cmd.angular_z

        # Second-order Runge-Kutta / midpoint integration
        theta_mid = self.state.yaw + 0.5 * w * dt_s
        self.state.x += v * math.cos(theta_mid) * dt_s
        self.state.y += v * math.sin(theta_mid) * dt_s
        self.state.yaw += w * dt_s

        # Wrap yaw to [-pi, pi]
        self.state.yaw = math.atan2(math.sin(self.state.yaw), math.cos(self.state.yaw))
        self.state.linear_x = v
        self.state.angular_z = w
        self.state.timestamp += dt_s

        return RobotState2D(
            x=self.state.x,
            y=self.state.y,
            yaw=self.state.yaw,
            linear_x=self.state.linear_x,
            angular_z=self.state.angular_z,
            timestamp=self.state.timestamp,
        )

    def generate_deterministic_trajectory(
        self,
        total_duration_s: float = 10.0,
        dt_s: float = 0.05,
    ) -> List[Tuple[float, UGVCommand, RobotState2D]]:
        """Generates Section 34 deterministic trajectory:

        1. Straight 2.5 m (v=0.5 m/s for 5.0 s)
        2. Turn +30 deg (w=0.2618 rad/s for 2.0 s)
        3. Move 2.0 m (v=0.5 m/s for 4.0 s)
        4. Turn -45 deg (w=-0.3927 rad/s for 2.0 s)
        5. Move 2.0 m (v=0.5 m/s for 4.0 s)
        """
        self.state = RobotState2D()
        trajectory: List[Tuple[float, UGVCommand, RobotState2D]] = []

        segments = [
            (5.0, 0.50, 0.0),                  # Segment 1: Straight 2.5m
            (2.0, 0.00, math.radians(15.0)),   # Segment 2: Turn +30 deg
            (4.0, 0.50, 0.0),                  # Segment 3: Straight 2.0m
            (2.0, 0.00, -math.radians(22.5)),  # Segment 4: Turn -45 deg
            (4.0, 0.50, 0.0),                  # Segment 5: Straight 2.0m
        ]

        t_sim = 0.0
        for seg_dur, v_cmd, w_cmd in segments:
            steps = int(round(seg_dur / dt_s))
            for _ in range(steps):
                self.set_command(v_cmd, w_cmd)
                st = self.step(dt_s)
                trajectory.append((t_sim, UGVCommand(v_cmd, w_cmd), st))
                t_sim += dt_s

        return trajectory
