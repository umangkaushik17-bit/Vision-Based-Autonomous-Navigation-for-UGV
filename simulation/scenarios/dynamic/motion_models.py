"""Deterministic kinematic motion models for simulated ground-truth dynamic obstacles."""

from __future__ import annotations

import abc
import math
from typing import List, Optional, Tuple
import numpy as np

from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    GroundTruthObstacleState,
    MotionModelType,
)


class BaseObstacleMotionModel(abc.ABC):
    """Abstract evaluator for dynamic obstacle ground-truth states."""

    def __init__(self, spec: DynamicObstacleSpec) -> None:
        self.spec = spec

    @abc.abstractmethod
    def evaluate(self, timestamp_s: float) -> Optional[GroundTruthObstacleState]:
        """Evaluate ground truth state at timestamp_s.

        Returns None if obstacle has not spawned or has despawned.
        """
        pass


class ConstantVelocityMotionModel(BaseObstacleMotionModel):
    """Linear trajectory moving with constant velocity vector (vx, vy)."""

    def evaluate(self, timestamp_s: float) -> Optional[GroundTruthObstacleState]:
        if timestamp_s < self.spec.spawn_time_s or timestamp_s > self.spec.despawn_time_s:
            return None

        dt = timestamp_s - self.spec.spawn_time_s
        x0, y0, yaw0 = self.spec.initial_pose
        vx, vy = self.spec.initial_velocity

        x = x0 + vx * dt
        y = y0 + vy * dt
        speed = math.hypot(vx, vy)
        yaw = math.atan2(vy, vx) if speed > 1e-3 else yaw0

        return GroundTruthObstacleState(
            obstacle_id=self.spec.obstacle_id,
            timestamp_s=timestamp_s,
            x=x,
            y=y,
            yaw=yaw,
            vx=vx,
            vy=vy,
            ax=0.0,
            ay=0.0,
            length_m=self.spec.size[0],
            width_m=self.spec.size[1],
            is_active=True,
        )


class ConstantTurnRateMotionModel(BaseObstacleMotionModel):
    """Circular arc trajectory moving with speed v and constant yaw rate w."""

    def evaluate(self, timestamp_s: float) -> Optional[GroundTruthObstacleState]:
        if timestamp_s < self.spec.spawn_time_s or timestamp_s > self.spec.despawn_time_s:
            return None

        dt = timestamp_s - self.spec.spawn_time_s
        x0, y0, yaw0 = self.spec.initial_pose
        vx, vy = self.spec.initial_velocity
        speed = math.hypot(vx, vy)
        w = self.spec.yaw_rate_rad_s

        if abs(w) < 1e-5:
            # Straight line limit
            yaw = yaw0
            x = x0 + speed * math.cos(yaw) * dt
            y = y0 + speed * math.sin(yaw) * dt
            vx_t = speed * math.cos(yaw)
            vy_t = speed * math.sin(yaw)
        else:
            yaw = yaw0 + w * dt
            # Circular arc integration
            x = x0 + (speed / w) * (math.sin(yaw) - math.sin(yaw0))
            y = y0 - (speed / w) * (math.cos(yaw) - math.cos(yaw0))
            vx_t = speed * math.cos(yaw)
            vy_t = speed * math.sin(yaw)

        # Centripetal acceleration
        ax = -vy_t * w
        ay = vx_t * w

        return GroundTruthObstacleState(
            obstacle_id=self.spec.obstacle_id,
            timestamp_s=timestamp_s,
            x=x,
            y=y,
            yaw=math.atan2(math.sin(yaw), math.cos(yaw)),
            vx=vx_t,
            vy=vy_t,
            ax=ax,
            ay=ay,
            length_m=self.spec.size[0],
            width_m=self.spec.size[1],
            is_active=True,
        )


class AcceleratedMotionModel(BaseObstacleMotionModel):
    """Trajectory with initial velocity (vx, vy) and constant acceleration (ax, ay)."""

    def evaluate(self, timestamp_s: float) -> Optional[GroundTruthObstacleState]:
        if timestamp_s < self.spec.spawn_time_s or timestamp_s > self.spec.despawn_time_s:
            return None

        dt = timestamp_s - self.spec.spawn_time_s
        x0, y0, yaw0 = self.spec.initial_pose
        vx0, vy0 = self.spec.initial_velocity
        ax, ay = self.spec.acceleration

        x = x0 + vx0 * dt + 0.5 * ax * (dt ** 2)
        y = y0 + vy0 * dt + 0.5 * ay * (dt ** 2)
        vx = vx0 + ax * dt
        vy = vy0 + ay * dt
        speed = math.hypot(vx, vy)
        yaw = math.atan2(vy, vx) if speed > 1e-3 else yaw0

        return GroundTruthObstacleState(
            obstacle_id=self.spec.obstacle_id,
            timestamp_s=timestamp_s,
            x=x,
            y=y,
            yaw=yaw,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            length_m=self.spec.size[0],
            width_m=self.spec.size[1],
            is_active=True,
        )


class WaypointMotionModel(BaseObstacleMotionModel):
    """Piecewise linear interpolation through timed waypoints (t, x, y)."""

    def evaluate(self, timestamp_s: float) -> Optional[GroundTruthObstacleState]:
        if timestamp_s < self.spec.spawn_time_s or timestamp_s > self.spec.despawn_time_s:
            return None

        raw_wpts = self.spec.waypoints
        if not raw_wpts:
            x0, y0, yaw0 = self.spec.initial_pose
            return GroundTruthObstacleState(
                obstacle_id=self.spec.obstacle_id,
                timestamp_s=timestamp_s,
                x=x0, y=y0, yaw=yaw0,
                length_m=self.spec.size[0], width_m=self.spec.size[1],
            )

        if len(raw_wpts[0]) == 2:
            speed = math.hypot(self.spec.initial_velocity[0], self.spec.initial_velocity[1])
            if speed < 1e-3:
                speed = 1.0
            wpts = []
            cur_t = self.spec.spawn_time_s
            prev_pt = (self.spec.initial_pose[0], self.spec.initial_pose[1])
            for pt in raw_wpts:
                dist = math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1])
                cur_t += dist / speed
                wpts.append((cur_t, pt[0], pt[1]))
                prev_pt = pt
        else:
            wpts = sorted(raw_wpts, key=lambda w: w[0])

        # Before first waypoint
        if timestamp_s <= wpts[0][0]:
            return GroundTruthObstacleState(
                obstacle_id=self.spec.obstacle_id,
                timestamp_s=timestamp_s,
                x=wpts[0][1], y=wpts[0][2],
                yaw=self.spec.initial_pose[2],
                length_m=self.spec.size[0], width_m=self.spec.size[1],
            )

        # After last waypoint
        if timestamp_s >= wpts[-1][0]:
            return GroundTruthObstacleState(
                obstacle_id=self.spec.obstacle_id,
                timestamp_s=timestamp_s,
                x=wpts[-1][1], y=wpts[-1][2],
                yaw=self.spec.initial_pose[2],
                length_m=self.spec.size[0], width_m=self.spec.size[1],
            )

        # Find segment
        for i in range(len(wpts) - 1):
            t0, x0, y0 = wpts[i]
            t1, x1, y1 = wpts[i + 1]
            if t0 <= timestamp_s <= t1:
                frac = (timestamp_s - t0) / (t1 - t0) if t1 > t0 else 0.0
                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                dt_seg = t1 - t0
                vx = (x1 - x0) / dt_seg if dt_seg > 0 else 0.0
                vy = (y1 - y0) / dt_seg if dt_seg > 0 else 0.0
                yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-3 else self.spec.initial_pose[2]
                return GroundTruthObstacleState(
                    obstacle_id=self.spec.obstacle_id,
                    timestamp_s=timestamp_s,
                    x=x, y=y, yaw=yaw,
                    vx=vx, vy=vy,
                    length_m=self.spec.size[0], width_m=self.spec.size[1],
                )

        return None


class StationaryObstacleMotionModel(BaseObstacleMotionModel):
    """Static or temporarily stationary obstacle at a fixed position."""

    def evaluate(self, timestamp_s: float) -> Optional[GroundTruthObstacleState]:
        if timestamp_s < self.spec.spawn_time_s or timestamp_s > self.spec.despawn_time_s:
            return None

        x, y, yaw = self.spec.initial_pose
        return GroundTruthObstacleState(
            obstacle_id=self.spec.obstacle_id,
            timestamp_s=timestamp_s,
            x=x,
            y=y,
            yaw=yaw,
            vx=0.0,
            vy=0.0,
            ax=0.0,
            ay=0.0,
            length_m=self.spec.size[0],
            width_m=self.spec.size[1],
            is_active=True,
        )


def create_motion_model(spec: DynamicObstacleSpec) -> BaseObstacleMotionModel:
    """Factory creating the appropriate motion model for a given specification."""
    if spec.motion_model == MotionModelType.CONSTANT_VELOCITY:
        return ConstantVelocityMotionModel(spec)
    elif spec.motion_model == MotionModelType.CONSTANT_TURN_RATE:
        return ConstantTurnRateMotionModel(spec)
    elif spec.motion_model == MotionModelType.ACCELERATED:
        return AcceleratedMotionModel(spec)
    elif spec.motion_model == MotionModelType.WAYPOINT:
        return WaypointMotionModel(spec)
    elif spec.motion_model == MotionModelType.STATIONARY:
        return StationaryObstacleMotionModel(spec)
    else:
        raise ValueError(f"Unsupported motion model type: {spec.motion_model}")
