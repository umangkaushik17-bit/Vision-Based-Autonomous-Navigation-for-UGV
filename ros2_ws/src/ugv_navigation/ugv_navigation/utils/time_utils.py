"""Time conversion and ROS2 Header utilities for pure-Python and rclpy execution."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class RosTime:
    """Representation of builtin_interfaces/Time."""
    sec: int = 0
    nanosec: int = 0

    def to_sec(self) -> float:
        return self.sec + self.nanosec * 1e-9

    @classmethod
    def from_sec(cls, t_sec: float) -> RosTime:
        sec = int(t_sec)
        nanosec = int(round((t_sec - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return cls(sec=sec, nanosec=nanosec)


@dataclass
class RosHeader:
    """Representation of std_msgs/Header."""
    stamp: RosTime
    frame_id: str

    @classmethod
    def create(cls, frame_id: str, timestamp_s: float | None = None) -> RosHeader:
        if timestamp_s is None:
            timestamp_s = time.time()
        return cls(stamp=RosTime.from_sec(timestamp_s), frame_id=frame_id)
