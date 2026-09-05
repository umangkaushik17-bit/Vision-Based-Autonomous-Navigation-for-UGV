"""Common geometric, sensor, and state types for the UGV robotics stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class TraversabilityClass(IntEnum):
    """Discrete semantic classification of ground surface traversability."""
    SAFE = 0       # Flat ground, trail, dry dirt, benign grass
    UNKNOWN = 1    # Tall grass, obscured shadows, unmapped terrain
    UNSAFE = 2     # Boulders, ditches, water bodies, trees, steep drops

class RiskLevel(IntEnum):
    """Hierarchical risk level governing UGV speed and behavior."""
    LOW = 0       # Path is clear, nominal speed
    MEDIUM = 1    # Nearby hazards or elevated uncertainty, reduce speed
    HIGH = 2      # Hazard encroaching clearance corridor, trigger replanning
    CRITICAL = 3  # Immediate collision or unsafe cliff/obstacle, emergency halt

class MissionState(Enum):
    """High-level state machine state for autonomous navigation."""
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    REPLANNING = "REPLANNING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    GOAL_REACHED = "GOAL_REACHED"
    FAILED = "FAILED"

@dataclass
class Pose2D:
    """2D pose representation in robot odometry or map coordinate frame."""
    x: float = 0.0          # meters
    y: float = 0.0          # meters
    yaw: float = 0.0        # radians (-pi to pi)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=np.float32)

@dataclass
class Pose3D:
    """Full 6-DOF pose representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

@dataclass
class ObstacleBBox:
    """Detected 2D bounding box representing an obstacle in camera pixel coordinates."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    confidence: float
    estimated_distance_m: Optional[float] = None
    traversability_override: Optional[TraversabilityClass] = None

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.width / 2.0, self.ymin + self.height / 2.0)

@dataclass
class VelocityCommand:
    """Twist command (linear and angular velocities) sent to simulated UGV."""
    linear_x: float = 0.0   # m/s
    angular_z: float = 0.0  # rad/s
    timestamp: float = 0.0

    def clamp(self, max_linear: float, max_angular: float) -> VelocityCommand:
        """Return a new velocity command clamped within dynamic limits."""
        return VelocityCommand(
            linear_x=float(np.clip(self.linear_x, -max_linear, max_linear)),
            angular_z=float(np.clip(self.angular_z, -max_angular, max_angular)),
            timestamp=self.timestamp
        )

@dataclass
class CostmapGrid:
    """2D grid representation of terrain traversability and obstacle costs."""
    grid: np.ndarray             # 2D array of uint8 [0..254] (0=free/safe, 254=lethal)
    resolution: float            # meters per cell
    origin_x: float              # world x coordinate of grid[0, 0] in meters
    origin_y: float              # world y coordinate of grid[0, 0] in meters
    width_cells: int
    height_cells: int
    timestamp: float = 0.0

    def world_to_grid(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates (meters) to grid cell indices (col, row)."""
        gx = int(np.floor((wx - self.origin_x) / self.resolution))
        gy = int(np.floor((wy - self.origin_y) / self.resolution))
        if 0 <= gx < self.width_cells and 0 <= gy < self.height_cells:
            return gx, gy
        return None

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell coordinates (col, row) to world coordinates (meters)."""
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

@dataclass
class FramePacket:
    """Unified sensor data packet representing a single camera / sensor capture cycle."""
    timestamp: float
    sequence_id: int
    rgb_image: np.ndarray                        # H x W x 3 (BGR or RGB uint8)
    depth_image: Optional[np.ndarray] = None     # H x W (float32 meters)
    imu_data: Optional[Dict[str, Any]] = None    # Linear accel & angular velocity
    estimated_pose: Optional[Pose2D] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
