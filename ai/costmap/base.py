"""Base definitions, layer enums, and data contracts for the BEV Terrain Costmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CostmapLayerType(str, Enum):
    """Explicitly inspectable costmap layers."""

    TRAVERSABILITY = "traversability"            # Layer 1: Learned vision drivability (0.0=blocked, 1.0=safe)
    HAZARD = "hazard"                            # Layer 2: Explicit hazards (0.0=none, 1.0=critical)
    SLOPE = "slope"                              # Layer 3: Surface slope angle (degrees)
    ROUGHNESS = "roughness"                      # Layer 4: Surface roughness / coplanarity variance
    HEIGHT = "height"                            # Layer 5: Vertical step height variation (meters)
    UNKNOWN = "unknown"                          # Layer 6: Observation gap / model disagreement (bool)
    CONFIDENCE = "confidence"                    # Layer 7: Sensor / model confidence (0.0 to 1.0)
    TEMPORAL_PERSISTENCE = "temporal_persistence"# Layer 8: Multi-frame observation hit count
    FINAL_COST = "final_cost"                    # Fused cost field (0.0=preferred, 1.0=lethal)


class UnknownPolicyMode(str, Enum):
    """Handling strategy for unobserved or ambiguous cells."""

    CONSERVATIVE = "CONSERVATIVE"  # High cost (0.65), cautious exploration
    MODERATE = "MODERATE"          # Neutral cost (0.50), balanced path exploration
    EXPLORATORY = "EXPLORATORY"    # Low/moderate cost (0.35), favors discovery


class CoordinateFrame(str, Enum):
    """Standardized coordinate systems in the UGV architecture."""

    CAMERA_OPTICAL = "camera_optical"  # X right, Y down, Z forward
    ROBOT_BASE = "base_link"           # X forward, Y left, Z up (ISO 8855 / ROS standard)
    LOCAL_BEV = "local_bev"            # Top-down raster grid aligned with robot base
    MAP = "map"                        # Global fixed navigation frame


@dataclass
class VehicleFootprint:
    """Geometric dimensions and safety clearances for the autonomous rover."""

    length_m: float = 0.65
    width_m: float = 0.45
    inscribed_radius_m: float = 0.35
    inflation_radius_m: float = 0.55
    cost_scaling_factor: float = 3.0

    @property
    def footprint_polygon(self) -> np.ndarray:
        """Return 4-corner bounding rectangle vertices relative to robot center."""
        half_l = self.length_m / 2.0
        half_w = self.width_m / 2.0
        return np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ], dtype=np.float32)


@dataclass
class CostmapMetadata:
    """Metadata describing spatial extent, origin, resolution, and timing of a BEV costmap."""

    resolution_m: float
    shape: Tuple[int, int]  # (rows, cols) -> (n_x, n_y)
    x_range_m: Tuple[float, float]  # (min_x, max_x) forward bounds
    y_range_m: Tuple[float, float]  # (min_y, max_y) lateral bounds
    frame_id: str = "base_link"
    timestamp: float = 0.0
    coordinate_convention: str = "X=forward, Y=left, Z=up; Grid (0,0) at (max_x, max_y)"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return {
            "resolution_m": self.resolution_m,
            "shape": list(self.shape),
            "x_range_m": list(self.x_range_m),
            "y_range_m": list(self.y_range_m),
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "coordinate_convention": self.coordinate_convention,
            "extra": self.extra
        }
