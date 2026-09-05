"""Pure-Python data model and message adapter for WorldTerrainMap.msg."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ugv_navigation.utils.time_utils import RosHeader, RosTime


@dataclass
class WorldTerrainMapMsg:
    """In-memory representation of WorldTerrainMap.msg conforming to ROS2 message conventions."""

    header: RosHeader
    resolution: float
    origin_x: float
    origin_y: float
    origin_z: float
    width: int
    height: int
    cost: list[float] = field(default_factory=list)
    traversability: list[float] = field(default_factory=list)
    obstacle: list[bool] = field(default_factory=list)
    hazard: list[float] = field(default_factory=list)
    unknown: list[bool] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    slope: list[float] = field(default_factory=list)
    roughness: list[float] = field(default_factory=list)
    height_var: list[float] = field(default_factory=list)
    observation_count: list[int] = field(default_factory=list)

    @classmethod
    def from_world_terrain_map(cls, wmap, frame_id: str = "odom", timestamp_s: float | None = None) -> WorldTerrainMapMsg:
        """Construct from a Phase 9 WorldTerrainMap object."""
        if timestamp_s is None:
            # Extract latest observed timestamp or current time
            t_max = float(np.max(wmap.last_observed_timestamp))
            timestamp_s = t_max if t_max > 0.0 else 0.0

        header = RosHeader.create(frame_id=frame_id, timestamp_s=timestamp_s)
        H, W = wmap.shape
        origin_x = float(wmap.x_bounds_m[0])
        origin_y = float(wmap.y_bounds_m[0])
        origin_z = 0.0

        return cls(
            header=header,
            resolution=float(wmap.resolution_m),
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            width=W,
            height=H,
            cost=wmap.cost.flatten().tolist(),
            traversability=wmap.traversability.flatten().tolist(),
            obstacle=wmap.obstacle.flatten().tolist(),
            hazard=wmap.hazard.flatten().tolist(),
            unknown=wmap.unknown.flatten().tolist(),
            confidence=wmap.confidence.flatten().tolist(),
            slope=wmap.slope.flatten().tolist(),
            roughness=wmap.roughness.flatten().tolist(),
            height_var=wmap.height_var.flatten().tolist(),
            observation_count=wmap.observation_count.flatten().tolist(),
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert flat list fields back into (height, width) 2D NumPy raster layers."""
        shape = (self.height, self.width)
        return {
            "cost": np.array(self.cost, dtype=np.float32).reshape(shape),
            "traversability": np.array(self.traversability, dtype=np.float32).reshape(shape),
            "obstacle": np.array(self.obstacle, dtype=bool).reshape(shape),
            "hazard": np.array(self.hazard, dtype=np.float32).reshape(shape),
            "unknown": np.array(self.unknown, dtype=bool).reshape(shape),
            "confidence": np.array(self.confidence, dtype=np.float32).reshape(shape),
            "slope": np.array(self.slope, dtype=np.float32).reshape(shape),
            "roughness": np.array(self.roughness, dtype=np.float32).reshape(shape),
            "height_var": np.array(self.height_var, dtype=np.float32).reshape(shape),
            "observation_count": np.array(self.observation_count, dtype=np.int32).reshape(shape),
        }

    def to_world_terrain_map(self):
        """Reconstruct a Phase 9 WorldTerrainMap instance from this message."""
        from ai.spatial_fusion.world_map import WorldTerrainMap

        x_min = float(self.origin_x)
        x_max = float(self.origin_x + self.height * self.resolution)
        y_min = float(self.origin_y)
        y_max = float(self.origin_y + self.width * self.resolution)

        wmap = WorldTerrainMap(
            resolution_m=self.resolution,
            x_bounds_m=(x_min, x_max),
            y_bounds_m=(y_min, y_max),
            frame_id=self.header.frame_id,
            created_timestamp=self.header.stamp.to_sec(),
        )

        arrays = self.to_arrays()
        wmap.cost = arrays["cost"]
        wmap.traversability = arrays["traversability"]
        wmap.obstacle = arrays["obstacle"]
        wmap.hazard = arrays["hazard"]
        wmap.unknown = arrays["unknown"]
        wmap.confidence = arrays["confidence"]
        wmap.slope = arrays["slope"]
        wmap.roughness = arrays["roughness"]
        wmap.height_var = arrays["height_var"]
        wmap.observation_count = arrays["observation_count"]
        # Reconstruct observation epoch from header timestamp for non-zero observation cells
        wmap.last_observed_timestamp = np.where(
            wmap.observation_count > 0,
            self.header.stamp.to_sec(),
            0.0,
        )
        return wmap
