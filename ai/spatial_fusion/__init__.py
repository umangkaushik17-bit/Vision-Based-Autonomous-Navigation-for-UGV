"""Phase 9: Perception + Localization Spatial Fusion Module.

Provides world-referenced terrain representation, rigid SE(3) coordinate transformations,
temporal synchronization with motion compensation, and safety-precedence fusion.
"""

from ai.spatial_fusion.diagnostics import SpatialFusionDiagnostics
from ai.spatial_fusion.fusion_engine import SpatialFusionEngine, SpatialFusionResult
from ai.spatial_fusion.synchronization import SyncResult, TemporalSynchronizer
from ai.spatial_fusion.transforms import (
    SpatialCameraExtrinsics,
    create_se3_matrix,
    euler_to_rotation_matrix,
    invert_se3,
    rotation_matrix_to_euler,
    transform_points_3d,
)
from ai.spatial_fusion.world_map import WorldMapMetadata, WorldTerrainMap

__all__ = [
    "SpatialCameraExtrinsics",
    "create_se3_matrix",
    "euler_to_rotation_matrix",
    "invert_se3",
    "rotation_matrix_to_euler",
    "transform_points_3d",
    "SyncResult",
    "TemporalSynchronizer",
    "WorldMapMetadata",
    "WorldTerrainMap",
    "SpatialFusionEngine",
    "SpatialFusionResult",
    "SpatialFusionDiagnostics",
]
