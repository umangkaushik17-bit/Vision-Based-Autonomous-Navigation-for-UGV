"""Runtime performance diagnostics and spatial fusion metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ai.spatial_fusion.fusion_engine import SpatialFusionResult
from ai.spatial_fusion.world_map import WorldTerrainMap


@dataclass
class FusionLatencyRecord:
    """Latency breakdown for a single spatial fusion cycle in milliseconds."""

    transform_time_ms: float = 0.0
    raster_mapping_time_ms: float = 0.0
    fusion_update_time_ms: float = 0.0
    total_time_ms: float = 0.0


class SpatialFusionDiagnostics:
    """Diagnostics and telemetry collector for Phase 9 Spatial Fusion."""

    def __init__(self) -> None:
        self.cycle_latencies_ms: List[float] = []
        self.transform_latencies_ms: List[float] = []
        self.results_history: List[SpatialFusionResult] = []
        self.repeated_observation_displacements_m: List[float] = []

        # Event counts
        self.total_cycles = 0
        self.successful_fusions = 0
        self.blocked_lost_count = 0
        self.blocked_gap_count = 0
        self.degraded_fusions = 0
        self.total_cells_fused = 0

    def record_cycle(
        self,
        result: SpatialFusionResult,
        total_time_ms: float,
        transform_time_ms: float = 0.0
    ) -> None:
        """Record outcome and latency of a spatial fusion invocation."""
        self.total_cycles += 1
        self.cycle_latencies_ms.append(float(total_time_ms))
        self.transform_latencies_ms.append(float(transform_time_ms))
        self.results_history.append(result)

        if result.success:
            self.successful_fusions += 1
            self.total_cells_fused += result.fused_cells_count
            if result.tracking_state.value == "DEGRADED":
                self.degraded_fusions += 1
        else:
            if "lost" in result.reason.lower() or "untrusted" in result.reason.lower():
                self.blocked_lost_count += 1
            else:
                self.blocked_gap_count += 1

    def record_repeated_displacement(self, displacement_m: float) -> None:
        """Record metric spatial error between repeated observations of the same landmark."""
        self.repeated_observation_displacements_m.append(float(displacement_m))

    def compute_summary(self, world_map: Optional[WorldTerrainMap] = None) -> Dict[str, Any]:
        """Compute full statistical report over all recorded spatial fusion cycles."""
        latencies = np.array(self.cycle_latencies_ms, dtype=np.float64) if self.cycle_latencies_ms else np.zeros(1)
        tf_latencies = np.array(self.transform_latencies_ms, dtype=np.float64) if self.transform_latencies_ms else np.zeros(1)
        displacements = (
            np.array(self.repeated_observation_displacements_m, dtype=np.float64)
            if self.repeated_observation_displacements_m else np.zeros(1)
        )

        summary = {
            "total_cycles": self.total_cycles,
            "successful_fusions": self.successful_fusions,
            "blocked_lost_count": self.blocked_lost_count,
            "blocked_gap_count": self.blocked_gap_count,
            "degraded_fusions": self.degraded_fusions,
            "total_cells_fused": self.total_cells_fused,
            "mean_fused_cells_per_cycle": float(self.total_cells_fused / max(1, self.successful_fusions)),
            "fusion_latency_mean_ms": float(np.mean(latencies)),
            "fusion_latency_median_ms": float(np.median(latencies)),
            "fusion_latency_p95_ms": float(np.percentile(latencies, 95)),
            "transform_latency_mean_ms": float(np.mean(tf_latencies)),
            "repeated_displacement_mean_m": float(np.mean(displacements)),
            "repeated_displacement_max_m": float(np.max(displacements)),
        }

        if world_map is not None:
            summary.update(world_map.summary_stats())

        return summary
