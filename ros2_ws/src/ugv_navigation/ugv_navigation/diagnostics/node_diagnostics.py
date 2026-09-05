"""Diagnostic status and telemetry tracker for UGV ROS2 nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Any


@dataclass
class DiagnosticStatus:
    """Represents diagnostic_msgs/DiagnosticStatus."""
    OK = 0
    WARN = 1
    ERROR = 2

    level: int = OK
    name: str = "ugv_diagnostics"
    message: str = "System nominal"
    hardware_id: str = "ugv_chassis_core"
    values: Dict[str, Any] = field(default_factory=dict)


class NodeDiagnosticsTracker:
    """Aggregates telemetry across perception, localization, and spatial fusion nodes."""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.tracking_state_name: str = "UNINITIALIZED"
        self.confidence: float = 0.0
        self.tracked_features: int = 0
        self.inliers: int = 0
        self.stale_message_count: int = 0
        self.rejected_pose_count: int = 0
        self.tf_lookup_failures: int = 0
        self.last_update_time: float = time.time()
        self.update_count: int = 0
        self.latencies_ms: list[float] = []

    def record_cycle(
        self,
        latency_ms: float,
        tracking_state_name: str = "TRACKING_OK",
        confidence: float = 1.0,
        features: int = 0,
        inliers: int = 0,
        rejected: bool = False,
        tf_failed: bool = False,
        stale: bool = False,
    ) -> None:
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 100:
            self.latencies_ms.pop(0)

        self.tracking_state_name = tracking_state_name
        self.confidence = confidence
        self.tracked_features = features
        self.inliers = inliers
        self.last_update_time = time.time()
        self.update_count += 1

        if rejected:
            self.rejected_pose_count += 1
        if tf_failed:
            self.tf_lookup_failures += 1
        if stale:
            self.stale_message_count += 1

    def get_status(self) -> DiagnosticStatus:
        mean_lat = sum(self.latencies_ms) / max(1, len(self.latencies_ms))

        level = DiagnosticStatus.OK
        msg = "Nominal operation"

        if self.tracking_state_name == "LOST":
            level = DiagnosticStatus.ERROR
            msg = "Localization tracking LOST - Zero-trust safety engaged"
        elif self.tracking_state_name == "DEGRADED":
            level = DiagnosticStatus.WARN
            msg = "Localization tracking DEGRADED - Caution speed & conservative fusion active"
        elif self.tf_lookup_failures > 0 or self.stale_message_count > 0:
            level = DiagnosticStatus.WARN
            msg = "Warning: Stale messages or TF lookup failures detected"

        values = {
            "node_name": self.node_name,
            "tracking_state": self.tracking_state_name,
            "confidence": f"{self.confidence:.2f}",
            "features": self.tracked_features,
            "inliers": self.inliers,
            "mean_latency_ms": f"{mean_lat:.2f}",
            "rejected_poses": self.rejected_pose_count,
            "tf_failures": self.tf_lookup_failures,
            "stale_messages": self.stale_message_count,
            "total_updates": self.update_count,
        }

        return DiagnosticStatus(
            level=level,
            name=f"{self.node_name}_diagnostics",
            message=msg,
            hardware_id="ugv_core",
            values=values,
        )
