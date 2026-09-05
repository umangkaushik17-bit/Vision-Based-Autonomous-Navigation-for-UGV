"""Tracking diagnostics, inlier statistics, and state timeline analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ai.localization.base import TrackingState
from ai.localization.result import LocalizationEstimate


@dataclass
class FrameDiagnosticRecord:
    timestamp: float
    state: TrackingState
    confidence: float
    features_detected: int
    inliers: int
    is_keyframe: bool
    position: np.ndarray


class TrackingDiagnostics:
    """Collects and analyzes visual tracking statistics over a localization session."""

    def __init__(self):
        self.records: List[FrameDiagnosticRecord] = []

    def record(self, estimate: LocalizationEstimate) -> None:
        self.records.append(FrameDiagnosticRecord(
            timestamp=estimate.timestamp,
            state=estimate.tracking_state,
            confidence=estimate.confidence,
            features_detected=estimate.tracked_features,
            inliers=estimate.inliers,
            is_keyframe=estimate.is_keyframe,
            position=estimate.position.copy()
        ))

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate diagnostic metrics across the sequence."""
        if not self.records:
            return {"total_frames": 0, "status": "no_data"}

        N = len(self.records)
        states = [r.state for r in self.records]
        inliers = np.array([r.inliers for r in self.records])
        confs = np.array([r.confidence for r in self.records])

        ok_count = states.count(TrackingState.TRACKING_OK)
        degraded_count = states.count(TrackingState.DEGRADED)
        lost_count = states.count(TrackingState.LOST)
        relocalizing_count = states.count(TrackingState.RELOCALIZING)
        recovered_count = states.count(TrackingState.RECOVERED)
        keyframe_count = sum(1 for r in self.records if r.is_keyframe)

        # Usable tracking frames: OK + DEGRADED + RECOVERED
        available_count = ok_count + degraded_count + recovered_count

        # Transition event counts and continuous segments
        longest_ok_streak = 0
        current_ok_streak = 0
        longest_avail_streak = 0
        current_avail_streak = 0
        loss_events = 0
        degraded_events = 0
        prev_state = None

        for s in states:
            if s == TrackingState.TRACKING_OK:
                current_ok_streak += 1
                if current_ok_streak > longest_ok_streak:
                    longest_ok_streak = current_ok_streak
            else:
                current_ok_streak = 0

            if s != TrackingState.LOST:
                current_avail_streak += 1
                if current_avail_streak > longest_avail_streak:
                    longest_avail_streak = current_avail_streak
            else:
                current_avail_streak = 0

            if prev_state is not None:
                if s == TrackingState.LOST and prev_state != TrackingState.LOST:
                    loss_events += 1
                elif s == TrackingState.DEGRADED and prev_state != TrackingState.DEGRADED:
                    degraded_events += 1
            prev_state = s

        return {
            "total_frames": N,
            "tracking_ok_count": ok_count,
            "degraded_count": degraded_count,
            "lost_count": lost_count,
            "relocalizing_count": relocalizing_count,
            "recovered_count": recovered_count,
            "tracking_ok_pct": (ok_count / N) * 100.0,
            "degraded_pct": (degraded_count / N) * 100.0,
            "lost_pct": (lost_count / N) * 100.0,
            "tracking_availability_pct": (available_count / N) * 100.0,
            "longest_continuous_ok_frames": longest_ok_streak,
            "longest_continuous_available_frames": longest_avail_streak,
            "loss_events": loss_events,
            "degraded_events": degraded_events,
            "mean_inliers": float(np.mean(inliers)),
            "min_inliers": int(np.min(inliers)),
            "max_inliers": int(np.max(inliers)),
            "mean_confidence": float(np.mean(confs)),
            "total_keyframes": keyframe_count,
        }

    def export_csv(self, filepath: Path) -> Path:
        """Export diagnostic timeline to CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        lines = ["timestamp,state,confidence,features,inliers,is_keyframe,x,y,z\n"]
        for r in self.records:
            pos = r.position
            lines.append(
                f"{r.timestamp:.4f},{r.state.value},{r.confidence:.3f},{r.features_detected},"
                f"{r.inliers},{int(r.is_keyframe)},{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}\n"
            )
        with open(filepath, "w") as f:
            f.writelines(lines)
        return filepath
