"""Audit real outdoor tracking across varying temporal frame intervals."""

from __future__ import annotations

import json
from pathlib import Path
from ai.localization.base import CameraIntrinsics
from ai.localization.datasets import RUGDSequenceLoader
from ai.localization.diagnostics import TrackingDiagnostics
from ai.localization.mono_vo import MonocularVisualOdometry


def audit_temporal_sampling() -> list[dict]:
    camera = CameraIntrinsics(
        fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480
    )

    configs = [
        {"seq": "trail", "gap_desc": "1.0s (30 frames)", "max_frames": 50},
        {"seq": "creek", "gap_desc": "2.2s (65 frames)", "max_frames": 50},
        {"seq": "park-1", "gap_desc": "1.7s (50 frames)", "max_frames": 50},
        {"seq": "village", "gap_desc": "0.16s (5 frames)", "max_frames": 50},
    ]

    results = []
    print("\n" + "=" * 95)
    print("         PHASE 8: RUGD TEMPORAL SAMPLING AUDIT & STRESS-TEST ANALYSIS")
    print("=" * 95)

    for cfg in configs:
        seq = cfg["seq"]
        seq_path = f"data/processed/sequences/rugd/{seq}"
        loader = RUGDSequenceLoader(seq_path, max_frames=cfg["max_frames"])
        vo = MonocularVisualOdometry(camera)
        diag = TrackingDiagnostics()

        for frame in loader:
            res = vo.process_frame(frame.image_left, frame.timestamp)
            diag.record(res)

        summary = diag.compute_summary()
        res_entry = {
            "sequence_id": seq,
            "temporal_gap": cfg["gap_desc"],
            "total_frames": len(loader),
            "track_ok_pct": summary.get("tracking_ok_pct", 0.0),
            "degraded_pct": summary.get("degraded_pct", 0.0),
            "lost_pct": summary.get("lost_pct", 0.0),
            "tracking_availability_pct": summary.get("tracking_availability_pct", 0.0),
            "mean_inliers": summary.get("mean_inliers", 0.0),
            "loss_events": summary.get("loss_events", 0),
        }
        results.append(res_entry)

    print(f"\n{'Sequence':<12} | {'Temporal Gap':<18} | {'Frames':<8} | {'Track OK%':<10} | {'Avail%':<8} | {'Mean Inliers':<13} | {'Loss Events':<11}")
    print("-" * 95)
    for r in results:
        print(f"{r['sequence_id']:<12} | {r['temporal_gap']:<18} | {r['total_frames']:<8} | {r['track_ok_pct']:<10.1f} | {r['tracking_availability_pct']:<8.1f} | {r['mean_inliers']:<13.1f} | {r['loss_events']:<11}")
    print("=" * 95 + "\n")

    out_file = Path("outputs/slam/benchmarks/rugd_sampling_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    audit_temporal_sampling()
