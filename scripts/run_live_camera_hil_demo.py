#!/usr/bin/env python3
"""Plug-and-Play Live Webcam / Stereo Hardware-in-the-Loop (HIL) Table Demo Runner (Subphase 16.3).

Usage:
  - Live USB webcam GUI mode:
      python scripts/run_live_camera_hil_demo.py --camera-id 0
  - Headless CI / automated verification mode (30 frames):
      python scripts/run_live_camera_hil_demo.py --headless --frames 30
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import cv2
import numpy as np

from production.perception.hil_camera_bridge import (
    HILCameraNavigationBridge,
    OpenCVCameraSource,
    SyntheticCameraSource,
)


def run_hil_demo(
    num_frames: int = 30,
    headless: bool = True,
    camera_id: int = 0,
    use_synthetic: bool = True,
) -> Dict[str, Any]:
    """Execute closed-loop HIL camera navigation demo and return telemetry summary."""
    bridge = HILCameraNavigationBridge()
    synth_src = SyntheticCameraSource(width=320, height=240, integrator=bridge.integrator)
    cam_src = None if use_synthetic else OpenCVCameraSource(camera_id=camera_id, allow_fallback=True)

    # Warm-up 2 frames
    warm_rgb = synth_src.read_frame()
    bridge.step_frame(warm_rgb, timestamp_s=0.1, render_hud=True)
    bridge.step_frame(warm_rgb, timestamp_s=0.2, render_hud=True)
    bridge.frame_counter = 0

    latencies_ms: List[float] = []
    frame_records: List[Dict[str, Any]] = []
    last_hud = None

    for idx in range(num_frames):
        ts = 1.0 + idx * 0.033
        # Inject obstacle in frames 10..19 and simulated pitch tilt in frames 20..29
        obs_xy = (2.8, 0.0) if (10 <= idx < 20) else None
        pitch_deg = -8.0 if (idx >= 20) else 0.0
        roll_deg = 6.0 * math.sin(idx * 0.25) if (idx >= 20) else 0.0

        if cam_src is not None:
            rgb = cam_src.read(obstacle_ground_xy=obs_xy, roll_deg=roll_deg, pitch_deg=pitch_deg)
        else:
            rgb = synth_src.read_frame(obstacle_ground_xy=obs_xy, roll_deg=roll_deg, pitch_deg=pitch_deg)

        step_res = bridge.step_frame(
            rgb=rgb,
            timestamp_s=ts,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            render_hud=True,
        )
        latencies_ms.append(step_res.total_latency_ms)
        frame_records.append(step_res.to_dict())
        last_hud = step_res.composite_hud

        if not headless and last_hud is not None:
            cv2.imshow("SIH26126 Live Camera HIL Navigation Demo", cv2.cvtColor(last_hud, cv2.COLOR_RGB2BGR))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    if cam_src is not None:
        cam_src.release()
    if not headless:
        cv2.destroyAllWindows()

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p95_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    mean_fps = 1000.0 / max(1e-3, mean_ms)

    return {
        "frames_executed": len(frame_records),
        "headless": headless,
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "mean_fps": round(mean_fps, 1),
        "hud_shape": list(last_hud.shape) if last_hud is not None else None,
        "frame_records": frame_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26126 Live Camera HIL Demo Runner")
    parser.add_argument("--headless", action="store_true", help="Run in headless CI mode without GUI window")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to process")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV USB camera device ID")
    parser.add_argument("--live-webcam", action="store_true", help="Attempt live USB webcam capture")
    args = parser.parse_args()

    summary = run_hil_demo(
        num_frames=args.frames,
        headless=args.headless,
        camera_id=args.camera_id,
        use_synthetic=not args.live_webcam,
    )
    print(
        f"[HIL DEMO] Completed {summary['frames_executed']} frames | "
        f"Mean Latency: {summary['mean_latency_ms']:.2f} ms | "
        f"Mean FPS: {summary['mean_fps']:.1f} FPS"
    )


if __name__ == "__main__":
    main()
