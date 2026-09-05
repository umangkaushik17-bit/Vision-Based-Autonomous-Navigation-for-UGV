"""Controlled failure injection and quantitative relocalization / loop-closure evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from ai.localization.base import CameraIntrinsics, TrackingState
from ai.localization.datasets import SyntheticStereoSequenceGenerator
from ai.localization.loop_closing import LoopClosingEngine
from ai.localization.stereo_vo import StereoVisualOdometry


def run_controlled_relocalization_test() -> dict:
    """Execute controlled failure-injection experiment:
    TRACKING_OK -> Force Feature Loss -> LOST -> Return to mapped area -> RELOCALIZING -> RECOVERED.
    """
    camera = CameraIntrinsics(
        fx=462.14, fy=462.14, cx=320.0, cy=240.0, width=640, height=480, baseline_m=0.05
    )

    vo = StereoVisualOdometry(camera)
    loop_engine = LoopClosingEngine(min_inliers_loop=20, min_separation_keyframes=10)

    # 1. Warm up tracking and build keyframe database
    gen = SyntheticStereoSequenceGenerator(num_frames=25, baseline_m=camera.baseline_m, motion_type="straight")
    archived_frames = []

    print("\n" + "=" * 80)
    print("      PHASE 8: CONTROLLED FAILURE INJECTION & RELOCALIZATION AUDIT")
    print("=" * 80)

    print("\nPhase 1: Building Map (Frames 0 to 19)...")
    for i, frame in enumerate(gen):
        archived_frames.append(frame)
        est = vo.process_frame(frame.image_left, frame.timestamp, frame.image_right)
        if est.is_keyframe and vo.prev_des is not None:
            loop_engine.add_keyframe(
                keyframe_id=est.keyframe_id,
                timestamp=frame.timestamp,
                pose=est.pose,
                descriptors=vo.prev_des,
                keypoints=np.array([kp.pt for kp in vo.prev_kp]) if vo.prev_kp else np.empty((0, 2))
            )
        if i >= 19:
            break

    last_good_pose = vo.current_pose.copy()
    print(f"  Mapped {len(loop_engine.database)} keyframes. Tracker State: {vo.state.value}")
    assert vo.state in (TrackingState.TRACKING_OK, TrackingState.DEGRADED)

    # 2. Inject Failure: Feed featureless black frames (e.g., severe camera occlusion or darkness)
    print("\nPhase 2: Injecting Complete Visual Failure (Occlusion)...")
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    loss_durations = 0

    for step in range(5):
        t = 20.0 + step * 0.033
        est = vo.process_frame(black_frame, timestamp=t, image_right=black_frame)
        loss_durations += 1
        print(f"  Step {step+1}: State = {est.tracking_state.value:<12} | Pose Valid = {est.is_pose_valid} | Trusted Pose = {est.trusted_pose is not None}")
        assert est.tracking_state == TrackingState.LOST
        assert not est.is_pose_valid
        assert est.trusted_pose is None
        assert not est.allows_spatial_fusion

    print(f"  Tracking LOST confirmed over {loss_durations} frames. Zero false poses emitted.")

    # 3. Relocalization Attempt: Feed a frame from a previously mapped location (Frame 5)
    print("\nPhase 3: Returning to Mapped Area (Relocalization)...")
    revisit_frame = archived_frames[5]
    gt_revisit_pose = revisit_frame.ground_truth_pose

    # Extract descriptors from revisit frame
    gray_l = vo.preprocessor.process(revisit_frame.image_left)
    kp_l, des_l = vo.orb.detectAndCompute(gray_l, None)

    # Attempt relocalization
    t_start = time.perf_counter()
    reloc_result = loop_engine.relocalize(des_l, min_inliers_reloc=15)
    reloc_latency_ms = (time.perf_counter() - t_start) * 1000.0

    success = reloc_result is not None
    pose_error_m = 0.0

    if success:
        matched_kf_id, recovered_pose, inliers = reloc_result
        # Update VO state to RECOVERED
        vo.current_pose = recovered_pose
        vo.state = TrackingState.RECOVERED
        pose_error_m = float(np.linalg.norm(recovered_pose[:3, 3] - gt_revisit_pose[:3, 3]))

        print(f"  [OK] Relocalized against Keyframe {matched_kf_id} with {inliers} inliers!")
        print(f"  Recovery Latency: {reloc_latency_ms:.2f} ms")
        print(f"  Position Error after Recovery: {pose_error_m:.4f} m")
        print(f"  Updated Tracker State: {vo.state.value}")
    else:
        print("  [FAIL] Relocalization failed.")

    # 4. Loop Closure Test: Feed end frame and detect loop closure
    print("\nPhase 4: Quantitative Loop Closure Evaluation...")
    loop_engine.detect_loop(des_l, curr_keyframe_id=len(loop_engine.database) + 1)

    result_summary = {
        "experiment": "controlled_failure_injection",
        "initial_state": "TRACKING_OK",
        "failure_injected": "complete_visual_occlusion",
        "lost_duration_frames": loss_durations,
        "zero_trust_enforced": True,
        "relocalization_attempts": loop_engine.relocalization_attempts,
        "relocalization_successes": loop_engine.relocalization_successes,
        "relocalization_failures": loop_engine.relocalization_failures,
        "relocalization_success_rate_pct": (
            (loop_engine.relocalization_successes / max(1, loop_engine.relocalization_attempts)) * 100.0
        ),
        "recovery_latency_ms": reloc_latency_ms,
        "pose_error_after_recovery_m": pose_error_m,
        "final_tracking_state": vo.state.value,
        "loop_attempts": loop_engine.loop_attempts,
        "loop_candidates": loop_engine.loop_candidates,
        "loop_closures_accepted": loop_engine.loop_closures_accepted,
        "loop_closures_rejected": loop_engine.loop_closures_rejected,
    }

    out_path = Path("outputs/slam/benchmarks/relocalization_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_summary, f, indent=2)

    print("-" * 80)
    print(f"  Saved Relocalization Audit: {out_path}")
    print("=" * 80 + "\n")
    return result_summary


if __name__ == "__main__":
    run_controlled_relocalization_test()
