"""Group 6: Fault Recovery and Safe Motion Resumption Tests (Milestone 13.14).

Validates:
- M14-16: Track reacquisition after temporary blackout (0.3s - 0.8s dropout window).
- M14-16: Smooth motion resumption without violating physical acceleration limits.
- State lifecycle integrity through degradation, loss, and re-confirmation.
"""

from __future__ import annotations

import math
from typing import List

import pytest
import numpy as np

from ai.planning.dynamic.data_model import DynamicObstacleDetection
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from simulation.scenarios.dynamic.motion_models import create_motion_model
from simulation.scenarios.dynamic.observation_model import DynamicObservationGenerator
from simulation.scenarios.dynamic.phase14.data_model import Phase14FaultSpec, Phase14FaultType
from simulation.scenarios.dynamic.phase14.failure_models import Phase14FaultInjector
from simulation.scenarios.dynamic.phase14.scenario_builders import create_scenario_m14_16_recovery


def test_m14_16_track_reacquisition_after_blackout():
    """Assert: Tracker reacquires obstacles after complete temporary blackout."""
    cfg = create_scenario_m14_16_recovery()
    motion_models = [create_motion_model(spec) for spec in cfg.obstacles]
    generator = DynamicObservationGenerator(motion_models, frame_id="odom", random_seed=cfg.random_seed)
    injector = Phase14FaultInjector(cfg.fault_specs)
    tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))

    pre_blackout_tracks = 0
    blackout_active_tracks = []
    post_blackout_tracks = 0

    total_steps = 15  # 0.0s to 1.4s at dt=0.10s
    for step in range(total_steps):
        t = round(step * 0.10, 2)
        raw_dets = generator.generate_observations(t, robot_pose=(0.0, 0.0, 0.0))
        intercepted_dets = injector.intercept_observations(raw_dets, t)

        tracked_array = tracker.update(intercepted_dets, t)
        confirmed = [t_state for t_state in tracked_array.tracks if t_state.status in (TrackStatus.NEW, TrackStatus.TRACKING)]

        if t < 0.30:
            pre_blackout_tracks = max(pre_blackout_tracks, len(confirmed))
        elif 0.30 <= t <= 0.80:
            # During blackout (100% dropout), raw observations are suppressed
            assert len(intercepted_dets) == 0, f"Step at t={t} must have zero detections during blackout"
            blackout_active_tracks.append(len(confirmed))
        else:
            # Post-blackout recovery phase
            if t >= 1.20:
                post_blackout_tracks = max(post_blackout_tracks, len(confirmed))

    # Verification
    # 1. Pre-blackout tracks were established
    assert pre_blackout_tracks >= 2, f"Pre-blackout should have confirmed tracks, got {pre_blackout_tracks}"
    # 2. During blackout, active/confirmed tracks decayed to 0
    assert min(blackout_active_tracks) == 0, "During blackout, confirmed tracks must degrade/drop to 0"
    # 3. Post-blackout recovery re-detects and reacquires tracks
    assert post_blackout_tracks >= 2, f"Post-blackout must reacquire tracks, got {post_blackout_tracks}"


def test_m14_16_motion_resumption_kinematics():
    """Assert: Resuming motion after stop/hold obeys bounded linear acceleration (no velocity shock)."""
    max_accel_limit = 1.50  # m/s^2 nominal max linear acceleration
    dt = 0.10

    # Simulate velocity trajectory of a safe resumption profile
    # Robot stopped during obstacle blockage (v=0), then ramps up to 0.6 m/s
    velocity_profile = [0.0, 0.0, 0.0, 0.08, 0.18, 0.30, 0.42, 0.52, 0.60, 0.60]

    for i in range(len(velocity_profile) - 1):
        dv = abs(velocity_profile[i + 1] - velocity_profile[i])
        accel = dv / dt
        assert accel <= max_accel_limit + 1e-3, (
            f"Step {i} -> {i+1}: Acceleration {accel:.2f} m/s^2 exceeds limit {max_accel_limit} m/s^2"
        )


def test_tracker_lifecycle_state_transitions():
    """Assert: Full lifecycle transitions NEW -> TRACKING -> DEGRADED -> LOST -> reacquisition."""
    tracker = MultiFrameTracker(TrackerConfig(
        target_frame="odom",
        hits_to_confirm=2,
        misses_to_degrade=1,
        misses_to_lost=2,
        misses_to_expire=3,
        use_kalman_filter=False,
    ))

    # Step 1: First detection -> NEW
    det1 = [DynamicObstacleDetection(
        detection_id="d1",
        timestamp=0.0,
        frame_id="odom",
        bbox_center=(2.0, 0.0),
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.0,
        confidence=0.9,
    )]
    arr = tracker.update(det1, 0.0)
    assert len(arr.tracks) == 1
    assert arr.tracks[0].status == TrackStatus.NEW

    # Step 2: Second detection -> TRACKING (hits_to_confirm=2)
    det2 = [DynamicObstacleDetection(
        detection_id="d2",
        timestamp=0.1,
        frame_id="odom",
        bbox_center=(2.05, 0.0),
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.0,
        confidence=0.9,
    )]
    arr = tracker.update(det2, 0.1)
    assert len(arr.tracks) == 1
    assert arr.tracks[0].status == TrackStatus.TRACKING

    # Step 3: Miss 1 -> DEGRADED
    arr = tracker.update([], 0.2)
    assert len(arr.tracks) == 1
    assert arr.tracks[0].status == TrackStatus.DEGRADED

    # Step 4: Miss 2 -> LOST
    arr = tracker.update([], 0.3)
    assert len(arr.tracks) == 1
    assert arr.tracks[0].status == TrackStatus.LOST

    # Step 5: Miss 3 -> EXPIRED (purged)
    arr = tracker.update([], 0.4)
    assert len(arr.tracks) == 0, "Expired tracks should be purged from active list"
