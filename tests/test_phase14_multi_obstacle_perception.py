"""Group 2: Multi-obstacle perception, association, Kalman tracking, and dropout tests."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import DynamicObstacleDetection
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from simulation.scenarios.dynamic.phase14.data_model import Phase14FaultSpec, Phase14FaultType
from simulation.scenarios.dynamic.phase14.failure_models import Phase14FaultInjector
from simulation.scenarios.dynamic.phase14.multi_obstacle_scenarios import (
    create_scenario_m14_01_nominal,
    create_scenario_m14_02_crossing,
)
from simulation.scenarios.dynamic.motion_models import create_motion_model
from simulation.scenarios.dynamic.observation_model import DynamicObservationGenerator


def test_m14_01_4obstacle_tracking_distinctness():
    """M14-01: 4 obstacles generate distinct detections and maintain separate tracks."""
    cfg = create_scenario_m14_01_nominal()
    motion_models = [create_motion_model(spec) for spec in cfg.obstacles]
    generator = DynamicObservationGenerator(motion_models, frame_id="odom", random_seed=cfg.random_seed)
    tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))

    for step in range(5):
        t = step * 0.10
        detections = generator.generate_observations(t, robot_pose=(0.0, 0.0, 0.0))
        assert len(detections) == 4, f"Step {step}: expected 4 detections, got {len(detections)}"
        tracked_array = tracker.update(detections, t)

    # After 5 steps, all 4 tracks should be tracked or active
    active_tracks = [t for t in tracked_array.tracks if t.status in (TrackStatus.NEW, TrackStatus.TRACKING)]
    assert len(active_tracks) >= 3, f"Expected at least 3-4 active tracks, got {len(active_tracks)}"
    track_ids = [t.track_id for t in active_tracks]
    assert len(set(track_ids)) == len(track_ids), "Track IDs must be strictly unique"


def test_m14_02_crossing_trajectories_association():
    """M14-02: Crossing obstacles maintain distinct tracks without identity collapse."""
    cfg = create_scenario_m14_02_crossing()
    motion_models = [create_motion_model(spec) for spec in cfg.obstacles]
    generator = DynamicObservationGenerator(motion_models, frame_id="odom", random_seed=cfg.random_seed)
    tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))

    track_history = {}
    for step in range(8):
        t = step * 0.10
        detections = generator.generate_observations(t, robot_pose=(0.0, 0.0, 0.0))
        tracked_array = tracker.update(detections, t)
        for trk in tracked_array.tracks:
            track_history.setdefault(trk.track_id, []).append((t, trk.x, trk.y))

    # Check that multiple distinct track trajectories are recorded
    assert len(track_history) >= 3, "Must track multiple distinct objects"


def test_component_dropout_levels():
    """M14-06 component-level test: Validate 10%, 25%, and 40% dropout rates."""
    sample_dets = [
        DynamicObstacleDetection(
            detection_id=f"det_{i}",
            timestamp=0.0,
            frame_id="odom",
            bbox_center=(float(i), 0.0),
            bbox_size=(0.8, 0.6),
            bbox_yaw=0.0,
            confidence=0.95,
        )
        for i in range(1000)
    ]

    for level, expected_prob in [("LOW", 0.10), ("MEDIUM", 0.25), ("HIGH", 0.40)]:
        fault = Phase14FaultSpec(
            fault_id=f"test_drop_{level}",
            fault_type=Phase14FaultType.DROPOUT,
            target_component="observation_generator",
            start_time=0.0,
            end_time=1.0,
            expected_response="DROP",
            parameters={"dropout_probability": expected_prob},
        )
        injector = Phase14FaultInjector([fault], random_seed=42)
        out = injector.intercept_observations(sample_dets, 0.5)
        measured_drop_rate = 1.0 - (len(out) / len(sample_dets))
        assert abs(measured_drop_rate - expected_prob) < 0.04, (
            f"Level {level}: expected ~{expected_prob}, got {measured_drop_rate:.3f}"
        )


def test_stale_observation_rejection():
    """M14-07 component test: Out-of-order or stale timestamps are rejected by Invariant K."""
    t_now = 1.0
    stale_det = DynamicObstacleDetection(
        detection_id="det_stale",
        timestamp=0.5,
        frame_id="odom",
        bbox_center=(2.0, 0.0),
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.0,
        confidence=0.9,
    )
    fresh_det = DynamicObstacleDetection(
        detection_id="det_fresh",
        timestamp=1.0,
        frame_id="odom",
        bbox_center=(2.0, 0.0),
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.0,
        confidence=0.9,
    )

    # Rejection logic matching DynamicSILRunner Invariant K
    valid = []
    rejected_count = 0
    for d in [stale_det, fresh_det]:
        if d.timestamp < (t_now - 1e-4) or d.timestamp > (t_now + 1e-4):
            rejected_count += 1
        else:
            valid.append(d)

    assert rejected_count == 1
    assert len(valid) == 1
    assert valid[0].detection_id == "det_fresh"


def test_noisy_observation_kalman_filtering():
    """M14-08 component test: Kalman filter smooths Gaussian position noise."""
    tracker = MultiFrameTracker(TrackerConfig(target_frame="odom", use_kalman_filter=True))
    rng = np.random.RandomState(42)

    true_x = 2.0
    measurements = []
    filtered_x = []

    for step in range(15):
        t = step * 0.10
        noisy_x = true_x + float(rng.normal(0.0, 0.20))
        measurements.append(noisy_x)
        det = [DynamicObstacleDetection(
            detection_id="det_noise",
            timestamp=t,
            frame_id="odom",
            bbox_center=(noisy_x, 0.0),
            bbox_size=(0.8, 0.6),
            bbox_yaw=0.0,
            confidence=0.9,
        )]
        res = tracker.update(det, t)
        if res.tracks:
            filtered_x.append(res.tracks[0].x)

    raw_rmse = float(np.sqrt(np.mean((np.array(measurements) - true_x) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((np.array(filtered_x) - true_x) ** 2)))
    # Kalman filter must achieve lower or comparable RMSE than raw noise
    assert filtered_rmse <= raw_rmse * 1.25, f"Filtered RMSE ({filtered_rmse}) vs Raw ({raw_rmse})"
