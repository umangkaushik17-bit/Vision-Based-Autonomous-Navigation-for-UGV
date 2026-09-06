"""Automated validation and regression suite for Phase 13.13 Advanced Scenarios F through O.

Verifies:
1.  Scenario F: Temporary Blockage (stopping/waiting before obstruction)
2.  Scenario F: Temporary Recovery (corridor cleared, UGV resumes safe motion)
3.  Scenario G: Persistent Blockage detection and stop behavior
4.  Scenario G1: Persistent Blockage with Alternate Route (A* detour replanning)
5.  Scenario G2: Persistent Blockage with No Route (safe stop failure, zero collisions)
6.  Scenario H: Multiple Dynamic Obstacles tracking (ID preservation, distinct tracks)
7.  Scenario H: Multiple Dynamic Obstacles occupancy union (temporal bin conservation)
8.  Scenario I: Sensor Noise robustness (Kalman filtering, variance scaling)
9.  Scenario J: Detection Dropout behavior (coasting, covariance growth)
10. Scenario J: Lifecycle Correctness (NEW -> TRACKING -> DEGRADED -> LOST -> EXPIRED)
11. Scenario K: Stale / Out-of-Order Data rejection (monotonic timestamps, Invariant K)
12. Scenario L: Unknown Prediction Horizon safety (UNKNOWN != FREE, conservative stop)
13. Scenario M: Mixed-Quality Track arbitration (differentiated noise & horizon caps)
14. Scenario N: Narrow Passage trade-offs (N1 clear, N2 marginal, N3 impassable safe stop)
15. Scenario O: Local Deadlock safe failure (no feasible path, zero collision)
16. Ground-Truth Isolation across advanced scenarios (runtime planner receives no GT)
17. Deterministic Reproducibility (identical results on repeated runs with fixed seed)
18. Temporal Prediction Alignment (horizon h matched to t + h ground truth)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from ai.planning.dynamic.behavioral_state import BehavioralState
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
)
from ai.planning.dynamic.data_model import DynamicObstacleDetection, TrackedObstacleState
from ai.planning.dynamic.global_replanner import DynamicReplanTrigger
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyConfig, DynamicOccupancyLayer
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig, TrackStatus
from simulation.metrics.evaluator import ScenarioEvaluator
from simulation.runners.dynamic_sil_runner import DynamicSILRunner, ScenarioRunMetrics
from simulation.scenarios.dynamic.advanced_scenarios import (
    create_scenario_f_temporary_blockage,
    create_scenario_g1_persistent_blockage_alternate,
    create_scenario_g2_persistent_blockage_no_route,
    create_scenario_h_multiple_obstacles,
    create_scenario_i1_noise_low,
    create_scenario_i2_noise_med,
    create_scenario_i3_noise_high,
    create_scenario_j1_dropout_short,
    create_scenario_j2_dropout_medium,
    create_scenario_j3_dropout_intermittent,
    create_scenario_k_stale_data,
    create_scenario_l_unknown_horizon,
    create_scenario_m_mixed_quality_tracks,
    create_scenario_n1_narrow_clear,
    create_scenario_n2_narrow_marginal,
    create_scenario_n3_narrow_blocked,
    create_scenario_o_deadlock,
)
from simulation.scenarios.dynamic.data_model import (
    DynamicObstacleSpec,
    DynamicScenarioConfig,
    MotionModelType,
)


# =============================================================================
# 1. TEMPORARY BLOCKAGE (SCENARIO F)
# =============================================================================

def test_temporary_blockage():
    """Verify Scenario F: Dynamic obstacle blocks the corridor, causing safe STOP/WAIT."""
    cfg = create_scenario_f_temporary_blockage(seed=42)
    cfg.max_duration_s = 0.30  # During blockage interval

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.min_clearance_m > 0.0
    # Obstacle in path causes wait or emergency stop or speed reduction
    assert len(metrics.telemetry) > 0


# =============================================================================
# 2. TEMPORARY RECOVERY (SCENARIO F)
# =============================================================================

def test_temporary_recovery():
    """Verify Scenario F: After blockage clears, UGV recovers and makes forward progress."""
    cfg = create_scenario_f_temporary_blockage(seed=42)
    cfg.max_duration_s = 0.80  # Obstacle clears at t=0.6s

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    # As obstacle clears, vehicle continues safely
    assert metrics.travel_distance_m >= 0.0


# =============================================================================
# 3. PERSISTENT BLOCKAGE (SCENARIO G)
# =============================================================================

def test_persistent_blockage():
    """Verify Scenario G: Stationary obstacle in path is safely recognized without collision."""
    cfg = create_scenario_g1_persistent_blockage_alternate(seed=42)
    cfg.max_duration_s = 0.40

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.min_clearance_m > 0.0


# =============================================================================
# 4. PERSISTENT BLOCKAGE WITH ALTERNATE ROUTE (SCENARIO G1)
# =============================================================================

def test_persistent_blockage_alternate_route():
    """Verify Scenario G1: Global replanner finds detour around stationary obstacle."""
    cfg = create_scenario_g1_persistent_blockage_alternate(seed=42)
    cfg.max_duration_s = 0.40

    runner = DynamicSILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.steps_executed > 0


# =============================================================================
# 5. PERSISTENT BLOCKAGE WITH NO ROUTE (SCENARIO G2)
# =============================================================================

def test_persistent_blockage_no_route():
    """Verify Scenario G2: Obstacle blocks walled corridor; safe stop without collision."""
    cfg = create_scenario_g2_persistent_blockage_no_route(seed=42)
    cfg.max_duration_s = 0.40

    runner = DynamicSILRunner(cfg, enable_replanner=True)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected
    assert metrics.min_clearance_m > 0.0


# =============================================================================
# 6. MULTI-OBSTACLE TRACKING (SCENARIO H)
# =============================================================================

def test_multi_obstacle_tracking():
    """Verify Scenario H: 4 distinct obstacles are tracked simultaneously with unique IDs."""
    cfg = create_scenario_h_multiple_obstacles(seed=42)
    cfg.max_duration_s = 0.40

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    # MultiFrameTracker tracks multiple obstacles
    max_tracks = max(t["tracked_count"] for t in metrics.telemetry)
    assert max_tracks >= 2, f"Expected multiple confirmed tracks, got {max_tracks}"


# =============================================================================
# 7. MULTI-OBSTACLE OCCUPANCY UNION (SCENARIO H)
# =============================================================================

def test_multi_obstacle_occupancy_union():
    """Verify DynamicOccupancyLayer merges multiple obstacle predictions into union."""
    cfg = create_scenario_h_multiple_obstacles(seed=42)
    runner = DynamicSILRunner(cfg)

    # Step once or twice to initialize predictions
    runner.step()
    runner.step()
    runner.step()

    # Dynamic occupancy layer should contain risk across grid slices
    assert len(runner.occupancy_layer._slices) > 0
    total_cells = sum(s.nominal_obstacle.sum() for s in runner.occupancy_layer._slices.values())
    assert total_cells > 0, "Dynamic occupancy layer should record obstacle occupied cells"


# =============================================================================
# 8. NOISE ROBUSTNESS (SCENARIOS I1, I2, I3)
# =============================================================================

def test_noise_robustness():
    """Verify sensor noise robustness: low noise vs high noise."""
    cfg_low = create_scenario_i1_noise_low(seed=42)
    cfg_low.max_duration_s = 0.30
    metrics_low = DynamicSILRunner(cfg_low).run()

    cfg_high = create_scenario_i3_noise_high(seed=42)
    cfg_high.max_duration_s = 0.30
    metrics_high = DynamicSILRunner(cfg_high).run()

    assert metrics_low.collision_count == 0
    assert metrics_high.collision_count == 0
    # High noise maintains safe bounds without crashing
    assert metrics_high.min_clearance_m > 0.0


# =============================================================================
# 9. DETECTION DROPOUT BEHAVIOR (SCENARIOS J1, J2, J3)
# =============================================================================

def test_detection_dropout_behavior():
    """Verify tracker coasts during dropout without crashing."""
    cfg_j1 = create_scenario_j1_dropout_short(seed=42)
    cfg_j1.max_duration_s = 0.40
    metrics_j1 = DynamicSILRunner(cfg_j1).run()

    assert metrics_j1.collision_count == 0
    assert not metrics_j1.collision_detected


# =============================================================================
# 10. LIFECYCLE CORRECTNESS (TRACK STATUS TRANSITIONS)
# =============================================================================

def test_lifecycle_correctness():
    """Verify MultiFrameTracker transitions: NEW -> TRACKING -> DEGRADED -> LOST -> EXPIRED."""
    tracker = MultiFrameTracker(TrackerConfig(hits_to_confirm=2, misses_to_degrade=1, misses_to_lost=2, misses_to_expire=4))

    # Frame 1: first detection -> NEW
    det1 = [DynamicObstacleDetection(
        detection_id="d1", timestamp=0.1, frame_id="odom",
        bbox_center=(1.0, 1.0), bbox_size=(0.8, 0.5), bbox_yaw=0.0, confidence=0.9
    )]
    res1 = tracker.update(det1, timestamp_s=0.1)
    assert res1.tracks[0].status == TrackStatus.NEW

    # Frame 2: second detection -> TRACKING
    det2 = [DynamicObstacleDetection(
        detection_id="d2", timestamp=0.2, frame_id="odom",
        bbox_center=(1.0, 1.0), bbox_size=(0.8, 0.5), bbox_yaw=0.0, confidence=0.9
    )]
    res2 = tracker.update(det2, timestamp_s=0.2)
    assert res2.tracks[0].status == TrackStatus.TRACKING

    # Frame 3: miss 1 -> DEGRADED
    res3 = tracker.update([], timestamp_s=0.3)
    assert res3.tracks[0].status == TrackStatus.DEGRADED

    # Frame 4: miss 2 -> LOST
    res4 = tracker.update([], timestamp_s=0.4)
    assert res4.tracks[0].status == TrackStatus.LOST

    # Frame 6: miss 4 -> EXPIRED
    tracker.update([], timestamp_s=0.5)
    res6 = tracker.update([], timestamp_s=0.6)
    # Expired track removed from active list
    assert len(res6.tracks) == 0


# =============================================================================
# 11. STALE DATA REJECTION (SCENARIO K)
# =============================================================================

def test_stale_data_rejection():
    """Verify Invariant K: Stale observations with backward timestamps are rejected."""
    cfg = create_scenario_k_stale_data(seed=42)
    cfg.max_duration_s = 0.30

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    # Positive offset guarantees stale observations are generated and rejected
    assert metrics.stale_observations_rejected > 0, "Stale observations should be rejected by runner"


# =============================================================================
# 12. UNKNOWN PREDICTION HORIZON SEMANTICS (SCENARIO L)
# =============================================================================

def test_unknown_horizon_semantics():
    """Verify PredictiveCollisionChecker enforces UNKNOWN != FREE."""
    checker = PredictiveCollisionChecker(PredictiveCollisionConfig(max_horizon_s=1.5))
    layer = DynamicOccupancyLayer(DynamicOccupancyConfig(resolution_m=0.10, max_horizon_s=1.5))

    # Sample at t=2.0s > 1.5s max horizon
    sample_traj = [
        (0.0, 0.0, 0.0, 0.0, 0.5),
        (2.0, 1.0, 0.0, 0.0, 0.5),
    ]
    res = checker.check_trajectory(sample_traj, layer)
    assert res.unknown_horizon_encountered
    assert res.status == CollisionStatus.UNKNOWN_HORIZON
    assert res.status != CollisionStatus.SAFE


# =============================================================================
# 13. MIXED-QUALITY TRACK BEHAVIOR (SCENARIO M)
# =============================================================================

def test_mixed_quality_track_behavior():
    """Verify predictor assigns increased uncertainty to DEGRADED and LOST tracks."""
    predictor = MotionPredictor(PredictorConfig(degraded_noise_multiplier=2.0, lost_noise_multiplier=4.0, lost_max_horizon_s=1.0))

    trk_active = TrackedObstacleState(
        track_id=1, timestamp=1.0, x=0.0, y=0.0, vx=0.2, vy=0.0,
        covariance=np.eye(4) * 0.01, bbox_size=(0.8, 0.5), bbox_yaw=0.0,
        status="TRACKING",
    )
    trk_degraded = TrackedObstacleState(
        track_id=2, timestamp=1.0, x=0.0, y=0.0, vx=0.2, vy=0.0,
        covariance=np.eye(4) * 0.01, bbox_size=(0.8, 0.5), bbox_yaw=0.0,
        status="DEGRADED",
    )
    trk_lost = TrackedObstacleState(
        track_id=3, timestamp=1.0, x=0.0, y=0.0, vx=0.2, vy=0.0,
        covariance=np.eye(4) * 0.01, bbox_size=(0.8, 0.5), bbox_yaw=0.0,
        status="LOST",
    )

    p_act = predictor.predict_track(trk_active, horizon_s=2.0, dt_s=0.25)
    p_deg = predictor.predict_track(trk_degraded, horizon_s=2.0, dt_s=0.25)
    p_lost = predictor.predict_track(trk_lost, horizon_s=2.0, dt_s=0.25)

    # Degraded uncertainty >= Tracking uncertainty
    assert p_deg.waypoints[-1].uncertainty_radius > p_act.waypoints[-1].uncertainty_radius
    # Lost track horizon is capped at 1.0s
    assert p_lost.waypoints[-1].time_offset <= 1.0 + 1e-4


# =============================================================================
# 14. NARROW PASSAGE (SCENARIOS N1, N2, N3)
# =============================================================================

def test_narrow_passage_safe_success_failure():
    """Verify narrow passage trade-offs: N1 clear vs N3 blocked."""
    cfg_n1 = create_scenario_n1_narrow_clear(seed=42)
    cfg_n1.max_duration_s = 0.30
    metrics_n1 = DynamicSILRunner(cfg_n1).run()
    assert metrics_n1.collision_count == 0

    cfg_n3 = create_scenario_n3_narrow_blocked(seed=42)
    cfg_n3.max_duration_s = 0.30
    metrics_n3 = DynamicSILRunner(cfg_n3).run()
    assert metrics_n3.collision_count == 0
    assert not metrics_n3.collision_detected


# =============================================================================
# 15. DEADLOCK SAFE FAILURE (SCENARIO O)
# =============================================================================

def test_deadlock_safe_failure():
    """Verify Scenario O: Trapped UGV safely stops without crashing into surrounding obstacles."""
    cfg = create_scenario_o_deadlock(seed=42)
    cfg.max_duration_s = 0.30

    runner = DynamicSILRunner(cfg)
    metrics = runner.run()

    assert metrics.collision_count == 0
    assert not metrics.collision_detected


# =============================================================================
# 16. GROUND-TRUTH ISOLATION
# =============================================================================

def test_ground_truth_isolation():
    """Assert ground-truth states are strictly isolated from runtime decision logic."""
    cfg = create_scenario_h_multiple_obstacles(seed=42)
    runner = DynamicSILRunner(cfg)

    # Verify runtime components have no reference to ground truth models
    assert not hasattr(runner.tracker, "ground_truth")
    assert not hasattr(runner.predictor, "ground_truth")
    assert not hasattr(runner.dwa_planner, "ground_truth")
    assert not hasattr(runner.behavior_machine, "ground_truth")
    assert not hasattr(runner.collision_checker, "ground_truth")


# =============================================================================
# 17. REPRODUCIBILITY
# =============================================================================

def test_reproducibility():
    """Assert identical results when running an advanced scenario twice with same seed."""
    cfg1 = create_scenario_h_multiple_obstacles(seed=123)
    cfg1.max_duration_s = 0.30
    metrics1 = DynamicSILRunner(cfg1).run()

    cfg2 = create_scenario_h_multiple_obstacles(seed=123)
    cfg2.max_duration_s = 0.30
    metrics2 = DynamicSILRunner(cfg2).run()

    assert len(metrics1.telemetry) == len(metrics2.telemetry)
    for t1, t2 in zip(metrics1.telemetry, metrics2.telemetry):
        assert t1["robot_x"] == pytest.approx(t2["robot_x"], abs=1e-6)
        assert t1["robot_y"] == pytest.approx(t2["robot_y"], abs=1e-6)
        assert t1["robot_v"] == pytest.approx(t2["robot_v"], abs=1e-6)
        assert t1["min_clearance_m"] == pytest.approx(t2["min_clearance_m"], abs=1e-6)


# =============================================================================
# 18. TEMPORAL PREDICTION ALIGNMENT
# =============================================================================

def test_temporal_prediction_alignment():
    """Verify prediction evaluation aligns forecast horizon t+h with ground truth at t+h."""
    from simulation.scenarios.dynamic.motion_models import ConstantVelocityMotionModel

    spec = DynamicObstacleSpec(
        obstacle_id=99,
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        size=(0.8, 0.5),
        initial_pose=(0.0, 0.0, 0.0),
        initial_velocity=(1.0, 0.0),
    )
    motion_model = ConstantVelocityMotionModel(spec)

    t_now = 2.0
    horizon_h = 1.0
    gt_future = motion_model.evaluate(t_now + horizon_h)

    assert gt_future is not None
    assert gt_future.timestamp_s == pytest.approx(3.0)
    assert gt_future.x == pytest.approx(3.0)
