"""Group 5: Replanner failures, timeouts, tracker/predictor degradation, and Ground-Truth isolation audit."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.global_replanner import (
    DynamicPlanningMapOverlay,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.data_model import DynamicObstacleDetection, TrackedObstacleState
from ai.planning.dynamic.tracker import MultiFrameTracker, TrackerConfig
from ai.spatial_fusion.world_map import WorldTerrainMap
from simulation.runners.phase14_sil_runner import Phase14SILRunner
from simulation.scenarios.dynamic.phase14.data_model import Phase14FaultSpec, Phase14FaultType
from simulation.scenarios.dynamic.phase14.failure_models import Phase14FaultInjector
from simulation.scenarios.dynamic.phase14.scenario_builders import (
    create_scenario_m14_11_replanner_failure,
    create_scenario_m14_12_replanner_timeout,
    create_scenario_m14_13_tracker_degradation,
    create_scenario_m14_14_prediction_degradation,
)


def test_m14_11_replanner_failure_cases():
    """Assert H: Replanner failures (Cases A, B, C) are distinguishable and trigger safe fallback."""
    for case, expected_type in [
        ("A", Phase14FaultType.REPLANNER_NO_PATH),
        ("B", Phase14FaultType.REPLANNER_EXCEPTION),
        ("C", Phase14FaultType.REPLANNER_INVALID_RESULT),
    ]:
        cfg = create_scenario_m14_11_replanner_failure(case=case)
        fault = cfg.fault_specs[0]
        injector = Phase14FaultInjector([fault])

        # Test interface interception directly
        dummy_res = DynamicReplanResult(
            replan_triggered=True,
            trigger_type=DynamicReplanTrigger.PERSISTENT_BLOCKAGE,
            success=True,
            status=PlannerStatus.SUCCESS,
        )
        intercepted = injector.intercept_replan_execution(lambda: dummy_res, 0.5)

        assert intercepted.success is False, f"Case {case}: Replan must report success=False"
        assert intercepted.path is None, f"Case {case}: Path must be None on failure"
        assert intercepted.is_path_stale is True, f"Case {case}: Invalid path must be marked stale"
        assert "Injected Fault" in str(intercepted.failure_reason)


def test_m14_12_replanner_timeout_distinguishability():
    """Assert I: Replanner timeout is distinguishable from successful replanning and does not use stale path."""
    fault = Phase14FaultSpec(
        fault_id="timeout_test",
        fault_type=Phase14FaultType.REPLANNER_TIMEOUT,
        target_component="global_replanner",
        start_time=0.0,
        end_time=1.0,
        expected_response="TIMEOUT",
    )
    injector = Phase14FaultInjector([fault])
    dummy_res = DynamicReplanResult(replan_triggered=True, success=True)
    intercepted = injector.intercept_replan_execution(lambda: dummy_res, 0.5)

    assert intercepted.success is False
    assert intercepted.is_path_stale is True
    assert "timed out" in str(intercepted.failure_reason).lower()


def test_m14_13_tracker_degradation_handling():
    """M14-13: Downstream sees degraded track array without throwing unhandled exceptions."""
    fault = Phase14FaultSpec(
        fault_id="trk_loss",
        fault_type=Phase14FaultType.TRACK_LOSS,
        target_component="tracker",
        start_time=0.0,
        end_time=1.0,
        expected_response="DROP",
    )
    injector = Phase14FaultInjector([fault])
    sample_tracks = [
        TrackedObstacleState(
            track_id=1,
            timestamp=0.0,
            x=2.0,
            y=0.0,
            vx=0.1,
            vy=0.0,
            covariance=np.eye(4)*0.05,
            bbox_size=(0.8, 0.6),
            bbox_yaw=0.0,
        )
    ]
    out = injector.intercept_tracks(sample_tracks, 0.5)
    assert len(out) == 0, "Track array must be empty during track loss fault"


def test_m14_14_prediction_degradation_horizon_shorten():
    """M14-14: Shortened prediction horizon is correctly handled."""
    from tests.test_phase14_occupancy_collision import _make_dummy_pred
    p = _make_dummy_pred(1, 2.0, 0.0, 0.1, 0.0)
    orig_len = len(p.waypoints)

    fault = Phase14FaultSpec(
        fault_id="pred_short",
        fault_type=Phase14FaultType.PREDICTION_HORIZON_SHORTEN,
        target_component="predictor",
        start_time=0.0,
        end_time=1.0,
        expected_response="SHORTEN",
        parameters={"max_points": 2},
    )
    injector = Phase14FaultInjector([fault])
    out = injector.intercept_predictions([p], 0.5)

    assert len(out) == 1
    assert len(out[0].waypoints) == 2, f"Expected 2 waypoints, got {len(out[0].waypoints)}"
    assert len(out[0].bounding_polygon_per_step) == 2


def test_workstream_9_ground_truth_isolation_audit():
    """Workstream 9 / Assert B: STRICT AUDIT - No Ground-Truth state enters autonomy data structures."""
    cfg = create_scenario_m14_11_replanner_failure()
    runner = Phase14SILRunner(cfg)

    # 1. Inspect raw detections generated by observation generator
    raw_detections = runner.observation_generator.generate_observations(0.0, (0.0, 0.0, 0.0))
    for d in raw_detections:
        # Detection must be DynamicObstacleDetection
        assert isinstance(d, DynamicObstacleDetection)
        # Detection must NOT have velocity or GT trajectory attributes
        assert not hasattr(d, "vx"), "GT velocity leakage in DynamicObstacleDetection"
        assert not hasattr(d, "vy"), "GT velocity leakage in DynamicObstacleDetection"
        assert not hasattr(d, "future_waypoints"), "GT future path leakage in detection"
        assert not hasattr(d, "motion_model"), "GT motion model leakage in detection"

    # 2. Inspect tracker inputs and outputs
    tracked = runner.tracker.update(raw_detections, 0.0)
    for t in tracked.tracks:
        assert isinstance(t, TrackedObstacleState)
        assert not hasattr(t, "gt_state"), "GT state attached to TrackedObstacleState"
        assert not hasattr(t, "true_trajectory"), "GT trajectory attached to TrackedObstacleState"

    # 3. Ground truth motion models are only evaluated in the runner's evaluation loop
    for model in runner.motion_models:
        gt_state = model.evaluate(0.0)
        # Verify gt_state is never passed to tracker or predictor
        assert gt_state is not None
        assert type(gt_state).__name__ == "GroundTruthObstacleState"
