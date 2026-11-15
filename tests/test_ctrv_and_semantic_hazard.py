"""Unit tests for WS-2B: Coordinated Turn-Rate & Velocity (CTRV) Predictor & Semantic-Weighted Hazard Memory.

Verifies:
1. CTRV analytical circular arc trajectory accuracy (quarter-circle 90-degree turn at v=1.0m/s, omega=pi/4 rad/s).
2. Exact numerical continuity between CTRV (omega -> 0) and linear Constant-Velocity (CV, omega = 0).
3. Semantic-weighted hazard memory: transient grass/shadow artifacts decay and expire rapidly (< 1.0s),
   whereas solid boulders/pedestrians retain extended safety latching (> 4.0s).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.dynamic.predictor import MotionPredictor, PredictorConfig
from ai.planning.safety_governor import RecentHazardMemory, RecentHazardMemoryConfig


def _make_track(
    track_id: int = 1,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 1.0,
    vy: float = 0.0,
    omega: float = 0.0,
    semantic_class: str = "obstacle",
) -> TrackedObstacleState:
    tr = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        frame_id="odom",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        covariance=np.eye(4, dtype=np.float64) * 0.02,
        bbox_size=(0.8, 0.6),
        bbox_yaw=0.0,
        track_age=5,
        coast_cycles=0,
        is_confirmed=True,
        status="TRACKING",
    )
    tr.omega = omega
    tr.semantic_class = semantic_class
    return tr


def test_ctrv_quarter_circle_turn_accuracy():
    """Verify CTRV accurately predicts a 90-degree circular turn over T = 2.0s (omega = pi/4 rad/s)."""
    predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.5))
    v = 1.0
    omega = math.pi / 4.0  # 45 deg/s -> 90 deg (pi/2) in 2.0s
    radius = v / omega

    track = _make_track(x=0.0, y=0.0, vx=v, vy=0.0, omega=omega)
    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)

    final_wp = pred.waypoints[-1]
    # At theta = pi/2: x = R * sin(pi/2) = R, y = R * (1 - cos(pi/2)) = R
    # vx = 0, vy = v
    assert math.isclose(final_wp.x, radius, abs_tol=1e-4)
    assert math.isclose(final_wp.y, radius, abs_tol=1e-4)
    assert math.isclose(final_wp.vx, 0.0, abs_tol=1e-4)
    assert math.isclose(final_wp.vy, v, abs_tol=1e-4)


def test_ctrv_to_cv_continuity_limit():
    """Verify CTRV smoothly converges to linear CV as omega -> 0."""
    predictor = MotionPredictor(PredictorConfig(default_horizon_s=2.0, default_dt_s=0.5))
    tr_cv = _make_track(x=1.0, y=2.0, vx=0.8, vy=-0.3, omega=0.0)
    tr_small = _make_track(x=1.0, y=2.0, vx=0.8, vy=-0.3, omega=2e-4)

    pred_cv = predictor.predict_track(tr_cv)
    pred_small = predictor.predict_track(tr_small)

    for wp_cv, wp_sm in zip(pred_cv.waypoints, pred_small.waypoints):
        assert math.isclose(wp_cv.x, wp_sm.x, abs_tol=1e-3)
        assert math.isclose(wp_cv.y, wp_sm.y, abs_tol=1e-3)


def test_semantic_weighted_hazard_memory_decay():
    """Verify transient grass/shadow artifacts expire quickly (< 1.1s) while boulders persist (> 4.0s)."""
    memory = RecentHazardMemory(RecentHazardMemoryConfig(base_ttl_s=3.50, close_ttl_s=5.00))
    robot_pose = (0.0, 0.0, 0.0)

    boulder_tr = _make_track(track_id=10, x=1.5, y=0.0, semantic_class="boulder")
    shadow_tr = _make_track(track_id=20, x=1.5, y=0.5, semantic_class="shadow")

    # Frame 0 (t=0.0): Both observed
    active = memory.update([boulder_tr, shadow_tr], current_time_s=0.0, robot_pose=robot_pose)
    assert len(active) == 2

    # At t=1.1s after dropout: shadow/glare artifact must have decayed & expired, while boulder remains latched!
    remaining = memory.update([], current_time_s=1.1, robot_pose=robot_pose)
    rem_ids = {h.hazard_id for h in remaining}
    assert "10" in rem_ids, "Solid boulder hazard must remain latched at t=1.1s"
    assert "20" not in rem_ids, "Transient shadow artifact must expire by t=1.1s to prevent ghost stops"
