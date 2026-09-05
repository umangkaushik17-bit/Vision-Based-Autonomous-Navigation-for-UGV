"""Unit tests for Phase 13 Milestone 13.6 — Motion Prediction.

Validates the constant-velocity motion predictor for dynamic obstacles:
- Tests A through V matching specification requirements:
    A. Zero-horizon prediction
    B. Constant velocity prediction
    C. Diagonal motion prediction
    D. Arbitrary prediction dt
    E. Multiple horizons
    F. Covariance propagation (CWNA)
    G. Increasing uncertainty
    H. Uncertainty radius growth
    I. Timestamp validation
    J. Negative horizon rejection
    K. Duplicate horizon handling
    L. Non-monotonic horizon handling
    M. Stationary target
    N. Accelerating target model mismatch
    O. Turning target model mismatch
    P. Multiple tracks (no cross-track contamination)
    Q. Confidence/state handling (DEGRADED inflation)
    R. LOST track handling (horizon capping & noise inflation)
    S. EXPIRED track rejection
    T. Deterministic reproducibility
    U. NaN/Inf protection
    V. Ground-truth isolation audit
"""

import inspect
import math
import numpy as np
import pytest

from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
)
from ai.planning.dynamic.predictor import (
    MotionPredictor,
    PredictorConfig,
)


def make_test_track(
    track_id: int = 1,
    timestamp: float = 10.0,
    x: float = 5.0,
    y: float = 3.0,
    vx: float = 1.0,
    vy: float = 0.5,
    pos_var: float = 0.04,
    vel_var: float = 0.16,
    bbox_size: tuple[float, float] = (1.5, 0.8),
    bbox_yaw: float = 0.0,
    status: str = "TRACKING",
) -> TrackedObstacleState:
    """Helper to construct a valid TrackedObstacleState."""
    cov = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=timestamp,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        covariance=cov,
        bbox_size=bbox_size,
        bbox_yaw=bbox_yaw,
        status=status,
    )


# ============================================================================
# TEST A: Zero-horizon prediction
# ============================================================================
def test_a_zero_horizon_prediction() -> None:
    predictor = MotionPredictor()
    track = make_test_track(x=4.0, y=2.0, vx=1.5, vy=-0.5)

    pred = predictor.predict_track(track, horizon_s=0.0)

    assert pred.track_id == track.track_id
    assert pred.horizon_s == 0.0
    assert len(pred.waypoints) == 1

    wp = pred.waypoints[0]
    assert wp.time_offset == 0.0
    assert wp.timestamp == pytest.approx(10.0)
    assert wp.x == pytest.approx(4.0)
    assert wp.y == pytest.approx(2.0)
    assert wp.vx == pytest.approx(1.5)
    assert wp.vy == pytest.approx(-0.5)

    # Positional covariance at zero offset equals initial position covariance
    np.testing.assert_allclose(wp.covariance_2d, track.position_covariance(), atol=1e-9)
    assert wp.uncertainty_radius == pytest.approx(
        track.position_uncertainty_radius(predictor.config.sigma_scale)
    )
    assert len(pred.bounding_polygon_per_step) == 1
    assert pred.bounding_polygon_per_step[0].shape == (4, 2)


# ============================================================================
# TEST B: Constant velocity
# ============================================================================
def test_b_constant_velocity_prediction() -> None:
    predictor = MotionPredictor()
    track = make_test_track(x=10.0, y=5.0, vx=2.0, vy=0.0)

    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)

    assert len(pred.waypoints) == 5  # [0.0, 0.5, 1.0, 1.5, 2.0]
    expected_times = [0.0, 0.5, 1.0, 1.5, 2.0]

    for wp, dt in zip(pred.waypoints, expected_times):
        assert wp.time_offset == pytest.approx(dt)
        assert wp.timestamp == pytest.approx(track.timestamp + dt)
        assert wp.x == pytest.approx(10.0 + 2.0 * dt, abs=1e-7)
        assert wp.y == pytest.approx(5.0, abs=1e-7)
        assert wp.vx == pytest.approx(2.0, abs=1e-7)
        assert wp.vy == pytest.approx(0.0, abs=1e-7)


# ============================================================================
# TEST C: Diagonal motion
# ============================================================================
def test_c_diagonal_motion_prediction() -> None:
    predictor = MotionPredictor()
    track = make_test_track(x=0.0, y=0.0, vx=3.0, vy=4.0)

    pred = predictor.predict_track(track, horizon_s=1.0, dt_s=0.25)

    assert len(pred.waypoints) == 5
    for wp in pred.waypoints:
        dt = wp.time_offset
        assert wp.x == pytest.approx(3.0 * dt, abs=1e-7)
        assert wp.y == pytest.approx(4.0 * dt, abs=1e-7)
        assert wp.vx == pytest.approx(3.0, abs=1e-7)
        assert wp.vy == pytest.approx(4.0, abs=1e-7)


# ============================================================================
# TEST D: Arbitrary prediction dt
# ============================================================================
def test_d_arbitrary_prediction_dt() -> None:
    predictor = MotionPredictor()
    track = make_test_track(x=1.0, y=1.0, vx=1.0, vy=-1.0)
    arbitrary_offsets = [0.0, 0.13, 0.37, 0.81, 1.45]

    pred = predictor.predict_track(track, time_offsets_s=arbitrary_offsets)

    assert len(pred.waypoints) == len(arbitrary_offsets)
    for wp, dt in zip(pred.waypoints, arbitrary_offsets):
        assert wp.time_offset == pytest.approx(dt)
        assert wp.x == pytest.approx(1.0 + 1.0 * dt, abs=1e-7)
        assert wp.y == pytest.approx(1.0 - 1.0 * dt, abs=1e-7)


# ============================================================================
# TEST E: Multiple horizons
# ============================================================================
def test_e_multiple_horizons() -> None:
    predictor = MotionPredictor()
    track = make_test_track()
    required_horizons = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0]

    pred = predictor.predict_track(track, time_offsets_s=required_horizons)

    assert len(pred.waypoints) == len(required_horizons)
    for i, h in enumerate(required_horizons):
        assert pred.waypoints[i].time_offset == pytest.approx(h)


# ============================================================================
# TEST F: Covariance propagation (CWNA)
# ============================================================================
def test_f_covariance_propagation() -> None:
    q = 1.5
    config = PredictorConfig(process_noise_spectral_density=q)
    predictor = MotionPredictor(config)

    P0 = np.array([
        [0.05, 0.01, 0.00, 0.00],
        [0.01, 0.05, 0.00, 0.00],
        [0.00, 0.00, 0.20, 0.02],
        [0.00, 0.00, 0.02, 0.20],
    ], dtype=np.float64)

    track = TrackedObstacleState(
        track_id=1,
        timestamp=0.0,
        x=0.0,
        y=0.0,
        vx=1.0,
        vy=1.0,
        covariance=P0,
        bbox_size=(1.0, 1.0),
        bbox_yaw=0.0,
    )

    dt = 0.75
    pred = predictor.predict_track(track, time_offsets_s=[0.0, dt])
    wp = pred.waypoints[1]

    # Manual analytical calculation:
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    dt2 = dt * dt
    dt3 = dt2 * dt
    Q = q * np.array([
        [dt3 / 3.0, 0.0, dt2 / 2.0, 0.0],
        [0.0, dt3 / 3.0, 0.0, dt2 / 2.0],
        [dt2 / 2.0, 0.0, dt, 0.0],
        [0.0, dt2 / 2.0, 0.0, dt],
    ])
    expected_P_full = F @ P0 @ F.T + Q
    expected_P_pos = expected_P_full[:2, :2]

    np.testing.assert_allclose(wp.covariance_2d, expected_P_pos, atol=1e-8)


# ============================================================================
# TEST G: Increasing uncertainty
# ============================================================================
def test_g_increasing_uncertainty() -> None:
    predictor = MotionPredictor()
    track = make_test_track()

    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.25)

    traces = [float(np.trace(wp.covariance_2d)) for wp in pred.waypoints]

    # Trace of position covariance must be strictly monotonically increasing with horizon
    for i in range(len(traces) - 1):
        assert traces[i + 1] > traces[i], (
            f"Covariance trace did not increase: step {i} ({traces[i]:.6f}) "
            f"-> step {i+1} ({traces[i+1]:.6f})"
        )


# ============================================================================
# TEST H: Uncertainty radius growth
# ============================================================================
def test_h_uncertainty_radius_growth() -> None:
    predictor = MotionPredictor()
    track = make_test_track()

    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.25)

    radii = [wp.uncertainty_radius for wp in pred.waypoints]

    for i in range(len(radii) - 1):
        assert radii[i + 1] > radii[i], (
            f"Uncertainty radius did not increase: step {i} ({radii[i]:.4f}) "
            f"-> step {i+1} ({radii[i+1]:.4f})"
        )

    # Check that bounding polygon expands outward
    base_poly = pred.bounding_polygon_per_step[0]
    final_poly = pred.bounding_polygon_per_step[-1]
    # Area or bounding size of final polygon must be larger than initial
    base_span_x = np.ptp(base_poly[:, 0])
    final_span_x = np.ptp(final_poly[:, 0])
    assert final_span_x > base_span_x


# ============================================================================
# TEST I: Timestamp validation
# ============================================================================
def test_i_timestamp_validation() -> None:
    predictor = MotionPredictor()
    track = make_test_track(timestamp=42.5)

    pred = predictor.predict_track(track, horizon_s=1.0, dt_s=0.25)

    for wp in pred.waypoints:
        assert wp.timestamp == pytest.approx(42.5 + wp.time_offset)


# ============================================================================
# TEST J: Negative horizon rejection
# ============================================================================
def test_j_negative_horizon_rejection() -> None:
    predictor = MotionPredictor()
    track = make_test_track()

    with pytest.raises(ValueError, match="non-negative"):
        predictor.predict_track(track, horizon_s=-0.5)

    with pytest.raises(ValueError, match="Negative"):
        predictor.predict_track(track, time_offsets_s=[0.0, -0.25, 0.5])


# ============================================================================
# TEST K: Duplicate horizon handling
# ============================================================================
def test_k_duplicate_horizon_handling() -> None:
    predictor = MotionPredictor()
    track = make_test_track()

    with pytest.raises(ValueError, match="Duplicate"):
        predictor.predict_track(track, time_offsets_s=[0.0, 0.5, 0.5, 1.0])


# ============================================================================
# TEST L: Non-monotonic horizon handling
# ============================================================================
def test_l_non_monotonic_horizon_handling() -> None:
    predictor = MotionPredictor()
    track = make_test_track()

    with pytest.raises(ValueError, match="Non-monotonic"):
        predictor.predict_track(track, time_offsets_s=[0.0, 0.8, 0.4, 1.0])


# ============================================================================
# TEST M: Stationary target
# ============================================================================
def test_m_stationary_target() -> None:
    predictor = MotionPredictor()
    track = make_test_track(x=7.0, y=-3.0, vx=0.0, vy=0.0)

    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)

    for wp in pred.waypoints:
        assert wp.x == pytest.approx(7.0)
        assert wp.y == pytest.approx(-3.0)
        assert wp.vx == pytest.approx(0.0)
        assert wp.vy == pytest.approx(0.0)

    # Uncertainty still grows because of process noise Q(dt)
    assert pred.waypoints[-1].uncertainty_radius > pred.waypoints[0].uncertainty_radius


# ============================================================================
# TEST N: Accelerating target model mismatch
# ============================================================================
def test_n_accelerating_target_model_mismatch() -> None:
    predictor = MotionPredictor()
    # True target accelerates with ax = 2.0 m/s^2 from vx0 = 0.0
    ax = 2.0
    track = make_test_track(x=0.0, y=0.0, vx=0.0, vy=0.0)

    pred = predictor.predict_track(track, horizon_s=2.0, dt_s=0.5)

    # Predictor assumes vx = 0 constantly, so predicted x(t) = 0.0
    # True position is x_gt(t) = 0.5 * ax * t^2
    errors = []
    for wp in pred.waypoints:
        t = wp.time_offset
        x_gt = 0.5 * ax * (t ** 2)
        err = abs(wp.x - x_gt)
        errors.append((t, err))

    # Error must grow quadratically with horizon
    assert errors[0][1] == pytest.approx(0.0)  # t=0 -> err=0
    assert errors[-1][1] == pytest.approx(0.5 * ax * (2.0 ** 2))  # t=2 -> err=4.0m
    # Verify predictor didn't cheat by knowing ax
    assert pred.waypoints[-1].x == pytest.approx(0.0)


# ============================================================================
# TEST O: Turning target model mismatch
# ============================================================================
def test_o_turning_target_model_mismatch() -> None:
    predictor = MotionPredictor()
    # Target moving in circle of radius R=5m at omega=1 rad/s (speed v=5m/s)
    # Initial state at (0, 0) moving along +X: vx=5.0, vy=0.0
    R = 5.0
    omega = 1.0
    track = make_test_track(x=0.0, y=0.0, vx=5.0, vy=0.0)

    pred = predictor.predict_track(track, horizon_s=1.5, dt_s=0.25)

    # Ground truth circular trajectory:
    # x_gt(t) = R * sin(omega * t)
    # y_gt(t) = R * (1 - cos(omega * t))
    for wp in pred.waypoints:
        t = wp.time_offset
        if t == 0.0:
            continue
        x_gt = R * math.sin(omega * t)
        y_gt = R * (1.0 - math.cos(omega * t))
        euclidean_err = math.hypot(wp.x - x_gt, wp.y - y_gt)
        # CV predicts straight line y=0, so error in y grows
        assert wp.y == pytest.approx(0.0)
        assert euclidean_err > 0.0


# ============================================================================
# TEST P: Multiple tracks
# ============================================================================
def test_p_multiple_tracks() -> None:
    predictor = MotionPredictor()
    tracks = [
        make_test_track(track_id=1, x=0.0, y=0.0, vx=1.0, vy=0.0),
        make_test_track(track_id=2, x=10.0, y=10.0, vx=0.0, vy=2.0),
        make_test_track(track_id=3, x=-5.0, y=2.0, vx=-1.0, vy=1.0),
        make_test_track(track_id=4, x=20.0, y=-10.0, vx=0.5, vy=-0.5),
    ]

    arr = TrackedObstacleArray(header_timestamp=10.0, tracks=tracks)
    pred_arr = predictor.predict_tracks(arr, horizon_s=1.0, dt_s=0.5)

    assert len(pred_arr) == 4
    for i, t in enumerate(tracks):
        pred_single = predictor.predict_track(t, horizon_s=1.0, dt_s=0.5)
        p = pred_arr[i]
        assert p.track_id == t.track_id
        assert len(p.waypoints) == len(pred_single.waypoints)
        for wp1, wp2 in zip(p.waypoints, pred_single.waypoints):
            assert wp1.x == pytest.approx(wp2.x)
            assert wp1.y == pytest.approx(wp2.y)
            assert wp1.uncertainty_radius == pytest.approx(wp2.uncertainty_radius)


# ============================================================================
# TEST Q: Confidence/state handling
# ============================================================================
def test_q_confidence_state_handling() -> None:
    predictor = MotionPredictor()
    track_normal = make_test_track(track_id=1, status="TRACKING")
    track_degraded = make_test_track(track_id=2, status="DEGRADED")

    pred_normal = predictor.predict_track(track_normal, horizon_s=2.0, dt_s=0.5)
    pred_degraded = predictor.predict_track(track_degraded, horizon_s=2.0, dt_s=0.5)

    # State positions must be identical (same nominal state)
    for wp_n, wp_d in zip(pred_normal.waypoints, pred_degraded.waypoints):
        assert wp_n.x == pytest.approx(wp_d.x)
        assert wp_n.y == pytest.approx(wp_d.y)

    # For t > 0, degraded track must have strictly higher covariance and uncertainty radius
    for wp_n, wp_d in zip(pred_normal.waypoints[1:], pred_degraded.waypoints[1:]):
        assert np.trace(wp_d.covariance_2d) > np.trace(wp_n.covariance_2d)
        assert wp_d.uncertainty_radius > wp_n.uncertainty_radius


# ============================================================================
# TEST R: LOST track handling
# ============================================================================
def test_r_lost_track_handling() -> None:
    predictor = MotionPredictor()
    track_lost = make_test_track(track_id=3, status="LOST")

    # Request 2.0s horizon on LOST track
    pred = predictor.predict_track(track_lost, horizon_s=2.0, dt_s=0.25)

    # Capped at lost_max_horizon_s (1.0s by default)
    assert pred.horizon_s == pytest.approx(predictor.config.lost_max_horizon_s)
    assert pred.waypoints[-1].time_offset <= predictor.config.lost_max_horizon_s + 1e-6


# ============================================================================
# TEST S: EXPIRED track handling
# ============================================================================
def test_s_expired_track_handling() -> None:
    predictor = MotionPredictor()
    track_expired = make_test_track(track_id=4, status="EXPIRED")

    # Direct predict_track must raise ValueError
    with pytest.raises(ValueError, match="EXPIRED"):
        predictor.predict_track(track_expired)

    # Batch predict_tracks must gracefully exclude EXPIRED tracks
    tracks = [
        make_test_track(track_id=1, status="TRACKING"),
        track_expired,
        make_test_track(track_id=2, status="TRACKING"),
    ]
    pred_arr = predictor.predict_tracks(tracks)
    assert len(pred_arr) == 2
    assert [p.track_id for p in pred_arr.predictions] == [1, 2]


# ============================================================================
# TEST T: Deterministic reproducibility
# ============================================================================
def test_t_deterministic_reproducibility() -> None:
    predictor = MotionPredictor()
    track = make_test_track(x=3.5, y=-2.1, vx=1.2, vy=-0.8)

    pred1 = predictor.predict_track(track, horizon_s=1.5, dt_s=0.25)
    pred2 = predictor.predict_track(track, horizon_s=1.5, dt_s=0.25)

    assert len(pred1.waypoints) == len(pred2.waypoints)
    for wp1, wp2 in zip(pred1.waypoints, pred2.waypoints):
        assert wp1.x == wp2.x
        assert wp1.y == wp2.y
        assert wp1.vx == wp2.vx
        assert wp1.vy == wp2.vy
        assert wp1.uncertainty_radius == wp2.uncertainty_radius
        np.testing.assert_array_equal(wp1.covariance_2d, wp2.covariance_2d)


# ============================================================================
# TEST U: NaN/Inf protection
# ============================================================================
def test_u_nan_inf_protection() -> None:
    predictor = MotionPredictor()
    track = make_test_track()

    with pytest.raises(ValueError, match="finite"):
        predictor.predict_track(track, horizon_s=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        predictor.predict_track(track, horizon_s=float("inf"))

    with pytest.raises(ValueError, match="finite"):
        predictor.predict_track(track, dt_s=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        predictor.predict_track(track, time_offsets_s=[0.0, float("nan"), 1.0])


# ============================================================================
# TEST V: Ground-truth isolation
# ============================================================================
def test_v_ground_truth_isolation() -> None:
    import ai.planning.dynamic.predictor as predictor_module

    source_code = inspect.getsource(predictor_module)
    forbidden_terms = [
        "ground_truth",
        "gt_registry",
        "simulator",
        "ground_truth_pose",
        "ai.simulation",
        "gazebo",
    ]

    for term in forbidden_terms:
        assert term not in source_code.lower(), (
            f"Forbidden term '{term}' detected in predictor module! "
            "Ground-truth isolation violated."
        )
