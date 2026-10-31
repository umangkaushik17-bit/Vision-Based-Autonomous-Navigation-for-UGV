"""Unit tests for Lost-Obstacle Hazard Latching and Authoritative Safety Governor (Phase 13.23).

Mandatory Scenarios Tested:
A. Obstacle continuously detected (Normal tracking).
B. Obstacle lost for 1 frame (Latched hazard persists, safety stop maintained).
C. Obstacle lost for several frames (Persists across multiple frames, no acceleration into obstacle).
D. Obstacle ID changes (Proximity memory associates hazard seamlessly).
E. Complete detector blackout (All close hazards latched, UGV holds stop).
F. Tracker temporarily loses object (Zero unsafe STOP -> PROCEED transitions).
G. Obstacle disappears after moving far away (> 2.5m: safely expires / not latched).
H. Obstacle remains close (< 1.2m: extended conservative TTL).
I. Obstacle reappears (Reacquisition updates memory seamlessly).
J. Stale hazard expires safely after TTL (Free space safely restored after timeout).
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from ai.planning.dynamic.data_model import TrackedObstacleState
from ai.planning.safety_governor import (
    LatchedHazard,
    RecentHazardMemory,
    RecentHazardMemoryConfig,
    SafetyGovernor,
    SafetyGovernorConfig,
)


def _make_track(track_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0, timestamp: float = 0.0) -> TrackedObstacleState:
    return TrackedObstacleState(
        track_id=track_id,
        timestamp=timestamp,
        frame_id="odom",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        covariance=np.eye(4) * 0.05,
        bbox_size=(0.8, 0.8),
        bbox_yaw=0.0,
        track_age=5,
        coast_cycles=0,
        is_confirmed=True,
        status="TRACKING",
    )


def test_scenario_a_continuous_detection():
    """Scenario A: Obstacle continuously detected maintains normal governor clearance checks."""
    memory = RecentHazardMemory()
    gov = SafetyGovernor()
    robot_pose = (0.0, 0.0, 0.0)

    # Obstacle at x=1.5m, y=0.0 (in front, comfortable clearance)
    tracks = [_make_track(1, x=1.5, y=0.0, timestamp=1.0)]
    hazards = memory.update(tracks, current_time_s=1.0, robot_pose=robot_pose)

    v_cmd, w_cmd, decision, diag = gov.evaluate(
        requested_v=0.50, requested_w=0.0, robot_pose=robot_pose, hazards=hazards, current_time_s=1.0
    )
    assert decision == "ALLOW"
    assert v_cmd == 0.50


def test_scenario_b_c_lost_single_and_multiple_frames():
    """Scenarios B & C: Close obstacle lost for 1 or several frames must NOT allow acceleration."""
    memory = RecentHazardMemory()
    gov = SafetyGovernor()
    robot_pose = (0.0, 0.0, 0.0)

    # Obstacle close ahead at x=0.80m (stopping distance is 0.40m, radius is 0.40m -> effective dist ~0.40m)
    tracks = [_make_track(1, x=0.75, y=0.0, timestamp=10.0)]
    hazards = memory.update(tracks, current_time_s=10.0, robot_pose=robot_pose)

    # Frame 1: Detected -> STOP enforced
    v, w, dec, _ = gov.evaluate(0.50, 0.0, robot_pose, hazards, 10.0)
    assert v == 0.0
    assert "STOP_FORWARD_HAZARD" in dec

    # Frame 2: Lost for 1 frame (empty tracks)
    hazards_f2 = memory.update([], current_time_s=10.10, robot_pose=robot_pose)
    assert len(hazards_f2) == 1, "Close obstacle must be latched upon dropout"
    v2, w2, dec2, _ = gov.evaluate(0.50, 0.0, robot_pose, hazards_f2, 10.10)
    assert v2 == 0.0, "Must NOT accelerate when obstacle dropped out for 1 frame"
    assert "STOP_FORWARD_HAZARD" in dec2

    # Frame 3-10: Lost for 1.0 second (10 frames)
    hazards_f10 = memory.update([], current_time_s=11.00, robot_pose=robot_pose)
    assert len(hazards_f10) == 1
    v10, w10, dec10, _ = gov.evaluate(0.50, 0.0, robot_pose, hazards_f10, 11.00)
    assert v10 == 0.0, "Must NOT accelerate when obstacle dropped out for 10 frames"


def test_scenario_d_id_change():
    """Scenario D: When obstacle tracker reassigns ID, memory tracks both seamlessly without free-space gap."""
    memory = RecentHazardMemory()
    gov = SafetyGovernor()
    robot_pose = (0.0, 0.0, 0.0)

    # Track 1 at x=0.70m
    t1 = [_make_track(1, x=0.70, y=0.0, timestamp=5.0)]
    memory.update(t1, 5.0, robot_pose)

    # Track 1 lost, Track 2 appears nearby at x=0.72m
    t2 = [_make_track(2, x=0.72, y=0.0, timestamp=5.10)]
    hazards = memory.update(t2, 5.10, robot_pose)

    v, _, dec, _ = gov.evaluate(0.50, 0.0, robot_pose, hazards, 5.10)
    assert v == 0.0
    assert "STOP_FORWARD_HAZARD" in dec


def test_scenario_e_detector_blackout():
    """Scenario E: Complete detector blackout (100% loss) maintains safe stop for all close obstacles."""
    memory = RecentHazardMemory()
    gov = SafetyGovernor()
    robot_pose = (0.0, 0.0, 0.0)

    # Two close obstacles
    obs = [_make_track(1, x=0.70, y=0.0, timestamp=1.0), _make_track(2, x=0.90, y=0.20, timestamp=1.0)]
    memory.update(obs, 1.0, robot_pose)

    # Blackout for 2 seconds
    for step in range(20):
        t_now = 1.0 + step * 0.10
        hazards = memory.update([], t_now, robot_pose)
        v, _, dec, _ = gov.evaluate(0.50, 0.0, robot_pose, hazards, t_now)
        assert v == 0.0, f"At step {step} during blackout, vehicle must remain stopped"


def test_scenario_f_zero_stop_to_proceed_inversions():
    """Scenario F: Assert strictly 0 unsafe STOP -> PROCEED inversions during tracking dropout."""
    memory = RecentHazardMemory()
    gov = SafetyGovernor()
    robot_pose = (0.0, 0.0, 0.0)

    # Obstacle pinning UGV front
    memory.update([_make_track(1, x=0.60, y=0.0, timestamp=0.0)], 0.0, robot_pose)

    # Simulate 30 cycles of complete dropout
    for step in range(1, 30):
        t = step * 0.10
        hazards = memory.update([], t, robot_pose)
        # Even if DWA proposed requested_v = 0.80 m/s (thinking space is free)
        gov_v, _, dec, _ = gov.evaluate(requested_v=0.80, requested_w=0.0, robot_pose=robot_pose, hazards=hazards, current_time_s=t)
        assert gov_v == 0.0, f"Unsafe STOP -> PROCEED occurred at t={t}!"


def test_scenario_g_far_obstacle_not_latched():
    """Scenario G: Obstacle that moved away (> 2.5m) is not latched as critical proximity hazard."""
    memory = RecentHazardMemory(RecentHazardMemoryConfig(proximity_threshold_m=2.50))
    robot_pose = (0.0, 0.0, 0.0)

    # Obstacle at x=4.0m (far away)
    tracks = [_make_track(1, x=4.0, y=0.0, timestamp=1.0)]
    hazards = memory.update(tracks, 1.0, robot_pose)
    assert len(hazards) == 0, "Far obstacles (> 2.5m) should not populate proximity latch"


def test_scenario_h_close_obstacle_extended_ttl():
    """Scenario H: Obstacle < 1.2m gets extended conservative TTL (5.0s)."""
    memory = RecentHazardMemory(RecentHazardMemoryConfig(critical_proximity_m=1.20, close_ttl_s=5.00, base_ttl_s=3.00))
    robot_pose = (0.0, 0.0, 0.0)

    # Critical close obstacle at x=0.8m
    memory.update([_make_track(1, x=0.8, y=0.0, timestamp=1.0)], 1.0, robot_pose)
    hazards = memory.update([], 1.0, robot_pose)
    assert hazards[0].ttl_s == 5.00


def test_scenario_i_obstacle_reappears():
    """Scenario I: Obstacle drops out then reappears; memory updates seamlessly."""
    memory = RecentHazardMemory()
    robot_pose = (0.0, 0.0, 0.0)

    # t=1.0: Seen
    memory.update([_make_track(1, x=0.8, y=0.0, timestamp=1.0)], 1.0, robot_pose)
    # t=1.5: Dropped
    h1 = memory.update([], 1.5, robot_pose)
    assert len(h1) == 1
    # t=2.0: Reappears with new observation
    h2 = memory.update([_make_track(1, x=0.82, y=0.0, timestamp=2.0)], 2.0, robot_pose)
    assert len(h2) == 1
    assert h2[0].last_observed_time_s == 2.0
    assert pytest.approx(h2[0].confidence, abs=1e-2) == 0.95


def test_scenario_j_stale_hazard_expires_safely():
    """Scenario J: After TTL expires, hazard is safely pruned, allowing valid free-space recovery."""
    memory = RecentHazardMemory(RecentHazardMemoryConfig(proximity_threshold_m=2.50, base_ttl_s=2.00, close_ttl_s=2.00))
    gov = SafetyGovernor()
    robot_pose = (0.0, 0.0, 0.0)

    memory.update([_make_track(1, x=0.75, y=0.0, timestamp=1.0)], 1.0, robot_pose)

    # At t=2.0 (age = 1.0s <= 2.0s): still held
    h_held = memory.update([], 2.0, robot_pose)
    assert len(h_held) == 1
    v, _, _, _ = gov.evaluate(0.50, 0.0, robot_pose, h_held, 2.0)
    assert v == 0.0

    # At t=3.5 (age = 2.5s > 2.0s): expired!
    h_expired = memory.update([], 3.5, robot_pose)
    assert len(h_expired) == 0, "Hazard must expire after TTL"
    v_free, _, dec_free, _ = gov.evaluate(0.50, 0.0, robot_pose, h_expired, 3.5)
    assert v_free == 0.50
    assert dec_free == "ALLOW"
