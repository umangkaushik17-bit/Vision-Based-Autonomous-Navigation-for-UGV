"""Unit tests for Phase 11 simulated sensors (Camera, IMU, Joint States, Clock)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from simulation.sim_engine import GazeboSimEngine


def test_simulated_camera_generation():
    """Verify camera rendering generates valid 640x480 stereo frames and CameraInfo."""
    engine = GazeboSimEngine()
    img_l, img_r, info = engine.render_camera_frame()

    assert isinstance(img_l, np.ndarray)
    assert img_l.shape == (480, 640, 3)
    assert img_l.dtype == np.uint8
    assert np.mean(img_l) > 10.0  # Non-blank outdoor terrain

    assert isinstance(img_r, np.ndarray)
    assert img_r.shape == (480, 640, 3)

    # Check CameraInfo
    assert info["width"] == 640
    assert info["height"] == 480
    assert info["header"]["frame_id"] == "camera_optical"
    assert info["K"][0] == pytest.approx(525.0)  # fx
    assert info["K"][4] == pytest.approx(525.0)  # fy
    assert info["K"][2] == pytest.approx(320.0)  # cx
    assert info["K"][5] == pytest.approx(240.0)  # cy


def test_simulated_imu_generation():
    """Verify IMU rendering outputs 200 Hz messages with realistic gravity and rotation."""
    engine = GazeboSimEngine()

    # Stationary state
    imu = engine.render_imu()
    assert imu["header"]["frame_id"] == "imu_link"
    accel = imu["linear_acceleration"]
    # Z-axis should experience approximately 1g (9.81 m/s^2)
    assert accel[2] == pytest.approx(9.81, abs=0.20)
    assert abs(accel[0]) < 0.20
    assert abs(accel[1]) < 0.20

    # Turning state
    engine.set_command(linear_x=0.5, angular_z=0.3)
    engine.step_physics(0.10)
    imu_turning = engine.render_imu()
    gyro = imu_turning["angular_velocity"]
    assert gyro[2] == pytest.approx(0.30, abs=0.05)


def test_simulated_joint_states_and_wheel_motion():
    """Verify 4-wheel joint state reporting and wheel angle progression."""
    engine = GazeboSimEngine()
    js0 = engine.render_joint_states()

    assert len(js0["name"]) == 4
    assert "wheel_fl_joint" in js0["name"]
    assert "wheel_fr_joint" in js0["name"]
    assert "wheel_rl_joint" in js0["name"]
    assert "wheel_rr_joint" in js0["name"]
    assert js0["position"] == [0.0, 0.0, 0.0, 0.0]

    # Command forward motion (0.5 m/s) for 1 second
    engine.set_command(linear_x=0.5, angular_z=0.0)
    for _ in range(200):
        engine.step_physics(0.005)

    js1 = engine.render_joint_states()
    # Expected wheel angle: distance / radius = 0.5m / 0.125m = 4.0 radians
    for pos in js1["position"]:
        assert pos == pytest.approx(4.0, abs=0.10)


def test_simulation_clock_monotonicity():
    """Verify simulation clock strictly advances without wall-clock leakage."""
    engine = GazeboSimEngine()
    prev_t = engine.sim_time_s

    for _ in range(10):
        stream = engine.get_next_sensor_cycle()
        curr_t = stream.timestamp
        assert curr_t > prev_t
        assert curr_t == pytest.approx(prev_t + engine.camera_dt_s, abs=engine.dt_s)
        prev_t = curr_t
