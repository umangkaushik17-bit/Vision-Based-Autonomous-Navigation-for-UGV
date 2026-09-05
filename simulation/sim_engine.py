"""Deterministic Kinematic and Sensor Software-in-the-Loop (SIL) Simulation Engine for Outdoor 4WD UGV Navigation.

Provides physically consistent simulated sensor streams (Stereo/RGB Camera,
200 Hz IMU, Joint States, Clock) matching Gazebo Sim SDF specifications,
with isolated ground truth for objective evaluation.
"""

from __future__ import annotations

import cv2
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ai.localization.base import CameraIntrinsics
from ai.spatial_fusion.transforms import (
    SpatialCameraExtrinsics,
    create_se3_matrix,
    euler_to_rotation_matrix,
    transform_points_3d,
)
from simulation.motion_controller import RobotState2D, UGVMotionController, UGVCommand


class SimulatedSensorStream:
    """Encapsulates a single time-stamped simulation frame."""

    def __init__(
        self,
        timestamp_s: float,
        image_left: np.ndarray,
        image_right: np.ndarray,
        camera_info: Dict[str, Any],
        imu_data: Dict[str, Any],
        joint_states: Dict[str, Any],
        ground_truth_pose: np.ndarray,
        ground_truth_state: RobotState2D,
    ):
        self.timestamp = float(timestamp_s)
        self.image_left = image_left
        self.image_right = image_right
        self.camera_info = camera_info
        self.imu_data = imu_data
        self.joint_states = joint_states
        self.ground_truth_pose = ground_truth_pose  # 4x4 SE(3) in odom frame
        self.ground_truth_state = ground_truth_state


class GazeboSimEngine:
    """Deterministic simulation engine driving outdoor terrain environments,

    robot physics, camera ray casting, and sensor generation.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parent
        cfg_file = Path(config_path) if config_path else base_dir / "config" / "simulation_params.yaml"
        gt_file = Path(ground_truth_path) if ground_truth_path else base_dir / "config" / "ground_truth_obstacles.json"

        # Load Obstacle Registry
        if gt_file.exists():
            with open(gt_file, "r") as f:
                self.gt_registry = json.load(f)
        else:
            self.gt_registry = {"obstacles": [], "safe_regions": []}

        # Motion & Kinematics
        self.motion = UGVMotionController(
            track_width_m=0.48,
            wheel_radius_m=0.125,
            max_linear_vel_m_s=1.5,
            max_angular_vel_rad_s=1.0,
        )

        # Extrinsics and Intrinsics
        self.extrinsics = SpatialCameraExtrinsics(
            forward_offset_m=0.15,
            lateral_offset_m=0.0,
            height_m=0.40,
            pitch_deg=15.0,
        )
        self.intrinsics = CameraIntrinsics(
            fx=462.14,
            fy=462.14,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
            baseline_m=0.05,
        )

        # Persistent outdoor terrain texture landmarks for consistent visual odometry tracking
        rng = np.random.RandomState(42)
        self.num_landmarks = 1200
        self.landmarks = rng.uniform(
            low=[-5.0, -8.0, -0.05],
            high=[25.0, 8.0, 0.20],
            size=(self.num_landmarks, 3)
        )

        # Simulation Time and State
        self.sim_time_s = 0.0
        self.dt_s = 0.005  # 200 Hz physics step
        self.camera_dt_s = 1.0 / 30.0  # 30 Hz camera
        self.joint_dt_s = 1.0 / 50.0   # 50 Hz joint states
        self.last_camera_time = -1.0
        self.last_joint_time = -1.0

        # Failure Injection Controls
        self.camera_enabled = True
        self.imu_enabled = True
        self.tf_enabled = True
        self.tracking_stress = False

        # Wheel positions (rad)
        self.wheel_angles = {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}

    def set_command(self, linear_x: float, angular_z: float) -> None:
        """Forward /cmd_vel velocity commands to motion controller."""
        self.motion.set_command(linear_x, angular_z)

    def step_physics(self, dt_s: Optional[float] = None) -> RobotState2D:
        """Step robot kinematics and simulation clock forward."""
        dt = dt_s if dt_s is not None else self.dt_s
        st = self.motion.step(dt)
        self.sim_time_s += dt

        # Update wheel joint angles
        w_left, w_right = self.motion.compute_wheel_speeds()
        self.wheel_angles["fl"] += w_left * dt
        self.wheel_angles["rl"] += w_left * dt
        self.wheel_angles["fr"] += w_right * dt
        self.wheel_angles["rr"] += w_right * dt

        return st

    def get_ground_truth_se3_pose(self) -> np.ndarray:
        """Returns 4x4 SE(3) pose matrix in odom frame (Ground Truth only)."""
        st = self.motion.state
        R = euler_to_rotation_matrix(0.0, 0.0, st.yaw)
        t = np.array([st.x, st.y, 0.125], dtype=np.float64)
        return create_se3_matrix(R, t)

    def render_imu(self) -> Dict[str, Any]:
        """Generates simulated 200 Hz IMU message with noise."""
        if not self.imu_enabled:
            return {"status": "FAULT_DISCONNECTED", "timestamp": self.sim_time_s}

        st = self.motion.state
        # Accelerometer includes gravity + forward acceleration + centripetal
        a_fwd = 0.0
        w_z = st.angular_z
        a_centripetal = st.linear_x * w_z

        # Add Gaussian sensor noise
        noise_a = np.random.normal(0, 0.005, 3)
        noise_w = np.random.normal(0, 0.001, 3)

        accel = np.array([a_fwd + noise_a[0], a_centripetal + noise_a[1], 9.81 + noise_a[2]])
        gyro = np.array([noise_w[0], noise_w[1], w_z + noise_w[2]])

        # Orientation quaternion (yaw around Z)
        qz = math.sin(st.yaw / 2.0)
        qw = math.cos(st.yaw / 2.0)

        return {
            "header": {"frame_id": "imu_link", "stamp": self.sim_time_s},
            "linear_acceleration": accel.tolist(),
            "angular_velocity": gyro.tolist(),
            "orientation": [0.0, 0.0, qz, qw],
            "linear_acceleration_covariance": [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.0001],
            "angular_velocity_covariance": [0.00001, 0, 0, 0, 0.00001, 0, 0, 0, 0.00001],
        }

    def render_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Renders synthetic stereo camera pair and CameraInfo based on current pose

        and obstacle positions.
        """
        W, H = self.intrinsics.width, self.intrinsics.height
        if not self.camera_enabled:
            # Black frame on camera failure
            img_left = np.zeros((H, W, 3), dtype=np.uint8)
            img_right = np.zeros((H, W, 3), dtype=np.uint8)
            info = {"status": "CAMERA_FAILURE", "stamp": self.sim_time_s}
            return img_left, img_right, info

        # Background: Realistic outdoor grass/dirt gradient
        img_left = np.full((H, W, 3), [40, 70, 45], dtype=np.uint8)
        img_right = np.full((H, W, 3), [40, 70, 45], dtype=np.uint8)

        # Sky in top rows
        img_left[:120, :] = [180, 160, 100]
        img_right[:120, :] = [180, 160, 100]

        # Calculate camera pose in odom frame
        T_odom_base = self.get_ground_truth_se3_pose()
        T_odom_cam = T_odom_base @ self.extrinsics.T_body_camera

        # Optical frame: X right, Y down, Z forward
        R_wc = T_odom_cam[:3, :3]
        t_wc = T_odom_cam[:3, 3]
        R_cw = R_wc.T
        t_cw_l = -R_cw @ t_wc
        t_cw_r = t_cw_l - np.array([self.intrinsics.baseline_m, 0.0, 0.0], dtype=np.float64)

        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        # 1. Project persistent terrain visual landmarks with true stereo disparity
        for lm in self.landmarks:
            p_l = R_cw @ lm + t_cw_l
            p_r = R_cw @ lm + t_cw_r
            if p_l[2] > 0.4 and p_r[2] > 0.4:
                ul = int(fx * (p_l[0] / p_l[2]) + cx)
                vl = int(fy * (p_l[1] / p_l[2]) + cy)
                ur = int(fx * (p_r[0] / p_r[2]) + cx)
                vr = int(fy * (p_r[1] / p_r[2]) + cy)

                if 10 <= ul < W - 10 and 10 <= vl < H - 10:
                    cv2.drawMarker(img_left, (ul, vl), (230, 230, 230), markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=2)
                    cv2.circle(img_left, (ul, vl), 2, (30, 30, 30), -1)

                if 10 <= ur < W - 10 and 10 <= vr < H - 10:
                    cv2.drawMarker(img_right, (ur, vr), (230, 230, 230), markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=2)
                    cv2.circle(img_right, (ur, vr), 2, (30, 30, 30), -1)

        # 2. Project obstacles into stereo camera pair
        for obs in self.gt_registry.get("obstacles", []):
            p_world = np.array(obs["position"], dtype=np.float64)
            p_l = R_cw @ p_world + t_cw_l
            p_r = R_cw @ p_world + t_cw_r

            obs_type = obs.get("type", "rock")
            color = (120, 120, 130) if obs_type == "rock" else (30, 80, 40)
            if obs_type in ("water", "puddle"):
                color = (160, 100, 40)
            elif obs_type == "ditch":
                color = (25, 35, 45)

            if 0.5 < p_l[2] < 25.0:
                ul = int(fx * (p_l[0] / p_l[2]) + cx)
                vl = int(fy * (p_l[1] / p_l[2]) + cy)
                rad_l = max(4, int((obs.get("bounding_radius_m", 0.5) / p_l[2]) * fx))
                if 0 <= ul < W and 0 <= vl < H:
                    cv2.circle(img_left, (ul, vl), rad_l, color, -1)

            if 0.5 < p_r[2] < 25.0:
                ur = int(fx * (p_r[0] / p_r[2]) + cx)
                vr = int(fy * (p_r[1] / p_r[2]) + cy)
                rad_r = max(4, int((obs.get("bounding_radius_m", 0.5) / p_r[2]) * fx))
                if 0 <= ur < W and 0 <= vr < H:
                    cv2.circle(img_right, (ur, vr), rad_r, color, -1)

        # Add subtle sensor noise
        noise_l = np.random.randint(-3, 3, (H, W, 3), dtype=np.int16)
        noise_r = np.random.randint(-3, 3, (H, W, 3), dtype=np.int16)
        img_left = np.clip(img_left.astype(np.int16) + noise_l, 0, 255).astype(np.uint8)
        img_right = np.clip(img_right.astype(np.int16) + noise_r, 0, 255).astype(np.uint8)

        info = {
            "header": {"frame_id": "camera_optical", "stamp": self.sim_time_s},
            "width": W,
            "height": H,
            "distortion_model": "plumb_bob",
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "K": [
                self.intrinsics.fx, 0.0, self.intrinsics.cx,
                0.0, self.intrinsics.fy, self.intrinsics.cy,
                0.0, 0.0, 1.0
            ],
            "P": [
                self.intrinsics.fx, 0.0, self.intrinsics.cx, 0.0,
                0.0, self.intrinsics.fy, self.intrinsics.cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ],
        }

        return img_left, img_right, info

    def render_joint_states(self) -> Dict[str, Any]:
        """Generates simulated 50 Hz JointState message."""
        w_left, w_right = self.motion.compute_wheel_speeds()
        return {
            "header": {"frame_id": "base_link", "stamp": self.sim_time_s},
            "name": ["wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"],
            "position": [
                self.wheel_angles["fl"],
                self.wheel_angles["fr"],
                self.wheel_angles["rl"],
                self.wheel_angles["rr"],
            ],
            "velocity": [w_left, w_right, w_left, w_right],
            "effort": [0.0, 0.0, 0.0, 0.0],
        }

    def get_next_sensor_cycle(self) -> SimulatedSensorStream:
        """Step forward to next camera cycle (30 Hz) and assemble sensor payload."""
        target_time = self.sim_time_s + self.camera_dt_s
        while self.sim_time_s < target_time - 1e-6:
            self.step_physics(self.dt_s)

        img_l, img_r, cinfo = self.render_camera_frame()
        imu = self.render_imu()
        joints = self.render_joint_states()
        gt_pose = self.get_ground_truth_se3_pose()

        return SimulatedSensorStream(
            timestamp_s=self.sim_time_s,
            image_left=img_l,
            image_right=img_r,
            camera_info=cinfo,
            imu_data=imu,
            joint_states=joints,
            ground_truth_pose=gt_pose,
            ground_truth_state=self.motion.state,
        )
