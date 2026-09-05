"""TF2 Broadcaster utilities and static extrinsics initialization."""

from __future__ import annotations

import numpy as np

from ai.spatial_fusion.transforms import SpatialCameraExtrinsics, euler_to_rotation_matrix, create_se3_matrix
from ugv_navigation.messages.localization_status_msg import Point3D, Quaternion
from ugv_navigation.tf.tf_tree import Transform3D, TransformStamped, TfTreeBuffer
from ugv_navigation.utils.time_utils import RosHeader, RosTime


class StaticCameraExtrinsicsPublisher:
    """Publishes nominal static transforms:

    base_link -> camera_link (Mount translation + pitch angle)
    camera_link -> camera_optical (Standard ROS optical frame rotation)
    """

    @staticmethod
    def configure_extrinsics(
        tf_buffer: TfTreeBuffer,
        forward_offset_m: float = 0.15,
        lateral_offset_m: float = 0.0,
        height_offset_m: float = 0.40,
        pitch_deg: float = 15.0,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        timestamp_s: float = 0.0,
    ) -> None:
        stamp = RosTime.from_sec(timestamp_s)

        # 1. base_link -> camera_link (Mount pose)
        R_mount = euler_to_rotation_matrix(np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg))
        t_mount = np.array([forward_offset_m, lateral_offset_m, height_offset_m], dtype=np.float64)
        T_base_camlink = create_se3_matrix(R_mount, t_mount)

        tf_base_camlink = TransformStamped(
            header=RosHeader(stamp=stamp, frame_id=TfTreeBuffer.FRAME_BASE_LINK),
            child_frame_id=TfTreeBuffer.FRAME_CAMERA_LINK,
            transform=Transform3D.from_matrix(T_base_camlink),
        )
        tf_buffer.set_static_transform(tf_base_camlink)

        # 2. camera_link -> camera_optical
        # Standard ROS conversion from robot camera_link (X forward, Y left, Z up)
        # to camera_optical (X right, Y down, Z forward):
        # x_opt = -y_link, y_opt = -z_link, z_opt = x_link
        # Inverse: x_link = z_opt, y_link = -x_opt, z_link = -y_opt
        R_link_opt = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)
        T_camlink_opt = create_se3_matrix(R_link_opt, np.zeros(3))

        tf_camlink_opt = TransformStamped(
            header=RosHeader(stamp=stamp, frame_id=TfTreeBuffer.FRAME_CAMERA_LINK),
            child_frame_id=TfTreeBuffer.FRAME_CAMERA_OPTICAL,
            transform=Transform3D.from_matrix(T_camlink_opt),
        )
        tf_buffer.set_static_transform(tf_camlink_opt)


class DynamicOdomBroadcaster:
    """Broadcasts dynamic odom -> base_link transforms from localization estimates."""

    def __init__(self, tf_buffer: TfTreeBuffer):
        self.tf_buffer = tf_buffer

    def broadcast_pose(self, pose_matrix_4x4: np.ndarray, timestamp_s: float, frame_id: str = "odom", child_frame_id: str = "base_link") -> TransformStamped:
        stamp = RosTime.from_sec(timestamp_s)
        tf = TransformStamped(
            header=RosHeader(stamp=stamp, frame_id=frame_id),
            child_frame_id=child_frame_id,
            transform=Transform3D.from_matrix(pose_matrix_4x4),
        )
        self.tf_buffer.set_transform(tf)
        return tf
