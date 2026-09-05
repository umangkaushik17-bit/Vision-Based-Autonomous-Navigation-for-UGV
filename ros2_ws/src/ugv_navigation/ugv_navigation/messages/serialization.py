"""High-performance serialization and deserialization codecs for UGV ROS2 messages."""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any, Dict

from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg, Point3D, Quaternion, Pose, Twist, PoseWithCovariance, TwistWithCovariance
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.utils.time_utils import RosHeader, RosTime


class MessageSerializer:
    """Provides fast, deterministic binary and JSON codecs for UGV ROS2 messages."""

    @staticmethod
    def serialize_terrain_map_json(msg: WorldTerrainMapMsg) -> str:
        data = {
            "header": {"stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec}, "frame_id": msg.header.frame_id},
            "resolution": msg.resolution,
            "origin": {"x": msg.origin_x, "y": msg.origin_y, "z": msg.origin_z},
            "width": msg.width,
            "height": msg.height,
            "cost": msg.cost,
            "traversability": msg.traversability,
            "obstacle": [int(b) for b in msg.obstacle],
            "hazard": msg.hazard,
            "unknown": [int(b) for b in msg.unknown],
            "confidence": msg.confidence,
            "slope": msg.slope,
            "roughness": msg.roughness,
            "height_var": msg.height_var,
            "observation_count": msg.observation_count,
        }
        return json.dumps(data)

    @staticmethod
    def deserialize_terrain_map_json(payload: str) -> WorldTerrainMapMsg:
        d = json.loads(payload)
        h = d["header"]
        stamp = RosTime(sec=h["stamp"]["sec"], nanosec=h["stamp"]["nanosec"])
        header = RosHeader(stamp=stamp, frame_id=h["frame_id"])
        orig = d["origin"]

        return WorldTerrainMapMsg(
            header=header,
            resolution=d["resolution"],
            origin_x=orig["x"],
            origin_y=orig["y"],
            origin_z=orig["z"],
            width=d["width"],
            height=d["height"],
            cost=d.get("cost", []),
            traversability=d.get("traversability", []),
            obstacle=[bool(b) for b in d.get("obstacle", [])],
            hazard=d.get("hazard", []),
            unknown=[bool(b) for b in d.get("unknown", [])],
            confidence=d.get("confidence", []),
            slope=d.get("slope", []),
            roughness=d.get("roughness", []),
            height_var=d.get("height_var", []),
            observation_count=d.get("observation_count", []),
        )

    @staticmethod
    def serialize_terrain_map_binary(msg: WorldTerrainMapMsg, compress: bool = True) -> bytes:
        """Packs WorldTerrainMapMsg into an efficient compact binary payload."""
        # Header: magic(4s), sec(I), nanosec(I), frame_len(H), frame(s), res(f), orig(3f), W(I), H(I)
        frame_bytes = msg.header.frame_id.encode("utf-8")
        hdr_fmt = f"<4sIIH{len(frame_bytes)}sffffII"
        hdr_bytes = struct.pack(
            hdr_fmt,
            b"WTMP",
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            len(frame_bytes),
            frame_bytes,
            msg.resolution,
            msg.origin_x,
            msg.origin_y,
            msg.origin_z,
            msg.width,
            msg.height,
        )

        n = msg.width * msg.height
        if n == 0:
            payload = hdr_bytes
        else:
            # Pack arrays: cost(f), trav(f), obst(?), haz(f), unk(?), conf(f), slope(f), rough(f), hvar(f), count(I)
            body_fmt = f"<{n}f{n}f{n}?{n}f{n}?{n}f{n}f{n}f{n}f{n}I"
            body_bytes = struct.pack(
                body_fmt,
                *msg.cost,
                *msg.traversability,
                *msg.obstacle,
                *msg.hazard,
                *msg.unknown,
                *msg.confidence,
                *msg.slope,
                *msg.roughness,
                *msg.height_var,
                *msg.observation_count,
            )
            payload = hdr_bytes + body_bytes

        if compress:
            return b"ZLIB" + zlib.compress(payload, level=1)
        return b"RAW_" + payload

    @staticmethod
    def deserialize_terrain_map_binary(data: bytes) -> WorldTerrainMapMsg:
        if data.startswith(b"ZLIB"):
            raw = zlib.decompress(data[4:])
        elif data.startswith(b"RAW_"):
            raw = data[4:]
        else:
            raise ValueError("Unknown binary encoding prefix")

        # Unpack fixed header prefix up to frame_len
        magic, sec, nanosec, flen = struct.unpack("<4sIIH", raw[:14])
        if magic != b"WTMP":
            raise ValueError("Invalid magic header for WorldTerrainMap binary")

        offset = 14
        frame_id = raw[offset : offset + flen].decode("utf-8")
        offset += flen

        res, ox, oy, oz, w, h = struct.unpack("<ffffII", raw[offset : offset + 24])
        offset += 24

        n = w * h
        if n == 0 or offset >= len(raw):
            return WorldTerrainMapMsg(
                header=RosHeader(stamp=RosTime(sec, nanosec), frame_id=frame_id),
                resolution=res,
                origin_x=ox,
                origin_y=oy,
                origin_z=oz,
                width=w,
                height=h,
            )

        body_fmt = f"<{n}f{n}f{n}?{n}f{n}?{n}f{n}f{n}f{n}f{n}I"
        unpacked = struct.unpack_from(body_fmt, raw, offset)

        cost = list(unpacked[0:n])
        trav = list(unpacked[n : 2 * n])
        obst = [bool(b) for b in unpacked[2 * n : 3 * n]]
        haz = list(unpacked[3 * n : 4 * n])
        unk = [bool(b) for b in unpacked[4 * n : 5 * n]]
        conf = list(unpacked[5 * n : 6 * n])
        slope = list(unpacked[6 * n : 7 * n])
        rough = list(unpacked[7 * n : 8 * n])
        hvar = list(unpacked[8 * n : 9 * n])
        count = list(unpacked[9 * n : 10 * n])

        return WorldTerrainMapMsg(
            header=RosHeader(stamp=RosTime(sec, nanosec), frame_id=frame_id),
            resolution=res,
            origin_x=ox,
            origin_y=oy,
            origin_z=oz,
            width=w,
            height=h,
            cost=cost,
            traversability=trav,
            obstacle=obst,
            hazard=haz,
            unknown=unk,
            confidence=conf,
            slope=slope,
            roughness=rough,
            height_var=hvar,
            observation_count=count,
        )

    @staticmethod
    def serialize_localization_status_json(msg: LocalizationStatusMsg) -> str:
        d = {
            "header": {"stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec}, "frame_id": msg.header.frame_id},
            "child_frame_id": msg.child_frame_id,
            "tracking_state": msg.tracking_state,
            "confidence": msg.confidence,
            "is_pose_valid": msg.is_pose_valid,
            "allows_spatial_fusion": msg.allows_spatial_fusion,
            "pose": {
                "position": {"x": msg.pose.pose.position.x, "y": msg.pose.pose.position.y, "z": msg.pose.pose.position.z},
                "orientation": {"x": msg.pose.pose.orientation.x, "y": msg.pose.pose.orientation.y, "z": msg.pose.pose.orientation.z, "w": msg.pose.pose.orientation.w},
                "covariance": msg.pose.covariance,
            },
            "twist": {
                "linear": {"x": msg.twist.twist.linear.x, "y": msg.twist.twist.linear.y, "z": msg.twist.twist.linear.z},
                "angular": {"x": msg.twist.twist.angular.x, "y": msg.twist.twist.angular.y, "z": msg.twist.twist.angular.z},
                "covariance": msg.twist.covariance,
            },
            "tracked_features": msg.tracked_features,
            "inliers": msg.inliers,
            "keyframe_id": msg.keyframe_id,
            "is_keyframe": msg.is_keyframe,
            "relocalized": msg.relocalized,
        }
        return json.dumps(d)

    @staticmethod
    def deserialize_localization_status_json(payload: str) -> LocalizationStatusMsg:
        d = json.loads(payload)
        h = d["header"]
        stamp = RosTime(sec=h["stamp"]["sec"], nanosec=h["stamp"]["nanosec"])
        header = RosHeader(stamp=stamp, frame_id=h["frame_id"])

        p_pos = d["pose"]["position"]
        p_ori = d["pose"]["orientation"]
        pos = Point3D(x=p_pos["x"], y=p_pos["y"], z=p_pos["z"])
        ori = Quaternion(x=p_ori["x"], y=p_ori["y"], z=p_ori["z"], w=p_ori["w"])
        pose_cov = PoseWithCovariance(pose=Pose(position=pos, orientation=ori), covariance=d["pose"].get("covariance", [0.0]*36))

        t_lin = d["twist"]["linear"]
        t_ang = d["twist"]["angular"]
        twist_cov = TwistWithCovariance(
            twist=Twist(linear=Point3D(x=t_lin["x"], y=t_lin["y"], z=t_lin["z"]), angular=Point3D(x=t_ang["x"], y=t_ang["y"], z=t_ang["z"])),
            covariance=d["twist"].get("covariance", [0.0]*36)
        )

        return LocalizationStatusMsg(
            header=header,
            child_frame_id=d["child_frame_id"],
            tracking_state=d["tracking_state"],
            confidence=d["confidence"],
            is_pose_valid=d["is_pose_valid"],
            allows_spatial_fusion=d["allows_spatial_fusion"],
            pose=pose_cov,
            twist=twist_cov,
            tracked_features=d["tracked_features"],
            inliers=d["inliers"],
            keyframe_id=d["keyframe_id"],
            is_keyframe=d["is_keyframe"],
            relocalized=d["relocalized"],
        )

    @staticmethod
    def serialize_dynamic_detections_json(msg: Any) -> str:
        return json.dumps(msg.to_dict() if hasattr(msg, "to_dict") else msg)

    @staticmethod
    def deserialize_dynamic_detections_json(payload: str) -> Any:
        from ugv_navigation.messages.dynamic_messages import DynamicObstacleDetectionArrayMsg
        return DynamicObstacleDetectionArrayMsg.from_dict(json.loads(payload))

    @staticmethod
    def serialize_tracked_obstacles_json(msg: Any) -> str:
        return json.dumps(msg.to_dict() if hasattr(msg, "to_dict") else msg)

    @staticmethod
    def deserialize_tracked_obstacles_json(payload: str) -> Any:
        from ugv_navigation.messages.dynamic_messages import TrackedObstacleArrayMsg
        return TrackedObstacleArrayMsg.from_dict(json.loads(payload))

    @staticmethod
    def serialize_predicted_trajectories_json(msg: Any) -> str:
        return json.dumps(msg.to_dict() if hasattr(msg, "to_dict") else msg)

    @staticmethod
    def deserialize_predicted_trajectories_json(payload: str) -> Any:
        from ugv_navigation.messages.dynamic_messages import PredictedTrajectoryArrayMsg
        return PredictedTrajectoryArrayMsg.from_dict(json.loads(payload))

    @staticmethod
    def serialize_behavioral_decision_json(msg: Any) -> str:
        return json.dumps(msg.to_dict() if hasattr(msg, "to_dict") else msg)

    @staticmethod
    def deserialize_behavioral_decision_json(payload: str) -> Any:
        from ugv_navigation.messages.dynamic_messages import BehavioralDecisionMsg
        return BehavioralDecisionMsg.from_dict(json.loads(payload))

    @staticmethod
    def serialize_replan_event_json(msg: Any) -> str:
        return json.dumps(msg.to_dict() if hasattr(msg, "to_dict") else msg)

    @staticmethod
    def deserialize_replan_event_json(payload: str) -> Any:
        from ugv_navigation.messages.dynamic_messages import DynamicReplanEventMsg
        return DynamicReplanEventMsg.from_dict(json.loads(payload))


def message_to_json(msg: Any) -> str:
    """Serialize any ROS2 message dataclass to a JSON string."""
    if hasattr(msg, "to_dict"):
        return json.dumps(msg.to_dict())
    return json.dumps(msg)


def message_from_json(json_str: str, msg_cls: Any) -> Any:
    """Deserialize a JSON string into the specified ROS2 message dataclass."""
    data = json.loads(json_str)
    if hasattr(msg_cls, "from_dict"):
        return msg_cls.from_dict(data)
    return msg_cls(**data)


def to_json_dict(msg: Any) -> Dict[str, Any]:
    """Convert message dataclass to JSON-serializable dictionary."""
    if hasattr(msg, "to_dict"):
        return msg.to_dict()
    return dict(msg)


def from_json_dict(d: Dict[str, Any], msg_cls: Any) -> Any:
    """Reconstruct message dataclass from dictionary."""
    if hasattr(msg_cls, "from_dict"):
        return msg_cls.from_dict(d)
    return msg_cls(**d)
