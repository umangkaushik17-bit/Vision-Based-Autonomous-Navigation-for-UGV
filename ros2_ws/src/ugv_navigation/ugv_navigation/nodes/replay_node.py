"""Offline Replay Node for deterministic sensor stream playback."""

from __future__ import annotations

import sys
from typing import Iterator, Optional
import numpy as np

from ai.localization.datasets import SyntheticStereoSequenceGenerator, SequenceFrame, RUGDSequenceLoader

try:
    import rclpy
    from rclpy.node import Node
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    Node = object


class ReplayNode(Node if RCLPY_AVAILABLE else object):
    """Offline sensor replay generator supporting synthetic sequences and real RUGD imagery."""

    def __init__(
        self,
        mode: str = "synthetic",
        sequence_path: Optional[str] = None,
        num_frames: int = 20,
        fps: float = 10.0,
    ):
        self.mode = mode
        self.sequence_path = sequence_path
        self.num_frames = num_frames
        self.fps = fps

        if self.mode == "synthetic":
            self.generator = iter(SyntheticStereoSequenceGenerator(num_frames=self.num_frames, baseline_m=0.05))
        elif self.mode == "rugd":
            if not self.sequence_path:
                raise ValueError("sequence_path required for RUGD replay mode")
            self.generator = iter(RUGDSequenceLoader(self.sequence_path, max_frames=self.num_frames))
        else:
            raise ValueError(f"Unknown replay mode: {self.mode}")

        if RCLPY_AVAILABLE:
            super().__init__("replay_node")
            self.get_logger().info(f"Replay Node initialized in '{self.mode}' mode.")

    def get_next_frame(self) -> Optional[SequenceFrame]:
        """Fetches the next frame in the replay stream."""
        try:
            return next(self.generator)
        except StopIteration:
            return None


def main(args=None):
    if not RCLPY_AVAILABLE:
        print("[WARN] rclpy is not installed. Replay Node requires ROS2 runtime.")
        sys.exit(0)

    rclpy.init(args=args)
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
