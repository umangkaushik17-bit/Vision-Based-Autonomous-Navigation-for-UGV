"""Test configuration adding ROS2 package to sys.path."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
ugv_nav_path = root_dir / "ros2_ws" / "src" / "ugv_navigation"
if ugv_nav_path.exists() and str(ugv_nav_path) not in sys.path:
    sys.path.insert(0, str(ugv_nav_path))

import ugv_navigation
