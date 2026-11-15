"""Root pytest configuration adding ROS2 package paths to sys.path."""

import sys
import types
from pathlib import Path
from typing import Any, List, Optional

root_dir = Path(__file__).resolve().parent
ugv_nav_path = root_dir / "ros2_ws" / "src" / "ugv_navigation"
if ugv_nav_path.exists() and str(ugv_nav_path) not in sys.path:
    sys.path.insert(0, str(ugv_nav_path))


def _ensure_ros2_launch_stubs() -> None:
    """Provide lightweight ROS 2 launch / launch_ros stubs on Windows when native ROS 2 is absent."""
    try:
        from launch import LaunchDescription as _LD  # noqa: F401
        return
    except Exception:
        pass

    launch_mod = types.ModuleType("launch")
    actions_mod = types.ModuleType("launch.actions")
    conditions_mod = types.ModuleType("launch.conditions")
    substitutions_mod = types.ModuleType("launch.substitutions")
    launch_ros_mod = types.ModuleType("launch_ros")
    launch_ros_actions_mod = types.ModuleType("launch_ros.actions")

    class LaunchDescription:
        def __init__(self, entities: Optional[List[Any]] = None) -> None:
            self.entities = list(entities or [])

    class DeclareLaunchArgument:
        def __init__(self, name: str, default_value: Any = None, description: str = "", **kwargs: Any) -> None:
            self.name = name
            self.default_value = default_value
            self.description = description

    class ExecuteProcess:
        def __init__(self, cmd: List[Any], name: Optional[str] = None, output: str = "screen",
                     additional_env: Any = None, condition: Any = None, on_exit: Any = None, **kwargs: Any) -> None:
            self.cmd = cmd
            self.name = name
            self.output = output
            self.additional_env = additional_env
            self.condition = condition
            self.on_exit = on_exit

    class LogInfo:
        def __init__(self, msg: Any = "", **kwargs: Any) -> None:
            self.msg = msg

    class SetEnvironmentVariable:
        def __init__(self, name: str, value: Any = "", **kwargs: Any) -> None:
            self.name = name
            self.value = value

    class IfCondition:
        def __init__(self, predicate: Any) -> None:
            self.predicate = predicate

    class UnlessCondition:
        def __init__(self, predicate: Any) -> None:
            self.predicate = predicate

    class LaunchConfiguration:
        def __init__(self, variable_name: str, default: Any = None) -> None:
            self.variable_name = variable_name
            self.default = default
            self.text = str(default) if default is not None else f"$({variable_name})"

        def __str__(self) -> str:
            return self.text

    class Node:
        def __init__(self, package: str = "", executable: str = "", name: Optional[str] = None,
                     output: str = "screen", parameters: Optional[List[Any]] = None, **kwargs: Any) -> None:
            self.package = package
            self.executable = executable
            self.name = name
            self._Node__node_name = name
            self.output = output
            self.parameters = parameters or []

    launch_mod.LaunchDescription = LaunchDescription  # type: ignore[attr-defined]
    launch_mod.actions = actions_mod  # type: ignore[attr-defined]
    launch_mod.conditions = conditions_mod  # type: ignore[attr-defined]
    launch_mod.substitutions = substitutions_mod  # type: ignore[attr-defined]

    actions_mod.DeclareLaunchArgument = DeclareLaunchArgument  # type: ignore[attr-defined]
    actions_mod.ExecuteProcess = ExecuteProcess  # type: ignore[attr-defined]
    actions_mod.LogInfo = LogInfo  # type: ignore[attr-defined]
    actions_mod.SetEnvironmentVariable = SetEnvironmentVariable  # type: ignore[attr-defined]

    conditions_mod.IfCondition = IfCondition  # type: ignore[attr-defined]
    conditions_mod.UnlessCondition = UnlessCondition  # type: ignore[attr-defined]

    substitutions_mod.LaunchConfiguration = LaunchConfiguration  # type: ignore[attr-defined]

    launch_ros_mod.actions = launch_ros_actions_mod  # type: ignore[attr-defined]
    launch_ros_actions_mod.Node = Node  # type: ignore[attr-defined]

    sys.modules["launch"] = launch_mod
    sys.modules["launch.actions"] = actions_mod
    sys.modules["launch.conditions"] = conditions_mod
    sys.modules["launch.substitutions"] = substitutions_mod
    sys.modules["launch_ros"] = launch_ros_mod
    sys.modules["launch_ros.actions"] = launch_ros_actions_mod

    try:
        import geometry_msgs.msg  # noqa: F401
    except Exception:
        geom_mod = types.ModuleType("geometry_msgs")
        geom_msg_mod = types.ModuleType("geometry_msgs.msg")

        class _Vec3:
            def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
                self.x, self.y, self.z = float(x), float(y), float(z)

        class Twist:
            def __init__(self) -> None:
                self.linear = _Vec3()
                self.angular = _Vec3()

        geom_msg_mod.Twist = Twist  # type: ignore[attr-defined]
        geom_msg_mod.Vector3 = _Vec3  # type: ignore[attr-defined]
        geom_msg_mod.Point = _Vec3  # type: ignore[attr-defined]
        geom_mod.msg = geom_msg_mod  # type: ignore[attr-defined]
        sys.modules["geometry_msgs"] = geom_mod
        sys.modules["geometry_msgs.msg"] = geom_msg_mod



_ensure_ros2_launch_stubs()

