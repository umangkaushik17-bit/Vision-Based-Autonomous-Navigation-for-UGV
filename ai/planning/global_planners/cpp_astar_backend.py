"""Native C++17 Global A* Planner Backend FFI.

Provides high-performance C-types bindings to astar_cpp.dll (AVX2 SIMD)
and astar_cpp_scalar.dll (portable CPU fallback) for sub-millisecond global route search.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_float,
    c_int32,
    c_uint8,
)
import math
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ai.planning.base import GoalPose, GoalState, PlanningConfig, PlannerStatus, RobotState
from ai.planning.path import Path


class C_AStarCostConfig(Structure):
    _fields_ = [
        ("w_distance", c_double),
        ("w_traversability", c_double),
        ("w_cost", c_double),
        ("w_hazard", c_double),
        ("w_slope", c_double),
        ("w_roughness", c_double),
        ("w_height", c_double),
        ("w_unknown", c_double),
        ("w_curvature", c_double),
        ("critical_hazard_threshold", c_double),
        ("lethal_cost_threshold", c_double),
        ("max_safe_slope_deg", c_double),
        ("max_safe_step_m", c_double),
        ("unknown_cost_penalty", c_double),
        ("unknown_is_lethal", c_int32),
        ("heuristic_type", c_int32),
        ("heuristic_weight", c_double),
        ("enable_smoothing", c_int32),
        ("max_smoothing_iterations", c_int32),
        ("smoothing_step_m", c_double),
        ("enable_jps", c_int32),
    ]


class C_AStarGridMeta(Structure):
    _fields_ = [
        ("resolution_m", c_double),
        ("x_min", c_double),
        ("x_max", c_double),
        ("y_min", c_double),
        ("y_max", c_double),
        ("n_rows", c_int32),
        ("n_cols", c_int32),
        ("is_cartesian", c_int32),
    ]


class C_AStarVehicleFootprint(Structure):
    _fields_ = [
        ("length_m", c_double),
        ("width_m", c_double),
        ("collision_margin_m", c_double),
        ("circumscribed_radius_m", c_double),
        ("inflation_radius_m", c_double),
        ("max_safe_cost", c_double),
    ]


class C_AStarResult(Structure):
    _fields_ = [
        ("status", c_int32),
        ("raw_waypoint_count", c_int32),
        ("smoothed_waypoint_count", c_int32),
        ("nodes_explored", c_int32),
        ("path_cost", c_double),
        ("path_length_m", c_double),
        ("planning_latency_ms", c_double),
        ("smoothing_latency_ms", c_double),
    ]


_DLL_HANDLE_SIMD: Optional[ctypes.CDLL] = None
_DLL_HANDLE_SCALAR: Optional[ctypes.CDLL] = None


def has_avx2_support() -> bool:
    """Detect whether host CPU supports AVX2 vector instructions."""
    system = platform.system()
    if system == "Windows":
        try:
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40 in Windows WinNT.h
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        except Exception:
            return False
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
            return "avx2" in content
        except Exception:
            return False
    return False


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    """Set argument and return types for the native C++ C-linkage entry points."""
    lib.astar_plan_c.argtypes = [
        c_double, c_double,
        c_double, c_double, c_double,
        POINTER(C_AStarCostConfig),
        POINTER(C_AStarGridMeta),
        POINTER(C_AStarVehicleFootprint),
        POINTER(c_uint8),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint8),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint8),
        POINTER(c_uint8),
        POINTER(C_AStarResult),
        POINTER(c_double),
        POINTER(c_double),
        c_int32,
        POINTER(c_double),
        POINTER(c_double),
        c_int32,
    ]
    lib.astar_plan_c.restype = c_int32

    lib.astar_evaluate_cell_cost_c.argtypes = [
        c_int32, c_int32,
        POINTER(C_AStarCostConfig),
        POINTER(C_AStarGridMeta),
        POINTER(c_uint8),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint8),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
    ]
    lib.astar_evaluate_cell_cost_c.restype = c_double

    lib.astar_is_footprint_collision_c.argtypes = [
        c_double, c_double, c_double,
        POINTER(C_AStarCostConfig),
        POINTER(C_AStarGridMeta),
        POINTER(C_AStarVehicleFootprint),
        POINTER(c_uint8),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint8),
    ]
    lib.astar_is_footprint_collision_c.restype = c_int32

    lib.astar_is_segment_collision_c.argtypes = [
        c_double, c_double, c_double, c_double,
        c_double, c_int32, c_double,
        POINTER(C_AStarCostConfig),
        POINTER(C_AStarGridMeta),
        POINTER(C_AStarVehicleFootprint),
        POINTER(c_uint8),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_uint8),
    ]
    lib.astar_is_segment_collision_c.restype = c_int32


def load_astar_cpp_lib(variant: str = "auto") -> Tuple[Optional[ctypes.CDLL], str]:
    """Load native C++ Global A* library with target variant selection.

    Args:
        variant: 'simd' (AVX2), 'scalar' (generic portable), or 'auto' (hardware detected).

    Returns:
        Tuple of (cdll_handle or None, actual_variant_loaded)
    """
    global _DLL_HANDLE_SIMD, _DLL_HANDLE_SCALAR

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "native_cpp"))
    system = platform.system()

    simd_name = "astar_cpp.dll" if system == "Windows" else "libastar_cpp.so"
    scalar_name = "astar_cpp_scalar.dll" if system == "Windows" else "libastar_cpp_scalar.so"

    simd_path = os.path.join(base_dir, simd_name)
    scalar_path = os.path.join(base_dir, scalar_name)

    chosen = variant.lower().strip()
    if chosen == "auto":
        if has_avx2_support() and os.path.exists(simd_path):
            chosen = "simd"
        elif os.path.exists(scalar_path):
            chosen = "scalar"
        else:
            return None, "NONE"

    if chosen == "simd":
        if not has_avx2_support():
            return None, "NONE"
        if _DLL_HANDLE_SIMD is not None:
            return _DLL_HANDLE_SIMD, "SIMD"
        if not os.path.exists(simd_path):
            return None, "NONE"
        try:
            lib = ctypes.CDLL(simd_path)
            _configure_prototypes(lib)
            _DLL_HANDLE_SIMD = lib
            return lib, "SIMD"
        except Exception:
            return None, "NONE"

    elif chosen == "scalar":
        if _DLL_HANDLE_SCALAR is not None:
            return _DLL_HANDLE_SCALAR, "SCALAR"
        if not os.path.exists(scalar_path):
            return None, "NONE"
        try:
            lib = ctypes.CDLL(scalar_path)
            _configure_prototypes(lib)
            _DLL_HANDLE_SCALAR = lib
            return lib, "SCALAR"
        except Exception:
            return None, "SCALAR"

    return None, "UNKNOWN"


class CPPAStarBackend:
    """High-performance Native C++17 Global A* Planner Backend."""

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        variant: str = "auto"
    ) -> None:
        self.config = config or PlanningConfig()
        self.variant = variant
        self.lib, self.active_variant = load_astar_cpp_lib(variant=variant)
        if self.lib is None:
            raise RuntimeError(
                f"Native C++ A* library ({variant}) could not be loaded. "
                f"Please compile it using: python scripts/build_astar_cpp.py"
            )

        # Preallocate output buffers for 20,000 maximum path waypoints
        self._max_waypoints = 20000
        self._raw_x = (c_double * self._max_waypoints)()
        self._raw_y = (c_double * self._max_waypoints)()
        self._smooth_x = (c_double * self._max_waypoints)()
        self._smooth_y = (c_double * self._max_waypoints)()

        # Build constant vehicle configuration
        v = self.config.vehicle
        self.c_vehicle = C_AStarVehicleFootprint(
            length_m=float(v.length_m),
            width_m=float(v.width_m),
            collision_margin_m=float(getattr(v, "collision_margin_m", 0.05)),
            circumscribed_radius_m=float(getattr(v, "circumscribed_radius_m", 0.40)),
            inflation_radius_m=float(getattr(v, "inflation_radius_m", 0.45)),
            max_safe_cost=float(self.config.tolerances.get("max_safe_cost", 0.65)),
        )

    def _build_cost_config(self) -> C_AStarCostConfig:
        cfg = self.config
        w = cfg.cost_weights
        astar_cfg = cfg.astar
        unknown_policy = getattr(cfg, "unknown_policy", {})
        tolerances = getattr(cfg, "tolerances", {})

        h_type = 1 if str(astar_cfg.get("heuristic_type", "octile")).lower() == "euclidean" else 0

        return C_AStarCostConfig(
            w_distance=float(w.get("w_distance", 1.00)),
            w_traversability=float(w.get("w_traversability", 0.80)),
            w_cost=float(w.get("w_cost", 1.50)),
            w_hazard=float(w.get("w_hazard", 2.50)),
            w_slope=float(w.get("w_slope", 1.20)),
            w_roughness=float(w.get("w_roughness", 0.60)),
            w_height=float(w.get("w_height", 1.00)),
            w_unknown=float(w.get("w_unknown", 1.00)),
            w_curvature=float(w.get("w_curvature", 0.40)),
            critical_hazard_threshold=float(tolerances.get("critical_hazard_threshold", 0.70)),
            lethal_cost_threshold=float(tolerances.get("lethal_cost_threshold", 0.85)),
            max_safe_slope_deg=20.0,
            max_safe_step_m=0.12,
            unknown_cost_penalty=float(unknown_policy.get("unknown_cost_penalty", 0.80)),
            unknown_is_lethal=1 if bool(unknown_policy.get("unknown_is_lethal", False)) else 0,
            heuristic_type=h_type,
            heuristic_weight=float(astar_cfg.get("heuristic_weight", 1.00)),
            enable_smoothing=1 if bool(astar_cfg.get("enable_smoothing", True)) else 0,
            max_smoothing_iterations=int(astar_cfg.get("max_smoothing_iterations", 100)),
            smoothing_step_m=0.08,
            enable_jps=1 if bool(astar_cfg.get("enable_jps", False)) else 0,
        )

    def _build_grid_meta(self, costmap: Any) -> C_AStarGridMeta:
        res = float(getattr(costmap, "resolution_m", getattr(costmap, "resolution", 0.10)))

        x_min, x_max = 0.0, 0.0
        y_min, y_max = 0.0, 0.0

        if hasattr(costmap, "x_range_m"):
            x_min, x_max = float(costmap.x_range_m[0]), float(costmap.x_range_m[1])
        elif hasattr(costmap, "x_bounds_m"):
            x_min, x_max = float(costmap.x_bounds_m[0]), float(costmap.x_bounds_m[1])
        elif hasattr(costmap, "metadata"):
            meta = costmap.metadata
            if hasattr(meta, "x_range_m"):
                x_min, x_max = float(meta.x_range_m[0]), float(meta.x_range_m[1])
            elif hasattr(meta, "x_bounds_m"):
                x_min, x_max = float(meta.x_bounds_m[0]), float(meta.x_bounds_m[1])

        if hasattr(costmap, "y_range_m"):
            y_min, y_max = float(costmap.y_range_m[0]), float(costmap.y_range_m[1])
        elif hasattr(costmap, "y_bounds_m"):
            y_min, y_max = float(costmap.y_bounds_m[0]), float(costmap.y_bounds_m[1])
        elif hasattr(costmap, "metadata"):
            meta = costmap.metadata
            if hasattr(meta, "y_range_m"):
                y_min, y_max = float(meta.y_range_m[0]), float(meta.y_range_m[1])
            elif hasattr(meta, "y_bounds_m"):
                y_min, y_max = float(meta.y_bounds_m[0]), float(meta.y_bounds_m[1])

        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        n_rows, n_cols = int(shape[0]), int(shape[1])

        # If still zero, check explicit attributes or origin
        if x_min == 0.0 and x_max == 0.0:
            if hasattr(costmap, "origin"):
                origin = costmap.origin
                x_min = float(origin[0])
                y_min = float(origin[1])
                x_max = x_min + n_rows * res
                y_max = y_min + n_cols * res
            elif hasattr(costmap, "x_min") and hasattr(costmap, "x_max"):
                x_min = float(costmap.x_min)
                x_max = float(costmap.x_max)
                y_min = float(costmap.y_min)
                y_max = float(costmap.y_max)

        return C_AStarGridMeta(
            resolution_m=res,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_rows=n_rows,
            n_cols=n_cols,
            is_cartesian=0,
        )

    def plan(
        self,
        costmap: Any,
        start: Union[RobotState, GoalPose, Tuple[float, float]],
        goal: Union[GoalPose, GoalState, Tuple[float, float]],
    ) -> Tuple[Optional[Path], PlannerStatus, Dict[str, Any]]:
        """Execute compiled Native C++ A* search on costmap."""
        t_py_start = time.perf_counter()

        if costmap is None:
            return None, PlannerStatus.MAP_UNAVAILABLE, {"error": "Map is None"}

        start_x = float(start.x if hasattr(start, "x") else start[0])
        start_y = float(start.y if hasattr(start, "y") else start[1])
        goal_x = float(goal.x if hasattr(goal, "x") else goal[0])
        goal_y = float(goal.y if hasattr(goal, "y") else goal[1])
        goal_yaw = float(goal.yaw) if hasattr(goal, "yaw") and goal.yaw is not None else 0.0

        grid_meta = self._build_grid_meta(costmap)
        cost_cfg = self._build_cost_config()

        # Enforce hard 2.5D terrain safety cutoffs on slope >= 28 deg and step >= 0.12 m
        slope_layer = getattr(costmap, "slope_deg", getattr(costmap, "slope", None))
        hvar_layer = getattr(costmap, "height_var", None)
        lethal_terrain = None
        if slope_layer is not None:
            s_mask = slope_layer >= 28.0
            if np.any(s_mask):
                lethal_terrain = s_mask
        if hvar_layer is not None:
            h_mask = hvar_layer >= cost_cfg.max_safe_step_m
            if np.any(h_mask):
                lethal_terrain = h_mask if lethal_terrain is None else (lethal_terrain | h_mask)
        if lethal_terrain is not None and hasattr(costmap, "obstacle") and not np.all(costmap.obstacle[lethal_terrain]):
            costmap.obstacle[lethal_terrain] = True
            if hasattr(costmap, "cost"):
                costmap.cost[lethal_terrain] = 1.0
            try:
                costmap._cached_infl_obstacle = None
            except Exception:
                pass

        # Extract layers ensuring contiguous arrays
        # Obstacle
        if hasattr(costmap, "obstacle"):
            obs_arr = np.ascontiguousarray(costmap.obstacle.astype(np.uint8))
            obs_ptr = obs_arr.ctypes.data_as(POINTER(c_uint8))
        else:
            obs_arr = None
            obs_ptr = None

        # Cost
        if hasattr(costmap, "cost"):
            cost_arr = np.ascontiguousarray(costmap.cost.astype(np.float32))
            cost_ptr = cost_arr.ctypes.data_as(POINTER(c_float))
        else:
            cost_arr = None
            cost_ptr = None

        # Hazard
        if hasattr(costmap, "hazard"):
            haz_arr = np.ascontiguousarray(costmap.hazard.astype(np.float32))
            haz_ptr = haz_arr.ctypes.data_as(POINTER(c_float))
        else:
            haz_arr = None
            haz_ptr = None

        # Unknown
        if hasattr(costmap, "unknown"):
            unk_arr = np.ascontiguousarray(costmap.unknown.astype(np.uint8))
            unk_ptr = unk_arr.ctypes.data_as(POINTER(c_uint8))
        else:
            unk_arr = None
            unk_ptr = None

        # Traversability
        if hasattr(costmap, "traversability"):
            trav_arr = np.ascontiguousarray(costmap.traversability.astype(np.float32))
            trav_ptr = trav_arr.ctypes.data_as(POINTER(c_float))
        else:
            trav_arr = None
            trav_ptr = None

        # Slope
        if hasattr(costmap, "slope_deg"):
            slope_arr = np.ascontiguousarray(costmap.slope_deg.astype(np.float32))
            slope_ptr = slope_arr.ctypes.data_as(POINTER(c_float))
        elif hasattr(costmap, "slope"):
            slope_arr = np.ascontiguousarray(costmap.slope.astype(np.float32))
            slope_ptr = slope_arr.ctypes.data_as(POINTER(c_float))
        else:
            slope_arr = None
            slope_ptr = None

        # Roughness
        if hasattr(costmap, "roughness"):
            rough_arr = np.ascontiguousarray(costmap.roughness.astype(np.float32))
            rough_ptr = rough_arr.ctypes.data_as(POINTER(c_float))
        else:
            rough_arr = None
            rough_ptr = None

        # Height var
        if hasattr(costmap, "height_var"):
            hvar_arr = np.ascontiguousarray(costmap.height_var.astype(np.float32))
            hvar_ptr = hvar_arr.ctypes.data_as(POINTER(c_float))
        else:
            hvar_arr = None
            hvar_ptr = None

        # Chassis clearance inflation
        res = grid_meta.resolution_m
        infl_radius = self.c_vehicle.circumscribed_radius_m + self.c_vehicle.collision_margin_m
        infl_cells = max(4, int(np.ceil(infl_radius / res)))

        infl_obs_arr = None
        infl_obs_ptr = None
        if hasattr(costmap, "obstacle") and costmap.obstacle.any():
            cached_infl = getattr(costmap, "_cached_infl_obstacle", None)
            cached_cells = getattr(costmap, "_cached_infl_cells", None)
            if cached_infl is not None and cached_cells == infl_cells:
                infl_obs_arr = np.ascontiguousarray(cached_infl.astype(np.uint8))
            else:
                from scipy.ndimage import binary_dilation
                dil = binary_dilation(costmap.obstacle, iterations=infl_cells)
                infl_obs_arr = np.ascontiguousarray(dil.astype(np.uint8))
                try:
                    costmap._cached_infl_obstacle = dil
                    costmap._cached_infl_cells = infl_cells
                except Exception:
                    pass
            infl_obs_ptr = infl_obs_arr.ctypes.data_as(POINTER(c_uint8))

        infl_haz_arr = None
        infl_haz_ptr = None
        crit_thresh = cost_cfg.critical_hazard_threshold
        if hasattr(costmap, "hazard") and (costmap.hazard >= crit_thresh).any():
            cached_h = getattr(costmap, "_cached_infl_hazard", None)
            cached_h_cells = getattr(costmap, "_cached_infl_hazard_cells", None)
            if cached_h is not None and cached_h_cells == infl_cells:
                infl_haz_arr = np.ascontiguousarray(cached_h.astype(np.uint8))
            else:
                from scipy.ndimage import binary_dilation
                crit_mask = costmap.hazard >= crit_thresh
                dil_h = binary_dilation(crit_mask, iterations=infl_cells)
                infl_haz_arr = np.ascontiguousarray(dil_h.astype(np.uint8))
                try:
                    costmap._cached_infl_hazard = dil_h
                    costmap._cached_infl_hazard_cells = infl_cells
                except Exception:
                    pass
            infl_haz_ptr = infl_haz_arr.ctypes.data_as(POINTER(c_uint8))

        c_result = C_AStarResult()

        ret = self.lib.astar_plan_c(
            c_double(start_x), c_double(start_y),
            c_double(goal_x), c_double(goal_y), c_double(goal_yaw),
            ctypes.byref(cost_cfg),
            ctypes.byref(grid_meta),
            ctypes.byref(self.c_vehicle),
            obs_ptr,
            cost_ptr,
            haz_ptr,
            unk_ptr,
            trav_ptr,
            slope_ptr,
            rough_ptr,
            hvar_ptr,
            infl_obs_ptr,
            infl_haz_ptr,
            ctypes.byref(c_result),
            self._raw_x,
            self._raw_y,
            c_int32(self._max_waypoints),
            self._smooth_x,
            self._smooth_y,
            c_int32(self._max_waypoints),
        )

        t_py_total_ms = (time.perf_counter() - t_py_start) * 1000.0

        status_map = {
            0: PlannerStatus.SUCCESS,
            1: PlannerStatus.NO_PATH,
            2: PlannerStatus.INVALID_START,
            3: PlannerStatus.INVALID_GOAL,
            4: PlannerStatus.START_BLOCKED,
            5: PlannerStatus.GOAL_BLOCKED,
        }
        status = status_map.get(c_result.status, PlannerStatus.NO_PATH)

        if status != PlannerStatus.SUCCESS:
            error_messages = {
                PlannerStatus.NO_PATH: "No feasible path found",
                PlannerStatus.INVALID_START: "Start out of map bounds",
                PlannerStatus.INVALID_GOAL: "Goal out of map bounds",
                PlannerStatus.START_BLOCKED: "Start point in collision/hazard",
                PlannerStatus.GOAL_BLOCKED: "Goal point in collision/hazard",
            }
            return None, status, {
                "error": error_messages.get(status, "Planning failed"),
                "nodes_explored": c_result.nodes_explored,
                "planning_latency_ms": c_result.planning_latency_ms,
                "cpp_total_ms": c_result.planning_latency_ms + c_result.smoothing_latency_ms,
                "py_total_ms": t_py_total_ms,
                "backend": f"native_cpp_{self.active_variant.lower()}",
            }

        n_smooth = c_result.smoothed_waypoint_count
        n_raw = c_result.raw_waypoint_count

        if cost_cfg.enable_smoothing and n_smooth > 0:
            coords = np.column_stack([
                np.ctypeslib.as_array(self._smooth_x)[:n_smooth],
                np.ctypeslib.as_array(self._smooth_y)[:n_smooth],
            ]).astype(np.float32)
            final_path = Path.from_xy_array(coords).interpolate(step_m=0.10)
            final_path.is_smoothed = True
        else:
            coords = np.column_stack([
                np.ctypeslib.as_array(self._raw_x)[:n_raw],
                np.ctypeslib.as_array(self._raw_y)[:n_raw],
            ]).astype(np.float32)
            final_path = Path.from_xy_array(coords)

        diagnostics = {
            "nodes_explored": c_result.nodes_explored,
            "planning_latency_ms": c_result.planning_latency_ms,
            "smoothing_latency_ms": c_result.smoothing_latency_ms,
            "cpp_total_ms": c_result.planning_latency_ms + c_result.smoothing_latency_ms,
            "py_total_ms": t_py_total_ms,
            "raw_waypoints": n_raw,
            "smoothed_waypoints": len(final_path),
            "is_smoothed": final_path.is_smoothed,
            "path_length_m": float(final_path.total_length_m),
            "path_cost": float(c_result.path_cost),
            "backend": f"native_cpp_{self.active_variant.lower()}",
        }
        return final_path, PlannerStatus.SUCCESS, diagnostics
