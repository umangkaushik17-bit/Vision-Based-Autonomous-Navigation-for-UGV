"""Native C++ CPU DWA Backend.

Executes candidate trajectory rollout, static footprint collision checking,
exact 8-axis polygon SAT dynamic collision checking, and multi-objective scoring
in compiled C++ using AVX2 SIMD and OpenMP parallelization.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_double,
    c_float,
    c_int32,
    c_uint8,
)
import math
import os
import platform
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ai.planning.backends.backend_interface import BackendHealth, DWABackend, DWABackendType
from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import BehavioralStateMachine
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
)
from ai.planning.dynamic.dwa_filter import DynamicPlanningConfig
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.path import Path
from ai.planning.trajectory import Trajectory, TrajectoryPoint


class C_RobotState(Structure):
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
        ("yaw", c_double),
        ("v", c_double),
        ("omega", c_double),
        ("timestamp", c_double),
    ]


class C_VehicleConfig(Structure):
    _fields_ = [
        ("length_m", c_double),
        ("width_m", c_double),
        ("collision_margin_m", c_double),
        ("circumscribed_radius_m", c_double),
        ("max_linear_velocity_mps", c_double),
        ("min_linear_velocity_mps", c_double),
        ("max_angular_velocity_radps", c_double),
        ("max_linear_acceleration_mps2", c_double),
        ("max_angular_acceleration_radps2", c_double),
        ("degraded_max_linear_velocity_mps", c_double),
    ]


class C_DWAConfig(Structure):
    _fields_ = [
        ("prediction_horizon_s", c_double),
        ("simulation_dt_s", c_double),
        ("v_samples", c_int32),
        ("omega_samples", c_int32),
    ]


class C_ScoringWeights(Structure):
    _fields_ = [
        ("w_path", c_double),
        ("w_heading", c_double),
        ("w_dist", c_double),
        ("w_clear", c_double),
        ("w_vel", c_double),
        ("w_terrain", c_double),
    ]


class C_DynamicConfig(Structure):
    _fields_ = [
        ("enabled", c_bool),
        ("enable_broad_phase", c_bool),
        ("reject_high_risk", c_bool),
        ("unknown_horizon_policy", c_int32),
        ("unknown_horizon_penalty", c_double),
        ("low_risk_penalty", c_double),
        ("medium_risk_penalty", c_double),
        ("high_risk_penalty", c_double),
        ("w_dynamic_risk", c_double),
        ("w_dynamic_clearance", c_double),
        ("broad_phase_margin_m", c_double),
        ("max_clearance_search_m", c_double),
    ]


class C_CostmapMeta(Structure):
    _fields_ = [
        ("resolution_m", c_double),
        ("x_min", c_double),
        ("x_max", c_double),
        ("y_min", c_double),
        ("y_max", c_double),
        ("n_rows", c_int32),
        ("n_cols", c_int32),
        ("is_cartesian", c_bool),
    ]


class C_ObstacleWaypoint(Structure):
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
        ("time_offset", c_double),
        ("uncertainty_radius", c_double),
    ]


class C_ObstacleTrack(Structure):
    _fields_ = [
        ("track_id", c_int32),
        ("horizon_s", c_double),
        ("bbox_length", c_double),
        ("bbox_width", c_double),
        ("bbox_yaw", c_double),
        ("n_waypoints", c_int32),
        ("waypoints", POINTER(C_ObstacleWaypoint)),
    ]


class C_CandidateDiagnostic(Structure):
    _fields_ = [
        ("v", c_double),
        ("omega", c_double),
        ("score", c_double),
        ("static_clearance", c_double),
        ("terrain_cost", c_double),
        ("dyn_clearance", c_double),
        ("dyn_cost", c_double),
        ("dyn_risk_level", c_int32),
        ("is_valid", c_bool),
        ("static_valid", c_bool),
        ("dyn_collision", c_bool),
        ("unknown_horizon", c_bool),
    ]


class C_DWAResult(Structure):
    _fields_ = [
        ("best_idx", c_int32),
        ("best_v", c_double),
        ("best_w", c_double),
        ("best_score", c_double),
        ("best_static_clearance", c_double),
        ("best_terrain_cost", c_double),
        ("best_dyn_clearance", c_double),
        ("best_dyn_cost", c_double),
        ("best_risk_level", c_int32),
        ("total_candidates", c_int32),
        ("statically_valid_count", c_int32),
        ("valid_count", c_int32),
        ("broad_phase_cleared_count", c_int32),
        ("expensive_collision_checks", c_int32),
        ("dynamically_rejected_count", c_int32),
        ("dynamically_penalized_count", c_int32),
        ("unknown_horizon_count", c_int32),
        ("dwa_latency_ms", c_double),
    ]


_DLL_HANDLE = None


def _load_dwa_cpp_lib():
    global _DLL_HANDLE
    if _DLL_HANDLE is not None:
        return _DLL_HANDLE

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "native_cpp"))
    system = platform.system()
    if system == "Windows":
        dll_name = "dwa_cpp.dll"
    else:
        dll_name = "libdwa_cpp.so"

    dll_path = os.path.join(base_dir, dll_name)
    if not os.path.exists(dll_path):
        return None

    try:
        lib = ctypes.CDLL(dll_path)
        # Configure function prototypes
        lib.dwa_plan_c.argtypes = [
            POINTER(C_RobotState),
            POINTER(C_VehicleConfig),
            POINTER(C_DWAConfig),
            POINTER(C_ScoringWeights),
            POINTER(c_double),
            POINTER(c_double),
            c_int32,
            c_double,
            c_double,
            c_double,
            POINTER(C_CostmapMeta),
            POINTER(c_uint8),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_uint8),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(C_DynamicConfig),
            POINTER(C_ObstacleTrack),
            c_int32,
            c_int32,
            POINTER(C_DWAResult),
            POINTER(C_CandidateDiagnostic),
            POINTER(c_double),
        ]
        lib.dwa_plan_c.restype = c_int32

        lib.dwa_sat_intersect_c.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.dwa_sat_intersect_c.restype = c_int32

        lib.dwa_polygon_distance_c.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib.dwa_polygon_distance_c.restype = c_double

        lib.dwa_footprint_collision_c.argtypes = [
            c_double,
            c_double,
            c_double,
            POINTER(C_VehicleConfig),
            POINTER(C_CostmapMeta),
            POINTER(c_uint8),
            POINTER(c_float),
            POINTER(c_float),
        ]
        lib.dwa_footprint_collision_c.restype = c_int32

        _DLL_HANDLE = lib
        return lib
    except Exception:
        return None


class CPPDWABackend(DWABackend):
    """Native C++ CPU accelerated batched DWA local planner backend."""

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        super().__init__(DWABackendType.CPP_CPU)
        self.config = config or PlanningConfig()

        self.lib = _load_dwa_cpp_lib()
        if self.lib is None:
            raise RuntimeError(
                "CPP_CPU backend explicitly requested, but compiled native C++ library "
                "(dwa_cpp.dll or libdwa_cpp.so) was not found or failed to load. "
                "Compile it using: python scripts/build_dwa_cpp.py"
            )

        # Vehicle Kinematics
        v_cfg = self.config.vehicle
        self.c_vehicle = C_VehicleConfig(
            length_m=float(v_cfg.length_m),
            width_m=float(v_cfg.width_m),
            collision_margin_m=float(getattr(v_cfg, "collision_margin_m", 0.05)),
            circumscribed_radius_m=float(getattr(v_cfg, "circumscribed_radius_m", 0.40)),
            max_linear_velocity_mps=float(v_cfg.max_linear_velocity_mps),
            min_linear_velocity_mps=float(v_cfg.min_linear_velocity_mps),
            max_angular_velocity_radps=float(v_cfg.max_angular_velocity_radps),
            max_linear_acceleration_mps2=float(v_cfg.max_linear_acceleration_mps2),
            max_angular_acceleration_radps2=float(v_cfg.max_angular_acceleration_radps2),
            degraded_max_linear_velocity_mps=float(getattr(v_cfg, "degraded_max_linear_velocity_mps", 0.35)),
        )

        # DWA Config
        dwa_cfg = self.config.dwa
        self.c_dwa = C_DWAConfig(
            prediction_horizon_s=float(dwa_cfg.get("prediction_horizon_s", 1.50)),
            simulation_dt_s=float(dwa_cfg.get("simulation_dt_s", 0.10)),
            v_samples=int(dwa_cfg.get("v_samples", 15)),
            omega_samples=int(dwa_cfg.get("omega_samples", 25)),
        )
        self.total_k = self.c_dwa.v_samples * self.c_dwa.omega_samples
        self.steps = max(1, int(math.ceil(self.c_dwa.prediction_horizon_s / self.c_dwa.simulation_dt_s)))

        # Scoring Weights
        self.c_weights = C_ScoringWeights(
            w_path=float(dwa_cfg.get("path_cost_weight", 0.30)),
            w_heading=float(dwa_cfg.get("heading_cost_weight", 0.15)),
            w_dist=float(dwa_cfg.get("goal_dist_cost_weight", 0.25)),
            w_clear=float(dwa_cfg.get("clearance_cost_weight", 0.20)),
            w_vel=float(dwa_cfg.get("velocity_cost_weight", 0.15)),
            w_terrain=float(dwa_cfg.get("terrain_cost_weight", 0.20)),
        )

        # Behavioral state machine
        behavior_cfg = getattr(self.config, "behavior", None)
        self.behavioral_state_machine = BehavioralStateMachine(behavior_cfg)
        self.enable_behavioral_arbitration = bool(getattr(self.config, "enable_behavioral_arbitration", False))

    @property
    def name(self) -> str:
        return "Native C++ CPU"

    @property
    def device_name(self) -> str:
        return "Host CPU (Native C++ / AVX2 / OpenMP)"

    @classmethod
    def is_available(cls) -> bool:
        return _load_dwa_cpp_lib() is not None

    def check_health(self) -> BackendHealth:
        avail = self.is_available()
        return BackendHealth(
            backend_type=self.backend_type,
            is_available=avail,
            device_name=self.device_name,
            safety_validated=avail,
            benchmark_latency_ms=10.0 if avail else None,
            error_message=None if avail else "Native C++ library not compiled (run: python scripts/build_dwa_cpp.py)",
        )

    def is_footprint_collision(self, x: float, y: float, yaw: float, costmap: Any) -> bool:
        """Evaluate static footprint collision at a single pose in native C++."""
        res = float(getattr(costmap, "resolution_m", getattr(costmap, "resolution", 0.10)))
        x_bounds = getattr(costmap, "x_bounds_m", getattr(costmap, "x_range_m", None))
        y_bounds = getattr(costmap, "y_bounds_m", getattr(costmap, "y_range_m", None))
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        n_rows, n_cols = int(shape[0]), int(shape[1])

        is_cartesian = False
        if x_bounds is not None and y_bounds is not None:
            x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
            y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        elif hasattr(costmap, "origin_x") and hasattr(costmap, "origin_y"):
            is_cartesian = True
            x_min = float(costmap.origin_x)
            y_min = float(costmap.origin_y)
            x_max = x_min + n_cols * res
            y_max = y_min + n_rows * res
        else:
            x_min, x_max = -20.0, 20.0
            y_min, y_max = -20.0, 20.0

        c_meta = C_CostmapMeta(
            resolution_m=res,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_rows=n_rows,
            n_cols=n_cols,
            is_cartesian=is_cartesian,
        )

        obs_arr = np.ascontiguousarray(costmap.obstacle, dtype=np.uint8) if hasattr(costmap, "obstacle") else None
        obs_ptr = obs_arr.ctypes.data_as(POINTER(c_uint8)) if obs_arr is not None else None

        cost_arr = np.ascontiguousarray(costmap.cost, dtype=np.float32) if hasattr(costmap, "cost") else None
        cost_ptr = cost_arr.ctypes.data_as(POINTER(c_float)) if cost_arr is not None else None

        haz_arr = np.ascontiguousarray(costmap.hazard, dtype=np.float32) if hasattr(costmap, "hazard") else None
        haz_ptr = haz_arr.ctypes.data_as(POINTER(c_float)) if haz_arr is not None else None

        ret = self.lib.dwa_footprint_collision_c(
            float(x), float(y), float(yaw),
            ctypes.byref(self.c_vehicle),
            ctypes.byref(c_meta),
            obs_ptr,
            cost_ptr,
            haz_ptr,
        )
        return ret == 1

    def _find_local_target(
        self,
        state: RobotState,
        global_path: Path,
        lookahead_dist: float = 0.60,
    ) -> Tuple[float, float]:
        if len(global_path) == 0:
            return (state.x + 1.0 * math.cos(state.yaw), state.y + 1.0 * math.sin(state.yaw))

        xy = global_path.waypoints_xy
        dists = np.hypot(xy[:, 0] - state.x, xy[:, 1] - state.y)
        closest_idx = int(np.argmin(dists))

        target_idx = closest_idx
        accum_dist = 0.0
        while target_idx < len(xy) - 1:
            segment_d = float(np.hypot(xy[target_idx+1, 0] - xy[target_idx, 0], xy[target_idx+1, 1] - xy[target_idx, 1]))
            accum_dist += segment_d
            target_idx += 1
            if accum_dist >= lookahead_dist:
                break

        return (float(xy[target_idx, 0]), float(xy[target_idx, 1]))

    def compute_trajectory(
        self,
        current_state: RobotState,
        global_path: Path,
        costmap: Any,
        is_degraded: bool = False,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[Union[PredictedTrajectoryArray, Sequence[PredictedObstacleTrajectory]]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
    ) -> Tuple[Optional[Trajectory], float, float, PlannerStatus, Dict[str, Any]]:
        t_start = time.perf_counter()

        # Local target and path
        local_target = self._find_local_target(current_state, global_path)
        d_curr = math.hypot(local_target[0] - current_state.x, local_target[1] - current_state.y)

        path_len = len(global_path)
        if path_len > 0:
            path_xy = np.ascontiguousarray(global_path.waypoints_xy, dtype=np.float64)
            path_x_ptr = path_xy[:, 0].ctypes.data_as(POINTER(c_double))
            path_y_ptr = path_xy[:, 1].ctypes.data_as(POINTER(c_double))
        else:
            path_x_ptr = None
            path_y_ptr = None

        # Robot State struct
        c_state = C_RobotState(
            x=float(current_state.x),
            y=float(current_state.y),
            yaw=float(current_state.yaw),
            v=float(current_state.v),
            omega=float(current_state.omega),
            timestamp=float(current_state.timestamp),
        )

        # Costmap metadata and buffers
        res = float(getattr(costmap, "resolution_m", getattr(costmap, "resolution", 0.10)))
        x_bounds = getattr(costmap, "x_bounds_m", getattr(costmap, "x_range_m", None))
        y_bounds = getattr(costmap, "y_bounds_m", getattr(costmap, "y_range_m", None))
        shape = getattr(costmap, "shape", (costmap.cost.shape[0], costmap.cost.shape[1]))
        n_rows, n_cols = int(shape[0]), int(shape[1])

        is_cartesian = False
        if x_bounds is not None and y_bounds is not None:
            x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
            y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        elif hasattr(costmap, "origin_x") and hasattr(costmap, "origin_y"):
            is_cartesian = True
            x_min = float(costmap.origin_x)
            y_min = float(costmap.origin_y)
            x_max = x_min + n_cols * res
            y_max = y_min + n_rows * res
        else:
            x_min, x_max = -20.0, 20.0
            y_min, y_max = -20.0, 20.0

        c_meta = C_CostmapMeta(
            resolution_m=res,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_rows=n_rows,
            n_cols=n_cols,
            is_cartesian=is_cartesian,
        )

        # Costmap layer arrays
        obs_arr = np.ascontiguousarray(costmap.obstacle, dtype=np.uint8) if hasattr(costmap, "obstacle") else None
        obs_ptr = obs_arr.ctypes.data_as(POINTER(c_uint8)) if obs_arr is not None else None

        cost_arr = np.ascontiguousarray(costmap.cost, dtype=np.float32) if hasattr(costmap, "cost") else None
        cost_ptr = cost_arr.ctypes.data_as(POINTER(c_float)) if cost_arr is not None else None

        haz_arr = np.ascontiguousarray(costmap.hazard, dtype=np.float32) if hasattr(costmap, "hazard") else None
        haz_ptr = haz_arr.ctypes.data_as(POINTER(c_float)) if haz_arr is not None else None

        unk_arr = np.ascontiguousarray(costmap.unknown, dtype=np.uint8) if hasattr(costmap, "unknown") else None
        unk_ptr = unk_arr.ctypes.data_as(POINTER(c_uint8)) if unk_arr is not None else None

        trav_arr = np.ascontiguousarray(costmap.traversability, dtype=np.float32) if hasattr(costmap, "traversability") else None
        trav_ptr = trav_arr.ctypes.data_as(POINTER(c_float)) if trav_arr is not None else None

        slope_arr = np.ascontiguousarray(costmap.slope_deg if hasattr(costmap, "slope_deg") else (costmap.slope if hasattr(costmap, "slope") else np.zeros((n_rows, n_cols))), dtype=np.float32)
        slope_ptr = slope_arr.ctypes.data_as(POINTER(c_float))

        rough_arr = np.ascontiguousarray(costmap.roughness if hasattr(costmap, "roughness") else np.zeros((n_rows, n_cols)), dtype=np.float32)
        rough_ptr = rough_arr.ctypes.data_as(POINTER(c_float))

        hvar_arr = np.ascontiguousarray(costmap.height_var if hasattr(costmap, "height_var") else np.zeros((n_rows, n_cols)), dtype=np.float32)
        hvar_ptr = hvar_arr.ctypes.data_as(POINTER(c_float))

        # Dynamic configuration
        dyn_cfg = dynamic_config or DynamicPlanningConfig()
        has_dynamic = dyn_cfg.enabled and ((dynamic_layer is not None) or (predicted_tracks is not None))

        policy_map = {"ALLOW": 0, "PENALIZE": 1, "REJECT": 2}
        c_dyn = C_DynamicConfig(
            enabled=has_dynamic,
            enable_broad_phase=dyn_cfg.enable_broad_phase,
            reject_high_risk=dyn_cfg.reject_high_risk,
            unknown_horizon_policy=policy_map.get(dyn_cfg.unknown_horizon_policy, 1),
            unknown_horizon_penalty=float(dyn_cfg.unknown_horizon_penalty),
            low_risk_penalty=float(dyn_cfg.low_risk_penalty),
            medium_risk_penalty=float(dyn_cfg.medium_risk_penalty),
            high_risk_penalty=float(dyn_cfg.high_risk_penalty),
            w_dynamic_risk=float(dyn_cfg.w_dynamic_risk),
            w_dynamic_clearance=float(dyn_cfg.w_dynamic_clearance),
            broad_phase_margin_m=float(dyn_cfg.broad_phase_margin_m),
            max_clearance_search_m=5.0,
        )

        # Dynamic Obstacle Tracks
        tracks_list: List[PredictedObstacleTrajectory] = []
        if predicted_tracks is not None:
            if isinstance(predicted_tracks, PredictedTrajectoryArray):
                tracks_list = [t for t in predicted_tracks.predictions if t.current_state.status.upper() != "EXPIRED"]
            else:
                tracks_list = [t for t in predicted_tracks if t.current_state.status.upper() != "EXPIRED"]

        num_tracks = len(tracks_list)
        c_tracks = (C_ObstacleTrack * num_tracks)() if num_tracks > 0 else None
        # Keep waypoint buffers alive in Python memory during call
        _wp_buffers = []

        if num_tracks > 0:
            for idx, trk in enumerate(tracks_list):
                wps = trk.waypoints
                n_wp = len(wps)
                wp_array = (C_ObstacleWaypoint * n_wp)()
                for w_i, wp in enumerate(wps):
                    wp_array[w_i].x = float(wp.x)
                    wp_array[w_i].y = float(wp.y)
                    wp_array[w_i].time_offset = float(wp.time_offset)
                    wp_array[w_i].uncertainty_radius = float(getattr(wp, "uncertainty_radius", getattr(wp, "position_uncertainty_radius", 0.10)))
                _wp_buffers.append(wp_array)

                bbox = trk.current_state.bbox_size
                c_tracks[idx].track_id = int(trk.track_id)
                c_tracks[idx].horizon_s = float(trk.horizon_s)
                c_tracks[idx].bbox_length = float(bbox[0])
                c_tracks[idx].bbox_width = float(bbox[1])
                c_tracks[idx].bbox_yaw = float(trk.current_state.bbox_yaw)
                c_tracks[idx].n_waypoints = n_wp
                c_tracks[idx].waypoints = ctypes.cast(wp_array, POINTER(C_ObstacleWaypoint))

        # Output buffers
        c_result = C_DWAResult()
        c_diagnostics = (C_CandidateDiagnostic * self.total_k)()
        out_traj = np.zeros((self.total_k, self.steps, 3), dtype=np.float64)
        out_traj_ptr = out_traj.ctypes.data_as(POINTER(c_double))

        # Invoke native C++ planner
        ret = self.lib.dwa_plan_c(
            ctypes.byref(c_state),
            ctypes.byref(self.c_vehicle),
            ctypes.byref(self.c_dwa),
            ctypes.byref(self.c_weights),
            path_x_ptr,
            path_y_ptr,
            path_len,
            float(local_target[0]),
            float(local_target[1]),
            d_curr,
            ctypes.byref(c_meta),
            obs_ptr,
            cost_ptr,
            haz_ptr,
            unk_ptr,
            trav_ptr,
            slope_ptr,
            rough_ptr,
            hvar_ptr,
            ctypes.byref(c_dyn),
            c_tracks,
            num_tracks,
            1 if is_degraded else 0,
            ctypes.byref(c_result),
            c_diagnostics,
            out_traj_ptr,
        )

        total_dwa_ms = (time.perf_counter() - t_start) * 1000.0

        if ret != 0 or c_result.best_idx < 0:
            # No safe trajectory case -> EMERGENCY STOP / WAIT
            zero_traj = Trajectory(
                points=[TrajectoryPoint(
                    t=current_state.timestamp + self.c_dwa.simulation_dt_s,
                    x=current_state.x, y=current_state.y, yaw=current_state.yaw,
                    v=0.0, omega=0.0, cost=1.0, clearance_m=0.0
                )],
                is_valid=False, total_cost=1.0, min_clearance_m=0.0
            )
            zero_diag = {
                "trajectories_evaluated": self.total_k,
                "statically_valid_candidates": c_result.statically_valid_count,
                "valid_trajectories": 0,
                "broad_phase_cleared_count": c_result.broad_phase_cleared_count,
                "expensive_collision_checks": c_result.expensive_collision_checks,
                "dynamically_rejected_candidates": c_result.dynamically_rejected_count,
                "dynamically_penalized_candidates": c_result.dynamically_penalized_count,
                "unknown_horizon_candidates": c_result.unknown_horizon_count,
                "dynamic_collision": c_result.dynamically_rejected_count > 0,
                "dynamic_risk": "LETHAL" if c_result.dynamically_rejected_count > 0 else ("UNKNOWN_HORIZON" if c_result.unknown_horizon_count > 0 else "FREE"),
                "unknown_horizon": c_result.unknown_horizon_count > 0,
                "blocking_obstacle_id": None,
                "earliest_collision_time": None,
                "minimum_dynamic_clearance": 0.0,
                "maximum_dynamic_cost": 1.0 if c_result.dynamically_rejected_count > 0 else 0.0,
                "dwa_latency_ms": total_dwa_ms,
                "cpp_core_latency_ms": c_result.dwa_latency_ms,
                "emergency_stop_triggered": True,
                "backend_device": self.device_name,
            }
            decision = self.behavioral_state_machine.evaluate(
                current_state=current_state,
                best_trajectory=zero_traj,
                best_v=0.0,
                best_w=0.0,
                planner_status=PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY,
                dwa_diagnostics=zero_diag,
                predicted_tracks=predicted_tracks,
                global_path=global_path,
                current_time_s=current_state.timestamp,
            )
            zero_diag["behavioral_state"] = decision.state.value
            zero_diag["behavioral_decision"] = decision.to_dict()
            zero_diag["deadlock_event"] = decision.deadlock_event
            cmd_v = decision.command_v if self.enable_behavioral_arbitration else 0.0
            cmd_w = decision.command_omega if self.enable_behavioral_arbitration else 0.0
            return zero_traj, cmd_v, cmd_w, PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY, zero_diag

        # Reconstruct optimal trajectory
        best_k = c_result.best_idx
        best_pts = out_traj[best_k]  # (S, 3)
        sim_dt = self.c_dwa.simulation_dt_s

        points: List[TrajectoryPoint] = []
        for s in range(self.steps):
            points.append(TrajectoryPoint(
                t=current_state.timestamp + (s + 1) * sim_dt,
                x=float(best_pts[s, 0]),
                y=float(best_pts[s, 1]),
                yaw=float(best_pts[s, 2]),
                v=float(c_result.best_v),
                omega=float(c_result.best_w),
                cost=float(c_result.best_terrain_cost),
                clearance_m=float(c_result.best_static_clearance),
            ))

        best_trajectory = Trajectory(
            points=points,
            is_valid=True,
            total_cost=float(c_result.best_terrain_cost),
            min_clearance_m=float(c_result.best_static_clearance),
        )

        risk_names = ["FREE", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "LETHAL"]
        risk_str = risk_names[c_result.best_risk_level] if c_result.best_risk_level < len(risk_names) else "FREE"

        diag = {
            "trajectories_evaluated": self.total_k,
            "statically_valid_candidates": c_result.statically_valid_count,
            "valid_trajectories": c_result.valid_count,
            "broad_phase_cleared_count": c_result.broad_phase_cleared_count,
            "expensive_collision_checks": c_result.expensive_collision_checks,
            "dynamically_rejected_candidates": c_result.dynamically_rejected_count,
            "dynamically_penalized_candidates": c_result.dynamically_penalized_count,
            "unknown_horizon_candidates": c_result.unknown_horizon_count,
            "dynamic_collision": False,
            "dynamic_risk": risk_str,
            "unknown_horizon": c_result.unknown_horizon_count > 0,
            "blocking_obstacle_id": None,
            "earliest_collision_time": None,
            "minimum_dynamic_clearance": float(c_result.best_dyn_clearance),
            "maximum_dynamic_cost": float(c_result.best_dyn_cost),
            "dwa_latency_ms": total_dwa_ms,
            "cpp_core_latency_ms": c_result.dwa_latency_ms,
            "emergency_stop_triggered": False,
            "backend_device": self.device_name,
            "active_backend": self.name,
            "best_trajectory_score": float(c_result.best_score),
            "selected_v": float(c_result.best_v),
            "selected_omega": float(c_result.best_w),
        }

        # Behavioral state machine arbitration
        decision = self.behavioral_state_machine.evaluate(
            current_state=current_state,
            best_trajectory=best_trajectory,
            best_v=float(c_result.best_v),
            best_w=float(c_result.best_w),
            planner_status=PlannerStatus.SUCCESS,
            dwa_diagnostics=diag,
            predicted_tracks=predicted_tracks,
            global_path=global_path,
            current_time_s=current_state.timestamp,
        )

        diag["behavioral_state"] = decision.state.value
        diag["behavioral_decision"] = decision.to_dict()
        diag["deadlock_event"] = decision.deadlock_event

        cmd_v = decision.command_v if self.enable_behavioral_arbitration else float(c_result.best_v)
        cmd_w = decision.command_omega if self.enable_behavioral_arbitration else float(c_result.best_w)

        return best_trajectory, cmd_v, cmd_w, PlannerStatus.SUCCESS, diag
