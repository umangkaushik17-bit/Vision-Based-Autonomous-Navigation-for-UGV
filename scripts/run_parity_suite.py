"""Comprehensive CPU/GPU Parity & Safety Validation Suite for Phase 13.17.1.

Evaluates mathematical rollout, candidate validity, collision safety, clearance/risk semantics,
multi-term trajectory scoring, velocity selection, behavioral states, and replanning triggers
between frozen CPU reference (ai/planning/dynamic/) and GPU accelerator (ai/planning/dynamic_gpu/).

Execution stages:
  STAGE A: 10 deterministic smoke cases
  STAGE B: 100 deterministic adversarial cases
  STAGE C: 100 randomized differential cases
  STAGE D: 1000 randomized differential cases

Critical Invariants Enforced:
  - ZERO GPU false negative collisions (CPU=COLLISION, GPU=FREE -> IMMEDIATE STOP)
  - ZERO unsafe behavioral state transitions (CPU=STOP, GPU=PROCEED -> SAFETY_DISCREPANCY)
  - ZERO suppressed safety-relevant replans
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ai.planning.base import PlanningConfig, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionResult,
)
from ai.planning.dynamic.data_model import (
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
)
from ai.planning.dynamic.dwa_filter import (
    DynamicBroadPhaseFilter,
    DynamicPlanningConfig,
)
from ai.planning.dynamic.global_replanner import (
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
)
from ai.planning.dynamic.occupancy_layer import DynamicOccupancyLayer
from ai.planning.dynamic_gpu.gpu_backend import GPUBackend
from ai.planning.dynamic_gpu.gpu_collision import GPUCollisionChecker
from ai.planning.dynamic_gpu.gpu_dwa import AcceleratedDWAPlanner
from ai.planning.dynamic_gpu.gpu_rollout import GPURollout
from ai.planning.dynamic_gpu.gpu_scoring import GPUScorer
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.path import Path
from ai.spatial_fusion.world_map import WorldTerrainMap

PARITY_OUTPUT_DIR = os.path.join("simulation", "outputs", "performance", "parity")
os.makedirs(PARITY_OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helper Utilities & Scenario Construction
# =============================================================================

def make_test_obstacle(
    track_id: int,
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    size: Tuple[float, float] = (0.6, 0.4),
    yaw: float = 0.0,
    horizon_s: float = 2.0,
    dt: float = 0.10,
    base_unc: float = 0.10,
    unc_growth: float = 0.05,
    status: str = "TRACKING",
) -> PredictedObstacleTrajectory:
    """Generate synthetic dynamic obstacle trajectory."""
    cov = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    state = TrackedObstacleState(
        track_id=track_id,
        timestamp=0.0,
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        covariance=cov,
        status=status,
        is_confirmed=True if status == "TRACKING" else False,
        bbox_size=size,
        bbox_yaw=yaw,
    )
    n_steps = int(round(horizon_s / dt))
    wps: List[TrajectoryWaypoint] = []
    polys: List[List[Tuple[float, float]]] = []

    for step in range(n_steps + 1):
        t = round(step * dt, 4)
        x = start_pos[0] + velocity[0] * t
        y = start_pos[1] + velocity[1] * t
        r_unc = base_unc + unc_growth * t
        cov_2d = np.diag([(r_unc / 3.0) ** 2, (r_unc / 3.0) ** 2]).astype(np.float64)
        wps.append(
            TrajectoryWaypoint(
                time_offset=t,
                timestamp=t,
                x=x,
                y=y,
                vx=velocity[0],
                vy=velocity[1],
                covariance_2d=cov_2d,
                uncertainty_radius=r_unc,
            )
        )
        polys.append(compute_obb_corners((x, y), size, yaw, margin_m=r_unc))

    return PredictedObstacleTrajectory(
        track_id=track_id,
        current_state=state,
        horizon_s=horizon_s,
        dt=dt,
        waypoints=wps,
        bounding_polygon_per_step=polys,
    )


def create_test_map(bounds: Tuple[float, float] = (-10.0, 10.0)) -> WorldTerrainMap:
    """Create a standardized clean WorldTerrainMap."""
    wmap = WorldTerrainMap(resolution_m=0.10, x_bounds_m=bounds, y_bounds_m=bounds)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    return wmap


# =============================================================================
# Benchmark Case Data Model
# =============================================================================

class ParityTestCase:
    """Encapsulates a complete local planning validation scenario."""

    def __init__(
        self,
        case_id: str,
        seed: int,
        scenario_name: str,
        robot_state: RobotState,
        global_path: Path,
        costmap: WorldTerrainMap,
        dynamic_layer: Optional[DynamicOccupancyLayer] = None,
        predicted_tracks: Optional[List[PredictedObstacleTrajectory]] = None,
        dynamic_config: Optional[DynamicPlanningConfig] = None,
        behavioral_config: Optional[BehavioralConfig] = None,
        description: str = "",
    ) -> None:
        self.case_id = case_id
        self.seed = seed
        self.scenario_name = scenario_name
        self.robot_state = robot_state
        self.global_path = global_path
        self.costmap = costmap
        self.dynamic_layer = dynamic_layer
        self.predicted_tracks = predicted_tracks or []
        self.dynamic_config = dynamic_config or DynamicPlanningConfig()
        self.behavioral_config = behavioral_config or BehavioralConfig()
        self.description = description


# =============================================================================
# Stage Generators
# =============================================================================

def build_stage_a_cases() -> List[ParityTestCase]:
    """STAGE A — 10 canonical smoke test cases."""
    cases = []
    default_path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    # Case 1: Empty Space
    wmap_1 = create_test_map()
    cases.append(ParityTestCase(
        case_id="case_01_empty_space",
        seed=1001,
        scenario_name="1. Empty Space",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_1,
        dynamic_layer=None,
        predicted_tracks=None,
        description="Unobstructed open terrain with zero obstacles",
    ))

    # Case 2: Stationary Obstacle
    wmap_2 = create_test_map()
    obs_2 = make_test_obstacle(track_id=1, start_pos=(1.2, 0.0), velocity=(0.0, 0.0), size=(0.6, 0.6))
    layer_2 = DynamicOccupancyLayer(); layer_2.update_predictions([obs_2])
    cases.append(ParityTestCase(
        case_id="case_02_stationary_obstacle",
        seed=1002,
        scenario_name="2. Stationary Obstacle",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_2,
        dynamic_layer=layer_2,
        predicted_tracks=[obs_2],
        description="Static obstacle blocking the forward centerline",
    ))

    # Case 3: Moving Obstacle
    wmap_3 = create_test_map()
    obs_3 = make_test_obstacle(track_id=2, start_pos=(1.5, 1.2), velocity=(0.0, -0.6), size=(0.7, 0.5))
    layer_3 = DynamicOccupancyLayer(); layer_3.update_predictions([obs_3])
    cases.append(ParityTestCase(
        case_id="case_03_moving_obstacle",
        seed=1003,
        scenario_name="3. Moving Obstacle",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_3,
        dynamic_layer=layer_3,
        predicted_tracks=[obs_3],
        description="Moving obstacle crossing the planned trajectory corridor",
    ))

    # Case 4: Near-Grazing Obstacle
    wmap_4 = create_test_map()
    # Obstacle passing parallel along the right edge within 0.12m of vehicle footprint
    obs_4 = make_test_obstacle(track_id=3, start_pos=(1.0, 0.38), velocity=(0.3, 0.0), size=(0.6, 0.4))
    layer_4 = DynamicOccupancyLayer(); layer_4.update_predictions([obs_4])
    cases.append(ParityTestCase(
        case_id="case_04_near_grazing_obstacle",
        seed=1004,
        scenario_name="4. Near-Grazing Obstacle",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_4,
        dynamic_layer=layer_4,
        predicted_tracks=[obs_4],
        description="Dynamic obstacle grazing parallel to vehicle corridor (clearance < 0.15m)",
    ))

    # Case 5: Head-On Obstacle
    wmap_5 = create_test_map()
    obs_5 = make_test_obstacle(track_id=4, start_pos=(2.8, 0.0), velocity=(-0.6, 0.0), size=(0.6, 0.5), yaw=math.pi)
    layer_5 = DynamicOccupancyLayer(); layer_5.update_predictions([obs_5])
    cases.append(ParityTestCase(
        case_id="case_05_head_on_obstacle",
        seed=1005,
        scenario_name="5. Head-On Obstacle",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_5,
        dynamic_layer=layer_5,
        predicted_tracks=[obs_5],
        description="Obstacle advancing directly opposite to robot heading",
    ))

    # Case 6: Crossing Obstacle
    wmap_6 = create_test_map()
    obs_6 = make_test_obstacle(track_id=5, start_pos=(1.2, -1.2), velocity=(0.0, 0.8), size=(0.6, 0.5), yaw=math.pi/2)
    layer_6 = DynamicOccupancyLayer(); layer_6.update_predictions([obs_6])
    cases.append(ParityTestCase(
        case_id="case_06_crossing_obstacle",
        seed=1006,
        scenario_name="6. Crossing Obstacle",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_6,
        dynamic_layer=layer_6,
        predicted_tracks=[obs_6],
        description="Fast crossing obstacle moving perpendicular across trajectory",
    ))

    # Case 7: Multiple Obstacles
    wmap_7 = create_test_map()
    obs_7a = make_test_obstacle(track_id=11, start_pos=(1.5, 1.2), velocity=(0.0, -0.6))
    obs_7b = make_test_obstacle(track_id=12, start_pos=(2.0, -1.0), velocity=(-0.2, 0.4))
    obs_7c = make_test_obstacle(track_id=13, start_pos=(3.0, 0.5), velocity=(-0.5, 0.0))
    obs_7d = make_test_obstacle(track_id=14, start_pos=(4.0, 4.0), velocity=(0.2, 0.2))
    # Add a static obstacle patch in the terrain
    r, c = wmap_7.world_to_grid(2.0, -0.3)
    wmap_7.obstacle[r-2:r+3, c-2:c+3] = True
    wmap_7.cost[r-2:r+3, c-2:c+3] = 1.00
    layer_7 = DynamicOccupancyLayer(); layer_7.update_predictions([obs_7a, obs_7b, obs_7c, obs_7d])
    cases.append(ParityTestCase(
        case_id="case_07_multiple_obstacles",
        seed=1007,
        scenario_name="7. Multiple Obstacles",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_7,
        dynamic_layer=layer_7,
        predicted_tracks=[obs_7a, obs_7b, obs_7c, obs_7d],
        description="Mixed static and dynamic obstacle environment with 4 agents",
    ))

    # Case 8: Narrow Clearance
    wmap_8 = create_test_map()
    # Flanking stationary obstacles creating narrow doorway of 0.85m (UGV is 0.45m wide)
    obs_8a = make_test_obstacle(track_id=21, start_pos=(1.5, 0.45), velocity=(0.0, 0.0), size=(0.5, 0.4))
    obs_8b = make_test_obstacle(track_id=22, start_pos=(1.5, -0.45), velocity=(0.0, 0.0), size=(0.5, 0.4))
    layer_8 = DynamicOccupancyLayer(); layer_8.update_predictions([obs_8a, obs_8b])
    cases.append(ParityTestCase(
        case_id="case_08_narrow_clearance",
        seed=1008,
        scenario_name="8. Narrow Clearance",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_8,
        dynamic_layer=layer_8,
        predicted_tracks=[obs_8a, obs_8b],
        description="Tight corridor with symmetrical flanking obstacles",
    ))

    # Case 9: Obstacle Outside Prediction Horizon
    wmap_9 = create_test_map()
    obs_9 = make_test_obstacle(track_id=31, start_pos=(5.0, 5.0), velocity=(0.5, 0.5), size=(0.8, 0.6))
    layer_9 = DynamicOccupancyLayer(); layer_9.update_predictions([obs_9])
    cases.append(ParityTestCase(
        case_id="case_09_outside_horizon",
        seed=1009,
        scenario_name="9. Outside Horizon",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_9,
        dynamic_layer=layer_9,
        predicted_tracks=[obs_9],
        description="Dynamic obstacle located beyond reachable lookahead envelope",
    ))

    # Case 10: Unknown-Horizon Case
    wmap_10 = create_test_map()
    # Obstacle trajectory only predicted up to 0.60s (planning horizon is 1.50s)
    obs_10 = make_test_obstacle(track_id=41, start_pos=(1.0, 0.0), velocity=(0.0, 0.0), size=(0.6, 0.6), horizon_s=0.60)
    layer_10 = DynamicOccupancyLayer(); layer_10.update_predictions([obs_10])
    dyn_cfg_10 = DynamicPlanningConfig(unknown_horizon_policy="PENALIZE", unknown_horizon_penalty=0.45)
    cases.append(ParityTestCase(
        case_id="case_10_unknown_horizon",
        seed=1010,
        scenario_name="10. Unknown Horizon",
        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
        global_path=default_path,
        costmap=wmap_10,
        dynamic_layer=layer_10,
        predicted_tracks=[obs_10],
        dynamic_config=dyn_cfg_10,
        description="Obstacle trajectory horizon ends at 0.6s before planning horizon 1.5s",
    ))

    return cases


def build_stage_b_cases(count: int = 100) -> List[ParityTestCase]:
    """STAGE B — 100 deterministic adversarial scenarios with fixed seeds."""
    cases = []
    base_path = Path.from_xy_array(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    distances = [0.8, 1.2, 1.6, 2.2, 3.0]
    lateral_offsets = [-0.6, -0.3, 0.0, 0.3, 0.6]
    speeds = [0.0, 0.3, 0.6, 1.0]
    headings = [0.0, math.pi / 2, math.pi, -math.pi / 2]

    case_idx = 0
    for d in distances:
        for lat in lateral_offsets:
            for spd in [0.0, 0.5]:
                for hd in [0.0, math.pi]:
                    if case_idx >= count:
                        break
                    case_idx += 1
                    seed = 2000 + case_idx
                    vx = -spd if hd == math.pi else spd
                    vy = 0.0

                    wmap = create_test_map()
                    obs = make_test_obstacle(
                        track_id=case_idx,
                        start_pos=(d, lat),
                        velocity=(vx, vy),
                        size=(0.6, 0.5),
                        yaw=hd,
                    )
                    layer = DynamicOccupancyLayer()
                    layer.update_predictions([obs])

                    # Interleave adversarial conditions
                    dyn_cfg = DynamicPlanningConfig()
                    if case_idx % 7 == 0:
                        obs.horizon_s = 0.50
                        dyn_cfg.unknown_horizon_policy = "PENALIZE"
                        dyn_cfg.unknown_horizon_penalty = 0.50
                    elif case_idx % 11 == 0:
                        # Add static obstacle wall
                        r, c = wmap.world_to_grid(d, lat + 0.5)
                        wmap.obstacle[r-1:r+2, c-1:c+2] = True
                        wmap.cost[r-1:r+2, c-1:c+2] = 1.0

                    cases.append(ParityTestCase(
                        case_id=f"stage_b_case_{case_idx:03d}",
                        seed=seed,
                        scenario_name=f"Det-Adversarial d={d:.1f}m lat={lat:.1f}m v={spd:.1f}",
                        robot_state=RobotState(x=0.0, y=0.0, yaw=0.0, v=0.30, omega=0.0),
                        global_path=base_path,
                        costmap=wmap,
                        dynamic_layer=layer,
                        predicted_tracks=[obs],
                        dynamic_config=dyn_cfg,
                        description=f"Deterministic adversarial variation #{case_idx}",
                    ))
                if case_idx >= count:
                    break
            if case_idx >= count:
                break
        if case_idx >= count:
            break

    return cases[:count]


def build_randomized_cases(count: int, master_seed: int, stage_prefix: str) -> List[ParityTestCase]:
    """STAGE C / D — Randomized differential scenarios with reproducible seeds."""
    rng = random.Random(master_seed)
    np_rng = np.random.default_rng(master_seed)
    cases = []
    base_path = Path.from_xy_array(np.array([[0.0, 0.0], [2.5, 0.0], [5.0, 0.0]], dtype=np.float32))

    for idx in range(1, count + 1):
        case_seed = int(rng.randint(100000, 999999))
        c_rng = np.random.default_rng(case_seed)

        # Randomized initial state
        init_v = float(c_rng.uniform(0.0, 0.50))
        init_w = float(c_rng.uniform(-0.40, 0.40))
        init_yaw = float(c_rng.uniform(-0.30, 0.30))
        state = RobotState(x=0.0, y=0.0, yaw=init_yaw, v=init_v, omega=init_w)

        wmap = create_test_map()

        # Random static clutter (10% chance per case)
        if c_rng.random() < 0.25:
            num_static = int(c_rng.integers(1, 4))
            for _ in range(num_static):
                ox = float(c_rng.uniform(0.5, 3.0))
                oy = float(c_rng.uniform(-1.0, 1.0))
                r, c = wmap.world_to_grid(ox, oy)
                wmap.obstacle[r-1:r+2, c-1:c+2] = True
                wmap.cost[r-1:r+2, c-1:c+2] = 1.0

        # Random dynamic obstacles (1 to 3)
        num_dyn = int(c_rng.integers(1, 4))
        obstacles = []
        for o_idx in range(num_dyn):
            start_x = float(c_rng.uniform(0.8, 3.5))
            start_y = float(c_rng.uniform(-1.5, 1.5))
            speed = float(c_rng.uniform(0.0, 0.8))
            angle = float(c_rng.uniform(-math.pi, math.pi))
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            size_x = float(c_rng.uniform(0.4, 0.8))
            size_y = float(c_rng.uniform(0.4, 0.6))
            horizon = float(c_rng.choice([0.6, 1.0, 1.5, 2.0]))

            obs = make_test_obstacle(
                track_id=100 * idx + o_idx + 1,
                start_pos=(start_x, start_y),
                velocity=(vx, vy),
                size=(size_x, size_y),
                yaw=angle,
                horizon_s=horizon,
            )
            obstacles.append(obs)

        layer = DynamicOccupancyLayer()
        layer.update_predictions(obstacles)

        dyn_cfg = DynamicPlanningConfig(
            unknown_horizon_policy=str(c_rng.choice(["PENALIZE", "REJECT", "ALLOW"])),
            unknown_horizon_penalty=float(c_rng.uniform(0.30, 0.60)),
        )

        cases.append(ParityTestCase(
            case_id=f"{stage_prefix}_case_{idx:04d}",
            seed=case_seed,
            scenario_name=f"Randomized #{idx} (dyn={num_dyn})",
            robot_state=state,
            global_path=base_path,
            costmap=wmap,
            dynamic_layer=layer,
            predicted_tracks=obstacles,
            dynamic_config=dyn_cfg,
            description=f"Differential random scenario seed={case_seed}",
        ))

    return cases


# =============================================================================
# Execution & Parity Evaluation Engine
# =============================================================================

def compare_rollouts(
    cpu_planner: DWAPlanner,
    gpu_planner: AcceleratedDWAPlanner,
    state: RobotState,
    v: float,
    omega: float,
    pred_horizon: float = 1.50,
    sim_dt: float = 0.10,
) -> Dict[str, float]:
    """Directly compare forward numerical kinematic rollouts for identical (v, omega)."""
    cpu_traj = cpu_planner._predict_trajectory(v, omega, state, create_test_map())
    if cpu_traj is None or len(cpu_traj.points) == 0:
        return {"max_abs_dx": 0.0, "max_abs_dy": 0.0, "max_abs_dyaw": 0.0, "rmse": 0.0}

    cpu_pts = np.array([[p.x, p.y, p.yaw] for p in cpu_traj.points])

    v_tensor = torch.tensor([v], device=gpu_planner.device, dtype=gpu_planner.dtype)
    w_tensor = torch.tensor([omega], device=gpu_planner.device, dtype=gpu_planner.dtype)
    gpu_trajs, _ = gpu_planner.rollout_kernel.rollout_trajectories(state, v_tensor, w_tensor, pred_horizon, sim_dt)
    gpu_pts = gpu_trajs[0, :, :3].detach().cpu().numpy()

    dx = np.abs(cpu_pts[:, 0] - gpu_pts[:, 0])
    dy = np.abs(cpu_pts[:, 1] - gpu_pts[:, 1])
    dyaw = np.abs((cpu_pts[:, 2] - gpu_pts[:, 2] + np.pi) % (2.0 * np.pi) - np.pi)
    rmse = float(np.sqrt(np.mean(dx**2 + dy**2)))

    return {
        "max_abs_dx": float(np.max(dx)),
        "max_abs_dy": float(np.max(dy)),
        "max_abs_dyaw": float(np.max(dyaw)),
        "rmse": rmse,
    }


def evaluate_case(
    case: ParityTestCase,
    cpu_planner: DWAPlanner,
    gpu_planner: AcceleratedDWAPlanner,
    global_replanner: EventDrivenGlobalReplanner,
) -> Dict[str, Any]:
    """Execute case on CPU and GPU planners and classify parity."""
    # 1. CPU Reference Execution
    t_cpu_0 = time.perf_counter()
    cpu_traj, cpu_v, cpu_w, cpu_status, cpu_diag = cpu_planner.compute_trajectory(
        case.robot_state,
        case.global_path,
        case.costmap,
        dynamic_layer=case.dynamic_layer,
        predicted_tracks=case.predicted_tracks,
        dynamic_config=case.dynamic_config,
    )
    cpu_latency_ms = (time.perf_counter() - t_cpu_0) * 1000.0

    # CPU Behavioral Evaluation
    cpu_decision = cpu_planner.behavioral_state_machine.evaluate(
        current_state=case.robot_state,
        best_trajectory=cpu_traj,
        best_v=cpu_v,
        best_w=cpu_w,
        planner_status=cpu_status,
        dwa_diagnostics=cpu_diag,
        predicted_tracks=case.predicted_tracks,
        global_path=case.global_path,
        current_time_s=case.robot_state.timestamp,
    )

    # Dynamic obstacles for replanner check
    dyn_obs = [t.current_state if isinstance(t, PredictedObstacleTrajectory) else t for t in case.predicted_tracks]

    # CPU Replanner Check
    global_replanner.reset(initial_path=case.global_path, timestamp=case.robot_state.timestamp)
    cpu_need_replan, cpu_replan_trigger, _, _ = global_replanner.check_trigger_conditions(
        case.robot_state.pose,
        (4.0, 0.0),
        case.global_path,
        case.costmap,
        behavioral_decision=cpu_decision,
        dynamic_obstacles=dyn_obs,
        timestamp=case.robot_state.timestamp,
    )

    # 2. GPU Accelerated Execution
    if gpu_planner.backend.is_cuda:
        torch.cuda.synchronize()
    t_gpu_0 = time.perf_counter()
    gpu_traj, gpu_v, gpu_w, gpu_status, gpu_diag = gpu_planner.compute_trajectory(
        case.robot_state,
        case.global_path,
        case.costmap,
        dynamic_layer=case.dynamic_layer,
        predicted_tracks=case.predicted_tracks,
        dynamic_config=case.dynamic_config,
    )
    if gpu_planner.backend.is_cuda:
        torch.cuda.synchronize()
    gpu_latency_ms = (time.perf_counter() - t_gpu_0) * 1000.0

    # GPU Behavioral Evaluation
    gpu_decision = gpu_planner.behavioral_state_machine.evaluate(
        current_state=case.robot_state,
        best_trajectory=gpu_traj,
        best_v=gpu_v,
        best_w=gpu_w,
        planner_status=gpu_status,
        dwa_diagnostics=gpu_diag,
        predicted_tracks=case.predicted_tracks,
        global_path=case.global_path,
        current_time_s=case.robot_state.timestamp,
    )

    # GPU Replanner Check
    global_replanner.reset(initial_path=case.global_path, timestamp=case.robot_state.timestamp)
    gpu_need_replan, gpu_replan_trigger, _, _ = global_replanner.check_trigger_conditions(
        case.robot_state.pose,
        (4.0, 0.0),
        case.global_path,
        case.costmap,
        behavioral_decision=gpu_decision,
        dynamic_obstacles=dyn_obs,
        timestamp=case.robot_state.timestamp,
    )

    # Rollout Parity
    rollout_metrics = compare_rollouts(
        cpu_planner, gpu_planner, case.robot_state, cpu_v, cpu_w
    )

    # 3. Parity Classification
    cpu_col = bool(cpu_diag.get("dynamic_collision", False) or (cpu_status == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY))
    gpu_col = bool(gpu_diag.get("dynamic_collision", False) or (gpu_status == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY))

    cpu_beh_str = cpu_decision.state.value
    gpu_beh_str = gpu_decision.state.value

    score_diff = abs(float(cpu_diag.get("best_score", 0.0)) - float(gpu_diag.get("best_score", 0.0)))
    v_diff = math.hypot(cpu_v - gpu_v, cpu_w - gpu_w)

    classification = "EXACT_MATCH"

    # CRITICAL SAFETY GATE: ZERO GPU FALSE NEGATIVE COLLISIONS
    if cpu_col and not gpu_col:
        classification = "SAFETY_DISCREPANCY"
        print(f"\n[!] CRITICAL SAFETY PARITY FAILURE: CPU=COLLISION, GPU=FREE on {case.case_id}!")
    elif cpu_beh_str == "STOP" and gpu_beh_str == "PROCEED":
        classification = "SAFETY_DISCREPANCY"
        print(f"\n[!] CRITICAL SAFETY PARITY FAILURE: Unsafe behavior transition CPU=STOP, GPU=PROCEED on {case.case_id}!")
    elif cpu_need_replan and not gpu_need_replan:
        classification = "SAFETY_DISCREPANCY"
        print(f"\n[!] CRITICAL SAFETY PARITY FAILURE: GPU suppressed safety replan on {case.case_id}!")
    elif cpu_diag.get("valid_trajectories", 0) != gpu_diag.get("valid_trajectories", 0):
        # Examine if difference is due to numerical edge near threshold
        val_diff = abs(cpu_diag.get("valid_trajectories", 0) - gpu_diag.get("valid_trajectories", 0))
        if val_diff > 3:
            classification = "VALIDITY_DISCREPANCY"
        else:
            classification = "EQUIVALENT_SAFE_ALTERNATIVE"
    elif v_diff > 0.02:
        if score_diff <= 0.05 and not gpu_col and gpu_status == PlannerStatus.SUCCESS:
            classification = "EQUIVALENT_SAFE_ALTERNATIVE"
        else:
            classification = "SCORE_DISCREPANCY"
    elif score_diff > 0.05:
        classification = "SCORE_DISCREPANCY"
    elif v_diff > 1e-4:
        classification = "NUMERICAL_TIE"
    else:
        classification = "EXACT_MATCH"

    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "scenario": case.scenario_name,
        "classification": classification,
        "cpu": {
            "selected_v": round(float(cpu_v), 4),
            "selected_w": round(float(cpu_w), 4),
            "status": cpu_status.name,
            "collision": cpu_col,
            "clearance": round(float(cpu_diag.get("min_clearance_m", 0.0)), 3),
            "cost": round(float(cpu_diag.get("maximum_dynamic_cost", 0.0)), 3),
            "score": round(float(cpu_diag.get("best_score", 0.0)), 4),
            "behavior": cpu_beh_str,
            "replan": cpu_replan_trigger.value,
            "valid_candidates": cpu_diag.get("valid_trajectories", 0),
            "latency_ms": round(cpu_latency_ms, 2),
        },
        "gpu": {
            "selected_v": round(float(gpu_v), 4),
            "selected_w": round(float(gpu_w), 4),
            "status": gpu_status.name,
            "collision": gpu_col,
            "clearance": round(float(gpu_diag.get("min_clearance_m", 0.0)), 3),
            "cost": round(float(gpu_diag.get("maximum_dynamic_cost", 0.0)), 3),
            "score": round(float(gpu_diag.get("best_score", 0.0)), 4),
            "behavior": gpu_beh_str,
            "replan": gpu_replan_trigger.value,
            "valid_candidates": gpu_diag.get("valid_trajectories", 0),
            "latency_ms": round(gpu_latency_ms, 2),
            "device": gpu_planner.backend.device_name,
        },
        "rollout_parity": rollout_metrics,
        "score_diff": round(score_diff, 5),
        "v_diff": round(v_diff, 5),
    }


# =============================================================================
# Stage Runner
# =============================================================================

def run_stage(
    stage_name: str,
    cases: List[ParityTestCase],
    preferred_device: str = "auto",
) -> Dict[str, Any]:
    """Execute an entire stage, print real-time report, and save output JSON."""
    print("\n" + "=" * 95)
    print(f"PHASE 13.17.1: RUNNING {stage_name} ({len(cases)} CASES)")
    print("=" * 95)

    config = PlanningConfig()
    cpu_planner = DWAPlanner(config)
    gpu_planner = AcceleratedDWAPlanner(config, preferred_device=preferred_device)
    global_replanner = EventDrivenGlobalReplanner()

    # Telemetry
    backend_name = gpu_planner.backend.device_name
    is_cuda = gpu_planner.backend.is_cuda
    print(f"GPU Backend Device: {backend_name} (CUDA Active: {is_cuda})")
    if is_cuda:
        mem_info = gpu_planner.backend.get_memory_info()
        print(f"Initial GPU Memory: {mem_info}")

    results = []
    counts = {
        "EXACT_MATCH": 0,
        "NUMERICAL_TIE": 0,
        "EQUIVALENT_SAFE_ALTERNATIVE": 0,
        "SCORE_DISCREPANCY": 0,
        "VALIDITY_DISCREPANCY": 0,
        "SAFETY_DISCREPANCY": 0,
    }

    header = (
        f"{'Case ID':<26} | {'CPU (v, w)':<14} | {'GPU (v, w)':<14} | "
        f"{'CPU/GPU Score':<15} | {'CPU/GPU Beh':<13} | {'Class':<14}"
    )
    print("\n" + header)
    print("-" * 95)

    out_filename = f"{stage_name.lower().replace(' ', '_')}.json"
    out_path = os.path.join(PARITY_OUTPUT_DIR, out_filename)

    for idx, case in enumerate(cases):
        res = evaluate_case(case, cpu_planner, gpu_planner, global_replanner)
        results.append(res)

        cls = res["classification"]
        counts[cls] = counts.get(cls, 0) + 1

        cpu_cmd = f"({res['cpu']['selected_v']:.2f}, {res['cpu']['selected_w']:.2f})"
        gpu_cmd = f"({res['gpu']['selected_v']:.2f}, {res['gpu']['selected_w']:.2f})"
        scores = f"{res['cpu']['score']:.3f}/{res['gpu']['score']:.3f}"
        behs = f"{res['cpu']['behavior'][:4]}/{res['gpu']['behavior'][:4]}"

        line = f"{case.case_id:<26} | {cpu_cmd:<14} | {gpu_cmd:<14} | {scores:<15} | {behs:<13} | {cls:<14}"
        print(line, flush=True)

        # Stop immediately on safety discrepancy
        if cls == "SAFETY_DISCREPANCY":
            print("\n" + "!" * 80)
            print("SAFETY PARITY FAILURE OCCURRED! STOPPING EXECUTION IMMEDIATELY.")
            print(f"Failing case: {case.case_id} (Seed: {case.seed})")
            print("!" * 80)
            break

        # Periodic checkpoint every 50 cases
        if (idx + 1) % 50 == 0 or (idx + 1) == len(cases):
            cpu_lats_chk = [r["cpu"]["latency_ms"] for r in results]
            gpu_lats_chk = [r["gpu"]["latency_ms"] for r in results]
            chk_summary = {
                "stage": stage_name,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "cases_evaluated": len(results),
                "device": backend_name,
                "is_cuda": is_cuda,
                "counts": counts,
                "pass_rate_pct": round(
                    100.0 * (counts["EXACT_MATCH"] + counts["NUMERICAL_TIE"] + counts["EQUIVALENT_SAFE_ALTERNATIVE"]) / max(1, len(results)),
                    2
                ),
                "safety_discrepancies": counts["SAFETY_DISCREPANCY"],
                "mean_cpu_latency_ms": round(float(np.mean(cpu_lats_chk)), 2),
                "mean_gpu_latency_ms": round(float(np.mean(gpu_lats_chk)), 2),
                "speedup_factor": round(float(np.mean(cpu_lats_chk)) / max(0.01, float(np.mean(gpu_lats_chk))), 2),
                "cases": results,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chk_summary, f, indent=2)
            if (idx + 1) % 50 == 0 and (idx + 1) < len(cases):
                print(f"[*] Checkpoint saved at {idx + 1}/{len(cases)} cases -> {out_path}", flush=True)

    # Latency aggregates
    cpu_lats = [r["cpu"]["latency_ms"] for r in results]
    gpu_lats = [r["gpu"]["latency_ms"] for r in results]

    summary = {
        "stage": stage_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cases_evaluated": len(results),
        "device": backend_name,
        "is_cuda": is_cuda,
        "counts": counts,
        "pass_rate_pct": round(
            100.0 * (counts["EXACT_MATCH"] + counts["NUMERICAL_TIE"] + counts["EQUIVALENT_SAFE_ALTERNATIVE"]) / max(1, len(results)),
            2
        ),
        "safety_discrepancies": counts["SAFETY_DISCREPANCY"],
        "mean_cpu_latency_ms": round(float(np.mean(cpu_lats)), 2),
        "mean_gpu_latency_ms": round(float(np.mean(gpu_lats)), 2),
        "speedup_factor": round(float(np.mean(cpu_lats)) / max(0.01, float(np.mean(gpu_lats))), 2),
        "cases": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved {stage_name} artifact: {out_path}")

    # Print summary table
    print("-" * 95)
    print(f"Summary: Total Cases={len(results)} | Exact={counts['EXACT_MATCH']} | Ties={counts['NUMERICAL_TIE']} | Equiv={counts['EQUIVALENT_SAFE_ALTERNATIVE']}")
    print(f"Discrepancies: Score={counts['SCORE_DISCREPANCY']} | Validity={counts['VALIDITY_DISCREPANCY']} | Safety={counts['SAFETY_DISCREPANCY']}")
    print(f"Performance: Mean CPU={summary['mean_cpu_latency_ms']:.1f}ms | Mean GPU={summary['mean_gpu_latency_ms']:.1f}ms | Speedup={summary['speedup_factor']:.2f}x")
    print("=" * 95 + "\n")

    return summary


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13.17.1 CPU/GPU Parity & Safety Suite")
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B", "C", "D", "all"],
                        help="Validation stage: A (10 cases), B (100 det), C (100 rand), D (1000 rand)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Target device for GPU planner ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    overall_results = {}

    # Stage A
    if args.stage in ("A", "all"):
        cases_a = build_stage_a_cases()
        summary_a = run_stage("Stage A 10 Cases", cases_a, preferred_device=args.device)
        overall_results["stage_a"] = summary_a
        if summary_a["safety_discrepancies"] > 0:
            print("[-] Stage A failed safety gate. Halting further stages.")
            sys.exit(1)

    # Stage B
    if args.stage in ("B", "all"):
        cases_b = build_stage_b_cases(100)
        summary_b = run_stage("Stage B 100 Deterministic", cases_b, preferred_device=args.device)
        overall_results["stage_b"] = summary_b
        if summary_b["safety_discrepancies"] > 0:
            print("[-] Stage B failed safety gate. Halting further stages.")
            sys.exit(1)

    # Stage C
    if args.stage in ("C", "all"):
        cases_c = build_randomized_cases(100, master_seed=42, stage_prefix="stage_c")
        summary_c = run_stage("Stage C 100 Randomized", cases_c, preferred_device=args.device)
        overall_results["stage_c"] = summary_c
        if summary_c["safety_discrepancies"] > 0:
            print("[-] Stage C failed safety gate. Halting further stages.")
            sys.exit(1)

    # Stage D
    if args.stage in ("D", "all"):
        cases_d = build_randomized_cases(1000, master_seed=1337, stage_prefix="stage_d")
        summary_d = run_stage("Stage D 1000 Randomized", cases_d, preferred_device=args.device)
        overall_results["stage_d"] = summary_d
        if summary_d["safety_discrepancies"] > 0:
            print("[-] Stage D failed safety gate.")
            sys.exit(1)

    # Load all existing stage summaries if present to form complete record
    for st_name, fname in [
        ("stage_a", "stage_a_10_cases.json"),
        ("stage_b", "stage_b_100_deterministic.json"),
        ("stage_c", "stage_c_100_randomized.json"),
        ("stage_d", "stage_d_1000_randomized.json"),
    ]:
        fpath = os.path.join(PARITY_OUTPUT_DIR, fname)
        if st_name not in overall_results and os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    overall_results[st_name] = json.load(f)
            except Exception:
                pass

    # Overall Summary
    parity_summary_path = os.path.join(PARITY_OUTPUT_DIR, "parity_summary.json")
    with open(parity_summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    print(f"[+] Overall parity summary written to: {parity_summary_path}")


if __name__ == "__main__":
    main()
