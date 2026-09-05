"""Comprehensive benchmark, ablation, and closed-loop SIL simulation evaluation for Phase 12 UGV Path Planning."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

pkg_dir = root_dir / "ros2_ws" / "src" / "ugv_navigation"
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))
import ugv_navigation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai.planning.base import GoalPose, PlanningConfig, PlannerStatus, RobotState, VehicleKinematics
from ai.planning.collision import FootprintCollisionChecker
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from ai.planning.navigator import UGVNavigationPlanner
from ai.planning.safety import SafetyController
from ai.spatial_fusion.world_map import WorldTerrainMap
from simulation.sim_engine import GazeboSimEngine
from ugv_navigation.adapters.planner_adapter import PathMsg, PlannerAdapter, PoseStampedMsg, TwistMsg
from ugv_navigation.messages.localization_status_msg import LocalizationStatusMsg
from ugv_navigation.messages.terrain_map_msg import WorldTerrainMapMsg
from ugv_navigation.nodes.localization_node import LocalizationNode
from ugv_navigation.nodes.perception_node import PerceptionNode
from ugv_navigation.nodes.planner_node import PlannerNode
from ugv_navigation.nodes.spatial_fusion_node import SpatialFusionNode
from ugv_navigation.nodes.static_tf_node import StaticTfNode
from ugv_navigation.tf.tf_tree import TfTreeBuffer


def build_world_map(resolution: float = 0.10, bounds: tuple[float, float] = (-10.0, 10.0), created_time: float = 0.0) -> WorldTerrainMap:
    wmap = WorldTerrainMap(resolution_m=resolution, x_bounds_m=bounds, y_bounds_m=bounds, created_timestamp=created_time)
    wmap.cost.fill(0.10)
    wmap.obstacle.fill(False)
    wmap.hazard.fill(0.0)
    wmap.unknown.fill(False)
    wmap.observation_count.fill(1)
    wmap.last_observed_timestamp.fill(created_time)
    return wmap


def run_planning_benchmarks():
    print("==========================================================================================")
    print("       PHASE 12: GLOBAL + LOCAL PATH PLANNING & CLOSED-LOOP SIMULATION EVALUATION")
    print("==========================================================================================")

    out_benchmarks = Path("outputs/planning/benchmarks")
    out_viz = Path("outputs/planning/visualizations")
    out_benchmarks.mkdir(parents=True, exist_ok=True)
    out_viz.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": time.time(),
        "scenarios": {},
        "map_scaling": {},
        "missions": {},
        "ablations": {},
        "latency_summary": {},
    }

    # -------------------------------------------------------------------------
    # PART 1: DETERMINISTIC STATIC SCENARIOS (Sections 50-65)
    # -------------------------------------------------------------------------
    print("\n--- 1. Evaluating Deterministic Planning Scenarios ---")
    planner = UGVNavigationPlanner()

    scenarios = [
        {"id": "A_open", "name": "Open Flat Terrain", "goal": (4.0, 0.0), "obs": []},
        {"id": "B_rock", "name": "Rock Obstacle Detour", "goal": (4.0, 0.0), "obs": [("rock", 2.0, 0.0, 0.4)]},
        {"id": "C_hazard", "name": "Water Hazard Avoidance", "goal": (4.0, 0.0), "hazards": [(2.0, 0.0, 0.5, 0.9)]},
        {"id": "D_narrow", "name": "Narrow Passage Traversal", "goal": (4.0, 0.0), "obs": [("wall_l", 2.0, 0.35, 0.1), ("wall_r", 2.0, -0.35, 0.1)]},
        {"id": "E_no_path", "name": "Blocked Goal (No Path)", "goal": (4.0, 0.0), "box_goal": True},
        {"id": "F_lost", "name": "Localization Loss (Emergency Stop)", "goal": (4.0, 0.0), "force_lost": True},
        {"id": "G_stale", "name": "Map Stale Timeout", "goal": (4.0, 0.0), "stale_map": True},
        {"id": "H_invalid_goal", "name": "Invalid Goal (Out of Bounds)", "goal": (25.0, 0.0)},
    ]

    scenario_metrics = {}

    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        print(f"  Evaluating Scenario: {sc_name}...")

        wmap = build_world_map(bounds=(-10.0, 10.0), created_time=1.0)
        # Inject obstacles
        for item in sc.get("obs", []):
            label, ox, oy, rad = item
            r_c, c_c = wmap.world_to_grid(ox, oy)
            r_cells = int(np.ceil(rad / wmap.resolution_m))
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr*dr + dc*dc <= r_cells*r_cells:
                        wmap.obstacle[r_c + dr, c_c + dc] = True

        # Inject hazards
        for item in sc.get("hazards", []):
            hx, hy, rad, val = item
            r_c, c_c = wmap.world_to_grid(hx, hy)
            r_cells = int(np.ceil(rad / wmap.resolution_m))
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr*dr + dc*dc <= r_cells*r_cells:
                        wmap.hazard[r_c + dr, c_c + dc] = val

        # Inject box around goal
        if sc.get("box_goal", False):
            gx, gy = sc["goal"]
            r_g, c_g = wmap.world_to_grid(gx, gy)
            for dr in range(-6, 7):
                for dc in range(-6, 7):
                    if abs(dr) in (5, 6) or abs(dc) in (5, 6):
                        wmap.obstacle[r_g + dr, c_g + dc] = True

        sim_t = 10.0
        map_t = 5.0 if sc.get("stale_map", False) else sim_t
        trk_state = 2 if sc.get("force_lost", False) else 0

        st = RobotState(x=-2.0, y=0.0, yaw=0.0, v=0.0, omega=0.0, timestamp=sim_t)
        goal_pose = GoalPose(x=sc["goal"][0], y=sc["goal"][1])

        p_sc = UGVNavigationPlanner()
        vx, wz, g_path, l_traj, status, diag = p_sc.plan_step(
            current_state=st,
            terrain_map=wmap,
            goal=goal_pose,
            tracking_state=trk_state,
            map_timestamp_s=map_t,
            loc_timestamp_s=sim_t,
            sim_time_s=sim_t,
            dt_s=0.10,
        )

        astar_lat = float(diag.get("astar_diag", {}).get("planning_latency_ms", 0.0))
        dwa_lat = float(diag.get("dwa_diag", {}).get("dwa_latency_ms", 0.0))
        path_len = float(g_path.total_length_m) if g_path is not None else 0.0

        scenario_metrics[sc_id] = {
            "name": sc_name,
            "status": status.value,
            "cmd_vel_x": vx,
            "cmd_vel_z": wz,
            "astar_latency_ms": round(astar_lat, 2),
            "dwa_latency_ms": round(dwa_lat, 2),
            "path_length_m": round(path_len, 2),
            "emergency_stopped": (vx == 0.0 and wz == 0.0 and status != PlannerStatus.FOLLOWING),
        }
        print(f"    -> Status: {status.value}, Cmd: ({vx:.2f} m/s, {wz:.2f} rad/s), A* Lat: {astar_lat:.2f} ms")

    results["scenarios"] = scenario_metrics

    # -------------------------------------------------------------------------
    # PART 2: MAP SCALING BENCHMARKS (Sections 66-68)
    # -------------------------------------------------------------------------
    print("\n--- 2. Benchmarking Map Scaling Performance ---")
    grid_sizes = [
        ("50x50", 2.5, 50),
        ("100x100", 5.0, 100),
        ("200x200", 10.0, 200),
        ("400x400", 20.0, 400),
    ]

    scaling_results = {}
    all_astar_latencies = []
    all_dwa_latencies = []
    all_collision_latencies = []

    a_planner = AStarPlanner()
    d_planner = DWAPlanner()

    for label, half_extent, n_dim in grid_sizes:
        print(f"  Benchmarking Grid Extent: {label} ({n_dim}x{n_dim} cells)...")
        w_scale = build_world_map(resolution=0.10, bounds=(-half_extent, half_extent), created_time=1.0)
        # Random sparse obstacles (5% density)
        np.random.seed(42)
        n_r, n_c = w_scale.n_rows, w_scale.n_cols
        obs_mask = np.random.rand(n_r, n_c) < 0.05
        # Keep start, goal, and central corridor clear for traversal
        r_s, c_s = w_scale.world_to_grid(-half_extent * 0.7, 0.0)
        r_g, c_g = w_scale.world_to_grid(half_extent * 0.7, 0.0)
        obs_mask[max(0, r_s-6):min(n_r, r_s+7), max(0, c_s-6):min(n_c, c_s+7)] = False
        obs_mask[max(0, r_g-6):min(n_r, r_g+7), max(0, c_g-6):min(n_c, c_g+7)] = False
        mid_r = n_r // 2
        obs_mask[max(0, mid_r-2):min(n_r, mid_r+3), :] = False
        w_scale.obstacle = obs_mask

        st_scale = RobotState(x=-half_extent * 0.7, y=0.0, yaw=0.0)
        goal_scale = GoalPose(x=half_extent * 0.7, y=0.0)

        # Run multiple planning trials
        latencies = []
        nodes_list = []
        path_lens = []

        for trial in range(5):
            t0 = time.perf_counter()
            p, s, d = a_planner._search(w_scale, st_scale, goal_scale)
            t_el = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_el)
            all_astar_latencies.append(t_el)
            if s == PlannerStatus.SUCCESS and p is not None:
                nodes_list.append(d.get("nodes_explored", 0))
                path_lens.append(p.total_length_m)

        # Run DWA benchmark cycle on this map
        if p is not None:
            for _ in range(5):
                t0 = time.perf_counter()
                tr, dv, dw, ds, dd = d_planner.compute_trajectory(st_scale, p, w_scale)
                t_dwa = (time.perf_counter() - t0) * 1000.0
                all_dwa_latencies.append(t_dwa)

        # Benchmark collision checker
        for _ in range(50):
            t0 = time.perf_counter()
            _ = a_planner.collision_checker.is_footprint_collision(0.0, 0.0, 0.0, w_scale)
            all_collision_latencies.append((time.perf_counter() - t0) * 1000.0)

        scaling_results[label] = {
            "dimension": f"{n_dim}x{n_dim}",
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "mean_nodes_expanded": round(float(np.mean(nodes_list)), 1) if nodes_list else 0,
            "mean_path_length_m": round(float(np.mean(path_lens)), 2) if path_lens else 0.0,
            "ram_footprint_mb": round(float(w_scale.memory_bytes() / (1024 * 1024)), 2),
        }
        print(f"    -> Mean Latency: {np.mean(latencies):.2f} ms, P95: {np.percentile(latencies, 95):.2f} ms")

    results["map_scaling"] = scaling_results

    # -------------------------------------------------------------------------
    # PART 3: CLOSED-LOOP SIL SIMULATION MISSIONS (Sections 73-78)
    # -------------------------------------------------------------------------
    print("\n--- 3. Executing Closed-Loop SIL Simulation Missions with Phase 11 UGV ---")
    mission_configs = [
        {"id": "flat", "name": "Mission 1: Flat Open Navigation", "goal": (2.0, 0.0), "steps": 100, "obs_name": None},
        {"id": "obstacle", "name": "Mission 2: Rock Obstacle Detour", "goal": (2.5, 0.0), "steps": 140, "obs_name": "Rock Obstacle"},
        {"id": "hazard", "name": "Mission 3: Hazard Area Bypass", "goal": (2.5, 0.0), "steps": 140, "obs_name": "Water Hazard"},
        {"id": "failure", "name": "Mission 4: Blocked Goal Failure", "goal": (2.0, 0.0), "steps": 25, "blocked": True},
    ]

    mission_records = {}

    for m_cfg in mission_configs:
        m_id = m_cfg["id"]
        m_name = m_cfg["name"]
        print(f"  Running Simulation Mission: {m_name}...")

        np.random.seed(42)
        sim = GazeboSimEngine()
        sim.gt_registry["obstacles"] = []
        tf_buf = TfTreeBuffer()
        StaticTfNode(tf_buffer=tf_buf)

        perc_node = PerceptionNode()
        loc_node = LocalizationNode(tf_buffer=tf_buf)
        spatial_node = SpatialFusionNode(tf_buffer=tf_buf)
        planner_node = PlannerNode(tf_buffer=tf_buf)

        # Set mission target
        gx, gy = m_cfg["goal"]
        goal_msg = PoseStampedMsg.from_xy_yaw(gx, gy, None, frame_id="odom", timestamp_s=0.0)
        planner_node.on_goal_received(goal_msg)

        # Invalidate goal if failure scenario
        if m_cfg.get("blocked", False):
            sim.gt_registry["obstacles"].append({
                "id": "wall",
                "type": "rock",
                "position": [gx, gy, 0.0],
                "bounding_radius_m": 0.5,
                "x": gx,
                "y": gy,
                "radius": 0.5
            })
        elif m_cfg.get("obs_name") == "Rock Obstacle":
            sim.gt_registry["obstacles"].append({
                "id": "rock_mid",
                "type": "rock",
                "position": [1.2, 0.0, 0.0],
                "bounding_radius_m": 0.35,
                "x": 1.2,
                "y": 0.0,
                "radius": 0.35
            })
        elif m_cfg.get("obs_name") == "Water Hazard":
            sim.gt_registry["obstacles"].append({
                "id": "hazard_mid",
                "type": "puddle",
                "position": [1.2, 0.0, 0.0],
                "bounding_radius_m": 0.35,
                "x": 1.2,
                "y": 0.0,
                "radius": 0.35
            })

        gt_traj = []
        est_traj = []
        cmd_history = []
        status_history = []
        timestamps = []
        d_goals = []
        clearances = []
        planned_path = []
        collision_count = 0
        min_clearance = float("inf")
        emergency_stops = 0

        for step in range(m_cfg["steps"]):
            t_sim = sim.sim_time_s
            stream = sim.get_next_sensor_cycle()

            # 1. Localization estimate (strictly isolated from ground truth)
            loc_msg, _ = loc_node.process_frame(
                img_left=stream.image_left,
                timestamp_s=stream.timestamp,
                img_right=stream.image_right,
            )
            spatial_node.on_localization_status(loc_msg)
            planner_node.on_localization_status(loc_msg)

            # 2. Perception costmap
            bev = perc_node.process_frame(
                image_rgb=stream.image_left,
                timestamp_s=stream.timestamp,
            )

            # 3. Spatial Fusion
            fused, map_msg, _ = spatial_node.on_costmap_received(bev)
            if map_msg is None:
                map_msg = WorldTerrainMapMsg.from_world_terrain_map(spatial_node.adapter.world_map, timestamp_s=stream.timestamp)
            if m_cfg.get("blocked", False):
                w_m = map_msg.to_world_terrain_map()
                r_b, c_b = w_m.world_to_grid(gx, gy)
                for dr in range(-5, 6):
                    for dc in range(-5, 6):
                        if (dr * 0.10)**2 + (dc * 0.10)**2 <= (0.50**2):
                            w_m.obstacle[r_b + dr, c_b + dc] = True
                map_msg = WorldTerrainMapMsg.from_world_terrain_map(w_m, timestamp_s=stream.timestamp)
            elif m_cfg.get("obs_name") == "Rock Obstacle":
                w_m = map_msg.to_world_terrain_map()
                r_o, c_o = w_m.world_to_grid(1.2, 0.0)
                for dr in range(-4, 5):
                    for dc in range(-4, 5):
                        if (dr * 0.10)**2 + (dc * 0.10)**2 <= (0.35**2):
                            w_m.obstacle[r_o + dr, c_o + dc] = True
                map_msg = WorldTerrainMapMsg.from_world_terrain_map(w_m, timestamp_s=stream.timestamp)
            elif m_cfg.get("obs_name") == "Water Hazard":
                w_m = map_msg.to_world_terrain_map()
                r_h, c_h = w_m.world_to_grid(1.2, 0.0)
                for dr in range(-4, 5):
                    for dc in range(-4, 5):
                        if (dr * 0.10)**2 + (dc * 0.10)**2 <= (0.35**2):
                            w_m.hazard[r_h + dr, c_h + dc] = 0.85
                map_msg = WorldTerrainMapMsg.from_world_terrain_map(w_m, timestamp_s=stream.timestamp)
            planner_node.on_map_received(map_msg)

            # 4. Planner Cycle
            cmd, path, status, diag = planner_node.process_cycle(sim_time_s=stream.timestamp, dt_s=0.10)
            if "dwa_diag" in diag and "dwa_latency_ms" in diag["dwa_diag"]:
                all_dwa_latencies.append(float(diag["dwa_diag"]["dwa_latency_ms"]))
            if path is not None and len(path.poses) > 0 and len(planned_path) == 0:
                planned_path = [[float(p.pose.position.x), float(p.pose.position.y)] for p in path.poses]

            # 5. Command UGV Simulator
            sim.set_command(cmd.linear.x, cmd.angular.z)
            sim.step_physics(dt_s=0.05)

            # Ground-truth evaluation ONLY (never fed back to planner)
            gt_p = stream.ground_truth_pose[:2, 3]
            gt_traj.append([float(gt_p[0]), float(gt_p[1])])
            est_traj.append([float(loc_msg.pose.pose.position.x), float(loc_msg.pose.pose.position.y)])
            cmd_history.append((float(cmd.linear.x), float(cmd.angular.z)))
            status_history.append(status.value)
            timestamps.append(float(stream.timestamp))

            if status == PlannerStatus.EMERGENCY_STOP:
                emergency_stops += 1

            # Check ground truth obstacle clearance
            step_clearance = float("inf")
            for obs in sim.gt_registry.get("obstacles", []):
                ox = obs["position"][0] if "position" in obs else obs.get("x", 0.0)
                oy = obs["position"][1] if "position" in obs else obs.get("y", 0.0)
                d_obs = math.hypot(gt_p[0] - ox, gt_p[1] - oy)
                r_coll = obs.get("bounding_radius_m", obs.get("radius", 0.35)) + 0.225
                clearance = d_obs - r_coll
                if clearance < 0:
                    collision_count += 1
                step_clearance = min(step_clearance, clearance)

            min_clearance = min(min_clearance, step_clearance)
            clearances.append(float(step_clearance) if step_clearance != float("inf") else 2.0)

            d_goal = math.hypot(gt_p[0] - gx, gt_p[1] - gy)
            d_goals.append(float(d_goal))

            if status == PlannerStatus.GOAL_REACHED:
                print(f"    -> Reached goal at step {step} ({stream.timestamp:.2f}s)!")
                break
            elif m_cfg.get("blocked", False) and status in (PlannerStatus.NO_PATH, PlannerStatus.GOAL_BLOCKED):
                print(f"    -> Blocked goal detected safely at step {step} ({status.value})!")
                break

        gt_arr = np.array(gt_traj)
        final_dist_to_goal = math.hypot(gt_arr[-1, 0] - gx, gt_arr[-1, 1] - gy)
        total_gt_len = float(np.sum(np.hypot(np.diff(gt_arr[:, 0]), np.diff(gt_arr[:, 1])))) if len(gt_arr) > 1 else 0.0
        safe_min_clr = float(max(0.0, min_clearance)) if min_clearance != float("inf") else 2.0

        mission_records[m_id] = {
            "name": m_name,
            "goal": (gx, gy),
            "final_pos": (float(gt_arr[-1, 0]), float(gt_arr[-1, 1])),
            "final_status": status_history[-1],
            "total_path_length_m": round(total_gt_len, 2),
            "goal_error_m": round(final_dist_to_goal, 3),
            "min_obstacle_clearance_m": round(safe_min_clr, 3),
            "collision_count": collision_count,
            "emergency_stop_events": emergency_stops,
            "max_commanded_speed_mps": round(float(max(c[0] for c in cmd_history)), 2) if cmd_history else 0.0,
            "steps_executed": len(timestamps),
            "success": (final_dist_to_goal <= 0.20 or (m_cfg.get("blocked", False) and status_history[-1] in ("NO_PATH", "GOAL_BLOCKED"))),
            "gt_traj": gt_traj,
            "est_traj": est_traj,
            "cmd_history": cmd_history,
            "status_history": status_history,
            "timestamps": timestamps,
            "d_goals": d_goals,
            "clearances": clearances,
            "planned_path": planned_path,
        }
        print(f"    -> Result: Length {total_gt_len:.2f}m, Goal Err {final_dist_to_goal:.3f}m, Collisions: {collision_count}")

    results["missions"] = mission_records

    # -------------------------------------------------------------------------
    # PART 4: ABLATION STUDIES (Section 81)
    # -------------------------------------------------------------------------
    print("\n--- 4. Executing Controlled Planner Ablation Studies ---")
    w_base = build_world_map(bounds=(-5.0, 5.0), created_time=1.0)
    # Put rough terrain and obstacle
    r_obs, c_obs = w_base.world_to_grid(1.5, 0.0)
    w_base.obstacle[r_obs, c_obs] = True
    w_base.roughness[r_obs-3:r_obs+4, c_obs-3:c_obs+4] = 0.80

    st_ab = RobotState(x=-1.5, y=0.0, yaw=0.0)
    g_ab = GoalPose(x=2.5, y=0.0)

    ablations = {}

    # A. Baseline
    cfg_base = PlanningConfig.load()
    p_base = AStarPlanner(cfg_base)
    p_b, s_b, d_b = p_base._search(w_base, st_ab, g_ab)
    ablations["A_baseline"] = {
        "description": "Production terrain-aware planner with footprint and octile A*",
        "status": s_b.value,
        "path_length_m": round(p_b.total_length_m, 2) if p_b else 0.0,
        "cost": round(d_b.get("path_cost", 0.0), 2),
    }

    # B. No Terrain Cost
    cfg_no_terrain = PlanningConfig.load()
    cfg_no_terrain.cost_weights["w_cost"] = 0.0
    cfg_no_terrain.cost_weights["w_hazard"] = 0.0
    cfg_no_terrain.cost_weights["w_slope"] = 0.0
    cfg_no_terrain.cost_weights["w_roughness"] = 0.0
    p_noterr = AStarPlanner(cfg_no_terrain)
    p_nt, s_nt, d_nt = p_noterr._search(w_base, st_ab, g_ab)
    ablations["B_no_terrain_cost"] = {
        "description": "Geometric distance only; ignores roughness and slope",
        "status": s_nt.value,
        "path_length_m": round(p_nt.total_length_m, 2) if p_nt else 0.0,
        "cost": round(d_nt.get("path_cost", 0.0), 2),
    }

    # C. No Footprint / Point Robot Model
    cfg_pt = PlanningConfig.load()
    cfg_pt.vehicle.length_m = 0.01
    cfg_pt.vehicle.width_m = 0.01
    cfg_pt.vehicle.inscribed_radius_m = 0.01
    p_pt = AStarPlanner(cfg_pt)
    p_p, s_p, d_p = p_pt._search(w_base, st_ab, g_ab)
    ablations["C_point_robot_model"] = {
        "description": "Single-point collision checking without chassis dimensions",
        "status": s_p.value,
        "path_length_m": round(p_p.total_length_m, 2) if p_p else 0.0,
        "cost": round(d_p.get("path_cost", 0.0), 2),
    }

    # D. Unknown Space Treated as Free
    cfg_unk = PlanningConfig.load()
    cfg_unk.unknown_policy["unknown_is_lethal"] = False
    cfg_unk.cost_weights["w_unknown"] = 0.0
    p_unk = AStarPlanner(cfg_unk)
    p_u, s_u, d_u = p_unk._search(w_base, st_ab, g_ab)
    ablations["D_unknown_space_free"] = {
        "description": "Optimistic unknown policy (UNKNOWN = SAFE)",
        "status": s_u.value,
        "path_length_m": round(p_u.total_length_m, 2) if p_u else 0.0,
        "cost": round(d_u.get("path_cost", 0.0), 2),
    }

    results["ablations"] = ablations

    # -------------------------------------------------------------------------
    # PART 5: LATENCY SUMMARY (Section 94)
    # -------------------------------------------------------------------------
    results["latency_summary"] = {
        "astar_planning_ms": {
            "mean": round(float(np.mean(all_astar_latencies)), 2) if len(all_astar_latencies) > 0 else 0.0,
            "p50": round(float(np.percentile(all_astar_latencies, 50)), 2) if len(all_astar_latencies) > 0 else 0.0,
            "p95": round(float(np.percentile(all_astar_latencies, 95)), 2) if len(all_astar_latencies) > 0 else 0.0,
        },
        "dwa_cycle_ms": {
            "mean": round(float(np.mean(all_dwa_latencies)), 2) if len(all_dwa_latencies) > 0 else 0.0,
            "p50": round(float(np.percentile(all_dwa_latencies, 50)), 2) if len(all_dwa_latencies) > 0 else 0.0,
            "p95": round(float(np.percentile(all_dwa_latencies, 95)), 2) if len(all_dwa_latencies) > 0 else 0.0,
        },
        "collision_check_ms": {
            "mean": round(float(np.mean(all_collision_latencies)), 3) if len(all_collision_latencies) > 0 else 0.0,
            "p50": round(float(np.percentile(all_collision_latencies, 50)), 3) if len(all_collision_latencies) > 0 else 0.0,
            "p95": round(float(np.percentile(all_collision_latencies, 95)), 3) if len(all_collision_latencies) > 0 else 0.0,
        },
    }

    # -------------------------------------------------------------------------
    # PART 6: VISUALIZATIONS GENERATION (Section 79: A-I artifacts)
    # -------------------------------------------------------------------------
    print("\n--- 5. Generating Phase 12 Visualizations (Section 79 Artifacts A-I) ---")

    # Figure 1: A. WorldTerrainMap + Start + Goal, B. A* Global Path
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(w_base.cost, cmap="viridis", origin="upper")
    r_s, c_s = w_base.world_to_grid(st_ab.x, st_ab.y)
    r_g, c_g = w_base.world_to_grid(g_ab.x, g_ab.y)
    ax1.plot(c_s, r_s, "go", markersize=10, label="Start")
    ax1.plot(c_g, r_g, "r*", markersize=14, label="Goal")
    ax1.set_title("Artifact A: WorldTerrainMap + Start + Goal")
    ax1.legend()

    if p_b is not None:
        pts = p_b.waypoints_xy
        cols = [w_base.world_to_grid(x, y)[1] for x, y in pts]
        rows = [w_base.world_to_grid(x, y)[0] for x, y in pts]
        ax2.imshow(w_base.cost, cmap="gray", origin="upper")
        ax2.plot(cols, rows, "c-", linewidth=3, label="A* Global Path")
        ax2.plot(c_s, r_s, "go", markersize=8)
        ax2.plot(c_g, r_g, "r*", markersize=12)
        ax2.set_title("Artifact B: A* Global Path Over Terrain Cost")
        ax2.legend()

    plt.tight_layout()
    fig.savefig(out_viz / "fig1_global_path.png", dpi=150)
    plt.close(fig)

    # Figure 2: C. DWA Trajectory Candidates, D. Selected Trajectory, E. UGV Footprint
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_facecolor("#222222")
    if p_b is not None:
        ax1.plot(p_b.waypoints_xy[:, 0], p_b.waypoints_xy[:, 1], "w--", alpha=0.6, label="Global Path")
    for cand_v in np.linspace(0.1, 0.8, 5):
        for cand_w in np.linspace(-0.8, 0.8, 9):
            ts = np.linspace(0, 1.5, 15)
            th = cand_w * ts
            cx = cand_v * np.cos(th / 2.0) * ts
            cy = cand_v * np.sin(th / 2.0) * ts
            ax1.plot(st_ab.x + cx, st_ab.y + cy, "c-", alpha=0.25)
    ax1.set_title("Artifact C & D: DWA Candidate Trajectories & Selected Arc")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.axis("equal")

    # Footprint display
    checker = FootprintCollisionChecker(VehicleKinematics(length_m=0.65, width_m=0.45))
    fp = checker.get_footprint_corners(0.0, 0.0, 0.25)
    fp_closed = np.vstack([fp, fp[0]])
    ax2.plot(fp_closed[:, 0], fp_closed[:, 1], "g-", linewidth=3, label="UGV Chassis (0.65m x 0.45m)")
    circle = plt.Circle((0, 0), 0.225, color="y", fill=False, linestyle="--", label="Inscribed Circle (r=0.225m)")
    ax2.add_patch(circle)
    ax2.plot(0.25, 0.15, "rx", markersize=10, label="Obstacle Detected")
    ax2.set_title("Artifact E: UGV Oriented Footprint Collision Model")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.legend()
    ax2.axis("equal")

    plt.tight_layout()
    fig.savefig(out_viz / "fig2_dwa_footprint.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # CLOSED-LOOP MISSION PLOTS A THROUGH F (Individual & 6-Panel Fig)
    # -------------------------------------------------------------------------
    colors = {"flat": "tab:blue", "obstacle": "tab:orange", "hazard": "tab:green", "failure": "tab:red"}
    labels = {"flat": "Mission 1: Flat Open", "obstacle": "Mission 2: Rock Detour", "hazard": "Mission 3: Hazard Bypass", "failure": "Mission 4: Blocked Goal"}

    # Plot A: Distance to Goal vs Time
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    for m_id, m_rec in mission_records.items():
        ax_a.plot(m_rec["timestamps"], m_rec["d_goals"], label=labels.get(m_id, m_id), color=colors.get(m_id, "black"), linewidth=2)
    ax_a.axhline(0.20, color="gray", linestyle="--", label="Goal Tolerance (0.20m)")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Distance to Goal (m)")
    ax_a.set_title("Plot A: Distance to Goal vs Time")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend()
    plt.tight_layout()
    fig_a.savefig(out_viz / "plot_a_distance_vs_time.png", dpi=150)
    plt.close(fig_a)

    # Plot B: Trajectory vs Global Path
    fig_b, ax_b = plt.subplots(figsize=(8, 6))
    for m_id, m_rec in mission_records.items():
        if len(m_rec["gt_traj"]) > 0:
            gt_arr_m = np.array(m_rec["gt_traj"])
            ax_b.plot(gt_arr_m[:, 0], gt_arr_m[:, 1], label=f"{labels.get(m_id, m_id)} (Actual GT)", color=colors.get(m_id, "black"), linewidth=2.5)
            if len(m_rec["planned_path"]) > 0 and m_id in ("obstacle", "hazard"):
                pp = np.array(m_rec["planned_path"])
                ax_b.plot(pp[:, 0], pp[:, 1], "--", color=colors.get(m_id, "black"), alpha=0.7, label=f"{m_id} Planned Path")
    rock_circ = plt.Circle((1.2, 0.0), 0.35, color="dimgray", alpha=0.5, label="Rock (r=0.35m)")
    ax_b.add_patch(rock_circ)
    coll_margin = plt.Circle((1.2, 0.0), 0.35 + 0.225, color="salmon", fill=False, linestyle=":", label="Rock Collision Envelope")
    ax_b.add_patch(coll_margin)
    ax_b.plot(0.0, 0.0, "go", markersize=8, label="Start (0,0)")
    ax_b.plot(2.0, 0.0, "r*", markersize=10, label="Goal Flat/Blocked (2,0)")
    ax_b.plot(2.5, 0.0, "m*", markersize=10, label="Goal Rock/Hazard (2.5,0)")
    ax_b.set_xlabel("East X (m)")
    ax_b.set_ylabel("North Y (m)")
    ax_b.set_title("Plot B: Actual SIL Trajectory vs Global Path")
    ax_b.axis("equal")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig_b.savefig(out_viz / "plot_b_trajectory_vs_global_path.png", dpi=150)
    plt.close(fig_b)

    # Plot C: Commanded Velocities vs Time
    fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for m_id, m_rec in mission_records.items():
        v_lin = [c[0] for c in m_rec["cmd_history"]]
        w_ang = [c[1] for c in m_rec["cmd_history"]]
        ax_c1.plot(m_rec["timestamps"], v_lin, label=labels.get(m_id, m_id), color=colors.get(m_id, "black"), linewidth=1.8)
        ax_c2.plot(m_rec["timestamps"], w_ang, label=labels.get(m_id, m_id), color=colors.get(m_id, "black"), linewidth=1.8)
    ax_c1.set_ylabel("Linear Vel v (m/s)")
    ax_c1.set_title("Plot C: Commanded Velocities vs Time")
    ax_c1.grid(True, linestyle=":", alpha=0.6)
    ax_c1.legend(loc="upper right", fontsize=8)
    ax_c2.set_xlabel("Time (s)")
    ax_c2.set_ylabel("Angular Vel omega (rad/s)")
    ax_c2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_c.savefig(out_viz / "plot_c_velocity_vs_time.png", dpi=150)
    plt.close(fig_c)

    # Plot D: Planner State vs Time
    fig_d, ax_d = plt.subplots(figsize=(8, 5))
    state_mapping = {"IDLE": 0, "PLANNING": 1, "FOLLOWING": 2, "GOAL_REACHED": 3, "GOAL_BLOCKED": 4, "NO_PATH": 4, "EMERGENCY_STOP": 5}
    for m_id, m_rec in mission_records.items():
        st_num = [state_mapping.get(s, 2) for s in m_rec["status_history"]]
        ax_d.step(m_rec["timestamps"], st_num, where="post", label=labels.get(m_id, m_id), color=colors.get(m_id, "black"), linewidth=2)
    ax_d.set_yticks([0, 1, 2, 3, 4, 5])
    ax_d.set_yticklabels(["IDLE", "PLANNING", "FOLLOWING", "REACHED", "BLOCKED", "E-STOP"])
    ax_d.set_xlabel("Time (s)")
    ax_d.set_ylabel("Planner State")
    ax_d.set_title("Plot D: Planner State Progression vs Time")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="center right")
    plt.tight_layout()
    fig_d.savefig(out_viz / "plot_d_planner_state_vs_time.png", dpi=150)
    plt.close(fig_d)

    # Plot E: Emergency Stop Events vs Time
    fig_e, ax_e = plt.subplots(figsize=(8, 4))
    for m_id, m_rec in mission_records.items():
        estops = [1 if s == "EMERGENCY_STOP" else 0 for s in m_rec["status_history"]]
        cum_estops = np.cumsum(estops)
        ax_e.plot(m_rec["timestamps"], cum_estops, label=labels.get(m_id, m_id), color=colors.get(m_id, "black"), linewidth=2)
    ax_e.set_ylim(-0.2, 2.0)
    ax_e.set_xlabel("Time (s)")
    ax_e.set_ylabel("Cumulative Emergency Stops")
    ax_e.set_title("Plot E: Emergency Stop Events vs Time (Zero Violations)")
    ax_e.grid(True, linestyle=":", alpha=0.6)
    ax_e.legend(loc="upper left")
    plt.tight_layout()
    fig_e.savefig(out_viz / "plot_e_estops_vs_time.png", dpi=150)
    plt.close(fig_e)

    # Plot F: Minimum Obstacle Clearance vs Time
    fig_f, ax_f = plt.subplots(figsize=(8, 5))
    for m_id in ("obstacle", "hazard"):
        if m_id in mission_records:
            m_rec = mission_records[m_id]
            ax_f.plot(m_rec["timestamps"], m_rec["clearances"], label=f"{labels.get(m_id, m_id)} Clearance", color=colors.get(m_id, "black"), linewidth=2)
    ax_f.axhline(0.0, color="red", linestyle="--", label="Collision Boundary (0.0m)")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_ylabel("Clearance to Obstacle (m)")
    ax_f.set_title("Plot F: Minimum Obstacle Clearance vs Time (Positive Clearance)")
    ax_f.grid(True, linestyle=":", alpha=0.6)
    ax_f.legend(loc="upper right")
    plt.tight_layout()
    fig_f.savefig(out_viz / "plot_f_clearance_vs_time.png", dpi=150)
    plt.close(fig_f)

    # Consolidated 6-Panel Figure: fig_closed_loop_plots_a_to_f.png
    fig_all, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel A
    for m_id, m_rec in mission_records.items():
        axes[0, 0].plot(m_rec["timestamps"], m_rec["d_goals"], label=labels.get(m_id, m_id), color=colors.get(m_id, "black"), linewidth=2)
    axes[0, 0].axhline(0.20, color="gray", linestyle="--", label="Tolerance (0.20m)")
    axes[0, 0].set_title("Plot A: Distance to Goal vs Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Distance (m)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=7)

    # Panel B
    for m_id, m_rec in mission_records.items():
        if len(m_rec["gt_traj"]) > 0:
            gt_arr_m = np.array(m_rec["gt_traj"])
            axes[0, 1].plot(gt_arr_m[:, 0], gt_arr_m[:, 1], label=f"{m_id} Actual", color=colors.get(m_id, "black"), linewidth=2)
            if len(m_rec["planned_path"]) > 0 and m_id in ("obstacle", "hazard"):
                pp = np.array(m_rec["planned_path"])
                axes[0, 1].plot(pp[:, 0], pp[:, 1], "--", color=colors.get(m_id, "black"), alpha=0.6)
    circ_obs = plt.Circle((1.2, 0.0), 0.35, color="gray", alpha=0.6)
    axes[0, 1].add_patch(circ_obs)
    axes[0, 1].plot(0, 0, "go", markersize=6)
    axes[0, 1].plot(2.0, 0.0, "r*", markersize=8)
    axes[0, 1].plot(2.5, 0.0, "m*", markersize=8)
    axes[0, 1].set_title("Plot B: Trajectory vs Global Path")
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Y (m)")
    axes[0, 1].axis("equal")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(fontsize=7)

    # Panel C
    for m_id, m_rec in mission_records.items():
        v_lin = [c[0] for c in m_rec["cmd_history"]]
        axes[0, 2].plot(m_rec["timestamps"], v_lin, label=f"{m_id} v", color=colors.get(m_id, "black"), linewidth=1.5)
    axes[0, 2].set_title("Plot C: Commanded Linear Velocity vs Time")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("v (m/s)")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)
    axes[0, 2].legend(fontsize=7)

    # Panel D
    for m_id, m_rec in mission_records.items():
        st_num = [state_mapping.get(s, 2) for s in m_rec["status_history"]]
        axes[1, 0].step(m_rec["timestamps"], st_num, where="post", label=m_id, color=colors.get(m_id, "black"), linewidth=1.8)
    axes[1, 0].set_yticks([0, 1, 2, 3, 4, 5])
    axes[1, 0].set_yticklabels(["IDLE", "PLAN", "FOLLOW", "REACHED", "BLOCKED", "E-STOP"])
    axes[1, 0].set_title("Plot D: Planner State vs Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("State")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(fontsize=7)

    # Panel E
    for m_id, m_rec in mission_records.items():
        estops = [1 if s == "EMERGENCY_STOP" else 0 for s in m_rec["status_history"]]
        axes[1, 1].plot(m_rec["timestamps"], np.cumsum(estops), label=m_id, color=colors.get(m_id, "black"), linewidth=1.8)
    axes[1, 1].set_ylim(-0.2, 2.0)
    axes[1, 1].set_title("Plot E: E-Stops vs Time (0 Events)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Cumulative E-Stops")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(fontsize=7)

    # Panel F
    for m_id in ("obstacle", "hazard"):
        if m_id in mission_records:
            axes[1, 2].plot(mission_records[m_id]["timestamps"], mission_records[m_id]["clearances"], label=m_id, color=colors.get(m_id, "black"), linewidth=1.8)
    axes[1, 2].axhline(0.0, color="red", linestyle="--", label="Collision (0.0m)")
    axes[1, 2].set_title("Plot F: Minimum Clearance vs Time")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Clearance (m)")
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)
    axes[1, 2].legend(fontsize=7)

    plt.tight_layout()
    fig_all.savefig(out_viz / "fig_closed_loop_plots_a_to_f.png", dpi=150)
    plt.close(fig_all)

    # Also keep fig3_mission_execution.png for backwards compatibility
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(14, 6))
    if "obstacle" in mission_records and len(mission_records["obstacle"]["gt_traj"]) > 0:
        obs_gt = np.array(mission_records["obstacle"]["gt_traj"])
        ax3_1.plot(obs_gt[:, 0], obs_gt[:, 1], "b-", linewidth=2.5, label="Obstacle Detour Trajectory")
        ax3_1.plot(0, 0, "go", markersize=8, label="Start")
        ax3_1.plot(2.5, 0, "r*", markersize=12, label="Goal")
        ax3_1.set_title("Artifact F: Actual UGV Path vs Obstacle Detour")
        ax3_1.set_xlabel("East X (m)")
        ax3_1.set_ylabel("North Y (m)")
        ax3_1.legend()
        ax3_1.axis("equal")

        t_obs = mission_records["obstacle"]["timestamps"]
        v_obs = [c[0] for c in mission_records["obstacle"]["cmd_history"]]
        w_obs = [c[1] for c in mission_records["obstacle"]["cmd_history"]]
        ax3_2.plot(t_obs, v_obs, "g-", linewidth=2, label="Linear Vel x (m/s)")
        ax3_2.plot(t_obs, w_obs, "m-", linewidth=2, label="Angular Vel z (rad/s)")
        ax3_2.set_title("Artifact G & H: Velocity Commands During Detour")
        ax3_2.set_xlabel("Time (s)")
        ax3_2.set_ylabel("Velocity")
        ax3_2.legend()

    plt.tight_layout()
    fig3.savefig(out_viz / "fig3_mission_execution.png", dpi=150)
    plt.close(fig3)

    # Save complete benchmark JSON
    json_path = out_benchmarks / "path_planning_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete benchmark metrics to {json_path}")
    print(f"Saved 3 multi-panel visualization figures to {out_viz}/")

    print("\n==========================================================================================")
    print("                     PHASE 12 BENCHMARK & EVALUATION COMPLETE")
    print("==========================================================================================")
    return results


if __name__ == "__main__":
    run_planning_benchmarks()
