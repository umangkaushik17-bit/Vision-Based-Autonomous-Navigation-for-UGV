"""CLI entry point for running deterministic closed-loop dynamic SIL scenarios.

Usage:
    python run_dynamic_sil.py --scenario scenario_a_static --seed 42
    python run_dynamic_sil.py --scenario scenario_c_head_on --output-dir simulation/outputs/dynamic_sil
    python run_dynamic_sil.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from simulation.runners.dynamic_sil_runner import DynamicSILRunner, ScenarioRunMetrics
from simulation.scenarios.dynamic.advanced_scenarios import ADVANCED_SCENARIO_BUILDERS
from simulation.scenarios.dynamic.core_scenarios import CORE_SCENARIO_BUILDERS
from simulation.scenarios.dynamic.data_model import DynamicScenarioConfig
from simulation.visualization.diagnostic_plotter import ScenarioDiagnosticPlotter

ALL_SCENARIO_BUILDERS = {**CORE_SCENARIO_BUILDERS, **ADVANCED_SCENARIO_BUILDERS}


def run_scenario(
    scenario_id_or_path: str,
    seed_override: Optional[int] = None,
    duration_override: Optional[float] = None,
    output_dir: str = "simulation/outputs/dynamic_sil",
    generate_viz: bool = True,
    save_csv: bool = False,
) -> ScenarioRunMetrics:
    """Execute a dynamic SIL scenario by ID or config path and export results."""
    # 1. Load scenario configuration
    if scenario_id_or_path in ALL_SCENARIO_BUILDERS:
        builder = ALL_SCENARIO_BUILDERS[scenario_id_or_path]
        seed = seed_override if seed_override is not None else 42
        cfg = builder(seed=seed)
    elif os.path.isfile(scenario_id_or_path):
        cfg = DynamicScenarioConfig.load_from_file(scenario_id_or_path)
        if seed_override is not None:
            cfg.random_seed = seed_override
    else:
        raise ValueError(
            f"Unknown scenario ID or file path '{scenario_id_or_path}'. "
            f"Available scenarios: {list(ALL_SCENARIO_BUILDERS.keys())}"
        )

    if duration_override is not None:
        cfg.max_duration_s = float(duration_override)

    print(f"\n============================================================")
    print(f"RUNNING SCENARIO: {cfg.name}")
    print(f"Scenario ID:      {cfg.scenario_id}")
    print(f"Random Seed:      {cfg.random_seed}")
    print(f"Max Duration:     {cfg.max_duration_s:.1f} s (dt = {cfg.control_dt_s:.2f} s)")
    print(f"Obstacle Count:   {len(cfg.obstacles)}")
    print(f"============================================================")

    # 2. Run simulation
    runner = DynamicSILRunner(cfg)
    t_start = time.perf_counter()
    metrics = runner.run()
    wall_duration = time.perf_counter() - t_start

    rtf = (metrics.total_time_s / wall_duration) if wall_duration > 0 else 0.0

    print(f"\nExecution Summary:")
    print(f"  Mission Outcome:     {metrics.mission_outcome} (Success: {metrics.mission_success})")
    print(f"  Safety Outcome:      {metrics.safety_outcome} (Collisions: {metrics.collision_count})")
    print(f"  Termination Reason:  {metrics.termination_reason} ({metrics.exit_reason})")
    print(f"  Goal Distance Error: {metrics.goal_position_error_m:.3f} m (Tolerance: {metrics.goal_tolerance_m:.3f} m)")
    print(f"  Min Clearance:       {metrics.min_clearance_m:.3f} m")
    print(f"  Travel Distance:     {metrics.travel_distance_m:.2f} m")
    print(f"  Sim Elapsed Time:    {metrics.actual_simulation_time_s:.2f} s / {metrics.configured_duration_s:.2f} s ({metrics.steps_executed} steps)")
    print(f"  Wall Time:           {wall_duration:.2f} s (Real-Time Factor: {rtf:.2f}x)")
    print(f"  Planner Cycles:      Local: {metrics.local_planner_cycles}, Global Replans: {metrics.global_replan_count} (Failed: {metrics.failed_global_replan_count})")
    print(f"  Behavior States:     PROCEED: {metrics.proceed_count}, AVOID: {metrics.avoid_count}, WAIT: {metrics.wait_count}, STOP: {metrics.stop_count}")
    if metrics.blocked_duration_s is not None:
        print(f"  Max Blocked Dur:     {metrics.blocked_duration_s:.2f} s")
    print(f"  Deadlock Count:      {metrics.deadlock_count}")

    # 3. Export results
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{cfg.scenario_id}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"  Metrics JSON saved:  {json_path}")

    if save_csv and metrics.telemetry:
        csv_path = os.path.join(output_dir, f"{cfg.scenario_id}_telemetry.csv")
        keys = [
            "step", "sim_time", "robot_x", "robot_y", "robot_yaw",
            "robot_v", "robot_w", "cmd_v", "cmd_w", "behavior_state",
            "min_clearance_m", "emergency_stop", "gt_collision",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics.telemetry)
        print(f"  Telemetry CSV saved: {csv_path}")

    # 4. Diagnostic Visualization
    if generate_viz:
        plot_path = os.path.join(output_dir, f"{cfg.scenario_id}.png")
        ScenarioDiagnosticPlotter.plot_scenario(
            scenario_id=cfg.scenario_id,
            telemetry=metrics.telemetry,
            motion_models=runner.motion_models,
            start_pose=cfg.start_pose,
            goal_pose=cfg.goal_pose,
            output_filepath=plot_path,
        )
        print(f"  Diagnostic Plot:     {plot_path}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13.13 Dynamic SIL Scenario Runner")
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_a_static",
        help="Scenario ID (e.g. scenario_a_static, scenario_b_parallel, scenario_c_head_on) or path to JSON",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--duration", type=float, default=None, help="Max simulation duration override (s)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation/outputs/dynamic_sil",
        help="Directory to save metrics, telemetry, and plots",
    )
    parser.add_argument("--no-viz", action="store_true", help="Disable diagnostic plot generation")
    parser.add_argument("--csv", action="store_true", help="Save telemetry CSV file")
    parser.add_argument("--list", action="store_true", help="List all available core scenarios and exit")

    args = parser.parse_args()

    if args.list:
        print("Available Scenarios (Phase 13.13 Core & Advanced):")
        for sc_id in ALL_SCENARIO_BUILDERS:
            cfg = ALL_SCENARIO_BUILDERS[sc_id]()
            print(f"  - {sc_id:44s} : {cfg.name}")
        return 0

    try:
        run_scenario(
            scenario_id_or_path=args.scenario,
            seed_override=args.seed,
            duration_override=args.duration,
            output_dir=args.output_dir,
            generate_viz=not args.no_viz,
            save_csv=args.csv,
        )
        return 0
    except Exception as e:
        print(f"Error executing scenario: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
