"""Regenerate authoritative Phase 13.13 dynamic SIL scenario evidence.

Executes representative scenarios under the repaired canonical metric schema
and exports JSON, telemetry CSV, and diagnostic plots to simulation/outputs/dynamic_sil/.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from run_dynamic_sil import run_scenario

AUTHORITATIVE_SCENARIOS = [
    # scenario_id, duration_s, description
    ("scenario_a_static", 10.0, "Full Goal-Arrival Static Baseline"),
    ("scenario_g1_persistent_blockage_alternate", 4.0, "Persistent Blockage Alternate Route Analysis"),
    ("scenario_g2_persistent_blockage_no_route", 4.0, "Persistent Blockage Walled Corridor Analysis"),
    ("scenario_n3_narrow_blocked", 4.0, "Narrow Passage Unconstrained Corridor Analysis"),
    ("scenario_o_deadlock", 5.5, "Local Deadlock Box Trap Safe Hold Analysis"),
]


def main():
    print("============================================================")
    print("REGENERATING AUTHORITATIVE PHASE 13.13 SIL EVIDENCE")
    print("============================================================")

    output_dir = "simulation/outputs/dynamic_sil"
    os.makedirs(output_dir, exist_ok=True)

    summary_records = []
    t_batch_start = time.perf_counter()

    for sc_id, duration, desc in AUTHORITATIVE_SCENARIOS:
        print(f"\n============================================================")
        print(f"RUNNING: {sc_id} ({desc})")
        print(f"Configured Horizon: {duration:.1f} s")
        print(f"============================================================")

        t0 = time.perf_counter()
        metrics = run_scenario(
            scenario_id_or_path=sc_id,
            duration_override=duration,
            output_dir=output_dir,
            generate_viz=True,
            save_csv=True,
        )
        wall_dur = time.perf_counter() - t0

        rec = {
            "scenario_id": sc_id,
            "description": desc,
            "configured_duration_s": duration,
            "actual_sim_time_s": metrics.actual_simulation_time_s,
            "wall_clock_s": wall_dur,
            "exit_reason": metrics.exit_reason,
            "mission_outcome": metrics.mission_outcome,
            "safety_outcome": metrics.safety_outcome,
            "mission_success": metrics.mission_success,
            "goal_error_m": metrics.goal_position_error_m,
            "collisions": metrics.collision_count,
            "local_cycles": metrics.local_planner_cycles,
            "global_replans": metrics.global_replan_count,
            "proceed_count": metrics.proceed_count,
            "avoid_count": metrics.avoid_count,
            "wait_count": metrics.wait_count,
            "stop_count": metrics.stop_count,
            "deadlock_count": metrics.deadlock_count,
            "blocked_duration_s": metrics.blocked_duration_s,
        }
        summary_records.append(rec)

    total_batch_wall = time.perf_counter() - t_batch_start
    print("\n=========================================================================================")
    print(f"AUTHORITATIVE REGENERATION COMPLETE (Total wall time: {total_batch_wall:.1f} s)")
    print("=========================================================================================")
    print(f"{'Scenario':42s} | {'Sim(s)':6s} | {'Wall(s)':7s} | {'Mission':12s} | {'Safety':9s} | {'Exit Reason':22s} | {'GoalErr':7s} | {'Coll':4s} | {'Local':5s} | {'Replan':6s}")
    print("-" * 135)
    for r in summary_records:
        print(
            f"{r['scenario_id']:42s} | {r['actual_sim_time_s']:6.2f} | {r['wall_clock_s']:7.1f} | "
            f"{r['mission_outcome']:12s} | {r['safety_outcome']:9s} | {r['exit_reason']:22s} | "
            f"{r['goal_error_m']:7.3f} | {r['collisions']:4d} | {r['local_cycles']:5d} | {r['global_replans']:6d}"
        )


if __name__ == "__main__":
    main()
