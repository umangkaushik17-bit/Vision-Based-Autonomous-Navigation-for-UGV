"""Batch runner to execute representative Advanced Scenarios F through O and export artifacts."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from run_dynamic_sil import run_scenario

ADVANCED_SCENARIOS = [
    ("scenario_f_temporary_blockage", 1.0),
    ("scenario_g1_persistent_blockage_alternate", 1.0),
    ("scenario_g2_persistent_blockage_no_route", 0.8),
    ("scenario_h_multiple_obstacles", 0.8),
    ("scenario_i1_noise_low", 0.8),
    ("scenario_i2_noise_med", 0.8),
    ("scenario_i3_noise_high", 0.8),
    ("scenario_j1_dropout_short", 0.8),
    ("scenario_j2_dropout_medium", 0.8),
    ("scenario_j3_dropout_intermittent", 0.8),
    ("scenario_k_stale_data", 0.8),
    ("scenario_l_unknown_horizon", 0.8),
    ("scenario_m_mixed_quality_tracks", 0.8),
    ("scenario_n1_narrow_clear", 0.8),
    ("scenario_n2_narrow_marginal", 0.8),
    ("scenario_n3_narrow_blocked", 0.8),
    ("scenario_o_deadlock", 0.8),
]


def main():
    print("============================================================")
    print("M13.13.7 BATCH EXECUTION OF ADVANCED SCENARIOS F THROUGH O")
    print("============================================================")

    results = []
    t0 = time.perf_counter()

    for sc_id, duration in ADVANCED_SCENARIOS:
        print(f"\n---> Executing {sc_id} (duration={duration}s)")
        try:
            m = run_scenario(
                scenario_id_or_path=sc_id,
                duration_override=duration,
                output_dir="simulation/outputs/dynamic_sil",
                generate_viz=True,
                save_csv=True,
            )
            results.append((sc_id, "PASS", m.collision_count, m.min_clearance_m, m.exit_reason))
        except Exception as e:
            print(f"FAILED {sc_id}: {e}")
            results.append((sc_id, "FAIL", -1, -1.0, str(e)))

    total_wall = time.perf_counter() - t0
    print("\n============================================================")
    print(f"ADVANCED SCENARIOS BATCH SUMMARY (Total time: {total_wall:.1f}s)")
    print("============================================================")
    print(f"{'Scenario':44s} | {'Result':6s} | {'Collisions':10s} | {'MinClear(m)':11s} | {'Exit Reason'}")
    print("-" * 95)
    for sc_id, res, coll, clr, reason in results:
        print(f"{sc_id:44s} | {res:6s} | {coll:10d} | {clr:11.3f} | {reason}")


if __name__ == "__main__":
    main()
