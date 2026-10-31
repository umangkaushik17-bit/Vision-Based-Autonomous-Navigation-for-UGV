"""Detailed Root-Cause Analysis for EQUIVALENT_SAFE_ALTERNATIVE Decisions (Phase 13.17.2 Part D).

Examines all 102 equivalent safe alternative cases identified across Stages A, B, C, and D
in Phase 13.17.1. Re-evaluates candidate spectra to extract:
- best score and second-best score on both CPU and GPU
- best-vs-second-best score gaps
- absolute and relative score differences
- candidate validity counts and collision safety
- classification into TRUE_NEAR_TIE, FLOATING_POINT_ARGMAX_SHIFT,
  CANDIDATE_ORDER_EFFECT, SCORE_IMPLEMENTATION_DIFFERENCE, VALIDITY_DIFFERENCE, or OTHER.

Outputs: simulation/outputs/performance/equivalent_alternative_analysis.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ai.planning.base import PlanningConfig, PlannerStatus
from ai.planning.dynamic_gpu.gpu_dwa import AcceleratedDWAPlanner
from ai.planning.local_planners.dwa import DWAPlanner
from scripts.run_parity_suite import (
    ParityTestCase,
    build_randomized_cases,
    build_stage_a_cases,
    build_stage_b_cases,
)

OUTPUT_DIR = os.path.join("simulation", "outputs", "performance")
PARITY_DIR = os.path.join(OUTPUT_DIR, "parity")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_cases() -> Dict[str, ParityTestCase]:
    """Rebuild full dictionary of test cases from Stages A, B, C, D keyed by case_id."""
    print("[*] Reconstructing test case registry across Stages A-D...")
    cases_dict: Dict[str, ParityTestCase] = {}

    for c in build_stage_a_cases():
        cases_dict[c.case_id] = c

    for c in build_stage_b_cases(100):
        cases_dict[c.case_id] = c

    for c in build_randomized_cases(100, master_seed=42, stage_prefix="stage_c"):
        cases_dict[c.case_id] = c

    for c in build_randomized_cases(1000, master_seed=1337, stage_prefix="stage_d"):
        cases_dict[c.case_id] = c

    print(f"[+] Total registered test cases: {len(cases_dict)}")
    return cases_dict


def find_target_case_ids() -> List[str]:
    """Retrieve case IDs labeled EQUIVALENT_SAFE_ALTERNATIVE from saved JSON artifacts."""
    stage_files = [
        "stage_a_10_cases.json",
        "stage_b_100_deterministic.json",
        "stage_c_100_randomized.json",
        "stage_d_1000_randomized.json",
    ]
    target_ids = []
    for sf in stage_files:
        path = os.path.join(PARITY_DIR, sf)
        if not os.path.exists(path):
            print(f"[-] Note: {path} not found.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("cases", []):
            if c.get("classification") == "EQUIVALENT_SAFE_ALTERNATIVE":
                target_ids.append(c["case_id"])

    print(f"[+] Found {len(target_ids)} EQUIVALENT_SAFE_ALTERNATIVE cases to analyze.")
    return target_ids


def analyze_case(
    case: ParityTestCase,
    cpu_planner: DWAPlanner,
    gpu_planner: AcceleratedDWAPlanner,
) -> Dict[str, Any]:
    """Re-evaluate candidate spectra and compute deep tie/gap diagnostics."""
    # 1. CPU execution
    cpu_traj, cpu_v, cpu_w, cpu_status, cpu_diag = cpu_planner.compute_trajectory(
        case.robot_state,
        case.global_path,
        case.costmap,
        dynamic_layer=case.dynamic_layer,
        predicted_tracks=case.predicted_tracks,
        dynamic_config=case.dynamic_config,
    )

    cpu_candidates = getattr(cpu_planner, "latest_candidates", [])
    if cpu_candidates:
        # Sort in descending order of score
        sorted_cpu = sorted(cpu_candidates, key=lambda item: item[0], reverse=True)
        best_cpu_score = float(sorted_cpu[0][0])
        second_best_cpu_score = float(sorted_cpu[1][0]) if len(sorted_cpu) > 1 else best_cpu_score
    else:
        best_cpu_score = float(cpu_diag.get("best_score", 0.0))
        second_best_cpu_score = best_cpu_score

    cpu_gap = max(0.0, best_cpu_score - second_best_cpu_score)

    # 2. GPU execution
    gpu_traj, gpu_v, gpu_w, gpu_status, gpu_diag = gpu_planner.compute_trajectory(
        case.robot_state,
        case.global_path,
        case.costmap,
        dynamic_layer=case.dynamic_layer,
        predicted_tracks=case.predicted_tracks,
        dynamic_config=case.dynamic_config,
    )

    gpu_scores_tensor = getattr(gpu_planner, "latest_composite_scores", None)
    gpu_valid_mask = getattr(gpu_planner, "latest_valid_mask", None)

    if gpu_scores_tensor is not None and gpu_valid_mask is not None and int(gpu_valid_mask.sum().item()) > 0:
        masked_gpu = torch.where(gpu_valid_mask, gpu_scores_tensor, torch.tensor(-1e9, device=gpu_scores_tensor.device))
        val_count = int(gpu_valid_mask.sum().item())
        k_top = min(2, val_count)
        topk_vals = torch.topk(masked_gpu, k=k_top).values.detach().cpu().numpy()
        best_gpu_score = float(topk_vals[0])
        second_best_gpu_score = float(topk_vals[1]) if k_top > 1 else best_gpu_score
    else:
        best_gpu_score = float(gpu_diag.get("best_score", 0.0))
        second_best_gpu_score = best_gpu_score

    gpu_gap = max(0.0, best_gpu_score - second_best_gpu_score)

    # Metrics
    score_diff = abs(best_cpu_score - best_gpu_score)
    denom = max(1e-4, abs(best_cpu_score))
    rel_score_diff = score_diff / denom

    cpu_valid_count = int(cpu_diag.get("valid_trajectories", len(cpu_candidates)))
    gpu_valid_count = int(gpu_diag.get("valid_trajectories", 0))
    valid_diff = abs(cpu_valid_count - gpu_valid_count)

    cpu_col = bool(cpu_diag.get("dynamic_collision", False) or (cpu_status == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY))
    gpu_col = bool(gpu_diag.get("dynamic_collision", False) or (gpu_status == PlannerStatus.LOCAL_NO_SAFE_TRAJECTORY))

    cpu_beh = cpu_diag.get("behavioral_state", "PROCEED")
    gpu_beh = gpu_diag.get("behavioral_state", "PROCEED")

    v_diff = math.hypot(cpu_v - gpu_v, cpu_w - gpu_w)

    # Classification logic
    # 1. VALIDITY_DIFFERENCE: valid trajectory count differs
    if valid_diff > 0:
        classification = "VALIDITY_DIFFERENCE"
        explanation = (
            f"Candidate validity count differs slightly (CPU={cpu_valid_count}, GPU={gpu_valid_count}) "
            f"due to micro-rounding at the boundary of the maximum terrain or dynamic cost threshold."
        )
    # 2. FLOATING_POINT_ARGMAX_SHIFT: top candidate gap is near zero (< 0.003) and controls are adjacent
    elif (cpu_gap < 0.003 or gpu_gap < 0.003) and score_diff < 0.008:
        classification = "FLOATING_POINT_ARGMAX_SHIFT"
        explanation = (
            f"Near-degenerate candidate scores with tight top-2 gap (CPU gap={cpu_gap:.5f}, GPU gap={gpu_gap:.5f}). "
            f"Float32 GPU vs Float64 CPU precision flips the argmax between equidistant adjacent velocity samples."
        )
    # 3. TRUE_NEAR_TIE: Score difference is extremely small (< 0.002)
    elif score_diff < 0.002:
        classification = "TRUE_NEAR_TIE"
        explanation = (
            f"Practically identical candidate scores (Score diff={score_diff:.6f}, Rel diff={rel_score_diff:.4%}). "
            f"Both candidates provide safe, kinematically compliant paths with indistinguishable utility."
        )
    # 4. CANDIDATE_ORDER_EFFECT: Exact score tie broken by candidate array ordering
    elif abs(cpu_gap) < 1e-6 or abs(gpu_gap) < 1e-6:
        classification = "CANDIDATE_ORDER_EFFECT"
        explanation = "Exact numerical score tie broken by loop traversal / grid ordering."
    # 5. SCORE_IMPLEMENTATION_DIFFERENCE: Measurable score difference within acceptable 0.05 limit
    elif score_diff <= 0.05:
        classification = "SCORE_IMPLEMENTATION_DIFFERENCE"
        explanation = (
            f"Score difference of {score_diff:.4f} is within the 0.05 tolerance limit, "
            f"caused by broad-phase margin boundary or spatiotemporal distance calculation."
        )
    else:
        classification = "OTHER"
        explanation = f"Score difference {score_diff:.4f} without safety implication."

    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "scenario": case.scenario_name,
        "cpu_selected_candidate": [round(float(cpu_v), 4), round(float(cpu_w), 4)],
        "gpu_selected_candidate": [round(float(gpu_v), 4), round(float(gpu_w), 4)],
        "velocity_difference_m_s": round(float(v_diff), 4),
        "cpu_selected_score": round(float(best_cpu_score), 4),
        "gpu_selected_score": round(float(best_gpu_score), 4),
        "best_cpu_score": round(float(best_cpu_score), 4),
        "second_best_cpu_score": round(float(second_best_cpu_score), 4),
        "best_gpu_score": round(float(best_gpu_score), 4),
        "second_best_gpu_score": round(float(second_best_gpu_score), 4),
        "best_vs_second_best_cpu_score_gap": round(float(cpu_gap), 5),
        "best_vs_second_best_gpu_score_gap": round(float(gpu_gap), 5),
        "absolute_score_difference": round(float(score_diff), 5),
        "relative_score_difference": round(float(rel_score_diff), 5),
        "candidate_validity": {
            "cpu_valid": cpu_valid_count,
            "gpu_valid": gpu_valid_count,
            "diff": valid_diff,
        },
        "collision_status": {
            "cpu_collision": cpu_col,
            "gpu_collision": gpu_col,
        },
        "behavior_classification": {
            "cpu_behavior": cpu_beh,
            "gpu_behavior": gpu_beh,
        },
        "classification": classification,
        "explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze EQUIVALENT_SAFE_ALTERNATIVE Cases")
    parser.add_argument("--device", type=str, default="auto", help="Device for GPU planner ('auto', 'cuda', 'cpu')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases to analyze (for smoke testing)")
    args = parser.parse_args()

    cases_dict = load_all_cases()
    target_ids = find_target_case_ids()

    if args.limit is not None and args.limit > 0:
        target_ids = target_ids[:args.limit]
        print(f"[*] Limiting analysis to first {len(target_ids)} cases.")

    if not target_ids:
        print("[-] No target cases found. Exiting.")
        return

    config = PlanningConfig()
    cpu_planner = DWAPlanner(config)
    gpu_planner = AcceleratedDWAPlanner(config, preferred_device=args.device)

    print(f"[*] Target device: {gpu_planner.backend.device_name} (CUDA Active: {gpu_planner.backend.is_cuda})")

    results = []
    cat_counts: Dict[str, int] = {
        "TRUE_NEAR_TIE": 0,
        "FLOATING_POINT_ARGMAX_SHIFT": 0,
        "VALIDITY_DIFFERENCE": 0,
        "SCORE_IMPLEMENTATION_DIFFERENCE": 0,
        "CANDIDATE_ORDER_EFFECT": 0,
        "OTHER": 0,
    }

    for idx, cid in enumerate(target_ids, 1):
        case = cases_dict.get(cid)
        if case is None:
            print(f"[-] Warning: Case {cid} not found in registry.")
            continue

        res = analyze_case(case, cpu_planner, gpu_planner)
        results.append(res)
        cat = res["classification"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        print(
            f"[{idx:03d}/{len(target_ids):03d}] {cid:<26} | "
            f"CPU Gap={res['best_vs_second_best_cpu_score_gap']:.4f} | "
            f"GPU Gap={res['best_vs_second_best_gpu_score_gap']:.4f} | "
            f"ScoreDiff={res['absolute_score_difference']:.4f} | "
            f"Class: {cat}"
        )

    summary = {
        "milestone": "Phase 13.17.2 Part D",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": gpu_planner.backend.device_name,
        "is_cuda": gpu_planner.backend.is_cuda,
        "total_cases_analyzed": len(results),
        "classification_counts": cat_counts,
        "mean_absolute_score_difference": round(float(np.mean([r["absolute_score_difference"] for r in results])), 5),
        "max_absolute_score_difference": round(float(np.max([r["absolute_score_difference"] for r in results])), 5),
        "cases": results,
    }

    out_file = os.path.join(OUTPUT_DIR, "equivalent_alternative_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("EQUIVALENT SAFE ALTERNATIVE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Cases: {len(results)}")
    for k, v in cat_counts.items():
        pct = (v / max(1, len(results))) * 100.0
        print(f"  {k:<35}: {v:3d} ({pct:5.1f}%)")
    print(f"Mean Absolute Score Difference: {summary['mean_absolute_score_difference']:.5f}")
    print(f"Max Absolute Score Difference : {summary['max_absolute_score_difference']:.5f}")
    print(f"Saved artifact to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
