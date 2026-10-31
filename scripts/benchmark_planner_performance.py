"""Dedicated performance profiling and benchmark harness for Phase 13 navigation stack.

Measures CPU execution, sub-component latencies, memory footprint,
and candidate trajectory statistics on canonical scenarios B01, B06, B07, B12.

Outputs structured JSONs to simulation/outputs/performance/:
- cpu_baseline.json
- cpu_profile.json
- memory_profile.json
- performance_summary.json
"""

from __future__ import annotations

import argparse
import cProfile
import datetime
import gc
import json
import math
import os
import platform
import pstats
import sys
import time
import tracemalloc
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ai.planning.base import GoalPose, PlannerStatus, RobotState
from ai.planning.dynamic.behavioral_state import BehavioralState
from ai.planning.dynamic.collision_checker import CollisionStatus
from simulation.runners.phase16_sil_runner import Phase16SILRunner
from simulation.scenarios.dynamic.phase16_scenarios import (
    BENCHMARK_SCENARIO_REGISTRY,
    Phase16ScenarioConfig,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "simulation", "outputs", "performance")


def get_hardware_environment() -> Dict[str, Any]:
    """Capture detailed hardware, OS, and Python environment metadata."""
    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    
    # GPU check
    gpu_name = "N/A"
    gpu_vram_total_mb = 0.0
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            if len(parts) > 1:
                gpu_vram_total_mb = float(parts[1].strip())
    except Exception:
        pass

    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "cpu_processor": platform.processor() or platform.machine(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "gpu_present": gpu_name != "N/A",
        "gpu_name": gpu_name,
        "gpu_vram_total_mb": gpu_vram_total_mb,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class TimedSILRunner:
    """Instruments Phase16SILRunner methods with high-resolution timers without modifying source files."""

    def __init__(self, config: Phase16ScenarioConfig, backend: str = "cpu", preferred_device: str = "auto") -> None:
        self.runner = Phase16SILRunner(config=config, enable_replanner=True)
        self.backend = backend
        if backend == "gpu":
            import torch
            from ai.planning.dynamic_gpu.gpu_dwa import AcceleratedDWAPlanner
            dev = preferred_device if preferred_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.runner.dwa_planner = AcceleratedDWAPlanner(self.runner.dwa_planner.config, preferred_device=dev)

        self.subtimers_accumulator: Dict[str, List[float]] = {
            "perception_generator_ms": [],
            "tracking_kalman_ms": [],
            "motion_prediction_ms": [],
            "dynamic_occupancy_layer_ms": [],
            "global_replan_check_ms": [],
            "dwa_total_ms": [],
            "behavioral_arbitration_ms": [],
            "kinematics_step_ms": [],
            "other_step_overhead_ms": [],
        }
        self._wrap_methods()

    def _wrap_methods(self) -> None:
        # Wrap observation generator
        orig_gen = self.runner.observation_generator.generate_observations
        def timed_gen(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_gen(*args, **kwargs)
            self._current_step_subtimers["perception_generator_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.observation_generator.generate_observations = timed_gen

        # Wrap tracker
        orig_tracker = self.runner.tracker.update
        def timed_tracker(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_tracker(*args, **kwargs)
            self._current_step_subtimers["tracking_kalman_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.tracker.update = timed_tracker

        # Wrap predictor
        orig_pred = self.runner.predictor.predict_tracks
        def timed_pred(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_pred(*args, **kwargs)
            self._current_step_subtimers["motion_prediction_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.predictor.predict_tracks = timed_pred

        # Wrap occupancy layer
        orig_occ = self.runner.occupancy_layer.update_predictions
        def timed_occ(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_occ(*args, **kwargs)
            self._current_step_subtimers["dynamic_occupancy_layer_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.occupancy_layer.update_predictions = timed_occ

        # Wrap replanner
        orig_replan = self.runner.replanner.check_and_replan
        def timed_replan(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_replan(*args, **kwargs)
            self._current_step_subtimers["global_replan_check_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.replanner.check_and_replan = timed_replan

        # Wrap DWA
        orig_dwa = self.runner.dwa_planner.compute_trajectory
        def timed_dwa(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_dwa(*args, **kwargs)
            self._current_step_subtimers["dwa_total_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.dwa_planner.compute_trajectory = timed_dwa

        # Wrap behavior machine
        orig_bm = self.runner.behavior_machine.evaluate
        def timed_bm(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_bm(*args, **kwargs)
            self._current_step_subtimers["behavioral_arbitration_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.behavior_machine.evaluate = timed_bm

        # Wrap motion
        orig_motion = self.runner.motion.step
        def timed_motion(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_motion(*args, **kwargs)
            self._current_step_subtimers["kinematics_step_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        self.runner.motion.step = timed_motion

    def step(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self._current_step_subtimers = {
            "perception_generator_ms": 0.0,
            "tracking_kalman_ms": 0.0,
            "motion_prediction_ms": 0.0,
            "dynamic_occupancy_layer_ms": 0.0,
            "global_replan_check_ms": 0.0,
            "dwa_total_ms": 0.0,
            "behavioral_arbitration_ms": 0.0,
            "kinematics_step_ms": 0.0,
            "other_step_overhead_ms": 0.0,
        }
        t0 = time.perf_counter()
        step_rec = self.runner.step()
        total_step_ms = (time.perf_counter() - t0) * 1000.0
        
        accounted_ms = sum(self._current_step_subtimers.values())
        self._current_step_subtimers["other_step_overhead_ms"] = max(0.0, total_step_ms - accounted_ms)
        self._current_step_subtimers["total_step_ms"] = total_step_ms

        for k, v in self._current_step_subtimers.items():
            if k not in self.subtimers_accumulator:
                self.subtimers_accumulator[k] = []
            self.subtimers_accumulator[k].append(v)

        return step_rec, self._current_step_subtimers


def run_bounded_scenario_benchmark(
    scenario_id: str,
    cycles: int = 30,
    seed: int = 42,
    backend: str = "cpu",
    preferred_device: str = "auto",
    enable_cprofile: bool = False,
) -> Dict[str, Any]:
    """Execute a strictly bounded performance benchmark on a single scenario."""
    if scenario_id not in BENCHMARK_SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")

    config = BENCHMARK_SCENARIO_REGISTRY[scenario_id](seed)
    runner_wrapper = TimedSILRunner(config=config, backend=backend, preferred_device=preferred_device)

    proc = psutil.Process(os.getpid())
    gc.collect()

    tracemalloc.start()
    rss_before_bytes = proc.memory_info().rss
    rss_checkpoints: Dict[str, float] = {
        "rss_before_mb": round(rss_before_bytes / (1024 * 1024), 2)
    }

    step_latencies_ms: List[float] = []
    dwa_latencies_ms: List[float] = []

    profiler: Optional[cProfile.Profile] = None
    if enable_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    t_benchmark_start = time.perf_counter()

    for cycle_idx in range(1, cycles + 1):
        step_rec, subtimers = runner_wrapper.step()
        step_latencies_ms.append(subtimers["total_step_ms"])
        dwa_latencies_ms.append(subtimers.get("dwa_total_ms", 0.0))

        # Memory checkpoints
        if cycle_idx == 10:
            rss_checkpoints["rss_after_10_cycles_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
        elif cycle_idx == 20:
            rss_checkpoints["rss_after_20_cycles_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
        elif cycle_idx == cycles:
            rss_checkpoints["rss_after_final_cycle_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)

    t_benchmark_end = time.perf_counter()
    total_wall_clock_s = t_benchmark_end - t_benchmark_start

    if profiler is not None:
        profiler.disable()

    current_mem_bytes, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    rss_checkpoints["peak_rss_mb"] = peak_rss_mb
    rss_checkpoints["tracemalloc_current_mb"] = round(current_mem_bytes / (1024 * 1024), 2)
    rss_checkpoints["tracemalloc_peak_mb"] = round(peak_mem_bytes / (1024 * 1024), 2)

    # Compute latency statistics
    step_latencies_arr = np.array(step_latencies_ms)
    dwa_latencies_arr = np.array(dwa_latencies_ms)

    simulated_time_s = cycles * config.control_dt_s
    rtf = simulated_time_s / max(total_wall_clock_s, 1e-6)

    # Sub-component mean breakdown
    subtimers_mean: Dict[str, float] = {
        k: round(float(np.mean(v)), 2) for k, v in runner_wrapper.subtimers_accumulator.items()
    }
    subtimers_pct: Dict[str, float] = {
        k: round(float(np.mean(v)) / max(np.mean(step_latencies_arr), 1e-6) * 100.0, 2)
        for k, v in runner_wrapper.subtimers_accumulator.items()
    }

    # Candidate statistics from runner history
    valid_history = runner_wrapper.runner.dwa_valid_candidates_history
    rej_history = runner_wrapper.runner.dwa_rejected_candidates_history
    cand_summary: Dict[str, Any] = {
        "evaluated": 375,
        "valid": round(float(np.mean(valid_history)), 1) if valid_history else 0.0,
        "rejected": round(float(np.mean(rej_history)), 1) if rej_history else 0.0,
    }

    # cProfile Top Functions Parsing
    cprofile_top10_cumulative: List[Dict[str, Any]] = []
    cprofile_top10_call_count: List[Dict[str, Any]] = []
    cprofile_top10_per_call: List[Dict[str, Any]] = []

    if profiler is not None:
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        
        # Sort by cumulative time
        ps.strip_dirs().sort_stats("cumulative")
        for func_key in ps.fcn_list[:10]:
            cc, nc, tt, ct, callers = ps.stats[func_key]
            file_name, line_num, func_name = func_key
            cprofile_top10_cumulative.append({
                "function": f"{func_name} ({file_name}:{line_num})",
                "calls": nc,
                "total_time_s": round(tt, 4),
                "cumulative_time_s": round(ct, 4),
                "per_call_ms": round((ct / max(nc, 1)) * 1000.0, 3),
            })

        # Sort by call count
        ps.sort_stats("calls")
        for func_key in ps.fcn_list[:10]:
            cc, nc, tt, ct, callers = ps.stats[func_key]
            file_name, line_num, func_name = func_key
            cprofile_top10_call_count.append({
                "function": f"{func_name} ({file_name}:{line_num})",
                "calls": nc,
                "total_time_s": round(tt, 4),
                "cumulative_time_s": round(ct, 4),
            })

        # Sort by per-call time (total_time / ncalls)
        ps.sort_stats("time")
        for func_key in ps.fcn_list[:10]:
            cc, nc, tt, ct, callers = ps.stats[func_key]
            file_name, line_num, func_name = func_key
            cprofile_top10_per_call.append({
                "function": f"{func_name} ({file_name}:{line_num})",
                "calls": nc,
                "total_time_s": round(tt, 4),
                "per_call_ms": round((tt / max(nc, 1)) * 1000.0, 3),
            })

    return {
        "scenario_id": scenario_id,
        "scenario_name": config.name,
        "seed": seed,
        "cycles_executed": cycles,
        "total_wall_clock_s": round(total_wall_clock_s, 2),
        "simulated_time_s": round(simulated_time_s, 2),
        "real_time_factor": round(rtf, 4),
        "latencies": {
            "mean_ms": round(float(np.mean(step_latencies_arr)), 2),
            "p50_ms": round(float(np.percentile(step_latencies_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(step_latencies_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(step_latencies_arr, 99)), 2),
            "min_ms": round(float(np.min(step_latencies_arr)), 2),
            "max_ms": round(float(np.max(step_latencies_arr)), 2),
            "std_ms": round(float(np.std(step_latencies_arr)), 2),
        },
        "dwa_latencies": {
            "mean_ms": round(float(np.mean(dwa_latencies_arr)), 2),
            "p50_ms": round(float(np.percentile(dwa_latencies_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(dwa_latencies_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(dwa_latencies_arr, 99)), 2),
            "max_ms": round(float(np.max(dwa_latencies_arr)), 2),
        },
        "subtimers_mean_ms": subtimers_mean,
        "subtimers_pct_of_cycle": subtimers_pct,
        "candidate_evaluation_summary": cand_summary,
        "memory_checkpoints": rss_checkpoints,
        "cprofile_top10_cumulative": cprofile_top10_cumulative,
        "cprofile_top10_call_count": cprofile_top10_call_count,
        "cprofile_top10_per_call": cprofile_top10_per_call,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13 Performance Profiling & Baseline Benchmark")
    parser.add_argument("--scenario", default="all", choices=["all", "B01", "B06", "B07", "B12"],
                        help="Canonical scenario to benchmark (default: all)")
    parser.add_argument("--cycles", type=int, default=30,
                        help="Number of bounded simulation cycles per scenario (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic random seed (default: 42)")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                        help="Planner backend to benchmark (default: cpu)")
    parser.add_argument("--preferred-device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Preferred tensor device for accelerated planner (default: auto)")
    parser.add_argument("--profile", action="store_true", default=True,
                        help="Enable detailed cProfile call tree extraction (default: True)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory where performance JSONs will be saved")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env_info = get_hardware_environment()

    scenarios = ["B01", "B06", "B07", "B12"] if args.scenario == "all" else [args.scenario]

    print("=" * 70)
    print("PHASE 13 NAVIGATION STACK PERFORMANCE PROFILING HARNESS")
    print("=" * 70)
    print(f"Host: {env_info['hostname']} | OS: {env_info['os']}")
    print(f"CPU: {env_info['cpu_processor']} ({env_info['cpu_physical_cores']} physical, {env_info['cpu_logical_cores']} logical cores)")
    print(f"RAM: {env_info['total_ram_gb']} GB | Python: {env_info['python_version']} | NumPy: {env_info['numpy_version']}")
    print(f"Backend: {args.backend.upper()} (Device: {args.preferred_device}) | Scenarios: {scenarios} | Cycles per run: {args.cycles} | Seed: {args.seed}")
    print("=" * 70)

    results: Dict[str, Any] = {}

    for sc_id in scenarios:
        print(f"\n>>> Running bounded benchmark on {sc_id} ({args.cycles} cycles, seed {args.seed}, backend={args.backend})...")
        t_start = time.perf_counter()
        sc_res = run_bounded_scenario_benchmark(
            scenario_id=sc_id,
            cycles=args.cycles,
            seed=args.seed,
            backend=args.backend,
            preferred_device=args.preferred_device,
            enable_cprofile=args.profile,
        )
        elapsed = time.perf_counter() - t_start
        results[sc_id] = sc_res
        lat = sc_res["latencies"]
        dwa_lat = sc_res["dwa_latencies"]
        print(f"    Completed in {elapsed:.1f}s | Step Mean: {lat['mean_ms']}ms | P95: {lat['p95_ms']}ms | RTF: {sc_res['real_time_factor']}x")
        print(f"    DWA Mean: {dwa_lat['mean_ms']}ms ({sc_res['subtimers_pct_of_cycle'].get('dwa_total_ms', 0.0)}% of step)")
        print(f"    Peak RSS: {sc_res['memory_checkpoints']['peak_rss_mb']} MB")

    # Aggregate JSON artifacts
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    baseline_manifest = {
        "benchmark_type": f"{args.backend.upper()}_PERFORMANCE_BASELINE",
        "timestamp_utc": timestamp_str,
        "environment": env_info,
        "backend": args.backend,
        "preferred_device": args.preferred_device,
        "scenarios_evaluated": scenarios,
        "cycles_per_scenario": args.cycles,
        "seed": args.seed,
        "results": results,
    }

    # 1. <backend>_baseline.json
    baseline_filename = f"{args.backend}_baseline.json"
    baseline_path = os.path.join(args.output_dir, baseline_filename)
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if "results" in existing:
                for k, v in results.items():
                    existing["results"][k] = v
                existing["scenarios_evaluated"] = sorted(list(set(existing.get("scenarios_evaluated", []) + scenarios)))
                existing["timestamp_utc"] = timestamp_str
                baseline_manifest = existing
        except Exception:
            pass
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_manifest, f, indent=2)
    print(f"\n[Saved] {args.backend.upper()} baseline written to: {baseline_path}")

    # 2. <backend>_profile.json (cProfile call graphs and top 10 rankings)
    profile_manifest = {
        "timestamp_utc": timestamp_str,
        "environment": env_info,
        "backend": args.backend,
        "profiles": {
            sc_id: {
                "scenario_name": results[sc_id]["scenario_name"],
                "subtimers_mean_ms": results[sc_id]["subtimers_mean_ms"],
                "subtimers_pct_of_cycle": results[sc_id]["subtimers_pct_of_cycle"],
                "cprofile_top10_cumulative": results[sc_id]["cprofile_top10_cumulative"],
                "cprofile_top10_call_count": results[sc_id]["cprofile_top10_call_count"],
                "cprofile_top10_per_call": results[sc_id]["cprofile_top10_per_call"],
            }
            for sc_id in scenarios
        }
    }
    profile_filename = f"{args.backend}_profile.json"
    profile_path = os.path.join(args.output_dir, profile_filename)
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if "profiles" in existing:
                for k, v in profile_manifest["profiles"].items():
                    existing["profiles"][k] = v
                existing["timestamp_utc"] = timestamp_str
                profile_manifest = existing
        except Exception:
            pass
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_manifest, f, indent=2)
    print(f"[Saved] CPU profile written to: {profile_path}")

    # 3. memory_profile.json
    memory_manifest = {
        "timestamp_utc": timestamp_str,
        "environment": env_info,
        "memory_profiles": {
            sc_id: results[sc_id]["memory_checkpoints"]
            for sc_id in scenarios
        }
    }
    memory_path = os.path.join(args.output_dir, "memory_profile.json")
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if "memory_profiles" in existing:
                for k, v in memory_manifest["memory_profiles"].items():
                    existing["memory_profiles"][k] = v
                existing["timestamp_utc"] = timestamp_str
                memory_manifest = existing
        except Exception:
            pass
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory_manifest, f, indent=2)
    print(f"[Saved] Memory profile written to: {memory_path}")

    # 4. performance_summary.json
    summary_path = os.path.join(args.output_dir, "performance_summary.json")
    existing_summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
        except Exception:
            existing_summary = {}

    if "backends" not in existing_summary:
        existing_summary["backends"] = {}
    if "speedups" not in existing_summary:
        existing_summary["speedups"] = {}

    backend_summary = {
        sc_id: {
            "name": results[sc_id]["scenario_name"],
            "mean_step_latency_ms": results[sc_id]["latencies"]["mean_ms"],
            "p50_step_latency_ms": results[sc_id]["latencies"]["p50_ms"],
            "p95_step_latency_ms": results[sc_id]["latencies"]["p95_ms"],
            "p99_step_latency_ms": results[sc_id]["latencies"]["p99_ms"],
            "max_step_latency_ms": results[sc_id]["latencies"]["max_ms"],
            "mean_dwa_latency_ms": results[sc_id]["dwa_latencies"]["mean_ms"],
            "dwa_share_pct": results[sc_id]["subtimers_pct_of_cycle"].get("dwa_total_ms", 0.0),
            "rtf": results[sc_id]["real_time_factor"],
            "peak_rss_mb": results[sc_id]["memory_checkpoints"]["peak_rss_mb"],
            "candidates_evaluated": results[sc_id]["candidate_evaluation_summary"].get("evaluated", 375),
            "candidates_valid": results[sc_id]["candidate_evaluation_summary"].get("valid", 0),
            "candidates_rejected": results[sc_id]["candidate_evaluation_summary"].get("rejected", 0),
        }
        for sc_id in scenarios
    }
    existing_summary["backends"][args.backend] = backend_summary
    existing_summary["timestamp_utc"] = timestamp_str
    existing_summary["environment"] = env_info

    # Compute speedup factors if both cpu and gpu backends are recorded
    if "cpu" in existing_summary["backends"] and "gpu" in existing_summary["backends"]:
        cpu_b = existing_summary["backends"]["cpu"]
        gpu_b = existing_summary["backends"]["gpu"]
        for sc_id in set(cpu_b.keys()).intersection(gpu_b.keys()):
            c_dwa = cpu_b[sc_id]["mean_dwa_latency_ms"]
            g_dwa = gpu_b[sc_id]["mean_dwa_latency_ms"]
            c_step = cpu_b[sc_id]["mean_step_latency_ms"]
            g_step = gpu_b[sc_id]["mean_step_latency_ms"]
            existing_summary["speedups"][sc_id] = {
                "dwa_speedup_factor": round(c_dwa / max(g_dwa, 1e-3), 2),
                "e2e_step_speedup_factor": round(c_step / max(g_step, 1e-3), 2),
                "cpu_dwa_mean_ms": c_dwa,
                "gpu_dwa_mean_ms": g_dwa,
                "cpu_step_mean_ms": c_step,
                "gpu_step_mean_ms": g_step,
            }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing_summary, f, indent=2)
    print(f"[Saved] Performance summary written to: {summary_path}")

    print("\n" + "=" * 70)
    print("ALL BOUNDED BASELINE RUNS COMPLETED SUCCESSFULLY.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
