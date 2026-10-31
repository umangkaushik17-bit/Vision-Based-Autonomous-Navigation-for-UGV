"""Orchestrator Suite for Milestone 13.17.2 Execution on Kaggle Tesla T4.

Supports modular execution stages:
- stage1: Equivalent alternative analysis + replan parity validation
- benchmark25: 25-cycle benchmark on B01, B06, B07, B12
- benchmark100: 100-cycle benchmark on B01, B06, B07, B12
- benchmark250: 250-cycle benchmark on B01, B06, B07, B12
- initial: stage1 + benchmark25 (the mandatory starting package)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PYTHON_EXE = sys.executable


def run_command(cmd: str) -> bool:
    print(f"\n{'='*80}\nEXEC: {cmd}\n{'='*80}", flush=True)
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, shell=True)
    dur = time.perf_counter() - t0
    print(f"[Done in {dur:.1f}s] Exit Code: {ret.returncode}\n", flush=True)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Phase 13.17.2 Orchestrator Suite")
    parser.add_argument(
        "--stage",
        type=str,
        default="initial",
        choices=["stage1", "benchmark25", "benchmark100", "benchmark250", "initial", "all"],
        help="Execution target stage",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Target device ('cuda', 'cpu')")
    args = parser.parse_args()

    dev = args.device

    if args.stage in ("stage1", "initial", "all"):
        print("\n>>> [1/3] Running Equivalent Alternative Analysis (Part D)...", flush=True)
        ok1 = run_command(f"{PYTHON_EXE} scripts/analyze_equivalent_alternatives.py --device {dev}")
        if not ok1:
            print("[-] Error in analyze_equivalent_alternatives.py. Halting.")
            sys.exit(1)

        print("\n>>> [2/3] Running Global Replan Parity Validation (Part E)...", flush=True)
        ok2 = run_command(f"{PYTHON_EXE} scripts/validate_replan_parity.py --device {dev}")
        if not ok2:
            print("[-] Error in validate_replan_parity.py. Halting.")
            sys.exit(1)

    if args.stage in ("benchmark25", "initial", "all"):
        print("\n>>> [3/3] Running 25-Cycle Performance Benchmark (Part F/G/H)...", flush=True)
        ok3 = run_command(f"{PYTHON_EXE} scripts/run_phase17_benchmark.py --cycles 25 --device {dev}")
        if not ok3:
            print("[-] Error in run_phase17_benchmark.py (25 cycles). Halting.")
            sys.exit(1)

    if args.stage in ("benchmark100", "all"):
        print("\n>>> Running 100-Cycle Performance Benchmark (Part I)...", flush=True)
        ok4 = run_command(f"{PYTHON_EXE} scripts/run_phase17_benchmark.py --cycles 100 --device {dev}")
        if not ok4:
            print("[-] Error in run_phase17_benchmark.py (100 cycles). Halting.")
            sys.exit(1)

    if args.stage in ("benchmark250", "all"):
        print("\n>>> Running 250-Cycle Extended Benchmark (Part J/K/L)...", flush=True)
        ok5 = run_command(f"{PYTHON_EXE} scripts/run_phase17_benchmark.py --cycles 250 --device {dev}")
        if not ok5:
            print("[-] Error in run_phase17_benchmark.py (250 cycles). Halting.")
            sys.exit(1)

    print("\n[+] Stage execution completed successfully!", flush=True)


if __name__ == "__main__":
    main()
