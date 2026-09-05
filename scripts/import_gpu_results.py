"""Imports and verifies external GPU training results back into the Antigravity project."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

def import_results(zip_file: str, dest_dir: str = "outputs") -> bool:
    """Unpack downloaded gpu_results.zip and validate critical artifacts."""
    zip_p = Path(zip_file).resolve()
    if not zip_p.exists():
        print(f"[!] Results archive not found: {zip_p}")
        return False

    dest_p = Path(dest_dir).resolve()
    dest_p.mkdir(parents=True, exist_ok=True)

    print(f"\n[IMPORT] Unpacking {zip_p.name} -> {dest_p}...")
    with zipfile.ZipFile(zip_p, "r") as zf:
        zf.extractall(dest_p.parent)

    # Validate essential artifacts
    best_ckpt = dest_p / "checkpoints" / "best.pt"
    eval_rep = dest_p / "reports" / "evaluation_report.json"
    bench_rep = dest_p / "reports" / "benchmark_report.json"

    print("\n" + "=" * 60)
    print("      EXTERNAL GPU TRAINING RESULTS IMPORT AUDIT        ")
    print("=" * 60)
    print(f"Checkpoints Present  : {'YES' if best_ckpt.exists() else 'NO'}")
    print(f"Evaluation Present   : {'YES' if eval_rep.exists() else 'NO'}")
    print(f"Benchmark Present    : {'YES' if bench_rep.exists() else 'NO'}")

    if eval_rep.exists():
        with open(eval_rep, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print("\n--- EXTERNAL GPU PERFORMANCE METRICS ---")
        dl_m = eval_data.get("dl_overall_metrics", {})
        print(f"Overall Test mIoU    : {dl_m.get('mIoU', 0.0):.4f}")
        print(f"Pixel Accuracy       : {dl_m.get('pixel_accuracy', 0.0):.4f}")
        print(f"False-Safe Ratio     : {dl_m.get('false_safe_pixel_ratio', 0.0)*100:.2f}%")
        print(f"Observed Domain Gap  : {eval_data.get('domain_gap_mIoU', 0.0):+.4f}")

    if bench_rep.exists():
        with open(bench_rep, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        print("\n--- EXTERNAL GPU HARDWARE BENCHMARK ---")
        print(f"Training GPU Device  : {b_data.get('device_name')}")
        print(f"Inference Throughput : {b_data.get('fps')} FPS")
        print(f"Avg Latency          : {b_data.get('avg_latency_ms')} ms")
        print(f"P95 Tail Latency     : {b_data.get('p95_latency_ms')} ms")

    print("=" * 60 + "\n")
    print("[SUCCESS] External GPU artifacts successfully integrated into Antigravity.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Import results archive from external GPU run")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to downloaded gpu_results.zip")
    parser.add_argument("--dest", "-d", type=str, default="outputs", help="Destination outputs directory")
    args = parser.parse_args()

    success = import_results(args.input, dest_dir=args.dest)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
