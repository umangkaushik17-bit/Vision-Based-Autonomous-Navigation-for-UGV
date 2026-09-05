"""CLI tool to audit dataset balance, class pixel distribution, and zero-leakage verification."""

import argparse
import sys
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.audit import DatasetAuditor

def main():
    parser = argparse.ArgumentParser(description="Audit UGV dataset health, balance, and sequence-level split integrity")
    parser.add_argument("--manifest", "-m", type=str, default="data/manifests/dataset_manifest.json", help="Path to manifest.json")
    parser.add_argument("--max-pixel-samples", type=int, default=30, help="Max masks to inspect for pixel coverage")
    args = parser.parse_args()

    auditor = DatasetAuditor(args.manifest)
    report = auditor.audit(max_pixel_samples=args.max_pixel_samples)

    print("\n" + report.summary_string() + "\n")

    if not report.zero_leakage_verified:
        print("[AUDIT ERROR] Sequence data leakage was detected! Resolve before training.")
        sys.exit(1)
    else:
        print("[AUDIT SUCCESS] Dataset passes all zero-leakage and format verification checks.")

if __name__ == "__main__":
    main()
