"""Pre-training dataset integrity and zero-leakage validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.base import TraversabilityTargetClass
from ai.datasets.split_manager import SplitManager
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("validate_dataset")

def validate_dataset(manifest_path: str, max_check: int = 100) -> Tuple[bool, str]:
    """Perform rigorous pre-training integrity audit."""
    manager = SplitManager(manifest_path)
    try:
        samples = manager.load_manifest()
    except Exception as e:
        return False, f"Manifest load failure: {e}"

    # 1. Zero-Leakage Proof
    leak_ok, leak_msg = manager.validate_zero_leakage()
    if not leak_ok:
        return False, leak_msg

    # 2. File Integrity & Dimension Consistency
    missing_images = 0
    corrupt_images = 0
    dimension_mismatches = 0
    invalid_classes = 0

    checked = 0
    for s in samples[:max_check]:
        img_p = Path(s.image_path)
        if not img_p.exists():
            missing_images += 1
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            corrupt_images += 1
            continue

        if s.mask_path and Path(s.mask_path).exists():
            mask = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                corrupt_images += 1
                continue

            if (img.shape[0], img.shape[1]) != (mask.shape[0], mask.shape[1]):
                dimension_mismatches += 1

            unique_classes = np.unique(mask)
            for c in unique_classes:
                if c not in (0, 1, 2, 255):
                    invalid_classes += 1

        checked += 1

    report_lines = [
        "================ PRE-TRAINING DATASET INTEGRITY AUDIT ================",
        f"Manifest Path       : {manifest_path}",
        f"Total Samples Indexed: {len(samples)}",
        f"Samples Checked     : {checked}",
        f"Zero-Leakage Status : {'PASS (Disjoint Sequences)' if leak_ok else 'FAIL (LEAK)'}",
        f"Missing Images      : {missing_images}",
        f"Corrupt Files       : {corrupt_images}",
        f"Dimension Mismatches: {dimension_mismatches}",
        f"Invalid Class IDs   : {invalid_classes}",
        "======================================================================="
    ]
    report_str = "\n".join(report_lines)
    print(report_str)

    if missing_images > 0 or corrupt_images > 0 or dimension_mismatches > 0 or invalid_classes > 0 or not leak_ok:
        return False, "Integrity validation FAILED. Correct issues before launching training."
    return True, "Integrity validation PASSED."

def main():
    parser = argparse.ArgumentParser(description="Validate dataset integrity and anti-leakage governance")
    parser.add_argument("--manifest", "-m", type=str, default="data/manifests/dataset_manifest.json")
    parser.add_argument("--max-check", type=int, default=150)
    args = parser.parse_args()

    passed, msg = validate_dataset(args.manifest, max_check=args.max_check)
    print(msg)
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
