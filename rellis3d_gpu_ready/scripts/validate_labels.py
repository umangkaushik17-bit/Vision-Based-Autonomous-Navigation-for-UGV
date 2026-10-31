#!/usr/bin/env python3
"""RELLIS-3D Label Validation and Class Frequency Audit.

Validates pixel ID values in pylon_camera_node_label_id against the class mapping,
identifies unexpected class IDs, and computes class balance statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set
import cv2
import numpy as np
import yaml

def validate_labels(dataset_root: Path, mapping_cfg_path: Path, max_samples: int = 100) -> Dict:
    """Audit label IDs across sample sequences."""
    with open(mapping_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    known_classes = {int(k): v["name"] for k, v in cfg.get("harmonized_mapping", {}).items()}

    label_files = list(dataset_root.rglob("pylon_camera_node_label_id/*.png"))
    if not label_files:
        print("[ERROR] No label ID PNG files found under", dataset_root)
        return {"error": "no_labels_found"}

    # Subsample if large
    samples_to_check = label_files[:max_samples] if max_samples > 0 else label_files
    print(f"Auditing label pixel values across {len(samples_to_check)} sample masks...")

    class_pixel_counts: Dict[int, int] = {}
    unexpected_ids: Set[int] = set()

    for idx, mask_file in enumerate(samples_to_check):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        unique_ids, counts = np.unique(mask, return_counts=True)
        for uid, count in zip(unique_ids, counts):
            uid = int(uid)
            class_pixel_counts[uid] = class_pixel_counts.get(uid, 0) + int(count)
            if uid not in known_classes:
                unexpected_ids.add(uid)

    total_pixels = sum(class_pixel_counts.values())
    frequency_table = []
    for uid, count in sorted(class_pixel_counts.items(), key=lambda x: x[1], reverse=True):
        name = known_classes.get(uid, f"UNEXPECTED_{uid}")
        ratio = count / total_pixels if total_pixels > 0 else 0.0
        frequency_table.append({
            "class_id": uid,
            "class_name": name,
            "pixel_count": count,
            "percentage": round(ratio * 100, 3),
            "is_known": uid in known_classes
        })

    return {
        "audited_masks": len(samples_to_check),
        "total_pixels_counted": total_pixels,
        "unexpected_class_ids": list(unexpected_ids),
        "frequency_table": frequency_table
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/raw/public/RELLIS-3D", help="RELLIS-3D root")
    parser.add_argument("--mapping", type=str, default="configs/rellis_class_mapping.yaml", help="Class mapping YAML")
    parser.add_argument("--output-csv", type=str, default="reports/rellis_class_frequency.csv", help="Output frequency CSV")
    parser.add_argument("--max-samples", type=int, default=200, help="Max masks to audit")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] RELLIS-3D root does not exist: {root}")
        return 1

    audit_result = validate_labels(root, Path(args.mapping), max_samples=args.max_samples)
    if "error" in audit_result:
        return 1

    # Write CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "pixel_count", "percentage", "is_known"])
        writer.writeheader()
        writer.writerows(audit_result["frequency_table"])

    print("=" * 65)
    print("           RELLIS-3D LABEL AUDIT & CLASS FREQUENCY              ")
    print("=" * 65)
    print(f"Audited Masks    : {audit_result['audited_masks']}")
    print(f"Unexpected IDs   : {audit_result['unexpected_class_ids']}")
    print(f"Top 5 Classes by Frequency:")
    for row in audit_result["frequency_table"][:5]:
        print(f"  - [{row['class_id']}] {row['class_name']:<12}: {row['percentage']}%")
    print(f"Report saved to  : {out_csv}")
    print("=" * 65)

    return 0

if __name__ == "__main__":
    sys.exit(main())
