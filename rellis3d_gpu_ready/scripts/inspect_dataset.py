#!/usr/bin/env python3
"""RELLIS-3D Dataset Inventory and Integrity Inspector.

Scans the dataset directory, pairs RGB images with label masks,
checks spatial dimensions, and outputs an inventory summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

def inspect_rellis(dataset_root: Path) -> Dict:
    """Inspect sequences in RELLIS-3D root and pair RGB images with label ID masks."""
    sequences = [d for d in dataset_root.iterdir() if d.is_dir() and (d / "pylon_camera_node").exists()]
    
    total_images = 0
    total_labels = 0
    paired_samples = []
    missing_labels = []

    sequence_stats = {}

    for seq in sorted(sequences):
        rgb_dir = seq / "pylon_camera_node"
        label_dir = seq / "pylon_camera_node_label_id"

        images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
        labels = sorted(label_dir.glob("*.png")) if label_dir.exists() else []

        seq_images = len(images)
        seq_labels = len(labels)
        total_images += seq_images
        total_labels += seq_labels

        seq_pairs = 0
        for img in images:
            expected_mask = label_dir / f"{img.stem}.png"
            if expected_mask.exists():
                paired_samples.append((str(img), str(expected_mask)))
                seq_pairs += 1
            else:
                missing_labels.append(str(img))

        sequence_stats[seq.name] = {
            "images": seq_images,
            "labels": seq_labels,
            "paired": seq_pairs
        }

    # Inspect sample image resolution if available
    sample_shape = None
    if paired_samples:
        img_path, mask_path = paired_samples[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if img is not None and mask is not None:
            sample_shape = {
                "rgb_height": img.shape[0],
                "rgb_width": img.shape[1],
                "mask_height": mask.shape[0],
                "mask_width": mask.shape[1],
                "aligned": (img.shape[:2] == mask.shape[:2])
            }

    return {
        "dataset_root": str(dataset_root),
        "total_sequences": len(sequences),
        "total_rgb_images": total_images,
        "total_label_masks": total_labels,
        "valid_pairs_count": len(paired_samples),
        "missing_labels_count": len(missing_labels),
        "sequence_breakdown": sequence_stats,
        "sample_spatial_alignment": sample_shape
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/raw/public/RELLIS-3D", help="RELLIS-3D root folder")
    parser.add_argument("--output", type=str, default="reports/dataset_inventory.json", help="Output inventory JSON")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] RELLIS-3D root not found: {root}")
        return 1

    report = inspect_rellis(root)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 65)
    print("             RELLIS-3D DATASET INVENTORY SUMMARY                 ")
    print("=" * 65)
    print(f"Total Sequences  : {report['total_sequences']}")
    print(f"Total Images     : {report['total_rgb_images']}")
    print(f"Total Masks      : {report['total_label_masks']}")
    print(f"Valid Pairs      : {report['valid_pairs_count']}")
    print(f"Missing Masks    : {report['missing_labels_count']}")
    print(f"Report saved to  : {out_file}")
    print("=" * 65)

    return 0

if __name__ == "__main__":
    sys.exit(main())
