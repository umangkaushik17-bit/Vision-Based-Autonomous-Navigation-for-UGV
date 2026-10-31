#!/usr/bin/env python3
"""Authoritative Manifest Builder for RELLIS-3D.

Reads official split lists (train.lst, val.lst, test.lst), verifies image
and annotation mask existence, extracts dimensions and unique class IDs,
and writes the authoritative data/rellis3d/rellis_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np


def build_rellis_manifest(
    dataset_root: Path,
    splits_dir: Path,
    output_csv: Path = Path("data/rellis3d/rellis_manifest.csv"),
    sample_classes: bool = True
) -> Path:
    print("=" * 70)
    print("             RELLIS-3D AUTHORITATIVE MANIFEST BUILDER                ")
    print("=" * 70)
    print(f"Dataset root : {dataset_root.resolve()}")
    print(f"Splits dir   : {splits_dir.resolve()}")
    print(f"Output CSV   : {output_csv.resolve()}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": splits_dir / "train.lst",
        "val": splits_dir / "val.lst",
        "test": splits_dir / "test.lst"
    }

    # Verify split files
    for sname, spath in split_files.items():
        if not spath.exists():
            # Search in subdirectories
            found = list(splits_dir.rglob(f"{sname}.lst"))
            if found:
                split_files[sname] = found[0]
            else:
                print(f"[ERROR] Split file not found: {spath}")
                sys.exit(1)

    records = []
    missing_count = 0
    valid_count = 0

    fieldnames = [
        "image_path",
        "annotation_path",
        "split",
        "width",
        "height",
        "valid",
        "class_ids_present"
    ]

    for split_name, spath in split_files.items():
        print(f"\n[*] Processing split: {split_name} from {spath.name}...")
        with open(spath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 2:
                continue

            rel_img = parts[0]
            rel_mask = parts[1]

            # Resolve paths
            img_path = dataset_root / rel_img
            mask_path = dataset_root / rel_mask

            # Also check alternative root layouts (e.g. without leading seq name duplicate)
            if not img_path.exists():
                candidates = list(dataset_root.rglob(Path(rel_img).name))
                if candidates:
                    img_path = candidates[0]
            if not mask_path.exists():
                candidates = list(dataset_root.rglob(Path(rel_mask).name))
                if candidates:
                    mask_path = candidates[0]

            is_img_valid = img_path.exists()
            is_mask_valid = mask_path.exists()
            is_pair_valid = is_img_valid and is_mask_valid

            width = 0
            height = 0
            class_ids_str = ""

            if is_pair_valid:
                valid_count += 1
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                    # Only sample mask classes occasionally to keep generation fast
                    if sample_classes and (idx % 25 == 0 or idx < 10):
                        with Image.open(mask_path) as mk:
                            arr = np.array(mk)
                            cids = sorted(np.unique(arr).tolist())
                            class_ids_str = ";".join(str(c) for c in cids)
                except Exception as e:
                    is_pair_valid = False
            else:
                missing_count += 1

            records.append({
                "image_path": str(img_path.resolve()) if is_img_valid else rel_img,
                "annotation_path": str(mask_path.resolve()) if is_mask_valid else rel_mask,
                "split": split_name,
                "width": width if is_pair_valid else "N/A",
                "height": height if is_pair_valid else "N/A",
                "valid": is_pair_valid,
                "class_ids_present": class_ids_str
            })

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 70)
    print(f"Manifest written to : {output_csv.resolve()}")
    print(f"Total entries       : {len(records)}")
    print(f"Valid pairs         : {valid_count}")
    print(f"Missing pairs       : {missing_count}")
    print("=" * 70)

    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Generate RELLIS-3D Dataset Manifest")
    parser.add_argument("--root", type=str, default="data/raw/public/RELLIS-3D", help="Extracted dataset root")
    parser.add_argument("--splits-dir", type=str, default="data/raw/public/RELLIS-3D", help="Path containing split lists")
    parser.add_argument("--output", type=str, default="data/rellis3d/rellis_manifest.csv", help="Output manifest CSV path")
    args = parser.parse_args()

    build_rellis_manifest(
        dataset_root=Path(args.root),
        splits_dir=Path(args.splits_dir),
        output_csv=Path(args.output)
    )


if __name__ == "__main__":
    main()
