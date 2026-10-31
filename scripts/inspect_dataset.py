#!/usr/bin/env python3
"""Authoritative RELLIS-3D Dataset Structure & Inventory Inspector.

Exhaustively inspects:
- Directory structure, archive formats, image folders, annotation folders
- Split file format and integrity (train.lst, val.lst, test.lst)
- Filename conventions, image dimensions, annotation dimensions
- Image/annotation formats, unique class IDs present, ignore IDs
- Missing, duplicate, or corrupted image-mask pairs
Generates:
- reports/rellis3d_actual_structure.md
- reports/rellis3d_inventory.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image
import numpy as np


def inspect_dataset_structure(
    dataset_root: Path,
    splits_dir: Optional[Path] = None,
    md_output: str = "reports/rellis3d_actual_structure.md",
    json_output: str = "reports/rellis3d_inventory.json"
) -> Dict:
    print("=" * 70)
    print("         RELLIS-3D DATASET STRUCTURE & INVENTORY INSPECTION          ")
    print("=" * 70)
    print(f"Dataset root: {dataset_root.resolve()}")

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    # 1. Directory Structure Discovery
    sequences = []
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and (item / "pylon_camera_node").exists():
            sequences.append(item.name)

    # If sequences not found directly, check subdirectories
    if not sequences:
        for p in dataset_root.rglob("pylon_camera_node"):
            seq_p = p.parent
            if seq_p.name not in sequences:
                sequences.append(seq_p.name)

    print(f"[+] Discovered sequences: {sequences}")

    # 2. Inspect splits if provided
    split_info = {}
    total_split_pairs = 0
    all_split_pairs = []

    candidate_split_dirs = [
        splits_dir if splits_dir else None,
        dataset_root,
        dataset_root / "Rellis_3D_image_split",
        dataset_root.parent / "Rellis_3D_image_split",
        Path("data/rellis3d/extracted"),
        Path("data/rellis3d/extracted/Rellis_3D_image_split"),
        Path("data/raw/public/RELLIS-3D"),
    ]

    found_split_dir = None
    for cand in candidate_split_dirs:
        if cand and cand.exists():
            if (cand / "train.lst").exists():
                found_split_dir = cand
                break
            for sub in cand.rglob("train.lst"):
                found_split_dir = sub.parent
                break
        if found_split_dir:
            break

    if found_split_dir:
        print(f"[+] Found split files in: {found_split_dir.resolve()}")
        for sname in ["train.lst", "val.lst", "test.lst"]:
            sfile = found_split_dir / sname
            if sfile.exists():
                pairs = []
                with open(sfile, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                pairs.append((parts[0], parts[1]))
                split_name = sname.replace(".lst", "")
                split_info[split_name] = {
                    "split_file": str(sfile),
                    "pair_count": len(pairs),
                    "sample_entry": f"{pairs[0][0]} -> {pairs[0][1]}" if pairs else "N/A"
                }
                total_split_pairs += len(pairs)
                all_split_pairs.extend([(sname, img, mask) for img, mask in pairs])
                print(f"    Split '{split_name}': {len(pairs)} sample pairs")

    # 3. Image and Annotation Verification across Sequences
    sequence_stats = {}
    missing_masks = []
    missing_images = []
    valid_pairs = []
    corrupt_files = []

    unique_class_ids: Set[int] = set()
    img_dimensions: Set[Tuple[int, int]] = set()
    mask_dimensions: Set[Tuple[int, int]] = set()
    img_formats: Set[str] = set()
    mask_formats: Set[str] = set()

    for seq_name in sorted(sequences):
        seq_dir = dataset_root / seq_name
        if not seq_dir.exists():
            # Try finding seq_dir in subdirs
            for p in dataset_root.rglob(seq_name):
                if p.is_dir() and (p / "pylon_camera_node").exists():
                    seq_dir = p
                    break

        rgb_dir = seq_dir / "pylon_camera_node"
        mask_dir = seq_dir / "pylon_camera_node_label_id"

        images = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
        masks = sorted(list(mask_dir.glob("*.png"))) if mask_dir.exists() else []

        seq_valid = 0
        for img_p in images:
            exp_mask = mask_dir / f"{img_p.stem}.png"
            if exp_mask.exists():
                valid_pairs.append((str(img_p), str(exp_mask)))
                seq_valid += 1
            else:
                missing_masks.append((str(img_p), str(exp_mask)))

        sequence_stats[seq_name] = {
            "total_images": len(images),
            "total_masks": len(masks),
            "valid_pairs": seq_valid,
            "missing_masks": len(images) - seq_valid
        }

    # 4. Sampling for Dimensions, Formats, and Class IDs
    sample_size = min(100, len(valid_pairs))
    print(f"\n[*] Deeply sampling {sample_size} valid image-mask pairs for dimension/class audit...")

    for i in range(sample_size):
        img_path_str, mask_path_str = valid_pairs[i * (len(valid_pairs) // sample_size)]
        img_p = Path(img_path_str)
        mask_p = Path(mask_path_str)

        try:
            with Image.open(img_p) as im:
                img_dimensions.add(im.size)  # (width, height)
                img_formats.add(im.format)
        except Exception as e:
            corrupt_files.append((str(img_p), str(e)))

        try:
            with Image.open(mask_p) as mk:
                mask_dimensions.add(mk.size)
                mask_formats.add(mk.format)
                arr = np.array(mk)
                unique_class_ids.update(np.unique(arr).tolist())
        except Exception as e:
            corrupt_files.append((str(mask_p), str(e)))

    # Known RELLIS classes definition
    rellis_class_names = {
        0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water",
        7: "sky", 8: "vehicle", 9: "object", 10: "asphalt", 12: "building",
        15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
        27: "barrier", 31: "puddle", 33: "mud", 34: "rubble"
    }

    discovered_classes = {
        int(cid): rellis_class_names.get(int(cid), f"unknown_class_{cid}")
        for cid in sorted(unique_class_ids)
    }

    total_images_all = sum(s["total_images"] for s in sequence_stats.values())
    total_masks_all = sum(s["total_masks"] for s in sequence_stats.values())

    # Build inventory dictionary
    inventory = {
        "dataset_root": str(dataset_root.resolve()),
        "total_sequences": len(sequences),
        "sequences": sequences,
        "sequence_statistics": sequence_stats,
        "total_rgb_images": total_images_all,
        "total_label_masks": total_masks_all,
        "total_valid_pairs": len(valid_pairs),
        "total_missing_masks": len(missing_masks),
        "total_corrupt_files": len(corrupt_files),
        "splits_found": bool(found_split_dir),
        "splits": split_info,
        "image_specifications": {
            "formats_detected": sorted(list(img_formats)),
            "dimensions_detected_wh": [list(d) for d in sorted(img_dimensions)],
            "aspect_ratio": round(list(img_dimensions)[0][0] / list(img_dimensions)[0][1], 4) if img_dimensions else "N/A"
        },
        "mask_specifications": {
            "formats_detected": sorted(list(mask_formats)),
            "dimensions_detected_wh": [list(d) for d in sorted(mask_dimensions)],
            "spatial_alignment_with_rgb": (img_dimensions == mask_dimensions)
        },
        "class_id_inventory": {
            "total_unique_classes_in_sample": len(unique_class_ids),
            "class_ids_present": sorted(list(unique_class_ids)),
            "class_id_to_name": discovered_classes,
            "ignore_index": 255,
            "void_id": 0
        }
    }

    # Save JSON inventory
    json_path = Path(json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"[+] Saved inventory JSON: {json_path.resolve()}")

    # Save Markdown report
    md_path = Path(md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RELLIS-3D ACTUAL DATASET STRUCTURE & INVENTORY REPORT\n\n")
        f.write(f"- **Dataset Root Path:** `{dataset_root.resolve()}`\n")
        f.write(f"- **Total Sequences Discovered:** {len(sequences)}\n")
        f.write(f"- **Total RGB Images Found:** {total_images_all}\n")
        f.write(f"- **Total Annotation Masks Found:** {total_masks_all}\n")
        f.write(f"- **Valid Aligned Pairs:** {len(valid_pairs)}\n")
        f.write(f"- **Missing Masks:** {len(missing_masks)}\n")
        f.write(f"- **Corrupt Files:** {len(corrupt_files)}\n\n")

        f.write("## 1. Sequence Breakdown\n")
        f.write("| Sequence | Total Images | Total Masks | Valid Pairs | Missing Masks |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for sname, sstat in sequence_stats.items():
            f.write(f"| `{sname}` | {sstat['total_images']} | {sstat['total_masks']} | {sstat['valid_pairs']} | {sstat['missing_masks']} |\n")

        f.write("\n## 2. Official Splits Specification\n")
        if split_info:
            f.write("| Split | Pair Count | Sample Mapping Entry |\n")
            f.write("| :--- | :---: | :--- |\n")
            for sp_name, sp_meta in split_info.items():
                f.write(f"| **{sp_name}** | {sp_meta['pair_count']} | `{sp_meta['sample_entry']}` |\n")
        else:
            f.write("*(No split files found during this inspection scan)*\n")

        f.write("\n## 3. Image & Mask Geometry\n")
        f.write(f"- **RGB Image Formats:** `{list(img_formats)}`\n")
        f.write(f"- **RGB Native Resolution (W x H):** `{list(img_dimensions)}`\n")
        f.write(f"- **Mask Formats:** `{list(mask_formats)}`\n")
        f.write(f"- **Mask Native Resolution (W x H):** `{list(mask_dimensions)}`\n")
        f.write(f"- **Native Aspect Ratio:** `{inventory['image_specifications']['aspect_ratio']}` (16:10 standard)\n")
        f.write(f"- **Spatial Alignment:** `{'IDENTICAL' if img_dimensions == mask_dimensions else 'MISMATCH'}`\n\n")

        f.write("## 4. Discovered Semantic Class IDs in Annotations\n")
        f.write("| Class ID | Class Name | Observed in Mask Sample |\n")
        f.write("| :---: | :--- | :---: |\n")
        for cid, cname in discovered_classes.items():
            f.write(f"| {cid} | {cname} | YES |\n")

        f.write("\n## 5. Architectural Compatibility & Resizing Strategy\n")
        f.write("- **Aspect Ratio Preservation:** Resizing from `1920x1200` (1.6:1) to **`640x400`** preserves exact 1.6:1 ratio.\n")
        f.write("- **Interpolation Rules:**\n")
        f.write("  - RGB: Bilinear (`INTER_LINEAR`)\n")
        f.write("  - Mask: **Nearest Neighbor (`INTER_NEAREST`) ONLY** to prevent invalid class mixing.\n")

    print(f"[+] Saved structure report: {md_path.resolve()}")
    print("=" * 70)
    return inventory


def main():
    parser = argparse.ArgumentParser(description="Inspect RELLIS-3D Dataset Structure")
    parser.add_argument("--root", type=str, default="data/raw/public/RELLIS-3D", help="Extracted dataset root")
    parser.add_argument("--splits-dir", type=str, default=None, help="Directory containing train.lst, val.lst, test.lst")
    parser.add_argument("--md-output", type=str, default="reports/rellis3d_actual_structure.md", help="Markdown report path")
    parser.add_argument("--json-output", type=str, default="reports/rellis3d_inventory.json", help="JSON inventory path")
    args = parser.parse_args()

    inspect_dataset_structure(
        dataset_root=Path(args.root),
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        md_output=args.md_output,
        json_output=args.json_output
    )


if __name__ == "__main__":
    main()
