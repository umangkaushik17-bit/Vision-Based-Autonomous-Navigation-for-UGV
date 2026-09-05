"""CLI utility to import and convert local RUGD DatasetNinja tar/zip archives directly into the project."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tarfile
import time
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.datasets.base import DatasetSample, TraversabilityTargetClass
from ai.datasets.split_manager import SplitManager
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("import_local_rugd")

# Map RUGD DatasetNinja classes to standard 3-tier traversability classes
RUGD_NINJA_CLASS_MAP: Dict[str, int] = {
    # 0: SAFE (traversable surfaces)
    "dirt": TraversabilityTargetClass.SAFE,
    "sand": TraversabilityTargetClass.SAFE,
    "grass": TraversabilityTargetClass.SAFE,
    "gravel": TraversabilityTargetClass.SAFE,
    "asphalt": TraversabilityTargetClass.SAFE,
    "concrete": TraversabilityTargetClass.SAFE,
    # 1: UNKNOWN (uncertain, tall brush, transition)
    "bush": TraversabilityTargetClass.UNKNOWN,
    "mulch": TraversabilityTargetClass.UNKNOWN,
    # 2: UNSAFE (lethal hazards, obstacles)
    "rock": TraversabilityTargetClass.UNSAFE,
    "rock-bed": TraversabilityTargetClass.UNSAFE,
    "tree": TraversabilityTargetClass.UNSAFE,
    "log": TraversabilityTargetClass.UNSAFE,
    "water": TraversabilityTargetClass.UNSAFE,
    "building": TraversabilityTargetClass.UNSAFE,
    "bridge": TraversabilityTargetClass.UNSAFE,
    "fence": TraversabilityTargetClass.UNSAFE,
    "pole": TraversabilityTargetClass.UNSAFE,
    "sign": TraversabilityTargetClass.UNSAFE,
    "person": TraversabilityTargetClass.UNSAFE,
    "bicycle": TraversabilityTargetClass.UNSAFE,
    "container": TraversabilityTargetClass.UNSAFE,
    "picnic-table": TraversabilityTargetClass.UNSAFE,
    "vehicle": TraversabilityTargetClass.UNSAFE,
    # 255: VOID (ignored in loss and costmap)
    "sky": TraversabilityTargetClass.VOID
}

def decode_datasetninja_mask(ann_data: dict) -> np.ndarray:
    """Decode Supervisely / DatasetNinja JSON bitmap objects into a single 2D uint8 mask."""
    h = ann_data["size"]["height"]
    w = ann_data["size"]["width"]
    # Default to UNKNOWN (1)
    mask = np.full((h, w), TraversabilityTargetClass.UNKNOWN, dtype=np.uint8)

    objects = ann_data.get("objects", [])
    # Render sky first if present, then terrain, then hazards on top
    # Sort order: VOID (sky) -> SAFE (dirt, grass) -> UNKNOWN (bush) -> UNSAFE (rocks, trees, logs)
    def class_priority(obj):
        cls_name = obj.get("classTitle", "")
        tgt = RUGD_NINJA_CLASS_MAP.get(cls_name, TraversabilityTargetClass.UNKNOWN)
        if tgt == TraversabilityTargetClass.VOID:
            return 0
        elif tgt == TraversabilityTargetClass.SAFE:
            return 1
        elif tgt == TraversabilityTargetClass.UNKNOWN:
            return 2
        return 3 # UNSAFE (obstacles rendered last to overwrite terrain)

    sorted_objects = sorted(objects, key=class_priority)

    for obj in sorted_objects:
        cls_name = obj.get("classTitle", "")
        target_cls = RUGD_NINJA_CLASS_MAP.get(cls_name, TraversabilityTargetClass.UNKNOWN)
        bmp_meta = obj.get("bitmap")
        if not bmp_meta:
            continue

        b64_data = bmp_meta.get("data")
        origin = bmp_meta.get("origin") # [x0, y0]
        if not b64_data or not origin:
            continue

        try:
            png_bytes = zlib.decompress(base64.b64decode(b64_data))
            bmp = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if bmp is None:
                continue

            x0, y0 = origin
            bh, bw = bmp.shape[:2]

            # Clip within boundaries
            x1 = min(x0 + bw, w)
            y1 = min(y0 + bh, h)
            sub_w = x1 - x0
            sub_h = y1 - y0

            if sub_w <= 0 or sub_h <= 0:
                continue

            # Alpha mask if RGBA, or binary mask
            if bmp.ndim == 3 and bmp.shape[2] == 4:
                alpha = bmp[:sub_h, :sub_w, 3] > 0
            else:
                alpha = bmp[:sub_h, :sub_w] > 0

            mask[y0:y1, x0:x1][alpha] = target_cls
        except Exception:
            continue

    return mask

def import_rugd_tar(
    tar_path: str,
    output_dir: str = "data/processed/sequences/rugd",
    target_train: int = 1200,
    target_val: int = 200,
    target_test: int = 250
) -> int:
    """Extract and convert RUGD sequences directly from tar into processed sequences."""
    tar_file = Path(tar_path).resolve()
    if not tar_file.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_file}")

    out_base = Path(output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"\n[RUGD CONVERTER] Target Allocation: Train={target_train} | Val={target_val} | Test={target_test}")
    print(f"[RUGD CONVERTER] Opening archive: {tar_file.name} ({tar_file.stat().st_size / (1024**3):.2f} GB)...")
    tf = tarfile.open(str(tar_file), "r")

    print("[RUGD CONVERTER] Indexing tar members...")
    ann_members: Dict[str, tarfile.TarInfo] = {}
    img_members: Dict[str, tarfile.TarInfo] = {}

    for m in tf.getmembers():
        if m.name.endswith(".png.json"):
            ann_members[m.name] = m
        elif m.name.endswith(".png") and "/img/" in m.name:
            img_members[m.name] = m

    # Group by sequence and split
    seq_pattern = re.compile(r'^(train|val|test)/ann/([a-zA-Z0-9\-_]+)_([0-9]+)\.png\.json$')
    sequences: Dict[str, List[Tuple[str, str, str]]] = {} # seq_name -> list of (split, frame_id, ann_key)

    for ann_name in ann_members.keys():
        match = seq_pattern.match(ann_name)
        if match:
            split, seq_name, frame_id = match.groups()
            sequences.setdefault(seq_name, []).append((split, frame_id, ann_name))

    # Calculate per-sequence targets
    split_seq_counts = {"train": 0, "val": 0, "test": 0}
    for seq_name, frame_list in sequences.items():
        s = frame_list[0][0]
        split_seq_counts[s] += 1

    per_seq_target = {
        "train": max(target_train // max(split_seq_counts["train"], 1), 1),
        "val": max(target_val // max(split_seq_counts["val"], 1), 1),
        "test": max(target_test // max(split_seq_counts["test"], 1), 1)
    }
    print(f"[RUGD CONVERTER] Per-Sequence Quotas: {per_seq_target}")

    # Re-initialize clean manifest with synthetic and own-real data preserved
    manifest_mgr = SplitManager("data/manifests/dataset_manifest.json")
    if Path("data/manifests/dataset_manifest.json").exists():
        try:
            existing = manifest_mgr.load_manifest()
            # Keep only synthetic and own real data, will re-add RUGD fresh
            manifest_mgr.samples = [s for s in existing if s.source_type != "public_rugd"]
            manifest_mgr._sequence_splits = {
                s.sequence_id: s.split for s in manifest_mgr.samples
            }
        except Exception:
            pass

    total_converted = 0
    t0 = time.time()

    for seq_name, frame_list in sequences.items():
        frame_list.sort(key=lambda x: int(x[1]))
        split = frame_list[0][0]
        quota = per_seq_target[split]

        # Calculate stride to spread samples evenly over sequence duration
        stride = max(len(frame_list) // quota, 1)

        seq_dir = out_base / seq_name
        rgb_dir = seq_dir / "rgb"
        mask_dir = seq_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        seq_converted = 0
        for idx in range(0, len(frame_list), stride):
            f_split, frame_id, ann_key = frame_list[idx]
            img_key = f"{f_split}/img/{seq_name}_{frame_id}.png"
            if img_key not in img_members:
                continue

            f_ann = tf.extractfile(ann_members[ann_key])
            if not f_ann:
                continue
            ann_data = json.load(f_ann)
            mask = decode_datasetninja_mask(ann_data)

            f_img = tf.extractfile(img_members[img_key])
            if not f_img:
                continue
            img_bytes = f_img.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            sample_stem = f"rugd_{seq_name}_{frame_id}"
            rgb_path = rgb_dir / f"{sample_stem}.png"
            mask_path = mask_dir / f"{sample_stem}.png"

            cv2.imwrite(str(rgb_path), img)
            cv2.imwrite(str(mask_path), mask)

            sample = DatasetSample(
                sample_id=sample_stem,
                sequence_id=f"rugd_{seq_name}",
                image_path=str(rgb_path),
                mask_path=str(mask_path),
                split=split,
                source_type="public_rugd",
                environment="trail" if "trail" in seq_name else ("park" if "park" in seq_name else "mixed"),
                lighting="sunlight" if "sun" in seq_name else "shadows",
                metadata={"sequence": seq_name, "source": "RUGD-DatasetNinja"}
            )
            manifest_mgr.add_sample(sample)

            seq_converted += 1
            total_converted += 1
            if seq_converted >= quota:
                break

        print(f" -> Converted '{seq_name}': {seq_converted} frames -> Split: {split}")

    tf.close()
    elapsed = time.time() - t0
    manifest_mgr.save_manifest()

    print("\n" + "=" * 65)
    print(f"SUCCESS: Converted {total_converted} RUGD frames in {elapsed:.1f}s")
    print(f"Updated Manifest: data/manifests/dataset_manifest.json")
    print("=" * 65 + "\n")
    return total_converted

def main():
    parser = argparse.ArgumentParser(description="Import and convert RUGD DatasetNinja archive into project")
    parser.add_argument("--input", "-i", type=str, default="rugd-DatasetNinja.tar", help="Path to RUGD tar file")
    parser.add_argument("--train-target", type=int, default=1200, help="Target images for training")
    parser.add_argument("--val-target", type=int, default=200, help="Target images for validation")
    parser.add_argument("--test-target", type=int, default=250, help="Target images for final testing")
    parser.add_argument("--output-dir", type=str, default="data/processed/sequences/rugd")
    args = parser.parse_args()

    input_tar = Path(args.input).resolve()
    if not input_tar.exists():
        input_tar = Path("rugd-DatasetNinja.tar").resolve()
        if not input_tar.exists():
            print(f"[ERROR] Could not find {args.input}")
            sys.exit(1)

    import_rugd_tar(
        tar_path=str(input_tar),
        output_dir=args.output_dir,
        target_train=args.train_target,
        target_val=args.val_target,
        target_test=args.test_target
    )

if __name__ == "__main__":
    main()
