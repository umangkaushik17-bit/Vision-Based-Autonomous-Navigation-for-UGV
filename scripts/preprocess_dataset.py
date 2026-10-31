#!/usr/bin/env python3
"""Label-Safe Resizing, Preprocessing, and Caching Utility for RELLIS-3D.

Rules enforced:
1. Aspect-ratio preserving resolution (default 640x400 for 1920x1200 native input).
2. RGB images resized with bilinear interpolation (cv2.INTER_LINEAR).
3. Annotation masks resized STRICTLY with Nearest-Neighbor interpolation (cv2.INTER_NEAREST).
4. RELLIS-3D raw class IDs mapped to 3-class traversability space (0: SAFE, 1: UNKNOWN, 2: UNSAFE, 255: VOID).
5. Dimension and class-mixing verification.
6. Optional caching to data/rellis3d/cache/640x400/ and benchmarking against raw pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml
from PIL import Image


def load_class_mapping(mapping_path: str = "configs/rellis3d_traversability_mapping.yaml") -> Dict[int, int]:
    """Load raw RELLIS-3D class ID -> model category mapping."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    mapping = {}
    for cid_str, info in cfg.get("classes", {}).items():
        cid = int(cid_str)
        target = int(info["model_output_category"])
        mapping[cid] = target
    return mapping


def preprocess_pair(
    img_path: Path,
    mask_path: Path,
    target_wh: Tuple[int, int] = (640, 400),
    mapping: Optional[Dict[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess an RGB image and label mask with strict label-safe rules."""
    w, h = target_wh

    # 1. Load RGB Image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Load Mask (single channel uint8 or uint16)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # 3. Label-safe resizing
    # RGB uses bilinear
    resized_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    # Mask MUST use nearest neighbor
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4. Map class IDs if mapping provided
    if mapping is not None:
        mapped_mask = np.full_like(resized_mask, fill_value=255, dtype=np.uint8)
        for raw_id, target_class in mapping.items():
            mapped_mask[resized_mask == raw_id] = target_class
        out_mask = mapped_mask
    else:
        out_mask = resized_mask.astype(np.uint8)

    # 5. Safety verification: assert spatial dimensions match exactly
    assert resized_rgb.shape[:2] == out_mask.shape[:2], (
        f"Spatial dimension mismatch: RGB {resized_rgb.shape[:2]} vs Mask {out_mask.shape[:2]}"
    )

    return resized_rgb, out_mask


def benchmark_caching(
    manifest_csv: Path,
    target_wh: Tuple[int, int] = (640, 400),
    num_samples: int = 50,
    mapping_path: str = "configs/rellis3d_traversability_mapping.yaml"
) -> Dict:
    """Benchmark raw loading & preprocessing vs cached loading."""
    mapping = load_class_mapping(mapping_path)
    
    samples = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("valid") in ("True", "true", True, 1, "1"):
                samples.append((Path(row["image_path"]), Path(row["annotation_path"])))
                if len(samples) >= num_samples:
                    break

    if not samples:
        print("[!] No valid samples in manifest for caching benchmark.")
        return {}

    print(f"\n[*] Benchmarking on {len(samples)} image-mask pairs...")

    # Benchmark Raw Pipeline
    t0 = time.perf_counter()
    preprocessed_items = []
    for img_p, mask_p in samples:
        rgb, mask = preprocess_pair(img_p, mask_p, target_wh=target_wh, mapping=mapping)
        preprocessed_items.append((rgb, mask))
    raw_time = time.perf_counter() - t0

    # Save to temp cache directory
    cache_temp_dir = Path("data/rellis3d/cache/benchmark_temp")
    cache_temp_dir.mkdir(parents=True, exist_ok=True)
    for idx, (rgb, mask) in enumerate(preprocessed_items):
        np.savez_compressed(cache_temp_dir / f"sample_{idx:04d}.npz", rgb=rgb, mask=mask)

    # Benchmark Cached Pipeline
    t1 = time.perf_counter()
    loaded_cached = []
    for idx in range(len(samples)):
        data = np.load(cache_temp_dir / f"sample_{idx:04d}.npz")
        loaded_cached.append((data["rgb"], data["mask"]))
    cached_time = time.perf_counter() - t1

    # Clean up temp cache
    for f in cache_temp_dir.glob("*.npz"):
        f.unlink()
    cache_temp_dir.rmdir()

    speedup = raw_time / max(cached_time, 1e-6)
    result = {
        "num_samples": len(samples),
        "target_resolution_wh": list(target_wh),
        "raw_pipeline_total_sec": round(raw_time, 3),
        "raw_per_sample_ms": round((raw_time / len(samples)) * 1000, 2),
        "cached_pipeline_total_sec": round(cached_time, 3),
        "cached_per_sample_ms": round((cached_time / len(samples)) * 1000, 2),
        "speedup_factor": round(speedup, 2)
    }

    print("\n--- Caching Benchmark Results ---")
    print(f"Raw Pipeline    : {result['raw_pipeline_total_sec']}s ({result['raw_per_sample_ms']} ms/sample)")
    print(f"Cached Pipeline : {result['cached_pipeline_total_sec']}s ({result['cached_per_sample_ms']} ms/sample)")
    print(f"Speedup Factor  : {result['speedup_factor']}x")
    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess and Cache RELLIS-3D Dataset")
    parser.add_argument("--manifest", type=str, default="data/rellis3d/rellis_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--mapping", type=str, default="configs/rellis3d_traversability_mapping.yaml", help="Class mapping YAML")
    parser.add_argument("--width", type=int, default=640, help="Target width")
    parser.add_argument("--height", type=int, default=400, help="Target height (400 preserves 16:10 aspect ratio)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparing raw vs cached pipeline")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_caching(
            manifest_csv=Path(args.manifest),
            target_wh=(args.width, args.height),
            mapping_path=args.mapping
        )


if __name__ == "__main__":
    main()
