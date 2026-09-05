"""Synthetic outdoor sequence generator and dataset loader with domain randomization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from ai.datasets.base import DatasetSample, TraversabilityTargetClass
try:
    from scripts.generate_synthetic_outdoor_data import generate_synthetic_frame
except ImportError:
    try:
        from generate_synthetic_outdoor_data import generate_synthetic_frame
    except ImportError:
        generate_synthetic_frame = None

class SyntheticSequenceGenerator:
    """Generates parameterized synthetic outdoor sequences for reproducible training and benchmarking."""

    def __init__(self, base_output_dir: str = "data/processed/sequences/synthetic"):
        self.base_output_dir = Path(base_output_dir).resolve()

    def generate_sequence(
        self,
        sequence_id: str,
        num_frames: int = 15,
        split: str = "train",
        environment: str = "trail",
        lighting: str = "sunlight",
        width: int = 640,
        height: int = 480
    ) -> List[DatasetSample]:
        """Generate a complete multi-frame outdoor synthetic sequence."""
        seq_dir = self.base_output_dir / sequence_id
        rgb_dir = seq_dir / "rgb"
        mask_dir = seq_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        samples: List[DatasetSample] = []

        for idx in range(num_frames):
            rgb, gt_mask, depth, meta = generate_synthetic_frame(
                step=idx,
                total_steps=num_frames,
                width=width,
                height=height
            )

            # Map raw GT mask to formal TraversabilityTargetClass:
            # generate_synthetic_frame returns: 0=SAFE, 1=UNKNOWN, 2=UNSAFE (and sky at y < horizon is 2)
            # Map sky to VOID (255) so loss doesn't penalize navigation over sky
            horizon_y = int(height * 0.35)
            gt_mask[0:horizon_y, :] = TraversabilityTargetClass.VOID

            frame_stem = f"{sequence_id}_{idx:04d}"
            rgb_path = rgb_dir / f"{frame_stem}.png"
            mask_path = mask_dir / f"{frame_stem}.png"

            cv2.imwrite(str(rgb_path), rgb)
            cv2.imwrite(str(mask_path), gt_mask)

            sample = DatasetSample(
                sample_id=frame_stem,
                sequence_id=sequence_id,
                image_path=str(rgb_path),
                mask_path=str(mask_path),
                split=split,
                source_type="synthetic",
                environment=environment,
                lighting=lighting,
                metadata={
                    "width": width,
                    "height": height,
                    "frame_index": idx,
                    "imu": meta.get("imu"),
                    "ground_truth_pose": meta.get("ground_truth_pose")
                }
            )
            samples.append(sample)

        return samples
