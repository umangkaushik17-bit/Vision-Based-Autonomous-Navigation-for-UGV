"""Ingestion and indexing framework for our own real-world outdoor recordings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from ai.datasets.base import DatasetSample
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.datasets.own_data")

class OwnDataImporter:
    """Imports real-world video files or image bursts into structured sequence datasets."""

    def __init__(self, output_root: str = "data/processed/sequences/own_data"):
        self.output_root = Path(output_root).resolve()

    def import_video_sequence(
        self,
        video_path: str,
        sequence_id: str,
        split: str = "train",
        environment: str = "dirt_trail",
        lighting: str = "sunlight",
        sample_every_n_frames: int = 5,
        max_frames: Optional[int] = None
    ) -> List[DatasetSample]:
        """Extract keyframes from a real-world video recording and assign to a sequence split.
        
        Args:
            video_path: Path to source .mp4 or .avi video.
            sequence_id: Unique sequence identifier (e.g. 'seq_own_dirt_01').
            split: 'train', 'val', or 'test'.
            environment: Terrain description.
            lighting: Illumination condition.
            sample_every_n_frames: Sampling stride to eliminate near-duplicate frames.
            max_frames: Optional cap on extracted frames.
            
        Returns:
            List of generated DatasetSample objects.
        """
        vid_file = Path(video_path).resolve()
        if not vid_file.exists():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        seq_dir = self.output_root / sequence_id
        rgb_dir = seq_dir / "rgb"
        mask_dir = seq_dir / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        samples: List[DatasetSample] = []
        frame_idx = 0
        extracted_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_every_n_frames == 0:
                sample_id = f"{sequence_id}_{extracted_idx:04d}"
                rgb_path = rgb_dir / f"{sample_id}.png"
                cv2.imwrite(str(rgb_path), frame)

                sample = DatasetSample(
                    sample_id=sample_id,
                    sequence_id=sequence_id,
                    image_path=str(rgb_path),
                    mask_path=None, # Populated upon annotation
                    split=split,
                    source_type="own_real",
                    environment=environment,
                    lighting=lighting,
                    metadata={
                        "source_video": vid_file.name,
                        "source_frame_idx": frame_idx,
                        "sample_stride": sample_every_n_frames,
                        "resolution": [frame.shape[1], frame.shape[0]]
                    }
                )
                samples.append(sample)
                extracted_idx += 1

                if max_frames and extracted_idx >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        # Save sequence manifest
        meta_file = seq_dir / "sequence_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "sequence_id": sequence_id,
                "split": split,
                "environment": environment,
                "lighting": lighting,
                "total_frames_extracted": len(samples),
                "source_file": str(vid_file)
            }, f, indent=2)

        logger.info(
            f"Imported own sequence '{sequence_id}' -> Split: {split} | "
            f"Extracted {len(samples)} frames (stride {sample_every_n_frames})"
        )
        return samples
