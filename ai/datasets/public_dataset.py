"""Public real-world outdoor robotics benchmark adapters (RUGD and RELLIS-3D).

Documented Licenses:
- RUGD: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- RELLIS-3D: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Both are verified for non-commercial research and academic hackathon prototypes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from ai.datasets.base import (
    DatasetSample,
    RUGD_LABEL_MAP,
    RELLIS_LABEL_MAP,
    remap_semantic_mask,
    TraversabilityTargetClass
)
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.datasets.public")

class RUGDAdapter:
    """Ingestion and label remapping parser for the RUGD Off-Road Dataset."""

    def __init__(self, root_dir: str = "data/raw/public/RUGD"):
        self.root_dir = Path(root_dir).resolve()

    def parse_sequence(
        self,
        sequence_name: str,
        split: str = "train",
        environment: str = "trail",
        lighting: str = "sunlight"
    ) -> List[DatasetSample]:
        """Index a single RUGD sequence folder and construct DatasetSample objects."""
        seq_dir = self.root_dir / sequence_name
        rgb_dir = seq_dir / "images"
        annot_dir = seq_dir / "annotations"

        if not rgb_dir.exists():
            logger.warning(f"RUGD sequence path does not exist: {rgb_dir}")
            return []

        samples: List[DatasetSample] = []
        image_files = sorted(rgb_dir.glob("*.png"))

        for img_path in image_files:
            sample_id = f"rugd_{sequence_name}_{img_path.stem}"
            mask_path = annot_dir / f"{img_path.stem}.png"

            sample = DatasetSample(
                sample_id=sample_id,
                sequence_id=f"rugd_{sequence_name}",
                image_path=str(img_path),
                mask_path=str(mask_path) if mask_path.exists() else None,
                split=split,
                source_type="public_rugd",
                environment=environment,
                lighting=lighting,
                metadata={"benchmark": "RUGD", "sequence": sequence_name}
            )
            samples.append(sample)

        return samples

    def convert_annotation(self, raw_mask: np.ndarray) -> np.ndarray:
        """Convert raw RUGD 24-class semantic mask to standard 3-class traversability."""
        return remap_semantic_mask(raw_mask, RUGD_LABEL_MAP)

class RELLIS3DAdapter:
    """Ingestion and label remapping parser for the RELLIS-3D Dataset."""

    def __init__(self, root_dir: str = "data/raw/public/RELLIS-3D"):
        self.root_dir = Path(root_dir).resolve()

    def parse_sequence(
        self,
        sequence_name: str,
        split: str = "train",
        environment: str = "mixed",
        lighting: str = "overcast"
    ) -> List[DatasetSample]:
        """Index a RELLIS-3D camera sequence."""
        seq_dir = self.root_dir / sequence_name
        rgb_dir = seq_dir / "pylon_camera_node"
        annot_dir = seq_dir / "pylon_camera_node_label_id"

        if not rgb_dir.exists():
            logger.warning(f"RELLIS-3D sequence path does not exist: {rgb_dir}")
            return []

        samples: List[DatasetSample] = []
        image_files = sorted(rgb_dir.glob("*.jpg"))

        for img_path in image_files:
            sample_id = f"rellis_{sequence_name}_{img_path.stem}"
            mask_path = annot_dir / f"{img_path.stem}.png"

            sample = DatasetSample(
                sample_id=sample_id,
                sequence_id=f"rellis_{sequence_name}",
                image_path=str(img_path),
                mask_path=str(mask_path) if mask_path.exists() else None,
                split=split,
                source_type="public_rellis",
                environment=environment,
                lighting=lighting,
                metadata={"benchmark": "RELLIS-3D", "sequence": sequence_name}
            )
            samples.append(sample)

        return samples

    def convert_annotation(self, raw_mask: np.ndarray) -> np.ndarray:
        """Convert raw RELLIS-3D 20-class semantic mask to standard 3-class traversability."""
        return remap_semantic_mask(raw_mask, RELLIS_LABEL_MAP)
