"""Sequence-level split manager enforcing strict zero-leakage governance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ai.datasets.base import DatasetSample
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.datasets.split_manager")

class SplitManager:
    """Manages dataset partitioning at the sequence level to prevent temporal data leakage."""

    def __init__(self, manifest_path: str = "data/manifests/dataset_manifest.json"):
        self.manifest_path = Path(manifest_path).resolve()
        self.samples: List[DatasetSample] = []
        self._sequence_splits: Dict[str, str] = {} # sequence_id -> split ('train', 'val', 'test')

    def add_sample(self, sample: DatasetSample):
        """Add a sample and register its sequence split mapping."""
        # Enforce consistent sequence split assignment
        seq_id = sample.sequence_id
        if seq_id in self._sequence_splits:
            expected_split = self._sequence_splits[seq_id]
            if sample.split != expected_split:
                raise ValueError(
                    f"Sequence Leakage Error: Sequence '{seq_id}' is already assigned to split "
                    f"'{expected_split}', but sample '{sample.sample_id}' has split '{sample.split}'!"
                )
        else:
            self._sequence_splits[seq_id] = sample.split

        self.samples.append(sample)

    def validate_zero_leakage(self) -> Tuple[bool, str]:
        """Perform programmatic proof that train, val, and test sequences are mutually disjoint."""
        train_seqs: Set[str] = set()
        val_seqs: Set[str] = set()
        test_seqs: Set[str] = set()

        for s in self.samples:
            if s.split == "train":
                train_seqs.add(s.sequence_id)
            elif s.split == "val":
                val_seqs.add(s.sequence_id)
            elif s.split == "test":
                test_seqs.add(s.sequence_id)
            else:
                raise ValueError(f"Unknown split '{s.split}' in sample '{s.sample_id}'")

        # Set intersection assertions
        train_val_overlap = train_seqs.intersection(val_seqs)
        train_test_overlap = train_seqs.intersection(test_seqs)
        val_test_overlap = val_seqs.intersection(test_seqs)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            err = (
                f"DATA LEAKAGE DETECTED!\n"
                f"Train-Val Overlap: {train_val_overlap}\n"
                f"Train-Test Overlap: {train_test_overlap}\n"
                f"Val-Test Overlap: {val_test_overlap}"
            )
            logger.error(err)
            return False, err

        msg = (
            f"Zero-Leakage Verified: {len(train_seqs)} Train Seqs, "
            f"{len(val_seqs)} Val Seqs, {len(test_seqs)} Test Seqs. No overlaps."
        )
        logger.info(msg)
        return True, msg

    def save_manifest(self) -> str:
        """Validate and write the unified dataset manifest to disk."""
        valid, msg = self.validate_zero_leakage()
        if not valid:
            raise RuntimeError(msg)

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "version": "1.0.0",
            "total_samples": len(self.samples),
            "sequence_splits": self._sequence_splits,
            "samples": [s.to_dict() for s in self.samples]
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Manifest successfully saved to: {self.manifest_path} ({len(self.samples)} samples)")
        return str(self.manifest_path)

    def load_manifest(self) -> List[DatasetSample]:
        """Load samples from existing manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = [DatasetSample.from_dict(d) for d in data.get("samples", [])]
        self._sequence_splits = data.get("sequence_splits", {})

        valid, msg = self.validate_zero_leakage()
        if not valid:
            raise RuntimeError(msg)

        logger.info(f"Loaded {len(self.samples)} samples from {self.manifest_path}")
        return self.samples

    def get_samples_by_split(self, split: str) -> List[DatasetSample]:
        """Return all samples belonging to a specific split ('train', 'val', or 'test')."""
        return [s for s in self.samples if s.split == split]
