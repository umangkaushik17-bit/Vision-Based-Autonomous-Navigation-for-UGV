"""Universal data representations and label mapping standards for UGV traversability."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class TraversabilityTargetClass(IntEnum):
    """Standardized 3-tier semantic traversability ground truth."""
    SAFE = 0       # Flat ground, trail, dry dirt, benign grass, asphalt
    UNKNOWN = 1    # Overgrown brush, obscured shadows, uncharted transition zones
    UNSAFE = 2     # Boulders, ditches, tree trunks, water bodies, obstacles
    VOID = 255     # Sky, ego-vehicle chassis, or unannotated pixels (ignored in loss)

@dataclass
class DatasetSample:
    """Standardized metadata for an individual image-mask sample in the dataset."""
    sample_id: str
    sequence_id: str
    image_path: str
    mask_path: Optional[str] = None
    split: str = "train"                         # 'train', 'val', or 'test'
    source_type: str = "synthetic"               # 'synthetic', 'public_rugd', 'public_rellis', 'own_real'
    environment: str = "trail"                   # 'trail', 'grass', 'gravel', 'rocks', 'mixed'
    lighting: str = "sunlight"                   # 'sunlight', 'shadows', 'overcast', 'glare'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sequence_id": self.sequence_id,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "split": self.split,
            "source_type": self.source_type,
            "environment": self.environment,
            "lighting": self.lighting,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DatasetSample:
        return cls(**d)

# ----------------------------------------------------------------------
# Public Benchmark Label Mappings
# ----------------------------------------------------------------------

# RUGD Dataset (24 Classes) -> UGV 3-Class Traversability
# Reference: Wigness et al., "RUGD: A Dataset for Autonomous Off-Road Navigation"
RUGD_LABEL_MAP: Dict[int, int] = {
    0: TraversabilityTargetClass.VOID,       # void
    1: TraversabilityTargetClass.SAFE,       # dirt
    2: TraversabilityTargetClass.SAFE,       # sand
    3: TraversabilityTargetClass.SAFE,       # grass
    4: TraversabilityTargetClass.UNSAFE,     # tree
    5: TraversabilityTargetClass.UNSAFE,     # pole
    6: TraversabilityTargetClass.UNSAFE,     # water
    7: TraversabilityTargetClass.VOID,       # sky
    8: TraversabilityTargetClass.SAFE,       # vehicle
    9: TraversabilityTargetClass.SAFE,       # container
    10: TraversabilityTargetClass.SAFE,      # asphalt
    11: TraversabilityTargetClass.SAFE,      # gravel
    12: TraversabilityTargetClass.UNSAFE,    # building
    13: TraversabilityTargetClass.UNKNOWN,   # mulched-path
    14: TraversabilityTargetClass.UNSAFE,    # rock-bed
    15: TraversabilityTargetClass.UNSAFE,    # log
    16: TraversabilityTargetClass.UNSAFE,    # bicycle
    17: TraversabilityTargetClass.UNSAFE,    # person
    18: TraversabilityTargetClass.UNSAFE,    # fence
    19: TraversabilityTargetClass.UNKNOWN,   # bush
    20: TraversabilityTargetClass.UNSAFE,    # sign
    21: TraversabilityTargetClass.UNSAFE,    # rock
    22: TraversabilityTargetClass.UNSAFE,    # bridge
    23: TraversabilityTargetClass.SAFE       # concrete
}

# RELLIS-3D Dataset (20 Classes) -> UGV 3-Class Traversability
# Reference: Jiang et al., "RELLIS-3D Dataset: Data, Multi-modal Sensor Fusion, and Benchmarks"
RELLIS_LABEL_MAP: Dict[int, int] = {
    0: TraversabilityTargetClass.VOID,       # void
    1: TraversabilityTargetClass.SAFE,       # dirt
    3: TraversabilityTargetClass.SAFE,       # grass
    4: TraversabilityTargetClass.UNSAFE,     # tree
    5: TraversabilityTargetClass.UNSAFE,     # pole
    6: TraversabilityTargetClass.UNSAFE,     # water
    7: TraversabilityTargetClass.VOID,       # sky
    8: TraversabilityTargetClass.UNSAFE,     # vehicle
    9: TraversabilityTargetClass.SAFE,       # object
    10: TraversabilityTargetClass.SAFE,      # asphalt
    12: TraversabilityTargetClass.UNSAFE,    # building
    15: TraversabilityTargetClass.UNSAFE,    # log
    17: TraversabilityTargetClass.UNSAFE,    # person
    18: TraversabilityTargetClass.UNSAFE,    # fence
    19: TraversabilityTargetClass.UNKNOWN,   # bush
    23: TraversabilityTargetClass.SAFE,      # concrete
    27: TraversabilityTargetClass.SAFE,      # barrier
    31: TraversabilityTargetClass.UNKNOWN,   # puddle
    33: TraversabilityTargetClass.SAFE,      # mud
    34: TraversabilityTargetClass.UNSAFE     # rubble
}

def remap_semantic_mask(mask: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Remap pixel labels from raw annotation format to standard UGV traversability classes."""
    output = np.full_like(mask, fill_value=TraversabilityTargetClass.UNKNOWN, dtype=np.uint8)
    for raw_id, target_cls in mapping.items():
        output[mask == raw_id] = target_cls
    return output
