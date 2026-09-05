"""Unit tests for dataset base models and label remapping standards."""

import numpy as np
import pytest

from ai.datasets.base import (
    TraversabilityTargetClass,
    DatasetSample,
    RUGD_LABEL_MAP,
    RELLIS_LABEL_MAP,
    remap_semantic_mask
)

def test_traversability_classes():
    assert TraversabilityTargetClass.SAFE == 0
    assert TraversabilityTargetClass.UNKNOWN == 1
    assert TraversabilityTargetClass.UNSAFE == 2
    assert TraversabilityTargetClass.VOID == 255

def test_dataset_sample_serialization():
    sample = DatasetSample(
        sample_id="test_001",
        sequence_id="seq_test",
        image_path="/path/to/img.png",
        mask_path="/path/to/mask.png",
        split="train",
        source_type="own_real",
        environment="gravel",
        lighting="shadows"
    )
    d = sample.to_dict()
    assert d["sample_id"] == "test_001"
    assert d["sequence_id"] == "seq_test"
    assert d["split"] == "train"

    restored = DatasetSample.from_dict(d)
    assert restored.sample_id == sample.sample_id
    assert restored.environment == sample.environment

def test_rugd_label_remapping():
    # RUGD classes: 1=dirt (SAFE), 4=tree (UNSAFE), 7=sky (VOID), 19=bush (UNKNOWN)
    raw_mask = np.array([
        [1, 1, 4],
        [7, 19, 1]
    ], dtype=np.uint8)

    remapped = remap_semantic_mask(raw_mask, RUGD_LABEL_MAP)
    assert remapped[0, 0] == TraversabilityTargetClass.SAFE
    assert remapped[0, 1] == TraversabilityTargetClass.SAFE
    assert remapped[0, 2] == TraversabilityTargetClass.UNSAFE
    assert remapped[1, 0] == TraversabilityTargetClass.VOID
    assert remapped[1, 1] == TraversabilityTargetClass.UNKNOWN
    assert remapped[1, 2] == TraversabilityTargetClass.SAFE

def test_rellis_label_remapping():
    # RELLIS classes: 1=dirt (SAFE), 4=tree (UNSAFE), 7=sky (VOID), 31=puddle (UNKNOWN)
    raw_mask = np.array([
        [1, 31],
        [4, 7]
    ], dtype=np.uint8)

    remapped = remap_semantic_mask(raw_mask, RELLIS_LABEL_MAP)
    assert remapped[0, 0] == TraversabilityTargetClass.SAFE
    assert remapped[0, 1] == TraversabilityTargetClass.UNKNOWN
    assert remapped[1, 0] == TraversabilityTargetClass.UNSAFE
    assert remapped[1, 1] == TraversabilityTargetClass.VOID
