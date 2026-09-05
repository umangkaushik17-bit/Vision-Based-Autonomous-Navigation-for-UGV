"""Unit tests for sequence-level split manager and zero-leakage enforcement."""

import pytest
from pathlib import Path

from ai.datasets.base import DatasetSample
from ai.datasets.split_manager import SplitManager

def test_sequence_split_consistency(tmp_path):
    manifest_file = tmp_path / "test_manifest.json"
    mgr = SplitManager(str(manifest_file))

    # Add samples from sequence A to train
    mgr.add_sample(DatasetSample(sample_id="a1", sequence_id="seq_A", image_path="img1.png", split="train"))
    mgr.add_sample(DatasetSample(sample_id="a2", sequence_id="seq_A", image_path="img2.png", split="train"))

    # Add sample from sequence B to val
    mgr.add_sample(DatasetSample(sample_id="b1", sequence_id="seq_B", image_path="img3.png", split="val"))

    # Add sample from sequence C to test
    mgr.add_sample(DatasetSample(sample_id="c1", sequence_id="seq_C", image_path="img4.png", split="test"))

    valid, _ = mgr.validate_zero_leakage()
    assert valid is True

    # Save and reload
    mgr.save_manifest()
    assert manifest_file.exists()

    mgr2 = SplitManager(str(manifest_file))
    samples = mgr2.load_manifest()
    assert len(samples) == 4
    assert len(mgr2.get_samples_by_split("train")) == 2
    assert len(mgr2.get_samples_by_split("val")) == 1
    assert len(mgr2.get_samples_by_split("test")) == 1

def test_leakage_detection_raises_error(tmp_path):
    manifest_file = tmp_path / "leak_manifest.json"
    mgr = SplitManager(str(manifest_file))

    # Sequence A assigned to train
    mgr.add_sample(DatasetSample(sample_id="a1", sequence_id="seq_A", image_path="img1.png", split="train"))

    # Attempting to add same sequence A to test MUST immediately raise ValueError
    with pytest.raises(ValueError, match="Sequence Leakage Error"):
        mgr.add_sample(DatasetSample(sample_id="a2", sequence_id="seq_A", image_path="img2.png", split="test"))
