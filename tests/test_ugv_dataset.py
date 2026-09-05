"""Unit tests for PyTorch UGVDataset, outdoor augmentations, and DataLoader batch streaming."""

import torch
import numpy as np
import pytest

from ai.datasets.base import DatasetSample, TraversabilityTargetClass
from ai.datasets.ugv_dataset import OutdoorAugmentation, UGVDataset, create_dataloader

def test_outdoor_augmentations():
    aug_train = OutdoorAugmentation(target_size=(240, 320), is_training=True)
    aug_eval = OutdoorAugmentation(target_size=(240, 320), is_training=False)

    raw_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    raw_mask = np.random.choice([0, 1, 2, 255], size=(480, 640)).astype(np.uint8)

    # Eval mode
    img_t, mask_t = aug_eval(raw_img, raw_mask)
    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape == (3, 240, 320)
    assert mask_t.shape == (240, 320)
    assert mask_t.dtype == torch.long

    # Train mode
    img_tr, mask_tr = aug_train(raw_img, raw_mask)
    assert img_tr.shape == (3, 240, 320)
    assert mask_tr.shape == (240, 320)

def test_dataloader_streaming():
    # Use existing compiled manifest
    manifest_path = "data/manifests/dataset_manifest.json"
    loader = create_dataloader(
        split="train",
        manifest_path=manifest_path,
        batch_size=4,
        target_size=(240, 320),
        num_workers=0
    )

    assert len(loader) > 0
    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    assert batch["image"].shape == (4, 3, 240, 320)
    assert batch["mask"].shape == (4, 240, 320)
    assert batch["image"].dtype == torch.float32
    assert batch["mask"].dtype == torch.long
