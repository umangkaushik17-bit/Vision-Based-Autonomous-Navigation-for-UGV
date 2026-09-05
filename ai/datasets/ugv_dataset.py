"""Unified PyTorch Dataset and DataLoader with multi-condition outdoor augmentations."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from ai.datasets.base import DatasetSample, TraversabilityTargetClass
from ai.datasets.split_manager import SplitManager
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.datasets.ugv_dataset")

class OutdoorAugmentation:
    """Robust data augmentation simulating shifting sunlight, tree shadows, and chassis vibrations."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (480, 640),
        is_training: bool = True,
        color_jitter_prob: float = 0.5,
        blur_prob: float = 0.3,
        hflip_prob: float = 0.5
    ):
        self.target_size = target_size # (height, width)
        self.is_training = is_training
        self.color_jitter_prob = color_jitter_prob
        self.blur_prob = blur_prob
        self.hflip_prob = hflip_prob

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply coordinated spatial and photometric augmentations to image and mask."""
        # Convert BGR (OpenCV) to RGB for PyTorch
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        # Resize both to target dimensions
        h_tgt, w_tgt = self.target_size
        if (rgb.shape[0], rgb.shape[1]) != (h_tgt, w_tgt):
            rgb = cv2.resize(rgb, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
            if mask is not None:
                mask = cv2.resize(mask, (w_tgt, h_tgt), interpolation=cv2.INTER_NEAREST)

        # Convert to PIL/Tensor representation for torchvision functional transforms
        # PyTorch image: (3, H, W) float32 in [0.0, 1.0]
        img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        if mask is not None:
            mask_tensor = torch.from_numpy(mask).long()
        else:
            mask_tensor = None

        if self.is_training:
            # 1. Random Horizontal Flip
            if random.random() < self.hflip_prob:
                img_tensor = TF.hflip(img_tensor)
                if mask_tensor is not None:
                    mask_tensor = TF.hflip(mask_tensor)

            # 2. Photometric Color Jitter (Sunlight / Shadow variation)
            if random.random() < self.color_jitter_prob:
                brightness_factor = random.uniform(0.75, 1.25)
                contrast_factor = random.uniform(0.75, 1.25)
                saturation_factor = random.uniform(0.80, 1.20)
                img_tensor = TF.adjust_brightness(img_tensor, brightness_factor)
                img_tensor = TF.adjust_contrast(img_tensor, contrast_factor)
                img_tensor = TF.adjust_saturation(img_tensor, saturation_factor)
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

            # 3. Gaussian Blur (Motion blur / dust / camera vibration)
            if random.random() < self.blur_prob:
                img_tensor = TF.gaussian_blur(img_tensor, kernel_size=[5, 5], sigma=[1.0, 1.5])

        # Standard ImageNet normalization
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return img_tensor, mask_tensor

class UGVDataset(Dataset):
    """PyTorch Dataset ingesting samples from the unified manifest."""

    def __init__(
        self,
        samples: List[DatasetSample],
        target_size: Tuple[int, int] = (480, 640),
        is_training: bool = False
    ):
        self.samples = samples
        self.transform = OutdoorAugmentation(target_size=target_size, is_training=is_training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load raw RGB image
        img = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {sample.image_path}")

        # Load or mock mask
        if sample.mask_path and Path(sample.mask_path).exists():
            mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Default to UNKNOWN (1) if unannotated
            mask = np.full((img.shape[0], img.shape[1]), TraversabilityTargetClass.UNKNOWN, dtype=np.uint8)

        img_tensor, mask_tensor = self.transform(img, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "sequence_id": sample.sequence_id,
            "split": sample.split,
            "source_type": sample.source_type,
            "environment": sample.environment,
            "lighting": sample.lighting
        }

def create_dataloader(
    split: str = "train",
    manifest_path: str = "data/manifests/dataset_manifest.json",
    batch_size: int = 8,
    target_size: Tuple[int, int] = (480, 640),
    num_workers: int = 0,
    max_samples: Optional[int] = None
) -> DataLoader:
    """Instantiate a validated PyTorch DataLoader for a specific sequence split."""
    manager = SplitManager(manifest_path)
    manager.load_manifest()
    samples = manager.get_samples_by_split(split)

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]

    is_train = (split == "train")
    dataset = UGVDataset(samples, target_size=target_size, is_training=is_train)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train and (len(dataset) > batch_size)
    )
