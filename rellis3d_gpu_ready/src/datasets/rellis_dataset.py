"""PyTorch Dataset loader for RELLIS-3D camera and semantic mask sequences."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import yaml

class RELLIS3DDataset(Dataset):
    """RELLIS-3D Dataset loader with verified traversability remapping."""

    def __init__(
        self,
        samples: List[Tuple[str, str]], # List of (image_path, mask_path)
        mapping_cfg_path: str = "configs/rellis_class_mapping.yaml",
        target_size: Tuple[int, int] = (480, 640),
        is_training: bool = False
    ):
        self.samples = samples
        self.target_size = target_size # (H, W)
        self.is_training = is_training

        # Load label mapping table
        with open(mapping_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.class_map = {int(k): v["target_class"] for k, v in cfg.get("harmonized_mapping", {}).items()}

    def __len__(self) -> int:
        return len(self.samples)

    def _remap_labels(self, raw_mask: np.ndarray) -> np.ndarray:
        """Map raw RELLIS-3D pixel class IDs to 3-class traversability."""
        out_mask = np.full_like(raw_mask, fill_value=1, dtype=np.uint8) # Default UNKNOWN
        for raw_id, tgt in self.class_map.items():
            out_mask[raw_mask == raw_id] = tgt
        return out_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # Map labels
        remapped_mask = self._remap_labels(raw_mask)

        # Resize to target dimensions
        h_tgt, w_tgt = self.target_size
        rgb_resized = cv2.resize(rgb, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(remapped_mask, (w_tgt, h_tgt), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        img_tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).long()

        # Training augmentations
        if self.is_training:
            # Random horizontal flip
            if random.random() < 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # Color jitter (sunlight & shadow variations)
            if random.random() < 0.5:
                img_tensor = TF.adjust_brightness(img_tensor, random.uniform(0.8, 1.2))
                img_tensor = TF.adjust_contrast(img_tensor, random.uniform(0.8, 1.2))
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # ImageNet normalization
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_path": img_path
        }
