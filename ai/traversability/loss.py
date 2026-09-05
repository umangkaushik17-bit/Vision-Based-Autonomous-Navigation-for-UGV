"""Class-weighted loss functions designed to penalize false-safe hazard misses."""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWeightedFocalLoss(nn.Module):
    """Multi-class Focal Loss with class weighting and sky ignore_index."""

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth class indices of shape (B, H, W)
        """
        # Cross entropy loss with reduction='none'
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none"
        ) # (B, H, W)

        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        # Mask valid pixels (excluding ignore_index)
        valid_mask = (targets != self.ignore_index)

        if self.reduction == "mean":
            valid_count = valid_mask.sum().float()
            if valid_count > 0:
                return (focal_loss * valid_mask.float()).sum() / valid_count
            return focal_loss.sum() * 0.0
        elif self.reduction == "sum":
            return (focal_loss * valid_mask.float()).sum()
        return focal_loss

class DiceLoss(nn.Module):
    """Soft Dice Loss for semantic segmentation with ignore_index masking."""

    def __init__(self, num_classes: int = 3, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1) # (B, C, H, W)
        valid_mask = (targets != self.ignore_index) # (B, H, W)

        total_dice_loss = 0.0
        classes_evaluated = 0

        for c in range(self.num_classes):
            target_c = ((targets == c) & valid_mask).float()
            prob_c = probs[:, c, :, :] * valid_mask.float()

            intersection = (prob_c * target_c).sum()
            union = prob_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_dice_loss += (1.0 - dice)
            classes_evaluated += 1

        return total_dice_loss / max(classes_evaluated, 1)

class CombinedTraversabilityLoss(nn.Module):
    """Combined Focal + Dice loss balancing cross-entropy probabilities with spatial overlap."""

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        ignore_index: int = 255
    ):
        super().__init__()
        self.focal = ClassWeightedFocalLoss(class_weights=class_weights, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=len(class_weights) if class_weights else 3, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return loss_focal + self.dice_weight * loss_dice
