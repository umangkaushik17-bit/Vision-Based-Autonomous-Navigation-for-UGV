"""Safety-weighted loss functions for UGV traversability segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedTraversabilityLoss(nn.Module):
    """Hybrid loss combining weighted Cross-Entropy (Focal variant) and multi-class Dice."""

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0
    ):
        super().__init__()
        weights_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.register_buffer("weights", weights_tensor)
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        # 1. Cross Entropy with class weighting
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        # Focal modulation (pt = exp(-ce))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma * ce_loss).mean()

        # 2. Multi-class Dice Loss
        valid_mask = (targets != self.ignore_index)
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        dice_total = 0.0
        active_classes = 0

        for c in range(num_classes):
            p_c = probs[:, c, :, :][valid_mask]
            t_c = (targets == c)[valid_mask].float()

            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum()
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            dice_total += (1.0 - dice)
            active_classes += 1

        dice_loss = dice_total / max(active_classes, 1)

        return self.ce_weight * focal_loss + self.dice_weight * dice_loss
