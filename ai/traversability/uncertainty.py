"""Uncertainty estimation and confidence thresholding for unknown terrain representation."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F

from ai.datasets.base import TraversabilityTargetClass

def apply_uncertainty_threshold(
    logits: torch.Tensor,
    confidence_threshold: float = 0.60
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate class probabilities, entropy, and mask low-confidence predictions into UNKNOWN.
    
    Args:
        logits: Tensor of shape (B, C, H, W) or (C, H, W)
        confidence_threshold: Probability below which a pixel is marked UNKNOWN
        
    Returns:
        calibrated_preds: (B, H, W) integer class predictions [0, 1, 2]
        confidence: (B, H, W) max probability in [0.0, 1.0]
        entropy: (B, H, W) normalized Shannon entropy in [0.0, 1.0]
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)

    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1) # (B, C, H, W)

    max_probs, initial_preds = torch.max(probs, dim=1) # (B, H, W)

    # Compute Normalized Shannon Entropy
    eps = 1e-7
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1) / torch.log(torch.tensor(float(num_classes)))
    entropy = torch.clamp(entropy, 0.0, 1.0)

    # Reclassify low-confidence pixels as UNKNOWN
    calibrated_preds = initial_preds.clone()
    low_confidence_mask = (max_probs < confidence_threshold)
    calibrated_preds[low_confidence_mask] = TraversabilityTargetClass.UNKNOWN

    return calibrated_preds.squeeze(0), max_probs.squeeze(0), entropy.squeeze(0)
