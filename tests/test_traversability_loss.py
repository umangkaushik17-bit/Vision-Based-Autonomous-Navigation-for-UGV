"""Unit tests for class-weighted focal loss and safety-critical hazard penalization."""

import pytest
import torch

from ai.traversability.loss import ClassWeightedFocalLoss, DiceLoss, CombinedTraversabilityLoss

def test_class_weighted_focal_loss():
    # Class weights: [SAFE: 1.0, UNKNOWN: 1.5, UNSAFE: 3.5]
    weights = [1.0, 1.5, 3.5]
    loss_fn = ClassWeightedFocalLoss(class_weights=weights, gamma=2.0, ignore_index=255)

    # 1. Test loss reduction and positive scalar
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0

    # Test backprop
    loss.backward()
    assert logits.grad is not None

def test_ignore_index_in_focal_loss():
    loss_fn = ClassWeightedFocalLoss(class_weights=[1.0, 1.0, 1.0], ignore_index=255)

    logits = torch.randn(1, 3, 16, 16)
    targets_all_sky = torch.full((1, 16, 16), 255, dtype=torch.long)
    loss = loss_fn(logits, targets_all_sky)
    assert loss.item() == 0.0 # Completely ignored

def test_combined_loss():
    loss_fn = CombinedTraversabilityLoss(class_weights=[1.0, 1.5, 3.5])
    logits = torch.randn(2, 3, 24, 24)
    targets = torch.randint(0, 3, (2, 24, 24))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
