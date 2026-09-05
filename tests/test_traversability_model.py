"""Unit tests for LightweightTraversabilityNet architecture, forward pass, and gradient flow."""

import pytest
import torch

from ai.models.traversability_net import LightweightTraversabilityNet, create_traversability_model

def test_model_forward_pass_and_shape():
    model = LightweightTraversabilityNet(num_classes=3, backbone_name="mobilenet_v3_small", pretrained=False)
    dummy_input = torch.randn(2, 3, 240, 320)

    logits = model(dummy_input)
    assert logits.shape == (2, 3, 240, 320)
    assert logits.dtype == torch.float32

def test_model_parameter_budget():
    model = LightweightTraversabilityNet(num_classes=3, backbone_name="mobilenet_v3_small", pretrained=False)
    num_params = sum(p.numel() for p in model.parameters())
    # Ensure model stays under 10 million parameters for edge CPU deployment
    assert num_params < 10_000_000

def test_model_backprop():
    model = LightweightTraversabilityNet(num_classes=3, backbone_name="mobilenet_v3_small", pretrained=False)
    dummy_input = torch.randn(1, 3, 128, 128)
    logits = model(dummy_input)
    loss = logits.sum()
    loss.backward()

    # Verify gradients computed on encoder and decoder
    assert model.enc1[0][0].weight.grad is not None
    assert model.dec1.conv[0].weight.grad is not None
