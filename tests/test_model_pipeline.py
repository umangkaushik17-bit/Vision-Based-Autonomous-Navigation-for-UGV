"""Unit tests for lightweight traversability model and ONNX export pipeline."""

from pathlib import Path
import tempfile
import torch
import pytest

from ai.training.train_traversability_a100 import LightweightTraversabilityNet
from ai.export.export_onnx import export_to_onnx

def test_model_forward_pass():
    model = LightweightTraversabilityNet(num_classes=3, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 3, 128, 128)

def test_onnx_export_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = Path(tmpdir) / "test_model.onnx"
        export_to_onnx(
            weights_path="",
            output_path=str(out_onnx),
            height=128,
            width=128
        )
        assert out_onnx.exists()
        assert out_onnx.stat().st_size > 1000
