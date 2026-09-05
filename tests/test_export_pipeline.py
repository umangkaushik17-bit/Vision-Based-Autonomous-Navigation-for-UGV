"""Unit tests for TorchScript and ONNX model export pipelines."""

from pathlib import Path
import pytest
import torch

from scripts.export_traversability import export_model

def test_export_torchscript_and_onnx(tmp_path):
    out_dir = tmp_path / "test_exports"

    success = export_model(
        checkpoint_path=None,
        output_dir=str(out_dir),
        height=128,
        width=128
    )
    assert success is True

    # Check that export artifacts exist
    ts_file = out_dir / "traversability_model.pt"
    onnx_file = out_dir / "traversability_model.onnx"

    assert ts_file.exists()
    assert onnx_file.exists()
    assert ts_file.stat().st_size > 1000
    assert onnx_file.stat().st_size > 1000
