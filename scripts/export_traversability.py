"""Export trained traversability model to TorchScript and verified ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.models.traversability_net import create_traversability_model

def export_model(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "outputs/exports",
    height: int = 480,
    width: int = 640
) -> bool:
    """Export and numerically verify TorchScript and ONNX formats."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    model = create_traversability_model()
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[EXPORT] Loaded weights from {checkpoint_path}")
    else:
        print("[EXPORT] Using base architecture initialization.")

    model.eval()
    dummy_input = torch.randn(1, 3, height, width)

    # 1. TorchScript Export
    ts_file = out_path / "traversability_model.pt"
    print(f"Exporting TorchScript -> {ts_file}...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(ts_file))

    # Parity check TorchScript
    with torch.no_grad():
        out_orig = model(dummy_input).numpy()
        out_ts = traced_model(dummy_input).numpy()
    assert np.allclose(out_orig, out_ts, atol=1e-4), "TorchScript parity check failed!"
    print("[OK] TorchScript export verified.")

    # 2. ONNX Export
    onnx_file = out_path / "traversability_model.onnx"
    print(f"Exporting ONNX -> {onnx_file}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print("[OK] ONNX export completed.")

    # Validate ONNX with onnxruntime if available
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_file))
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        assert np.allclose(out_orig, ort_outs[0], atol=1e-3), "ONNX numerical parity check failed!"
        print("[OK] ONNX numerical parity verified via ONNX Runtime.")
    except ImportError:
        print("[NOTICE] onnxruntime not installed in this environment; export file created.")

    print(f"\n[DONE] Model successfully exported to: {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Traversability model to TorchScript and ONNX")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt", help="Weights path")
    parser.add_argument("--output-dir", type=str, default="outputs/exports", help="Export target directory")
    args = parser.parse_args()

    export_model(checkpoint_path=args.checkpoint, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
