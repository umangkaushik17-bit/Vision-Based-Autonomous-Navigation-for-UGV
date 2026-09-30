"""Tests for Phase 13.26A: Intel GPU/XPU Perception Acceleration & Parity."""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.traversability.dense_predictor import DenseTraversabilityPredictor


class TestIntelGPUPerception(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = "outputs/rellis3d/checkpoints/best.pt"
        cls.has_xpu = hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)()

    def test_cpu_predictor_lifecycle(self):
        """Verify CPU predictor initialization, device property, and output shape."""
        pred = DenseTraversabilityPredictor(checkpoint=self.ckpt_path, device="cpu")
        self.assertEqual(pred.device, "cpu")

        dummy = np.full((400, 640, 3), 128, dtype=np.uint8)
        res = pred.predict(dummy)

        self.assertEqual(res.mask.shape, (400, 640))
        self.assertEqual(res.logits.shape, (400, 640))
        self.assertEqual(res.device, "cpu")
        self.assertFalse(np.isnan(res.mask).any())
        self.assertFalse(np.isinf(res.mask).any())
        self.assertGreaterEqual(res.mask.min(), 0.0)
        self.assertLessEqual(res.mask.max(), 1.0)

    def test_xpu_predictor_lifecycle(self):
        """Verify Intel XPU predictor initialization, execution, and output validity if XPU present."""
        if not self.has_xpu:
            # If XPU runtime unavailable, verify explicit request fails cleanly
            with self.assertRaises(RuntimeError) as ctx:
                _ = DenseTraversabilityPredictor(checkpoint=self.ckpt_path, device="xpu")
            self.assertIn("explicitly requested", str(ctx.exception))
            return

        pred = DenseTraversabilityPredictor(checkpoint=self.ckpt_path, device="xpu:0")
        self.assertEqual(pred.device, "xpu:0")

        dummy = np.full((400, 640, 3), 128, dtype=np.uint8)
        res = pred.predict(dummy)

        self.assertEqual(res.mask.shape, (400, 640))
        self.assertEqual(res.logits.shape, (400, 640))
        self.assertEqual(res.device, "xpu:0")
        self.assertFalse(np.isnan(res.mask).any())
        self.assertFalse(np.isinf(res.mask).any())
        self.assertGreaterEqual(res.mask.min(), 0.0)
        self.assertLessEqual(res.mask.max(), 1.0)

    def test_auto_device_selection(self):
        """Verify auto device selection selects XPU if available, else CPU (or CUDA)."""
        pred = DenseTraversabilityPredictor(checkpoint=self.ckpt_path, device="auto")
        if torch.cuda.is_available():
            self.assertEqual(pred.device, "cuda:0")
        elif self.has_xpu:
            self.assertEqual(pred.device, "xpu:0")
        else:
            self.assertEqual(pred.device, "cpu")

    def test_cpu_xpu_parity(self):
        """Verify numerical equivalence between CPU and XPU outputs on identical test input."""
        if not self.has_xpu:
            self.skipTest("Intel XPU hardware/runtime not available in this Python interpreter.")

        cpu_pred = DenseTraversabilityPredictor(checkpoint=self.ckpt_path, device="cpu")
        xpu_pred = DenseTraversabilityPredictor(checkpoint=self.ckpt_path, device="xpu:0")

        test_img = np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)
        cpu_res = cpu_pred.predict(test_img)
        xpu_res = xpu_pred.predict(test_img)

        max_diff = np.max(np.abs(cpu_res.mask - xpu_res.mask))
        self.assertLess(max_diff, 0.05, f"Disagreement between CPU and XPU mask: {max_diff}")

        # Class decision agreement
        cpu_safe = cpu_res.mask >= 0.5
        xpu_safe = xpu_res.mask >= 0.5
        agreement = np.mean(cpu_safe == xpu_safe)
        self.assertGreaterEqual(agreement, 0.98, f"Classification agreement too low: {agreement}")


if __name__ == "__main__":
    unittest.main()
