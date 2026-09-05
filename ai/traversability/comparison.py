"""Comparative evaluation and agreement analysis between Traversability Models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from ai.traversability.base import TraversabilityPredictorBase, TraversabilityResult


@dataclass
class ModelComparisonResult:
    """Quantitative comparison metrics between two traversability models on a frame."""

    result_a: TraversabilityResult
    result_b: TraversabilityResult
    agreement_ratio: float
    disagreement_ratio: float
    both_safe_ratio: float
    both_unsafe_ratio: float
    mean_absolute_difference: float
    disagreement_mask: np.ndarray
    agreement_safe_mask: np.ndarray
    agreement_unsafe_mask: np.ndarray
    comparison_overlay: np.ndarray
    metrics_a_vs_gt: Optional[Dict[str, float]] = None
    metrics_b_vs_gt: Optional[Dict[str, float]] = None


class TraversabilityModelComparator:
    """Executes side-by-side evaluation between Model A (SAM-TP) and Model B (RUGD)."""

    def __init__(
        self,
        predictor_a: TraversabilityPredictorBase,
        predictor_b: TraversabilityPredictorBase,
        threshold: float = 0.5
    ) -> None:
        self.predictor_a = predictor_a
        self.predictor_b = predictor_b
        self.threshold = threshold

    def compare_frame(
        self,
        image: Union[np.ndarray, str],
        gt_mask: Optional[np.ndarray] = None
    ) -> ModelComparisonResult:
        """Run both models on the input image and evaluate spatial agreement/disagreement."""
        res_a = self.predictor_a.predict(image)
        res_b = self.predictor_b.predict(image)

        mask_a = res_a.mask
        mask_b = res_b.mask

        bin_a = (mask_a >= self.threshold)
        bin_b = (mask_b >= self.threshold)

        agree_safe = bin_a & bin_b
        agree_unsafe = (~bin_a) & (~bin_b)
        disagree = bin_a ^ bin_b

        total_pixels = float(mask_a.size)
        agreement_ratio = float(np.sum(bin_a == bin_b)) / total_pixels
        disagreement_ratio = float(np.sum(disagree)) / total_pixels
        both_safe_ratio = float(np.sum(agree_safe)) / total_pixels
        both_unsafe_ratio = float(np.sum(agree_unsafe)) / total_pixels
        mad = float(np.mean(np.abs(mask_a - mask_b)))

        # Create composite visual comparison overlay
        rgb = res_a.image
        h, w = rgb.shape[:2]
        comp_vis = np.zeros((h, w, 3), dtype=np.uint8)
        # Green for Both Safe
        comp_vis[agree_safe] = [0, 220, 0]
        # Red for Both Unsafe
        comp_vis[agree_unsafe] = [220, 0, 0]
        # Yellow for Model Disagreement (Uncertainty)
        comp_vis[disagree] = [255, 200, 0]

        comparison_overlay = cv2.addWeighted(rgb, 0.55, comp_vis, 0.45, 0)

        # Draw legend on comparison overlay
        cv2.putText(comparison_overlay, "Green: Both Safe", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(comparison_overlay, "Red: Both Unsafe", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(comparison_overlay, "Yellow: Disagreement (UNKNOWN)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # If Ground Truth is provided, calculate exact comparative metrics
        metrics_a = None
        metrics_b = None
        if gt_mask is not None:
            # Assume GT: 0=SAFE, 1=UNKNOWN, 2=UNSAFE (or binary 1=SAFE, 0=UNSAFE)
            gt_safe = (gt_mask == 0) if np.max(gt_mask) > 1 else (gt_mask == 1)
            gt_unsafe = (gt_mask == 2) if np.max(gt_mask) > 1 else (gt_mask == 0)

            def calc_metrics(pred_bin: np.ndarray) -> Dict[str, float]:
                tp = np.logical_and(pred_bin, gt_safe).sum()
                fp = np.logical_and(pred_bin, gt_unsafe).sum()
                fn = np.logical_and(~pred_bin, gt_safe).sum()
                tn = np.logical_and(~pred_bin, gt_unsafe).sum()

                union = np.logical_or(pred_bin, gt_safe).sum()
                iou = float(tp) / float(union) if union > 0 else 0.0
                prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
                rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                false_safe_ratio = float(fp) / float(pred_bin.size)
                false_safe_area_ratio = float(fp) / float(gt_unsafe.sum()) if gt_unsafe.sum() > 0 else 0.0

                return {
                    "iou_safe": iou,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "false_safe_pixel_ratio": false_safe_ratio,
                    "false_safe_area_ratio": false_safe_area_ratio
                }

            metrics_a = calc_metrics(bin_a)
            metrics_b = calc_metrics(bin_b)

        return ModelComparisonResult(
            result_a=res_a,
            result_b=res_b,
            agreement_ratio=agreement_ratio,
            disagreement_ratio=disagreement_ratio,
            both_safe_ratio=both_safe_ratio,
            both_unsafe_ratio=both_unsafe_ratio,
            mean_absolute_difference=mad,
            disagreement_mask=disagree,
            agreement_safe_mask=agree_safe,
            agreement_unsafe_mask=agree_unsafe,
            comparison_overlay=comparison_overlay,
            metrics_a_vs_gt=metrics_a,
            metrics_b_vs_gt=metrics_b
        )
