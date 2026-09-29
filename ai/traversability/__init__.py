"""Traversability reasoning, SAM-TP integration, safety metrics, and baseline models."""

from ai.traversability.base import TraversabilityPredictorBase, TraversabilityResult
from ai.traversability.baseline_classifier import BaselineTerrainClassifier
from ai.traversability.loss import ClassWeightedFocalLoss, DiceLoss, CombinedTraversabilityLoss
from ai.traversability.metrics import (
    calculate_segmentation_metrics,
    calculate_navigation_safety_metrics
)
from ai.traversability.postprocess import refine_traversability_by_contrast, TemporalTraversabilityFilter
from ai.traversability.sam_tp import SAMTPTraversabilityPredictor
from ai.traversability.dense_predictor import DenseTraversabilityPredictor
from ai.traversability.temporal_stabilizer import TemporalStabilizer
from ai.traversability.uncertainty import apply_uncertainty_threshold
from ai.traversability.visualization import (
    create_difference_map,
    create_heatmap,
    create_traversability_overlay
)

__all__ = [
    "TraversabilityPredictorBase",
    "TraversabilityResult",
    "SAMTPTraversabilityPredictor",
    "DenseTraversabilityPredictor",
    "BaselineTerrainClassifier",
    "ClassWeightedFocalLoss",
    "DiceLoss",
    "CombinedTraversabilityLoss",
    "calculate_segmentation_metrics",
    "calculate_navigation_safety_metrics",
    "apply_uncertainty_threshold",
    "TemporalStabilizer",
    "TemporalTraversabilityFilter",
    "refine_traversability_by_contrast",
    "create_traversability_overlay",
    "create_heatmap",
    "create_difference_map"
]
