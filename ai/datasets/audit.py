"""Statistical audit and data gap analyzer for the hybrid UGV dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

from ai.datasets.base import DatasetSample, TraversabilityTargetClass
from ai.datasets.split_manager import SplitManager
from ai.utils.logging_utils import get_vision_logger

logger = get_vision_logger("ai.datasets.audit")

@dataclass
class AuditReport:
    """Comprehensive dataset health and gap diagnostic report."""
    total_samples: int
    split_counts: Dict[str, int]
    sequence_counts: Dict[str, int]
    source_distribution: Dict[str, int]
    environment_distribution: Dict[str, int]
    lighting_distribution: Dict[str, int]
    class_pixel_percentages: Dict[str, float]
    zero_leakage_verified: bool
    data_gaps_identified: List[str]

    def summary_string(self) -> str:
        lines = [
            "=================== UGV DATASET AUDIT REPORT ===================",
            f"Total Annotated Samples : {self.total_samples}",
            f"Split Breakdown         : Train={self.split_counts.get('train', 0)} | "
            f"Val={self.split_counts.get('val', 0)} | Test={self.split_counts.get('test', 0)}",
            f"Unique Sequences        : {sum(self.sequence_counts.values())} "
            f"(Train={self.sequence_counts.get('train', 0)}, Val={self.sequence_counts.get('val', 0)}, Test={self.sequence_counts.get('test', 0)})",
            f"Source Breakdown        : {dict(self.source_distribution)}",
            f"Environments Represented: {list(self.environment_distribution.keys())}",
            f"Lighting Distributions  : {dict(self.lighting_distribution)}",
            "----------------- Traversability Class Coverage -----------------",
            f"SAFE    (Class 0) : {self.class_pixel_percentages.get('SAFE', 0.0):5.2f}%",
            f"UNKNOWN (Class 1) : {self.class_pixel_percentages.get('UNKNOWN', 0.0):5.2f}%",
            f"UNSAFE  (Class 2) : {self.class_pixel_percentages.get('UNSAFE', 0.0):5.2f}%",
            f"VOID    (Sky/Ego) : {self.class_pixel_percentages.get('VOID', 0.0):5.2f}%",
            "------------------ Data Leakage Verification -------------------",
            f"Sequence-Disjointness Check: {'PASS (Zero Leakage)' if self.zero_leakage_verified else 'FAIL (LEAK DETECTED!)'}",
            "------------------- Identified Data Gaps -----------------------"
        ]
        if self.data_gaps_identified:
            for gap in self.data_gaps_identified:
                lines.append(f" [!] {gap}")
        else:
            lines.append(" [OK] Balanced distribution with no critical gaps detected.")
        lines.append("================================================================")
        return "\n".join(lines)

class DatasetAuditor:
    """Audits dataset balance, pixel statistics, and sequence separation."""

    def __init__(self, manifest_path: str = "data/manifests/dataset_manifest.json"):
        self.manifest_path = manifest_path

    def audit(self, max_pixel_samples: int = 20) -> AuditReport:
        """Run statistical audit over the manifest."""
        manager = SplitManager(self.manifest_path)
        samples = manager.load_manifest()

        split_counts: Dict[str, int] = {}
        seq_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        source_dist: Dict[str, int] = {}
        env_dist: Dict[str, int] = {}
        lighting_dist: Dict[str, int] = {}

        for s in samples:
            split_counts[s.split] = split_counts.get(s.split, 0) + 1
            if s.split in seq_by_split:
                seq_by_split[s.split].add(s.sequence_id)
            source_dist[s.source_type] = source_dist.get(s.source_type, 0) + 1
            env_dist[s.environment] = env_dist.get(s.environment, 0) + 1
            lighting_dist[s.lighting] = lighting_dist.get(s.lighting, 0) + 1

        seq_counts = {k: len(v) for k, v in seq_by_split.items()}
        zero_leak, _ = manager.validate_zero_leakage()

        # Measure pixel distribution on a sample of masks
        pixel_counts = {0: 0, 1: 0, 2: 0, 255: 0}
        samples_with_masks = [s for s in samples if s.mask_path and Path(s.mask_path).exists()]
        sample_subset = samples_with_masks[:max_pixel_samples]

        for s in sample_subset:
            mask = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                for cls_id in pixel_counts.keys():
                    pixel_counts[cls_id] += int(np.sum(mask == cls_id))

        total_pixels = sum(pixel_counts.values())
        if total_pixels > 0:
            class_percentages = {
                "SAFE": (pixel_counts[0] / total_pixels) * 100.0,
                "UNKNOWN": (pixel_counts[1] / total_pixels) * 100.0,
                "UNSAFE": (pixel_counts[2] / total_pixels) * 100.0,
                "VOID": (pixel_counts[255] / total_pixels) * 100.0
            }
        else:
            class_percentages = {"SAFE": 0.0, "UNKNOWN": 0.0, "UNSAFE": 0.0, "VOID": 0.0}

        # Gap identification
        gaps: List[str] = []
        if split_counts.get("test", 0) == 0:
            gaps.append("Missing final held-out TEST sequences.")
        if class_percentages.get("UNSAFE", 0.0) < 5.0 and total_pixels > 0:
            gaps.append("Under-representation of UNSAFE hazards (< 5% of pixels). Collect/synthesize more boulders/ditches.")
        if "shadows" not in lighting_dist or lighting_dist["shadows"] == 0:
            gaps.append("Missing challenging shadow/tree-canopy lighting conditions.")
        if "gravel" not in env_dist and "rocks" not in env_dist:
            gaps.append("Terrain diversity gap: gravel/rocky terrains are under-represented.")

        return AuditReport(
            total_samples=len(samples),
            split_counts=split_counts,
            sequence_counts=seq_counts,
            source_distribution=source_dist,
            environment_distribution=env_dist,
            lighting_distribution=lighting_dist,
            class_pixel_percentages=class_percentages,
            zero_leakage_verified=zero_leak,
            data_gaps_identified=gaps
        )
