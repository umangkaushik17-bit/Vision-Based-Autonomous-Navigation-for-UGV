"""Dataset management, sequence-level splitting, and PyTorch DataLoaders."""

from ai.datasets.base import (
    TraversabilityTargetClass,
    DatasetSample,
    RUGD_LABEL_MAP,
    RELLIS_LABEL_MAP,
    remap_semantic_mask
)
from ai.datasets.split_manager import SplitManager
from ai.datasets.synthetic_dataset import SyntheticSequenceGenerator
from ai.datasets.public_dataset import RUGDAdapter, RELLIS3DAdapter
from ai.datasets.own_dataset import OwnDataImporter
try:
    from ai.datasets.ugv_dataset import UGVDataset, OutdoorAugmentation, create_dataloader
except (ImportError, ModuleNotFoundError):
    UGVDataset, OutdoorAugmentation, create_dataloader = None, None, None
from ai.datasets.audit import DatasetAuditor, AuditReport

__all__ = [
    "TraversabilityTargetClass",
    "DatasetSample",
    "RUGD_LABEL_MAP",
    "RELLIS_LABEL_MAP",
    "remap_semantic_mask",
    "SplitManager",
    "SyntheticSequenceGenerator",
    "RUGDAdapter",
    "RELLIS3DAdapter",
    "OwnDataImporter",
    "UGVDataset",
    "OutdoorAugmentation",
    "create_dataloader",
    "DatasetAuditor",
    "AuditReport"
]
