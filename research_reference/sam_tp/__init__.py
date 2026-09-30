"""SAM-TP Research and Reference Asset.

SAM-TP is strictly a research/reference model for offline comparison, benchmarking,
and ablation studies.
It is QUARANTINED from production runtime:
  - Not imported by default
  - Not part of the production deployment package
  - Not loaded during production navigation
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("research_reference.sam_tp")

IS_RESEARCH_ONLY = True
IS_PRODUCTION_DEFAULT = False


def load_sam_tp_for_research(*args: Any, **kwargs: Any) -> Any:
    """Explicit research-only loader for SAM-TP."""
    logger.warning("Loading SAM-TP for offline research/comparison. Not for production runtime.")
    from ai.traversability.sam_tp import SAMTPSegmentor
    return SAMTPSegmentor(*args, **kwargs)


__all__ = [
    "IS_RESEARCH_ONLY",
    "IS_PRODUCTION_DEFAULT",
    "load_sam_tp_for_research",
]
