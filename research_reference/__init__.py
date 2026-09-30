"""Research and Reference Assets Package.

Contains components strictly quarantined from production autonomous navigation:
  - research_reference.sam_tp (Heavy offline foundation model comparison)
  - research_reference.python_dwa (Golden reference implementation for parity/regression)
  - research_reference.training (Offline dataset and model training assets)
  - research_reference.benchmark_only (Comparative evaluation scripts)
  - research_reference.experimental (Unvalidated prototypes)
"""

from __future__ import annotations

from research_reference.python_dwa import (
    CPUDWABackend as PythonCPUDWABackend,
    DWAPlanner as PythonDWAPlanner,
)
from research_reference.sam_tp import (
    IS_RESEARCH_ONLY as SAM_TP_IS_RESEARCH_ONLY,
    load_sam_tp_for_research,
)

__all__ = [
    "SAM_TP_IS_RESEARCH_ONLY",
    "load_sam_tp_for_research",
    "PythonDWAPlanner",
    "PythonCPUDWABackend",
]
