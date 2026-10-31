"""Python DWA Golden Reference Implementation.

Python DWA serves strictly as the GOLDEN REFERENCE and REGRESSION IMPLEMENTATION.
It is used for:
  - Trajectory rollout parity testing
  - SAT collision parity verification
  - STOP/PROCEED behavioral equivalence testing
  - Offline debugging

It is QUARANTINED from normal production navigation. Production runtime executes C++ DWA.
"""

from __future__ import annotations

from ai.planning.backends.cpu_backend import CPUDWABackend
from ai.planning.local_planners.dwa import DWAPlanner

IS_GOLDEN_REFERENCE = True
IS_PRODUCTION_DEFAULT = False

__all__ = [
    "IS_GOLDEN_REFERENCE",
    "IS_PRODUCTION_DEFAULT",
    "DWAPlanner",
    "CPUDWABackend",
]
