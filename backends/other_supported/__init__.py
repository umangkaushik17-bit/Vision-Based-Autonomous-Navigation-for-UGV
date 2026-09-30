"""Extensibility Stubs for Future Accelerators (Vulkan, OpenCL, Mobile GPU).

Explicitly declares status as UNSUPPORTED / EXPERIMENTAL until hardware and runtime
validation is completed.
"""

from __future__ import annotations

from typing import Dict, Any


def get_unsupported_accelerator_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "VULKAN_COMPUTE": {
            "status": "UNSUPPORTED",
            "reason": "Vulkan compute pipeline for DWA polygon SAT not implemented in this phase.",
        },
        "OPENCL": {
            "status": "UNSUPPORTED",
            "reason": "OpenCL kernels not implemented in this phase.",
        },
        "NPU_DWA": {
            "status": "UNSUPPORTED",
            "reason": "Branch-heavy dynamic DWA collision geometry is structurally incompatible with fixed-shape tensor NPU engines.",
        },
    }
