"""Zero-Copy Intra-Process and Shared-Memory IPC for ROS 2 Jazzy UGV Navigation (WS-4A / P1-2)."""

from ugv_navigation.ipc.zero_copy_ipc import (
    IntraProcessLoanedChannel,
    LoanedMessageHandle,
    SharedMemoryCameraFrameChannel,
    SharedMemoryWorldMapChannel,
)

__all__ = [
    "LoanedMessageHandle",
    "IntraProcessLoanedChannel",
    "SharedMemoryWorldMapChannel",
    "SharedMemoryCameraFrameChannel",
]
