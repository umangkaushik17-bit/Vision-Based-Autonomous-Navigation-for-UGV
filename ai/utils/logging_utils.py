"""Logging utilities for the AI perception subsystem."""

import sys
from pathlib import Path

# Ensure 'src' and root are in sys.path if not installed as a package
_root = Path(__file__).resolve().parent.parent.parent
_src = _root / "src"
for p in (str(_root), str(_src)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ugv_core.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str = "ai.vision"):
        """Fallback lightweight logger when ugv_core is not installed."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] [%(levelname)-8s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

def get_vision_logger(name: str = "ai.vision"):
    """Return a configured logger specifically for the vision pipeline."""
    return get_logger(name)

__all__ = ["get_vision_logger", "get_logger"]
