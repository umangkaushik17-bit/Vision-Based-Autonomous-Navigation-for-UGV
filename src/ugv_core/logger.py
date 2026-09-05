"""Structured logging utility for UGV navigation stack."""

import logging
import sys
from pathlib import Path
from typing import Optional

# ANSI Color codes for clean terminal output
COLOR_RESET = "\033[0m"
COLOR_DEBUG = "\033[36m"    # Cyan
COLOR_INFO = "\033[32m"     # Green
COLOR_WARNING = "\033[33m"  # Yellow
COLOR_ERROR = "\033[31m"    # Red
COLOR_CRITICAL = "\033[35m" # Magenta
COLOR_BOLD = "\033[1m"

class ColoredFormatter(logging.Formatter):
    """Logging formatter adding ANSI colors based on log level."""
    
    LEVEL_COLORS = {
        logging.DEBUG: COLOR_DEBUG,
        logging.INFO: COLOR_INFO,
        logging.WARNING: COLOR_WARNING,
        logging.ERROR: COLOR_ERROR,
        logging.CRITICAL: COLOR_CRITICAL,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, COLOR_RESET)
        record.levelname = f"{color}{record.levelname:<8}{COLOR_RESET}"
        return super().format(record)

def get_logger(name: str = "UGV", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Get or configure a structured logger instance.
    
    Args:
        name: Subsystem name (e.g. 'ugv.perception', 'ugv.planner')
        log_file: Optional file path to persist log messages.
        level: Minimum log level.
        
    Returns:
        logging.Logger configured instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Check existing handlers to avoid duplicates while allowing file handlers
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    existing_file_paths = [
        Path(h.baseFilename).resolve()
        for h in logger.handlers
        if isinstance(h, logging.FileHandler) and hasattr(h, "baseFilename")
    ]

    # Console handler
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = ColoredFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

    # File handler (plain text, no ANSI codes)
    if log_file:
        file_path = Path(log_file).resolve()
        if file_path not in existing_file_paths:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
            file_handler.setLevel(level)
            file_fmt = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

    return logger
