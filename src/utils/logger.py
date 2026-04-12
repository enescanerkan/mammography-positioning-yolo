"""
Logging factory for the breast positioning pipeline.

Provides a single ``get_logger`` function used consistently across all
modules so that log format and level are controlled from one place.
"""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    If the logger already has handlers attached (e.g., because it was
    requested earlier in the same process), no duplicate handlers are added.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
        level: Logging level.  Defaults to ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
