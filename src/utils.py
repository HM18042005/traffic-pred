"""Logging and shared utilities."""
from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def maybe_round(x: float, n: int = 6) -> float:
    """Tiny helper to round floats for tidy logging."""
    return float(round(x, n))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns default when denominator is 0."""
    try:
        if b == 0:
            return default
        return a / b
    except ZeroDivisionError:
        return default


__all__ = ["get_logger", "maybe_round", "safe_div"]
