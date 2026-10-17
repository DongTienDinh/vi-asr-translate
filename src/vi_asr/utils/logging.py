"""Logging utilities for vi_asr."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "vi_asr",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: Optional[int] = None,
) -> logging.Logger:
    """Setup structured logger with format and optional rank prefix."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    rank_prefix = f"[Rank {rank}] " if rank is not None else ""
    formatter = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)-7s | {rank_prefix}%(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
