"""
Centralized logging configuration for the Customer Retention Engine.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with both file and console handlers.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"retention_engine_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
