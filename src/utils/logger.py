"""
Logging configuration for the AMuFP project.

This module provides a centralized logging configuration that follows
Python best practices and provides consistent logging across the project.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "amufp",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.

    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "amufp") -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (will be prefixed with 'amufp.')

    Returns:
        Logger instance
    """
    full_name = f"amufp.{name}" if name != "amufp" else name
    return logging.getLogger(full_name)


# Initialize default logger
def init_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Initialize logging for the entire application.

    Args:
        level: Logging level as string
        log_file: Path to log file
        console_output: Whether to output to console
    """
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    # Set up root logger
    setup_logger(
        name="amufp",
        level=log_level,
        log_file=log_file,
        console_output=console_output
    )

    # Set up specific loggers for different components
    setup_logger(name="amufp.file_processing", level=log_level)
    setup_logger(name="amufp.graph", level=log_level)
    setup_logger(name="amufp.llm", level=log_level)
    setup_logger(name="amufp.utils", level=log_level)
    setup_logger(name="amufp.image_converter", level=log_level)


# Convenience function for getting component-specific loggers
def get_file_processing_logger() -> logging.Logger:
    """Get logger for file processing operations."""
    return get_logger("file_processing")


def get_graph_logger() -> logging.Logger:
    """Get logger for graph operations."""
    return get_logger("graph")


def get_llm_logger() -> logging.Logger:
    """Get logger for LLM operations."""
    return get_logger("llm")


def get_utils_logger() -> logging.Logger:
    """Get logger for utility operations."""
    return get_logger("utils")


def get_image_converter_logger() -> logging.Logger:
    """Get logger for image conversion operations."""
    return get_logger("image_converter")
