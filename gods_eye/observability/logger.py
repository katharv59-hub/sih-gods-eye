"""Structured JSON logging via structlog — §7 Observability Contract.

Every log entry includes: timestamp, level, subsystem, camera_id, event, data.
No print statements. No unstructured logging.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output per spec §7."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(subsystem: str, camera_id: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger bound to a subsystem.

    Args:
        subsystem: Module name (e.g. "detection", "tracking").
        camera_id: Optional camera identifier for per-camera logging.

    Returns:
        A bound structlog logger with subsystem and camera_id pre-bound.
    """
    logger: structlog.BoundLogger = structlog.get_logger()
    logger = logger.bind(subsystem=subsystem)
    if camera_id is not None:
        logger = logger.bind(camera_id=camera_id)
    return logger
