"""Observability — §7 of GODS_EYE_MASTER_SPEC.

Prometheus metrics + structured JSON logging via structlog.
No print statements in production code. No unstructured logging.
"""

from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry

__all__ = ["get_logger", "MetricsRegistry"]
