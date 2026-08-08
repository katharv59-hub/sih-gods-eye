"""Temporal Queue Admission Control — Phase 4.1 (§14, §9).

Implements bounded two-tier admission control evaluated BEFORE queue insertion.
Enforces 80% high-watermark thinning for PERIODIC observations while preserving CRITICAL events.
Operates 100% non-blockingly to protect perception pipeline throughput.
"""

from __future__ import annotations

import queue
from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.observation import (
    ObservationPriority,
    TemporalObservation,
)

_log = get_logger("memory.admission_control")


class TemporalAdmissionControl:
    """Bounded two-tier queue admission controller for TemporalObservations."""

    def __init__(
        self,
        maxsize: int = 100,
        high_watermark_pct: float = 0.80,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize MUST be a positive integer")
        if not (0.0 < high_watermark_pct < 1.0):
            raise ValueError("high_watermark_pct MUST be between 0.0 and 1.0")

        self._maxsize = maxsize
        self._watermark_pct = high_watermark_pct
        self._watermark_limit = int(maxsize * high_watermark_pct)
        self._metrics = metrics

        self._queue: queue.Queue[TemporalObservation] = queue.Queue(maxsize=maxsize)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def high_watermark_limit(self) -> int:
        return self._watermark_limit

    def qsize(self) -> int:
        return self._queue.qsize()

    def evaluate_admission(self, obs: TemporalObservation) -> bool:
        """Evaluate admission decision BEFORE queue insertion.

        Returns True if admitted, False if thinned/rejected.
        """
        current_depth = self._queue.qsize()

        # Update high watermark gauge
        is_watermark_active = current_depth >= self._watermark_limit
        if self._metrics is not None:
            self._metrics.temporal_queue_watermark_active.set(
                1.0 if is_watermark_active else 0.0
            )

        if obs.priority == ObservationPriority.PERIODIC:
            if current_depth >= self._watermark_limit:
                _log.warning(
                    "temporal_observation_thinned",
                    observation_id=obs.observation_id,
                    observation_type=obs.observation_type.value,
                    priority=obs.priority.value,
                    camera_id=obs.camera_id,
                    timestamp_ns=obs.timestamp_ns,
                    queue_depth=current_depth,
                    watermark_limit=self._watermark_limit,
                )
                if self._metrics is not None:
                    self._metrics.temporal_queue_drops_total.labels(
                        priority="periodic"
                    ).inc()
                return False
            return True

        # CRITICAL priority
        if current_depth >= self._maxsize:
            _log.error(
                "temporal_critical_event_dropped",
                observation_id=obs.observation_id,
                observation_type=obs.observation_type.value,
                priority=obs.priority.value,
                camera_id=obs.camera_id,
                timestamp_ns=obs.timestamp_ns,
                queue_depth=current_depth,
                maxsize=self._maxsize,
            )
            if self._metrics is not None:
                self._metrics.temporal_queue_drops_total.labels(
                    priority="critical"
                ).inc()
            return False

        return True

    def submit(self, obs: TemporalObservation) -> bool:
        """Evaluate admission control and insert into queue non-blockingly.

        Returns True if successfully queued, False if thinned or rejected.
        """
        if not self.evaluate_admission(obs):
            return False

        try:
            self._queue.put_nowait(obs)
            return True
        except queue.Full:
            # Fallback if queue fills concurrently between evaluate and put
            _log.error(
                "temporal_queue_concurrent_full_drop",
                observation_id=obs.observation_id,
                priority=obs.priority.value,
            )
            if self._metrics is not None:
                self._metrics.temporal_queue_drops_total.labels(
                    priority=obs.priority.value
                ).inc()
            return False

    def get_nowait(self) -> Optional[TemporalObservation]:
        """Fetch next observation non-blockingly."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
