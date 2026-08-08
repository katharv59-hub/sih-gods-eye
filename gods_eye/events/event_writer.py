"""Background Event Writer — Phase 4.2 (§14).

Provides thread-safe async batching writer to push pipeline events
to event store without blocking real-time video processing threads.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from gods_eye.events.event_store import BaseEventStore
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.event import Event

_log = get_logger("events.event_writer")


class EventWriter:
    """Thread-safe async batch writer for Event objects."""

    def __init__(
        self,
        store: BaseEventStore,
        metrics: Optional[MetricsRegistry] = None,
        batch_size: int = 100,
        flush_interval_s: float = 0.5,
        max_queue_size: int = 10_000,
    ) -> None:
        self._store = store
        self._metrics = metrics
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue[Optional[Event]] = queue.Queue(maxsize=max_queue_size)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background flush thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="EventWriterThread", daemon=True
        )
        self._thread.start()
        _log.info("event_writer_started", batch_size=self._batch_size)

    def emit(self, event: Event) -> bool:
        """Queue event for async background persistence.

        Returns True if successfully queued, False if queue full (dropped).
        """
        if not self._running:
            return False

        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            _log.warning("event_writer_queue_full_dropped", event_id=event.event_id)
            return False

    def _run(self) -> None:
        batch: list[Event] = []
        last_flush = time.time()

        while self._running:
            try:
                timeout = max(0.01, self._flush_interval_s - (time.time() - last_flush))
                event = self._queue.get(timeout=timeout)

                if event is None:
                    break

                batch.append(event)

                if len(batch) >= self._batch_size or (
                    time.time() - last_flush >= self._flush_interval_s
                ):
                    self._flush(batch)
                    batch = []
                    last_flush = time.time()

            except queue.Empty:
                if batch:
                    self._flush(batch)
                    batch = []
                    last_flush = time.time()

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    batch.append(item)
            except queue.Empty:
                break

        if batch:
            self._flush(batch)

    def _flush(self, batch: list[Event]) -> None:
        if not batch:
            return

        start_time = time.time()
        try:
            self._store.append_batch(batch)
            latency_ms = (time.time() - start_time) * 1000.0

            if self._metrics is not None:
                self._metrics.event_write_latency_ms.observe(latency_ms)

        except Exception as exc:
            _log.error("event_writer_flush_error", error=str(exc), count=len(batch))

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop writer and drain queued events."""
        if not self._running:
            return

        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

        _log.info("event_writer_stopped")
