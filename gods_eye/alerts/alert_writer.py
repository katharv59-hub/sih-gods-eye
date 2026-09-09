"""Alert Writer — Phase 10.1 (SIH 26187).

Thread-safe async batch writer for Alert objects.
Mirrors the EventWriter pattern from Phase 4.2 exactly.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from gods_eye.alerts.alert_store import AlertStore
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.alert import Alert


_log = get_logger("alerts.alert_writer")


class AlertWriter:
    """Thread-safe async batch writer for Alert objects.

    Mirrors EventWriter pattern: background thread drains a queue
    and flushes batches to AlertStore.
    """

    def __init__(
        self,
        store: AlertStore,
        batch_size: int = 50,
        flush_interval_s: float = 0.5,
        max_queue_size: int = 5_000,
    ) -> None:
        self._store = store
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue[Optional[Alert]] = queue.Queue(maxsize=max_queue_size)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background flush thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="AlertWriterThread", daemon=True
        )
        self._thread.start()
        _log.info("alert_writer_started", batch_size=self._batch_size)

    def emit(self, alert: Alert) -> bool:
        """Queue alert for async background persistence.

        Returns True if successfully queued, False if queue full.
        """
        if not self._running:
            return False

        try:
            self._queue.put_nowait(alert)
            return True
        except queue.Full:
            _log.warning("alert_writer_queue_full", alert_id=alert.alert_id)
            return False

    def _run(self) -> None:
        batch: list[Alert] = []
        last_flush = time.time()

        while self._running:
            try:
                timeout = max(0.01, self._flush_interval_s - (time.time() - last_flush))
                alert = self._queue.get(timeout=timeout)

                if alert is None:
                    break

                batch.append(alert)

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

        # Drain remaining
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    batch.append(item)
            except queue.Empty:
                break

        if batch:
            self._flush(batch)

    def _flush(self, batch: list[Alert]) -> None:
        if not batch:
            return

        try:
            self._store.append_batch(batch)
        except Exception as exc:
            _log.error("alert_writer_flush_error", error=str(exc), count=len(batch))

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop writer and drain queued alerts."""
        if not self._running:
            return

        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

        _log.info("alert_writer_stopped")
