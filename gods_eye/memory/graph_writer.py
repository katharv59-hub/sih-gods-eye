"""Background Graph Writer — Phase 4.3 (§14).

Provides thread-safe async batching writer to push spatiotemporal graph nodes and edges
to GraphStore without blocking real-time video processing or perception threads.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from gods_eye.memory.graph_store import BaseGraphStore
from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry

_log = get_logger("memory.graph_writer")


@dataclass(frozen=True)
class GraphRecord:
    """Encapsulates a graph write operation."""
    kind: str  # "identity_node", "camera_node", "zone_node", "obs_edge", "trans_edge", "zone_edge"
    payload: dict[str, Any]


class GraphWriter:
    """Thread-safe async batch writer for GraphStore."""

    def __init__(
        self,
        store: BaseGraphStore,
        metrics: Optional[MetricsRegistry] = None,
        batch_size: int = 100,
        flush_interval_s: float = 0.5,
        max_queue_size: int = 10_000,
    ) -> None:
        self._store = store
        self._metrics = metrics
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue[Optional[GraphRecord]] = queue.Queue(maxsize=max_queue_size)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="GraphWriterThread", daemon=True
        )
        self._thread.start()
        _log.info("graph_writer_started", batch_size=self._batch_size)

    def emit(self, record: GraphRecord) -> bool:
        """Queue graph record for async background persistence.

        Returns True if successfully queued, False if queue full (dropped).
        """
        if not self._running:
            return False

        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            _log.warning("graph_writer_queue_full_dropped", kind=record.kind)
            return False

    def _run(self) -> None:
        batch: list[GraphRecord] = []
        last_flush = time.time()

        while self._running:
            try:
                timeout = max(0.01, self._flush_interval_s - (time.time() - last_flush))
                item = self._queue.get(timeout=timeout)

                if item is None:
                    break

                batch.append(item)

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

    def _flush(self, batch: list[GraphRecord]) -> None:
        if not batch:
            return

        start_t = time.time()
        try:
            for rec in batch:
                kind = rec.kind
                p = rec.payload
                if kind == "identity_node":
                    self._store.upsert_identity_node(**p)
                elif kind == "camera_node":
                    self._store.upsert_camera_node(**p)
                elif kind == "zone_node":
                    self._store.upsert_zone_node(**p)
                elif kind == "obs_edge":
                    self._store.add_observation_edge(**p)
                elif kind == "trans_edge":
                    self._store.add_transition_edge(**p)
                elif kind == "zone_edge":
                    self._store.add_zone_occupancy_edge(**p)

            latency_ms = (time.time() - start_t) * 1000.0
            if self._metrics is not None:
                self._metrics.graph_write_latency_ms.observe(latency_ms)

        except Exception as exc:
            _log.error("graph_writer_flush_error", error=str(exc), count=len(batch))

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop background thread and drain queue."""
        if not self._running:
            return

        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

        _log.info("graph_writer_stopped")
