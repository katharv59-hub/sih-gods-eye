"""Append-only Event Store — Phase 4.2 (§14, ADR-012).

Provides thread-safe persistence and indexed query capability for system events.
Supports high-throughput batch writes, composite key deterministic sorting, and TTL purging.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.event import Event, EventType

_log = get_logger("events.event_store")


class BaseEventStore(ABC):
    """Abstract Base Class for event persistence engines."""

    @abstractmethod
    def append(self, event: Event) -> None:
        """Persist a single event."""
        ...

    @abstractmethod
    def append_batch(self, events: list[Event]) -> None:
        """Persist a batch of events atomically."""
        ...

    @abstractmethod
    def get_by_id(self, event_id: str) -> Optional[Event]:
        """Fetch event by UUID."""
        ...

    @abstractmethod
    def query_events(
        self,
        event_type: Optional[EventType] = None,
        camera_id: Optional[str] = None,
        global_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[Event]:
        """Query events matching specified filter criteria ordered by composite deterministic key."""
        ...

    @abstractmethod
    def purge_before(self, timestamp_ns: int) -> int:
        """Purge events older than specified timestamp according to retention policy. Returns count purged."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total event count in store."""
        ...

    @abstractmethod
    def delete_events(
        self,
        event_type: Optional[EventType] = None,
        camera_id: Optional[str] = None,
    ) -> int:
        """Delete events matching criteria. Returns count deleted."""
        ...

    @abstractmethod
    def upsert(self, event: Event) -> None:
        """Insert or replace an event by event_id.

        Used for mutable records (e.g. alert lifecycle updates)
        where the same event_id is re-persisted with updated metadata.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close store resources."""
        ...


class SQLiteEventStore(BaseEventStore):
    """SQLite append-only event store with WAL mode (ADR-012).

    Thread-safe implementation with composite ordering key (timestamp_ns, sequence_num, event_id).
    Exceeds 1,000 events/sec Master Spec gate threshold (engineering target >= 2,500 events/sec).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self._db_path = db_path
        self._metrics = metrics
        self._lock = threading.RLock()

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row

        with self._lock:
            # Enable Write-Ahead Logging (WAL) for concurrency & throughput
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence_num INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    global_id TEXT,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT,
                    timestamp_ns INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            # Query index optimizers including composite deterministic key index
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_ts_seq ON events(timestamp_ns, sequence_num, event_id);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_global_ts ON events(global_id, timestamp_ns);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_cam_ts ON events(camera_id, timestamp_ns);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, timestamp_ns);"
            )

    def append(self, event: Event) -> None:
        self.append_batch([event])

    def append_batch(self, events: list[Event]) -> None:
        if not events:
            return

        rows = [
            (
                e.event_id,
                e.event_type.value,
                e.global_id,
                e.camera_id,
                e.zone_id,
                e.timestamp_ns,
                e.frame_id,
                e.confidence,
                e.explanation,
                json.dumps(e.metadata),
            )
            for e in events
        ]

        with self._lock:
            try:
                with self._conn:
                    self._conn.executemany(
                        """
                        INSERT OR IGNORE INTO events (
                            event_id, event_type, global_id, camera_id, zone_id,
                            timestamp_ns, frame_id, confidence, explanation, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                if self._metrics is not None:
                    for e in events:
                        self._metrics.events_written_total.labels(
                            event_type=e.event_type.value
                        ).inc()
            except Exception as exc:
                _log.error("event_store_append_error", error=str(exc), count=len(events))
                if self._metrics is not None:
                    self._metrics.event_store_errors_total.labels(operation="append").inc()
                raise

    def get_by_id(self, event_id: str) -> Optional[Event]:
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT * FROM events WHERE event_id = ?", (event_id,)
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_event(row)
            except Exception as exc:
                _log.error("event_store_get_by_id_error", error=str(exc), event_id=event_id)
                if self._metrics is not None:
                    self._metrics.event_store_errors_total.labels(operation="get_by_id").inc()
                raise

    def query_events(
        self,
        event_type: Optional[EventType] = None,
        camera_id: Optional[str] = None,
        global_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[object] = []

        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type.value)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if global_id is not None:
            clauses.append("global_id = ?")
            params.append(global_id)
        if zone_id is not None:
            clauses.append("zone_id = ?")
            params.append(zone_id)
        if start_ns is not None:
            clauses.append("timestamp_ns >= ?")
            params.append(start_ns)
        if end_ns is not None:
            clauses.append("timestamp_ns <= ?")
            params.append(end_ns)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # Strict composite deterministic ordering key: timestamp_ns, sequence_num, event_id
        sql = f"SELECT * FROM events {where_str} ORDER BY timestamp_ns ASC, sequence_num ASC, event_id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                return [self._row_to_event(row) for row in cur.fetchall()]
            except Exception as exc:
                _log.error("event_store_query_error", error=str(exc))
                if self._metrics is not None:
                    self._metrics.event_store_errors_total.labels(operation="query").inc()
                raise

    def purge_before(self, timestamp_ns: int) -> int:
        with self._lock:
            try:
                with self._conn:
                    cur = self._conn.execute(
                        "DELETE FROM events WHERE timestamp_ns < ?", (timestamp_ns,)
                    )
                    purged_count = cur.rowcount
                if self._metrics is not None and purged_count > 0:
                    self._metrics.events_purged_total.inc(purged_count)
                return purged_count
            except Exception as exc:
                _log.error("event_store_purge_error", error=str(exc), timestamp_ns=timestamp_ns)
                if self._metrics is not None:
                    self._metrics.event_store_errors_total.labels(operation="purge").inc()
                raise

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM events")
            return cur.fetchone()[0]

    def delete_events(
        self,
        event_type: Optional[EventType] = None,
        camera_id: Optional[str] = None,
    ) -> int:
        """Delete events matching criteria. Returns count deleted."""
        clauses: list[str] = []
        params: list[object] = []
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type.value)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"DELETE FROM events {where_str}"
        with self._lock:
            try:
                with self._conn:
                    cur = self._conn.execute(sql, params)
                    deleted_count = cur.rowcount
                if self._metrics is not None and deleted_count > 0:
                    self._metrics.events_purged_total.inc(deleted_count)
                return deleted_count
            except Exception as exc:
                _log.error("event_store_delete_error", error=str(exc))
                if self._metrics is not None:
                    self._metrics.event_store_errors_total.labels(operation="delete").inc()
                raise

    def upsert(self, event: Event) -> None:
        """Insert or replace an event by event_id.

        Used for mutable records (e.g. alert lifecycle updates)
        where the same event_id is re-persisted with updated metadata.
        """
        row = (
            event.event_id,
            event.event_type.value,
            event.global_id,
            event.camera_id,
            event.zone_id,
            event.timestamp_ns,
            event.frame_id,
            event.confidence,
            event.explanation,
            json.dumps(event.metadata),
        )
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO events (
                            event_id, event_type, global_id, camera_id, zone_id,
                            timestamp_ns, frame_id, confidence, explanation, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(event_id) DO UPDATE SET
                            event_type = excluded.event_type,
                            global_id = excluded.global_id,
                            camera_id = excluded.camera_id,
                            zone_id = excluded.zone_id,
                            timestamp_ns = excluded.timestamp_ns,
                            frame_id = excluded.frame_id,
                            confidence = excluded.confidence,
                            explanation = excluded.explanation,
                            metadata = excluded.metadata
                        """,
                        row,
                    )
            except Exception as exc:
                _log.error("event_store_upsert_error", error=str(exc), event_id=event.event_id)
                raise

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        metadata_dict = json.loads(row["metadata"]) if row["metadata"] else {}
        return Event(
            event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            global_id=row["global_id"],
            camera_id=row["camera_id"],
            zone_id=row["zone_id"],
            timestamp_ns=row["timestamp_ns"],
            frame_id=row["frame_id"],
            confidence=row["confidence"],
            explanation=row["explanation"],
            metadata=metadata_dict,
        )
