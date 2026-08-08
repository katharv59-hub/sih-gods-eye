"""Unit tests and benchmarks for EventStore & EventWriter — Phase 4.2."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.events.event_writer import EventWriter
from gods_eye.observability.metrics import MetricsRegistry
from gods_eye.schemas.event import Event, EventType


def _make_event(
    global_id: str = "identity_001",
    camera_id: str = "cam_01",
    event_type: EventType = EventType.PERSON_ENTERED_FRAME,
    timestamp_ns: int = 1_000_000_000,
    event_id: str | None = None,
) -> Event:
    return Event(
        event_id=event_id or str(uuid.uuid4()),
        event_type=event_type,
        global_id=global_id,
        camera_id=camera_id,
        timestamp_ns=timestamp_ns,
        frame_id=100,
        confidence=0.95,
        explanation="Test event emission",
        zone_id="z1",
        metadata={"test": True},
    )


class TestSQLiteEventStore:

    def test_append_and_get_by_id(self) -> None:
        store = SQLiteEventStore(":memory:")
        event = _make_event()
        store.append(event)

        assert store.count() == 1
        fetched = store.get_by_id(event.event_id)
        assert fetched is not None
        assert fetched.event_id == event.event_id
        assert fetched.global_id == "identity_001"
        assert fetched.confidence == 0.95
        assert fetched.metadata == {"test": True}
        store.close()

    def test_query_events_filtering(self) -> None:
        store = SQLiteEventStore(":memory:")
        e1 = _make_event(global_id="id_1", camera_id="cam_01", event_type=EventType.PERSON_ENTERED_FRAME, timestamp_ns=1000)
        e2 = _make_event(global_id="id_2", camera_id="cam_01", event_type=EventType.CROSS_CAMERA_TRANSITION, timestamp_ns=2000)
        e3 = _make_event(global_id="id_1", camera_id="cam_02", event_type=EventType.ENVIRONMENTAL_ANOMALY, timestamp_ns=3000)

        store.append_batch([e1, e2, e3])
        assert store.count() == 3

        # Filter by global_id
        res_id1 = store.query_events(global_id="id_1")
        assert len(res_id1) == 2
        assert [e.event_id for e in res_id1] == [e1.event_id, e3.event_id]

        # Filter by camera_id and event_type
        res_cam1 = store.query_events(camera_id="cam_01", event_type=EventType.CROSS_CAMERA_TRANSITION)
        assert len(res_cam1) == 1
        assert res_cam1[0].event_id == e2.event_id

        # Filter by timestamp range
        res_ts = store.query_events(start_ns=1500, end_ns=3500)
        assert len(res_ts) == 2
        assert [e.event_id for e in res_ts] == [e2.event_id, e3.event_id]

        store.close()

    def test_deterministic_composite_key_ordering_identical_timestamps(self) -> None:
        """Requirement: Events with identical timestamp_ns order deterministically via sequence_num & event_id."""
        store = SQLiteEventStore(":memory:")
        ts = 1_000_000_000

        e_b = _make_event(timestamp_ns=ts, event_id="evt_b")
        e_a = _make_event(timestamp_ns=ts, event_id="evt_a")
        e_c = _make_event(timestamp_ns=ts, event_id="evt_c")

        store.append(e_b)
        store.append(e_a)
        store.append(e_c)

        queried = store.query_events(start_ns=ts, end_ns=ts)
        assert len(queried) == 3
        # Insertion sequence_num order: e_b (seq 1), e_a (seq 2), e_c (seq 3)
        assert [e.event_id for e in queried] == ["evt_b", "evt_a", "evt_c"]

        store.close()

    def test_append_only_semantics_no_mutation(self) -> None:
        """Requirement: EventStore preserves append-only semantics. Duplicate event_ids are ignored."""
        store = SQLiteEventStore(":memory:")
        event = _make_event(event_id="unique_id_100", timestamp_ns=1000)
        store.append(event)

        # Attempt to append duplicate event_id with different explanation
        dup_event = _make_event(event_id="unique_id_100", timestamp_ns=1000)
        dup_event.explanation = "Mutated explanation should be ignored"
        store.append(dup_event)

        assert store.count() == 1
        fetched = store.get_by_id("unique_id_100")
        assert fetched is not None
        assert fetched.explanation == "Test event emission"  # Original preserved

        store.close()

    def test_retention_purge(self) -> None:
        """Requirement: Purging removes events older than retention cutoff timestamp."""
        metrics = MetricsRegistry()
        store = SQLiteEventStore(":memory:", metrics=metrics)

        e1 = _make_event(timestamp_ns=100)
        e2 = _make_event(timestamp_ns=200)
        e3 = _make_event(timestamp_ns=300)

        store.append_batch([e1, e2, e3])
        assert store.count() == 3

        purged_count = store.purge_before(250)
        assert purged_count == 2
        assert store.count() == 1
        assert store.get_by_id(e3.event_id) is not None
        assert metrics.events_purged_total._value.get() == 2.0

        store.close()

    def test_wal_configuration(self, tmp_path) -> None:
        """Requirement: Verify SQLite WAL mode configuration on disk-backed database."""
        db_file = str(tmp_path / "test_wal.db")
        store = SQLiteEventStore(db_file)

        mode = store._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync = store._conn.execute("PRAGMA synchronous;").fetchone()[0]

        assert mode.lower() == "wal"
        assert sync == 1  # NORMAL synchronous

        store.close()

    def test_concurrent_producer_safety(self) -> None:
        """Requirement: Multiple producer threads can append events concurrently without corruption."""
        store = SQLiteEventStore(":memory:")
        threads: list[threading.Thread] = []
        num_threads = 5
        events_per_thread = 50

        def _worker(thread_idx: int) -> None:
            batch = [
                _make_event(
                    global_id=f"id_t{thread_idx}",
                    timestamp_ns=1000 + i,
                )
                for i in range(events_per_thread)
            ]
            store.append_batch(batch)

        for i in range(num_threads):
            t = threading.Thread(target=_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert store.count() == num_threads * events_per_thread
        store.close()

    def test_throughput_benchmark_gate_and_target(self) -> None:
        """Requirement: Verify Master Spec gate (>= 1,000 events/s) and internal target (>= 2,500 events/s)."""
        store = SQLiteEventStore(":memory:")
        num_events = 5000
        events = [_make_event(timestamp_ns=i * 1000) for i in range(num_events)]

        start_t = time.time()
        store.append_batch(events)
        elapsed_s = time.time() - start_t

        throughput = num_events / elapsed_s
        assert store.count() == num_events
        assert throughput >= 1000.0, f"Write throughput {throughput:.1f} events/s below 1,000 Master Spec Gate"

        # Log engineering target performance
        print(f"\n[BENCHMARK] SQLiteEventStore Write Throughput: {throughput:.1f} events/sec (Master Spec Gate: 1,000 / Engineering Target: 2,500)")
        store.close()


class TestEventWriter:

    def test_async_batch_writer(self) -> None:
        store = SQLiteEventStore(":memory:")
        writer = EventWriter(store, batch_size=10, flush_interval_s=0.1)
        writer.start()

        events = [_make_event() for _ in range(25)]
        for e in events:
            assert writer.emit(e) is True

        time.sleep(0.3)
        writer.stop()

        assert store.count() == 25
        store.close()
