"""Unit tests and benchmarks for TimelineReconstructionEngine & DeterministicReplayEngine — Phase 4.4."""

from __future__ import annotations

import time
import uuid
import pytest

from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.memory.replay_engine import DeterministicReplayEngine
from gods_eye.memory.timeline_engine import TimelineReconstructionEngine
from gods_eye.schemas.event import Event, EventType


def _make_event(
    global_id: str = "gid_01",
    camera_id: str = "cam_01",
    event_type: EventType = EventType.CROSS_CAMERA_TRANSITION,
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
        confidence=0.9,
        explanation="Test transition",
    )


class TestTimelineReconstructionEngine:

    def test_empty_history(self) -> None:
        """Requirement 12: Reconstructing empty history returns empty timeline cleanly."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = TimelineReconstructionEngine(es, gs)

        tl = engine.reconstruct_identity_timeline("gid_nonexistent")
        assert tl.global_id == "gid_nonexistent"
        assert len(tl.visits) == 0
        assert len(tl.transitions) == 0
        assert len(tl.events) == 0

        es.close()
        gs.close()

    def test_single_observation(self) -> None:
        """Requirement 13: Single observation produces single visit segment."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = TimelineReconstructionEngine(es, gs)

        gs.add_observation_edge("obs_1", "gid_01", "cam_01", 1000)

        tl = engine.reconstruct_identity_timeline("gid_01")
        assert len(tl.visits) == 1
        v = tl.visits[0]
        assert v.camera_id == "cam_01"
        assert v.start_ns == 1000
        assert v.dwell_s == 0.0
        assert v.observation_count == 1

        es.close()
        gs.close()

    def test_identity_timeline_visit_reconstruction_and_dwell(self) -> None:
        """Requirement 1, 2, 5, 6: Contiguous visit grouping, dwell calculation, and open vs complete visit handling."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = TimelineReconstructionEngine(es, gs)

        # Cam 1 visit (1000 to 3000 ns = 2.0 sec)
        gs.add_observation_edge("o1", "gid_01", "cam_01", 1000)
        gs.add_observation_edge("o2", "gid_01", "cam_01", 2000)
        gs.add_observation_edge("o3", "gid_01", "cam_01", 3000)

        # Transition to Cam 2
        gs.add_transition_edge("t1", "gid_01", "cam_01", "cam_02", 3500, transition_duration_s=0.5)

        # Cam 2 visit (4000 to 5000 ns = 1.0 sec)
        gs.add_observation_edge("o4", "gid_01", "cam_02", 4000)
        gs.add_observation_edge("o5", "gid_01", "cam_02", 5000)

        tl = engine.reconstruct_identity_timeline("gid_01")
        assert len(tl.visits) == 2

        v1 = tl.visits[0]
        assert v1.camera_id == "cam_01"
        assert v1.start_ns == 1000
        assert v1.end_ns == 3000
        assert abs(v1.dwell_s - 2.0e-6) < 1e-12
        assert v1.observation_count == 3
        assert v1.is_complete is True

        v2 = tl.visits[1]
        assert v2.camera_id == "cam_02"
        assert v2.start_ns == 4000
        assert v2.end_ns == 5000

        es.close()
        gs.close()

    def test_camera_and_zone_timelines(self) -> None:
        """Requirement 3, 4, 14, 15: Camera activity timeline and Zone occupancy timeline."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = TimelineReconstructionEngine(es, gs)

        e1 = _make_event(global_id="gid_01", camera_id="cam_01", timestamp_ns=1000)
        es.append(e1)
        gs.add_zone_occupancy_edge("ze1", "gid_01", "zone_a", 1000, dwell_s=15.0)
        gs.add_zone_occupancy_edge("ze2", "gid_02", "zone_a", 1200, dwell_s=10.0)

        cam_tl = engine.reconstruct_camera_timeline("cam_01")
        assert cam_tl.camera_id == "cam_01"
        assert "gid_01" in cam_tl.active_identities
        assert len(cam_tl.events) == 1

        zone_tl = engine.reconstruct_zone_timeline("zone_a")
        assert zone_tl.zone_id == "zone_a"
        assert set(zone_tl.unique_identities) == {"gid_01", "gid_02"}
        assert len(zone_tl.occupancy_segments) == 2

        es.close()
        gs.close()

    def test_time_range_filtering_and_retention_boundary(self) -> None:
        """Requirement 7, 17: Time range filtering respects start_ns / end_ns bounds."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = TimelineReconstructionEngine(es, gs)

        gs.add_observation_edge("o1", "gid_01", "cam_01", 1000)
        gs.add_observation_edge("o2", "gid_01", "cam_01", 2000)
        gs.add_observation_edge("o3", "gid_01", "cam_01", 3000)

        tl_filtered = engine.reconstruct_identity_timeline("gid_01", start_ns=1500, end_ns=2500)
        assert len(tl_filtered.visits) == 1
        assert tl_filtered.visits[0].observation_count == 1
        assert tl_filtered.visits[0].start_ns == 2000

        es.close()
        gs.close()


class TestDeterministicReplayEngine:

    def test_deterministic_replay_ordering(self) -> None:
        """Requirement 8, 9, 10, 16: Deterministic replay ordering across mixed events and graph records."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = DeterministicReplayEngine(es, gs)

        ts = 2000
        e1 = _make_event(global_id="gid_01", camera_id="cam_01", timestamp_ns=ts, event_id="evt_01")
        es.append(e1)
        gs.add_observation_edge("edge_01", "gid_01", "cam_01", ts)
        gs.add_transition_edge("trans_01", "gid_01", "cam_01", "cam_02", ts)

        stream_list_1 = list(engine.stream_replay(start_ns=1000, end_ns=3000, global_id="gid_01"))
        stream_list_2 = list(engine.stream_replay(start_ns=1000, end_ns=3000, global_id="gid_01"))

        assert len(stream_list_1) == 3
        # Strict deterministic replay invariant
        keys_1 = [(f.timestamp_ns, f.sequence_num, f.record_id) for f in stream_list_1]
        keys_2 = [(f.timestamp_ns, f.sequence_num, f.record_id) for f in stream_list_2]
        assert keys_1 == keys_2

        es.close()
        gs.close()

    def test_replay_read_only_guarantee(self) -> None:
        """Requirement 11: Replay execution leaves EventStore and GraphStore count completely unmodified."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = DeterministicReplayEngine(es, gs)

        es.append(_make_event(timestamp_ns=1000))
        gs.add_observation_edge("o1", "gid_01", "cam_01", 1000)

        c_event_before = es.count()
        c_nodes_before = gs.count_nodes()
        c_edges_before = gs.count_edges()

        # Run replay
        _ = list(engine.stream_replay(start_ns=0, end_ns=5000, global_id="gid_01"))

        assert es.count() == c_event_before
        assert gs.count_nodes() == c_nodes_before
        assert gs.count_edges() == c_edges_before

        es.close()
        gs.close()


class TestTimelinePerformanceBenchmark:

    def test_timeline_query_p95_latency_benchmark(self) -> None:
        """Requirement 18: Timeline query latency <= 200 ms p95 for 1-hour window."""
        es = SQLiteEventStore(":memory:")
        gs = SQLiteGraphStore(":memory:")
        engine = TimelineReconstructionEngine(es, gs)

        # Seed 1,000 observations and 100 events across 1 hour
        base_ts = 1_000_000_000
        for i in range(1000):
            gs.add_observation_edge(f"e_{i}", "gid_perf", "cam_01", base_ts + i * 3_600_000)

        for i in range(100):
            es.append(_make_event(global_id="gid_perf", timestamp_ns=base_ts + i * 36_000_000))

        latencies_ms: list[float] = []
        for _ in range(20):
            start_t = time.time()
            _ = engine.reconstruct_identity_timeline("gid_perf", start_ns=base_ts, end_ns=base_ts + 3_600_000_000_000)
            latencies_ms.append((time.time() - start_t) * 1000.0)

        latencies_ms.sort()
        p95_ms = latencies_ms[int(0.95 * len(latencies_ms))]
        print(f"\n[BENCHMARK] Timeline Reconstruction p95 Latency: {p95_ms:.2f} ms (Master Spec Gate: <= 200 ms)")

        assert p95_ms <= 200.0, f"Timeline p95 latency {p95_ms:.2f} ms exceeded 200 ms Master Spec Gate"

        es.close()
        gs.close()
