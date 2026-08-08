"""Unit tests and benchmarks for Spatiotemporal GraphStore & GraphWriter — Phase 4.3."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
import pytest
from prometheus_client import CollectorRegistry

from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.memory.graph_writer import GraphRecord, GraphWriter
from gods_eye.observability.metrics import MetricsRegistry


class TestSQLiteGraphStore:

    def test_node_creation_and_upsert(self) -> None:
        store = SQLiteGraphStore(":memory:")

        store.upsert_camera_node("cam_01", zone_id="z_east", status="ACTIVE")
        store.upsert_zone_node("z_east", name="East Entrance")
        store.upsert_identity_node(
            global_id="gid_01",
            first_seen_ns=1000,
            last_seen_ns=2000,
            primary_camera_id="cam_01",
            state="ACTIVE",
        )

        counts = store.count_nodes()
        assert counts["identities"] == 1
        assert counts["cameras"] == 1
        assert counts["zones"] == 1

        # Upsert identity node update
        store.upsert_identity_node(
            global_id="gid_01",
            first_seen_ns=1000,
            last_seen_ns=5000,
            primary_camera_id="cam_02",
            state="ACTIVE",
        )
        assert store.count_nodes()["identities"] == 1
        store.close()

    def test_observation_edges_and_trajectory_query(self) -> None:
        store = SQLiteGraphStore(":memory:")
        store.upsert_identity_node("gid_01", 1000, 5000, "cam_01")

        store.add_observation_edge("edge_1", "gid_01", "cam_01", 1000, confidence=0.9, track_id="t1")
        store.add_observation_edge("edge_2", "gid_01", "cam_01", 2000, confidence=0.95, track_id="t1")
        store.add_observation_edge("edge_3", "gid_01", "cam_02", 3000, confidence=0.92, track_id="t2")

        traj = store.get_identity_trajectory("gid_01")
        assert len(traj) == 3
        assert [e["camera_id"] for e in traj] == ["cam_01", "cam_01", "cam_02"]
        assert [e["edge_id"] for e in traj] == ["edge_1", "edge_2", "edge_3"]

        # Timestamp range filtering
        traj_sub = store.get_identity_trajectory("gid_01", start_ns=1500, end_ns=3500)
        assert len(traj_sub) == 2
        assert [e["edge_id"] for e in traj_sub] == ["edge_2", "edge_3"]

        store.close()

    def test_transition_edges_and_query(self) -> None:
        store = SQLiteGraphStore(":memory:")

        store.add_transition_edge("t_edge_1", "gid_01", "cam_01", "cam_02", 2500, transition_duration_s=1.5, confidence=0.88)
        store.add_transition_edge("t_edge_2", "gid_02", "cam_01", "cam_02", 3500, transition_duration_s=2.0, confidence=0.90)

        trans_route = store.get_camera_transitions(from_camera_id="cam_01", to_camera_id="cam_02")
        assert len(trans_route) == 2
        assert [e["global_id"] for e in trans_route] == ["gid_01", "gid_02"]

        trans_id = store.get_camera_transitions(global_id="gid_01")
        assert len(trans_id) == 1
        assert trans_id[0]["edge_id"] == "t_edge_1"

        store.close()

    def test_zone_occupancy_edges_and_query(self) -> None:
        store = SQLiteGraphStore(":memory:")

        store.add_zone_occupancy_edge("z_edge_1", "gid_01", "zone_a", 1000, dwell_s=10.5)
        store.add_zone_occupancy_edge("z_edge_2", "gid_02", "zone_a", 1500, dwell_s=5.0)

        occ = store.get_zone_occupancy(zone_id="zone_a")
        assert len(occ) == 2
        assert [e["global_id"] for e in occ] == ["gid_01", "gid_02"]

        store.close()

    def test_deterministic_composite_key_ordering_identical_timestamps(self) -> None:
        """Requirement: Graph edges sharing identical timestamps order deterministically via sequence_num & edge_id."""
        store = SQLiteGraphStore(":memory:")
        ts = 5000

        store.add_observation_edge("edge_b", "gid_01", "cam_01", ts)
        store.add_observation_edge("edge_a", "gid_01", "cam_01", ts)
        store.add_observation_edge("edge_c", "gid_01", "cam_01", ts)

        traj = store.get_identity_trajectory("gid_01")
        assert len(traj) == 3
        # Insertion sequence_num order tie-breaking
        assert [e["edge_id"] for e in traj] == ["edge_b", "edge_a", "edge_c"]

        store.close()

    def test_retention_purge(self) -> None:
        """Requirement: Purge graph edges older than retention cutoff timestamp."""
        metrics = MetricsRegistry(registry=CollectorRegistry())
        store = SQLiteGraphStore(":memory:", metrics=metrics)

        store.add_observation_edge("e1", "gid_01", "cam_01", 100)
        store.add_observation_edge("e2", "gid_01", "cam_01", 200)
        store.add_observation_edge("e3", "gid_01", "cam_01", 300)

        store.add_transition_edge("t1", "gid_01", "cam_01", "cam_02", 150)
        store.add_zone_occupancy_edge("z1", "gid_01", "z1", 180)

        assert store.count_edges()["observations"] == 3
        assert store.count_edges()["transitions"] == 1
        assert store.count_edges()["zone_occupancy"] == 1

        purged_count = store.purge_before(250)
        assert purged_count == 4  # e1, e2, t1, z1 purged
        assert store.count_edges()["observations"] == 1
        assert store.count_edges()["transitions"] == 0
        assert store.count_edges()["zone_occupancy"] == 0

        store.close()

    def test_wal_configuration(self, tmp_path) -> None:
        """Requirement: Verify SQLite WAL mode configuration on disk-backed database."""
        db_file = str(tmp_path / "test_graph_wal.db")
        store = SQLiteGraphStore(db_file)

        mode = store._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync = store._conn.execute("PRAGMA synchronous;").fetchone()[0]

        assert mode.lower() == "wal"
        assert sync == 1  # NORMAL

        store.close()

    def test_concurrent_writer_safety(self) -> None:
        """Requirement: Multiple concurrent writer threads append graph edges without corruption."""
        store = SQLiteGraphStore(":memory:")
        threads: list[threading.Thread] = []
        num_threads = 4
        records_per_thread = 50

        def _worker(t_idx: int) -> None:
            for i in range(records_per_thread):
                edge_id = f"e_t{t_idx}_{i}"
                store.add_observation_edge(edge_id, f"gid_{t_idx}", "cam_01", 1000 + i)

        for i in range(num_threads):
            t = threading.Thread(target=_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert store.count_edges()["observations"] == num_threads * records_per_thread
        store.close()

    def test_throughput_benchmark(self) -> None:
        """Requirement: Benchmark GraphStore write performance for graph records."""
        store = SQLiteGraphStore(":memory:")
        num_records = 5000

        start_t = time.time()
        with store._conn:
            for i in range(num_records):
                store._conn.execute(
                    "INSERT INTO edge_observations (edge_id, global_id, camera_id, timestamp_ns, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"e_{i}", "gid_bench", "cam_01", i * 1000, 0.9, "{}"),
                )
        elapsed_s = time.time() - start_t

        throughput = num_records / elapsed_s
        print(f"\n[BENCHMARK] SQLiteGraphStore Write Throughput: {throughput:.1f} records/sec")
        assert store.count_edges()["observations"] == num_records
        store.close()


class TestGraphWriter:

    def test_async_batch_writer(self) -> None:
        store = SQLiteGraphStore(":memory:")
        writer = GraphWriter(store, batch_size=10, flush_interval_s=0.1)
        writer.start()

        writer.emit(GraphRecord("identity_node", {"global_id": "gid_w1", "first_seen_ns": 1000, "last_seen_ns": 2000, "primary_camera_id": "cam_01"}))
        writer.emit(GraphRecord("camera_node", {"camera_id": "cam_01"}))

        for i in range(15):
            writer.emit(GraphRecord("obs_edge", {"edge_id": f"e_w_{i}", "global_id": "gid_w1", "camera_id": "cam_01", "timestamp_ns": 1000 + i}))

        time.sleep(0.3)
        writer.stop()

        assert store.count_nodes()["identities"] == 1
        assert store.count_nodes()["cameras"] == 1
        assert store.count_edges()["observations"] == 15

        store.close()
