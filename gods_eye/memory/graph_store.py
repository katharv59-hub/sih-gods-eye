"""Spatiotemporal Graph Store — Phase 4.3 (§14, ADR-012).

Provides thread-safe spatiotemporal graph relationship projection for identities,
cameras, zones, observations, cross-camera transitions, and zone occupancy.
Uses SQLite WAL mode with composite key deterministic ordering (timestamp_ns, sequence_num, edge_id).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from gods_eye.observability.logger import get_logger
from gods_eye.observability.metrics import MetricsRegistry

_log = get_logger("memory.graph_store")


class BaseGraphStore(ABC):
    """Abstract Base Class for spatiotemporal graph storage engines."""

    @abstractmethod
    def upsert_identity_node(
        self,
        global_id: str,
        first_seen_ns: int,
        last_seen_ns: int,
        primary_camera_id: str,
        state: str = "ACTIVE",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def upsert_camera_node(
        self,
        camera_id: str,
        zone_id: Optional[str] = None,
        status: str = "ACTIVE",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def upsert_zone_node(
        self,
        zone_id: str,
        name: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def add_observation_edge(
        self,
        edge_id: str,
        global_id: str,
        camera_id: str,
        timestamp_ns: int,
        confidence: float = 1.0,
        track_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def add_transition_edge(
        self,
        edge_id: str,
        global_id: str,
        from_camera_id: str,
        to_camera_id: str,
        timestamp_ns: int,
        transition_duration_s: float = 0.0,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def add_zone_occupancy_edge(
        self,
        edge_id: str,
        global_id: str,
        zone_id: str,
        timestamp_ns: int,
        dwell_s: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def get_identity_trajectory(
        self,
        global_id: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_camera_transitions(
        self,
        from_camera_id: Optional[str] = None,
        to_camera_id: Optional[str] = None,
        global_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_zone_occupancy(
        self,
        zone_id: Optional[str] = None,
        global_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def purge_before(self, timestamp_ns: int) -> int:
        ...

    @abstractmethod
    def count_nodes(self) -> dict[str, int]:
        ...

    @abstractmethod
    def count_edges(self) -> dict[str, int]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class SQLiteGraphStore(BaseGraphStore):
    """SQLite implementation of Spatiotemporal Graph Store.

    Stores spatiotemporal projection nodes (Identity, Camera, Zone) and edges
    (Observation, Transition, Zone Occupancy) with SQLite WAL mode.
    Orders temporal queries deterministically by (timestamp_ns, sequence_num, edge_id).
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
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._create_schema()

    def _create_schema(self) -> None:
        with self._conn:
            # Nodes
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_identities (
                    global_id TEXT PRIMARY KEY,
                    first_seen_ns INTEGER NOT NULL,
                    last_seen_ns INTEGER NOT NULL,
                    primary_camera_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_cameras (
                    camera_id TEXT PRIMARY KEY,
                    zone_id TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_zones (
                    zone_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )

            # Edges with AUTOINCREMENT sequence_num for tie-breaking
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_observations (
                    sequence_num INTEGER PRIMARY KEY AUTOINCREMENT,
                    edge_id TEXT UNIQUE NOT NULL,
                    global_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    track_id TEXT,
                    metadata TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_transitions (
                    sequence_num INTEGER PRIMARY KEY AUTOINCREMENT,
                    edge_id TEXT UNIQUE NOT NULL,
                    global_id TEXT NOT NULL,
                    from_camera_id TEXT NOT NULL,
                    to_camera_id TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    transition_duration_s REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_zone_occupancy (
                    sequence_num INTEGER PRIMARY KEY AUTOINCREMENT,
                    edge_id TEXT UNIQUE NOT NULL,
                    global_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    dwell_s REAL NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )

            # Indexes for deterministic composite key sorting and fast retrieval
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_ts_seq ON edge_observations(timestamp_ns, sequence_num, edge_id);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_global_ts ON edge_observations(global_id, timestamp_ns);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_cam_ts ON edge_observations(camera_id, timestamp_ns);"
            )

            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trans_ts_seq ON edge_transitions(timestamp_ns, sequence_num, edge_id);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trans_global_ts ON edge_transitions(global_id, timestamp_ns);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trans_route ON edge_transitions(from_camera_id, to_camera_id, timestamp_ns);"
            )

            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_zone_occ_ts_seq ON edge_zone_occupancy(timestamp_ns, sequence_num, edge_id);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_zone_occ_zone_ts ON edge_zone_occupancy(zone_id, timestamp_ns);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_zone_occ_global_ts ON edge_zone_occupancy(global_id, timestamp_ns);"
            )

    def upsert_identity_node(
        self,
        global_id: str,
        first_seen_ns: int,
        last_seen_ns: int,
        primary_camera_id: str,
        state: str = "ACTIVE",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO node_identities (global_id, first_seen_ns, last_seen_ns, primary_camera_id, state, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(global_id) DO UPDATE SET
                            last_seen_ns = MAX(node_identities.last_seen_ns, excluded.last_seen_ns),
                            primary_camera_id = excluded.primary_camera_id,
                            state = excluded.state,
                            metadata = excluded.metadata
                        """,
                        (global_id, first_seen_ns, last_seen_ns, primary_camera_id, state, meta_json),
                    )
                if self._metrics is not None:
                    self._metrics.graph_writes_total.labels(record_type="node_identity").inc()
            except Exception as exc:
                _log.error("upsert_identity_node_error", error=str(exc), global_id=global_id)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="upsert_identity_node").inc()
                raise

    def upsert_camera_node(
        self,
        camera_id: str,
        zone_id: Optional[str] = None,
        status: str = "ACTIVE",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO node_cameras (camera_id, zone_id, status, metadata)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(camera_id) DO UPDATE SET
                            zone_id = excluded.zone_id,
                            status = excluded.status,
                            metadata = excluded.metadata
                        """,
                        (camera_id, zone_id, status, meta_json),
                    )
                if self._metrics is not None:
                    self._metrics.graph_writes_total.labels(record_type="node_camera").inc()
            except Exception as exc:
                _log.error("upsert_camera_node_error", error=str(exc), camera_id=camera_id)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="upsert_camera_node").inc()
                raise

    def upsert_zone_node(
        self,
        zone_id: str,
        name: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO node_zones (zone_id, name, metadata)
                        VALUES (?, ?, ?)
                        ON CONFLICT(zone_id) DO UPDATE SET
                            name = excluded.name,
                            metadata = excluded.metadata
                        """,
                        (zone_id, name, meta_json),
                    )
                if self._metrics is not None:
                    self._metrics.graph_writes_total.labels(record_type="node_zone").inc()
            except Exception as exc:
                _log.error("upsert_zone_node_error", error=str(exc), zone_id=zone_id)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="upsert_zone_node").inc()
                raise

    def add_observation_edge(
        self,
        edge_id: str,
        global_id: str,
        camera_id: str,
        timestamp_ns: int,
        confidence: float = 1.0,
        track_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO edge_observations (
                            edge_id, global_id, camera_id, timestamp_ns, confidence, track_id, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (edge_id, global_id, camera_id, timestamp_ns, confidence, track_id, meta_json),
                    )
                if self._metrics is not None:
                    self._metrics.graph_writes_total.labels(record_type="edge_observation").inc()
            except Exception as exc:
                _log.error("add_observation_edge_error", error=str(exc), edge_id=edge_id)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="add_observation_edge").inc()
                raise

    def add_transition_edge(
        self,
        edge_id: str,
        global_id: str,
        from_camera_id: str,
        to_camera_id: str,
        timestamp_ns: int,
        transition_duration_s: float = 0.0,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO edge_transitions (
                            edge_id, global_id, from_camera_id, to_camera_id,
                            timestamp_ns, transition_duration_s, confidence, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            edge_id,
                            global_id,
                            from_camera_id,
                            to_camera_id,
                            timestamp_ns,
                            transition_duration_s,
                            confidence,
                            meta_json,
                        ),
                    )
                if self._metrics is not None:
                    self._metrics.graph_writes_total.labels(record_type="edge_transition").inc()
            except Exception as exc:
                _log.error("add_transition_edge_error", error=str(exc), edge_id=edge_id)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="add_transition_edge").inc()
                raise

    def add_zone_occupancy_edge(
        self,
        edge_id: str,
        global_id: str,
        zone_id: str,
        timestamp_ns: int,
        dwell_s: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO edge_zone_occupancy (
                            edge_id, global_id, zone_id, timestamp_ns, dwell_s, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (edge_id, global_id, zone_id, timestamp_ns, dwell_s, meta_json),
                    )
                if self._metrics is not None:
                    self._metrics.graph_writes_total.labels(record_type="edge_zone_occupancy").inc()
            except Exception as exc:
                _log.error("add_zone_occupancy_edge_error", error=str(exc), edge_id=edge_id)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="add_zone_occupancy_edge").inc()
                raise

    def get_identity_trajectory(
        self,
        global_id: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses = ["global_id = ?"]
        params: list[Any] = [global_id]

        if start_ns is not None:
            clauses.append("timestamp_ns >= ?")
            params.append(start_ns)
        if end_ns is not None:
            clauses.append("timestamp_ns <= ?")
            params.append(end_ns)

        where_str = f"WHERE {' AND '.join(clauses)}"
        sql = f"SELECT * FROM edge_observations {where_str} ORDER BY timestamp_ns ASC, sequence_num ASC, edge_id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def get_camera_transitions(
        self,
        from_camera_id: Optional[str] = None,
        to_camera_id: Optional[str] = None,
        global_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if from_camera_id is not None:
            clauses.append("from_camera_id = ?")
            params.append(from_camera_id)
        if to_camera_id is not None:
            clauses.append("to_camera_id = ?")
            params.append(to_camera_id)
        if global_id is not None:
            clauses.append("global_id = ?")
            params.append(global_id)
        if start_ns is not None:
            clauses.append("timestamp_ns >= ?")
            params.append(start_ns)
        if end_ns is not None:
            clauses.append("timestamp_ns <= ?")
            params.append(end_ns)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM edge_transitions {where_str} ORDER BY timestamp_ns ASC, sequence_num ASC, edge_id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def get_zone_occupancy(
        self,
        zone_id: Optional[str] = None,
        global_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if zone_id is not None:
            clauses.append("zone_id = ?")
            params.append(zone_id)
        if global_id is not None:
            clauses.append("global_id = ?")
            params.append(global_id)
        if start_ns is not None:
            clauses.append("timestamp_ns >= ?")
            params.append(start_ns)
        if end_ns is not None:
            clauses.append("timestamp_ns <= ?")
            params.append(end_ns)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM edge_zone_occupancy {where_str} ORDER BY timestamp_ns ASC, sequence_num ASC, edge_id ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def purge_before(self, timestamp_ns: int) -> int:
        with self._lock:
            try:
                with self._conn:
                    c1 = self._conn.execute(
                        "DELETE FROM edge_observations WHERE timestamp_ns < ?", (timestamp_ns,)
                    ).rowcount
                    c2 = self._conn.execute(
                        "DELETE FROM edge_transitions WHERE timestamp_ns < ?", (timestamp_ns,)
                    ).rowcount
                    c3 = self._conn.execute(
                        "DELETE FROM edge_zone_occupancy WHERE timestamp_ns < ?", (timestamp_ns,)
                    ).rowcount
                    total_purged = c1 + c2 + c3

                if self._metrics is not None and total_purged > 0:
                    self._metrics.graph_records_purged_total.inc(total_purged)
                return total_purged
            except Exception as exc:
                _log.error("graph_store_purge_error", error=str(exc), timestamp_ns=timestamp_ns)
                if self._metrics is not None:
                    self._metrics.graph_errors_total.labels(operation="purge").inc()
                raise

    def count_nodes(self) -> dict[str, int]:
        with self._lock:
            n_id = self._conn.execute("SELECT COUNT(*) FROM node_identities").fetchone()[0]
            n_cam = self._conn.execute("SELECT COUNT(*) FROM node_cameras").fetchone()[0]
            n_zone = self._conn.execute("SELECT COUNT(*) FROM node_zones").fetchone()[0]
            return {"identities": n_id, "cameras": n_cam, "zones": n_zone}

    def count_edges(self) -> dict[str, int]:
        with self._lock:
            e_obs = self._conn.execute("SELECT COUNT(*) FROM edge_observations").fetchone()[0]
            e_trans = self._conn.execute("SELECT COUNT(*) FROM edge_transitions").fetchone()[0]
            e_occ = self._conn.execute("SELECT COUNT(*) FROM edge_zone_occupancy").fetchone()[0]
            return {"observations": e_obs, "transitions": e_trans, "zone_occupancy": e_occ}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
