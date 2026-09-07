"""SQLite Alert Store — Phase 10.1 (SIH 26187).

Provides thread-safe alert persistence mirroring SQLiteEventStore pattern
(WAL mode, composite key ordering, append-only with lifecycle updates).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.alert import Alert, AlertStatus

_log = get_logger("alerts.alert_store")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    subject_ref TEXT,
    zone_id TEXT,
    camera_id TEXT NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    risk_signal_id TEXT NOT NULL,
    hypothesis_ids TEXT DEFAULT '[]',
    evidence_ids TEXT DEFAULT '[]',
    explanation TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    resolved_ns INTEGER,
    resolution_note TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id);
"""


class SQLiteAlertStore:
    """SQLite-backed alert persistence store.

    Mirrors the SQLiteEventStore pattern from Phase 4.2.
    """

    def __init__(self, db_path: str = "data/alerts.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(CREATE_TABLE_SQL)
        self._conn.commit()

    def append_alert(self, alert: Alert) -> None:
        """Persist a new alert."""
        import time

        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO alerts
                   (alert_id, severity, status, subject_ref, zone_id, camera_id,
                    timestamp_ns, risk_signal_id, hypothesis_ids, evidence_ids,
                    explanation, metadata, resolved_ns, resolution_note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert.alert_id,
                    alert.severity,
                    alert.status.value,
                    alert.subject_ref,
                    alert.zone_id,
                    alert.camera_id,
                    alert.timestamp_ns,
                    alert.risk_signal_id,
                    json.dumps(list(alert.hypothesis_ids)),
                    json.dumps(list(alert.evidence_ids)),
                    alert.explanation,
                    json.dumps(alert.metadata),
                    alert.resolved_ns,
                    alert.resolution_note,
                    time.time_ns(),
                ),
            )
            self._conn.commit()

    def append_batch(self, alerts: list[Alert]) -> None:
        """Persist a batch of alerts."""
        for alert in alerts:
            self.append_alert(alert)

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrieve a single alert by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()

        if row is None:
            return None
        return self._row_to_alert(row)

    def get_alerts_by_severity(
        self,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Retrieve alerts filtered by severity."""
        with self._lock:
            if severity:
                rows = self._conn.execute(
                    "SELECT * FROM alerts WHERE severity = ? ORDER BY timestamp_ns DESC LIMIT ?",
                    (severity, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM alerts ORDER BY timestamp_ns DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        return [self._row_to_alert(row) for row in rows]

    def get_active_alerts(self, limit: int = 100) -> list[Alert]:
        """Retrieve alerts that are not resolved."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM alerts WHERE status != 'resolved'
                   ORDER BY timestamp_ns DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        return [self._row_to_alert(row) for row in rows]

    def update_status(self, alert_id: str, status: AlertStatus,
                      resolved_ns: Optional[int] = None,
                      resolution_note: Optional[str] = None) -> None:
        """Update alert lifecycle status."""
        with self._lock:
            self._conn.execute(
                """UPDATE alerts SET status = ?, resolved_ns = ?, resolution_note = ?
                   WHERE alert_id = ?""",
                (status.value, resolved_ns, resolution_note, alert_id),
            )
            self._conn.commit()

    def count(self) -> int:
        """Return total alert count."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
            return row[0] if row else 0

    @staticmethod
    def _row_to_alert(row: tuple) -> Alert:
        """Convert a database row to an Alert object."""
        return Alert(
            alert_id=row[0],
            severity=row[1],
            status=AlertStatus(row[2]),
            subject_ref=row[3],
            zone_id=row[4],
            camera_id=row[5],
            timestamp_ns=row[6],
            risk_signal_id=row[7],
            hypothesis_ids=tuple(json.loads(row[8])) if row[8] else (),
            evidence_ids=tuple(json.loads(row[9])) if row[9] else (),
            explanation=row[10] or "",
            metadata=json.loads(row[11]) if row[11] else {},
            resolved_ns=row[12],
            resolution_note=row[13],
        )

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
