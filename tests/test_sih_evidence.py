"""Unit tests for Evidence Capture and Retention Manager (Phase 10.2 — SIH 26187)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from gods_eye.evidence.evidence_manager import EvidenceManager


class TestEvidenceManager:
    def test_capture_evidence_metadata_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EvidenceManager(base_path=tmpdir, retention_days=30)

            frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
            ev_dir = mgr.capture_evidence(
                alert_id="alt_123",
                camera_id="cam-01",
                timestamp_ns=1_000_000_000,
                frame=frame,
                metadata={"rule": "RESTRICTED_ZONE_INTRUSION", "plate": "DL01AB1234"},
            )

            assert Path(ev_dir).exists()
            meta_path = Path(ev_dir) / "metadata.json"
            assert meta_path.exists()

            with open(meta_path, "r") as f:
                meta = json.load(f)

            assert meta["alert_id"] == "alt_123"
            assert meta["camera_id"] == "cam-01"
            assert meta["plate"] == "DL01AB1234"
            assert meta["has_snapshot"] is True

            snapshot_path = Path(ev_dir) / "snapshot.jpg"
            assert snapshot_path.exists()

    def test_get_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EvidenceManager(base_path=tmpdir)

            mgr.capture_evidence(
                alert_id="alt_456",
                camera_id="cam-02",
                timestamp_ns=1_000_000_000,
                metadata={"index": 1},
            )
            mgr.capture_evidence(
                alert_id="alt_456",
                camera_id="cam-02",
                timestamp_ns=1_001_000_000,
                metadata={"index": 2},
            )

            records = mgr.get_evidence("alt_456")
            assert len(records) == 2
            assert {r["index"] for r in records} == {1, 2}

            # Non-existent alert returns empty list
            assert mgr.get_evidence("alt_missing") == []

    def test_cleanup_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Retention = 0 days forces everything in the past to be expired
            mgr = EvidenceManager(base_path=tmpdir, retention_days=0)

            ev_dir = mgr.capture_evidence(
                alert_id="alt_old",
                camera_id="cam-01",
                timestamp_ns=1_000_000_000,
            )
            # Rewrite captured_at to an ancient timestamp
            meta_path = Path(ev_dir) / "metadata.json"
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["captured_at"] = 1_000_000  # ancient nanoseconds
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            removed = mgr.cleanup_expired()
            assert removed == 1
            assert not Path(ev_dir).exists()
