"""Unit tests for Alert Engine, Alert Store, and Alert Writer (Phase 10.1 — SIH 26187).

Tests alert persistence through the canonical EventStore adapter architecture.
Includes 8 architectural regression tests proving no duplicate persistence store.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import pytest

from gods_eye.alerts.alert_engine import AlertEngine
from gods_eye.alerts.alert_store import AlertStore
from gods_eye.alerts.alert_writer import AlertWriter
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.schemas.alert import Alert, AlertStatus
from gods_eye.schemas.event import EventType
from gods_eye.schemas.situational import RiskSignal


class TestAlertEngine:
    def test_severity_filtering(self) -> None:
        engine = AlertEngine(alert_min_severity="HIGH")

        low_signal = RiskSignal(
            signal_id="sig_low",
            window_start_ns=0,
            window_end_ns=1_000_000_000,
            risk_level="LOW",
            risk_score=0.2,
            evidence_ids=(),
            contributing_factors=(),
            explanation="Low baseline risk",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )
        alert_low = engine.evaluate_risk_signal(low_signal, camera_id="cam-01")
        assert alert_low is None

        high_signal = RiskSignal(
            signal_id="sig_high",
            window_start_ns=0,
            window_end_ns=1_000_000_000,
            risk_level="HIGH",
            risk_score=0.85,
            evidence_ids=("ev-1",),
            contributing_factors=("trajectory_anomaly_sigma",),
            explanation="High risk detected",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )
        alert_high = engine.evaluate_risk_signal(high_signal, camera_id="cam-01")
        assert alert_high is not None
        assert alert_high.severity == "HIGH"
        assert alert_high.status == AlertStatus.DETECTED

    def test_deduplication_window(self) -> None:
        engine = AlertEngine(alert_min_severity="MEDIUM", incident_merge_window_s=10.0)

        sig1 = RiskSignal(
            signal_id="sig1",
            window_start_ns=0,
            window_end_ns=1_000_000_000,
            risk_level="HIGH",
            risk_score=0.8,
            evidence_ids=(),
            contributing_factors=(),
            explanation="Alert 1",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )
        # Alert 1: subject_ref="p1", zone_id="z1", t = 1s
        a1 = engine.evaluate_risk_signal(
            sig1, camera_id="cam-01", subject_ref="p1", zone_id="z1"
        )
        assert a1 is not None

        # Alert 2: same subject & zone, t = 5s (elapsed 4s < 10s window) -> Deduplicated
        sig2 = RiskSignal(
            signal_id="sig2",
            window_start_ns=0,
            window_end_ns=5_000_000_000,
            risk_level="HIGH",
            risk_score=0.85,
            evidence_ids=(),
            contributing_factors=(),
            explanation="Alert 2",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )
        a2 = engine.evaluate_risk_signal(
            sig2, camera_id="cam-01", subject_ref="p1", zone_id="z1"
        )
        assert a2 is None

        # Alert 3: same subject & zone, t = 15s (elapsed 14s > 10s window) -> New alert
        sig3 = RiskSignal(
            signal_id="sig3",
            window_start_ns=0,
            window_end_ns=15_000_000_000,
            risk_level="HIGH",
            risk_score=0.85,
            evidence_ids=(),
            contributing_factors=(),
            explanation="Alert 3",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )
        a3 = engine.evaluate_risk_signal(
            sig3, camera_id="cam-01", subject_ref="p1", zone_id="z1"
        )
        assert a3 is not None

    def test_lifecycle_transitions(self) -> None:
        engine = AlertEngine()
        sig = RiskSignal(
            signal_id="sig_test",
            window_start_ns=0,
            window_end_ns=1_000_000_000,
            risk_level="CRITICAL",
            risk_score=0.95,
            evidence_ids=(),
            contributing_factors=(),
            explanation="Critical risk",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )
        detected = engine.evaluate_risk_signal(sig, camera_id="cam-01")
        assert detected is not None
        assert detected.status == AlertStatus.DETECTED

        confirmed = engine.confirm_alert(detected)
        assert confirmed.status == AlertStatus.CONFIRMED

        resolved = engine.resolve_alert(confirmed, resolution_note="Investigated and secured")
        assert resolved.status == AlertStatus.RESOLVED
        assert resolved.resolution_note == "Investigated and secured"
        assert resolved.resolved_ns is not None


class TestAlertStore:
    """Tests for AlertStore adapter persisting through canonical EventStore."""

    def test_store_crud(self) -> None:
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        alert = Alert(
            alert_id="alt_test_01",
            severity="HIGH",
            status=AlertStatus.DETECTED,
            subject_ref="person_99",
            zone_id="zone_vault",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            risk_signal_id="sig_01",
            hypothesis_ids=("hyp_01",),
            evidence_ids=("ev_01", "ev_02"),
            explanation="Test intrusion alert",
        )

        store.append_alert(alert)

        # Retrieve by id
        retrieved = store.get_alert("alt_test_01")
        assert retrieved is not None
        assert retrieved.alert_id == "alt_test_01"
        assert retrieved.severity == "HIGH"
        assert retrieved.status == AlertStatus.DETECTED
        assert retrieved.evidence_ids == ("ev_01", "ev_02")

        # Query by severity
        high_alerts = store.get_alerts_by_severity(severity="HIGH")
        assert len(high_alerts) == 1

        # Active alerts
        active_alerts = store.get_active_alerts()
        assert len(active_alerts) == 1

        # Update status
        store.update_status(
            alert_id="alt_test_01",
            status=AlertStatus.RESOLVED,
            resolved_ns=2_000_000_000,
            resolution_note="False alarm confirmed",
        )

        updated = store.get_alert("alt_test_01")
        assert updated is not None
        assert updated.status == AlertStatus.RESOLVED
        assert updated.resolution_note == "False alarm confirmed"


class TestAlertWriter:
    def test_async_flush(self) -> None:
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)
        writer = AlertWriter(store=store, batch_size=2, flush_interval_s=0.1)
        try:
            writer.start()

            alert = Alert(
                alert_id="alt_async_01",
                severity="CRITICAL",
                status=AlertStatus.DETECTED,
                subject_ref="person_42",
                zone_id="zone_secure",
                camera_id="cam-02",
                timestamp_ns=1_000_000_000,
                risk_signal_id="sig_async",
                explanation="Async batch test alert",
            )

            success = writer.emit(alert)
            assert success is True

            writer.stop()

            saved = store.get_alert("alt_async_01")
            assert saved is not None
            assert saved.severity == "CRITICAL"
        finally:
            event_store.close()


class TestAlertArchitecturalRegression:
    """8 architectural regression tests proving no duplicate persistence store."""

    def test_1_no_independent_alert_database_required(self) -> None:
        """Test 1: No independent alert database is required."""
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        # AlertStore wraps the canonical event store — no separate db
        assert store.event_store is event_store
        assert not hasattr(store, "_conn")
        assert not hasattr(store, "_db_path")

    def test_2_alert_persistence_uses_canonical_event_store(self) -> None:
        """Test 2: Alert persistence uses the canonical event store."""
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        alert = Alert(
            alert_id="alt_arch_01",
            severity="HIGH",
            status=AlertStatus.DETECTED,
            subject_ref="person_1",
            zone_id="zone_a",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            risk_signal_id="sig_01",
            explanation="Architectural test",
        )
        store.append_alert(alert)

        # Verify the alert is stored as an Event in the canonical event store
        event = event_store.get_by_id("alt_arch_01")
        assert event is not None
        assert event.event_type == EventType.ALERT_CREATED
        assert event.metadata["alert_severity"] == "HIGH"

    def test_3_creating_alert_does_not_instantiate_second_database(self) -> None:
        """Test 3: Creating an alert does not instantiate a second database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")
            event_store = SQLiteEventStore(db_path=db_path)
            store = AlertStore(event_store=event_store)

            alert = Alert(
                alert_id="alt_no_db",
                severity="MEDIUM",
                status=AlertStatus.DETECTED,
                subject_ref=None,
                zone_id=None,
                camera_id="cam-01",
                timestamp_ns=1_000_000_000,
                risk_signal_id="sig_no_db",
                explanation="No second DB test",
            )
            store.append_alert(alert)

            # Only events.db should exist, NOT alerts.db
            files = list(Path(tmpdir).glob("*.db*"))
            db_names = {f.name for f in files}
            assert "events.db" in db_names or "events.db-wal" in db_names or any("events" in f.name for f in files)
            assert "alerts.db" not in db_names
            assert not any("alerts" in f.name for f in files)

            event_store.close()

    def test_4_alert_retrieval_works(self) -> None:
        """Test 4: Alert retrieval works."""
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        alert = Alert(
            alert_id="alt_retrieve",
            severity="CRITICAL",
            status=AlertStatus.ACTIVE,
            subject_ref="person_42",
            zone_id="zone_vault",
            camera_id="cam-01",
            timestamp_ns=2_000_000_000,
            risk_signal_id="sig_retrieve",
            hypothesis_ids=("hyp_1", "hyp_2"),
            evidence_ids=("ev_1",),
            explanation="Retrieval test",
        )
        store.append_alert(alert)

        retrieved = store.get_alert("alt_retrieve")
        assert retrieved is not None
        assert retrieved.alert_id == "alt_retrieve"
        assert retrieved.severity == "CRITICAL"
        assert retrieved.status == AlertStatus.ACTIVE
        assert retrieved.hypothesis_ids == ("hyp_1", "hyp_2")
        assert retrieved.evidence_ids == ("ev_1",)

    def test_5_alert_lifecycle_transitions_persistent(self) -> None:
        """Test 5: Alert lifecycle transitions remain persistent."""
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)
        engine = AlertEngine()

        sig = RiskSignal(
            signal_id="sig_lifecycle",
            window_start_ns=0,
            window_end_ns=1_000_000_000,
            risk_level="HIGH",
            risk_score=0.85,
            evidence_ids=(),
            contributing_factors=(),
            explanation="Lifecycle test",
            suppressed=False,
            suppression_reason=None,
            evaluator_version="v7.4.0",
        )

        # 1. Create & persist DETECTED
        detected = engine.evaluate_risk_signal(sig, camera_id="cam-01")
        assert detected is not None
        store.append_alert(detected)
        r1 = store.get_alert(detected.alert_id)
        assert r1.status == AlertStatus.DETECTED

        # 2. Confirm & persist
        confirmed = engine.confirm_alert(detected)
        store.append_alert(confirmed)
        r2 = store.get_alert(detected.alert_id)
        assert r2.status == AlertStatus.CONFIRMED

        # 3. Activate & persist
        activated = engine.activate_alert(confirmed)
        store.append_alert(activated)
        r3 = store.get_alert(detected.alert_id)
        assert r3.status == AlertStatus.ACTIVE

        # 4. Resolve via update_status & persist
        store.update_status(
            detected.alert_id,
            AlertStatus.RESOLVED,
            resolved_ns=time.time_ns(),
            resolution_note="All clear",
        )
        r4 = store.get_alert(detected.alert_id)
        assert r4.status == AlertStatus.RESOLVED
        assert r4.resolution_note == "All clear"
        assert r4.resolved_ns is not None

    def test_6_all_required_alert_fields_survive_persistence(self) -> None:
        """Test 6: All required alert fields survive persistence."""
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        alert = Alert(
            alert_id="alt_fields",
            severity="CRITICAL",
            status=AlertStatus.ACTIVE,
            subject_ref="subject_007",
            zone_id="zone_vault",
            camera_id="cam-03",
            timestamp_ns=5_000_000_000,
            risk_signal_id="sig_fields",
            hypothesis_ids=("hyp_a", "hyp_b", "hyp_c"),
            evidence_ids=("ev_x", "ev_y"),
            explanation="Full field preservation test",
            metadata={"custom_key": "custom_value"},
            resolved_ns=6_000_000_000,
            resolution_note="Confirmed false positive",
        )
        store.append_alert(alert)

        retrieved = store.get_alert("alt_fields")
        assert retrieved is not None
        assert retrieved.alert_id == "alt_fields"
        assert retrieved.severity == "CRITICAL"
        assert retrieved.status == AlertStatus.ACTIVE
        assert retrieved.subject_ref == "subject_007"
        assert retrieved.zone_id == "zone_vault"
        assert retrieved.camera_id == "cam-03"
        assert retrieved.timestamp_ns == 5_000_000_000
        assert retrieved.risk_signal_id == "sig_fields"
        assert retrieved.hypothesis_ids == ("hyp_a", "hyp_b", "hyp_c")
        assert retrieved.evidence_ids == ("ev_x", "ev_y")
        assert retrieved.explanation == "Full field preservation test"
        assert retrieved.metadata.get("custom_key") == "custom_value"
        assert retrieved.resolved_ns == 6_000_000_000
        assert retrieved.resolution_note == "Confirmed false positive"

    def test_7_reopen_canonical_store_preserves_alerts(self) -> None:
        """Test 7: Restart/reopen of canonical persistence does not lose alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")

            # Session 1: write alert
            es1 = SQLiteEventStore(db_path=db_path)
            store1 = AlertStore(event_store=es1)
            alert = Alert(
                alert_id="alt_persist",
                severity="HIGH",
                status=AlertStatus.DETECTED,
                subject_ref="person_1",
                zone_id="zone_1",
                camera_id="cam-01",
                timestamp_ns=1_000_000_000,
                risk_signal_id="sig_persist",
                explanation="Persistence across restart",
            )
            store1.append_alert(alert)
            es1.close()

            # Session 2: reopen and retrieve
            es2 = SQLiteEventStore(db_path=db_path)
            store2 = AlertStore(event_store=es2)
            retrieved = store2.get_alert("alt_persist")
            assert retrieved is not None
            assert retrieved.alert_id == "alt_persist"
            assert retrieved.severity == "HIGH"
            assert retrieved.explanation == "Persistence across restart"
            es2.close()

    def test_8_existing_event_persistence_unaffected(self) -> None:
        """Test 8: Existing event persistence remains unchanged."""
        from gods_eye.schemas.event import Event

        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        # Persist a normal event
        normal_event = Event(
            event_id="ev_normal_01",
            event_type=EventType.PERSON_ENTERED_FRAME,
            global_id="gid_01",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            frame_id=1,
            confidence=0.95,
            explanation="Person entered",
        )
        event_store.append(normal_event)

        # Persist an alert
        alert = Alert(
            alert_id="alt_coexist",
            severity="MEDIUM",
            status=AlertStatus.DETECTED,
            subject_ref=None,
            zone_id=None,
            camera_id="cam-01",
            timestamp_ns=2_000_000_000,
            risk_signal_id="sig_coexist",
            explanation="Coexistence test",
        )
        store.append_alert(alert)

        # Both should be independently retrievable
        ev = event_store.get_by_id("ev_normal_01")
        assert ev is not None
        assert ev.event_type == EventType.PERSON_ENTERED_FRAME

        al = store.get_alert("alt_coexist")
        assert al is not None
        assert al.severity == "MEDIUM"

        # Normal events are not alerts
        non_alert = store.get_alert("ev_normal_01")
        assert non_alert is None  # event_type != ALERT_CREATED

        # Total event count includes both
        assert event_store.count() == 2

    def test_9_lifecycle_transitions_across_store_restarts_and_preserves_ordering(self) -> None:
        """Test 9: Full lifecycle transitions persist across database reopens, preserve identity and sequence_num."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")

            # 1. Session 1: Create DETECTED
            es1 = SQLiteEventStore(db_path=db_path)
            s1 = AlertStore(event_store=es1)
            engine = AlertEngine()
            sig = RiskSignal(
                signal_id="sig_lc_disk",
                window_start_ns=0,
                window_end_ns=1_000_000_000,
                risk_level="HIGH",
                risk_score=0.85,
                evidence_ids=(),
                contributing_factors=(),
                explanation="Disk lifecycle test",
                suppressed=False,
                suppression_reason=None,
                evaluator_version="v7.4.0",
            )
            detected = engine.evaluate_risk_signal(sig, camera_id="cam-01")
            assert detected is not None
            s1.append_alert(detected)

            with es1._lock:
                cur = es1._conn.execute("SELECT sequence_num FROM events WHERE event_id = ?", (detected.alert_id,))
                orig_seq = cur.fetchone()[0]
            es1.close()

            # 2. Session 2: Reopen, Confirm, Close
            es2 = SQLiteEventStore(db_path=db_path)
            s2 = AlertStore(event_store=es2)
            al2 = s2.get_alert(detected.alert_id)
            assert al2 is not None
            assert al2.status == AlertStatus.DETECTED
            confirmed = engine.confirm_alert(al2)
            s2.append_alert(confirmed)
            es2.close()

            # 3. Session 3: Reopen, Activate, Close
            es3 = SQLiteEventStore(db_path=db_path)
            s3 = AlertStore(event_store=es3)
            al3 = s3.get_alert(detected.alert_id)
            assert al3 is not None
            assert al3.status == AlertStatus.CONFIRMED
            activated = engine.activate_alert(al3)
            s3.append_alert(activated)
            es3.close()

            # 4. Session 4: Reopen, Resolve via update_status, Close
            es4 = SQLiteEventStore(db_path=db_path)
            s4 = AlertStore(event_store=es4)
            al4 = s4.get_alert(detected.alert_id)
            assert al4 is not None
            assert al4.status == AlertStatus.ACTIVE
            s4.update_status(
                detected.alert_id,
                AlertStatus.RESOLVED,
                resolved_ns=5_000_000_000,
                resolution_note="Issue resolved and verified",
            )
            es4.close()

            # 5. Session 5: Reopen, verify final state and invariants
            es5 = SQLiteEventStore(db_path=db_path)
            s5 = AlertStore(event_store=es5)
            final_alert = s5.get_alert(detected.alert_id)
            assert final_alert is not None
            assert final_alert.alert_id == detected.alert_id
            assert final_alert.status == AlertStatus.RESOLVED
            assert final_alert.resolved_ns == 5_000_000_000
            assert final_alert.resolution_note == "Issue resolved and verified"

            # Invariant: exactly 1 event exists in the store (no duplication)
            assert es5.count() == 1

            # Invariant: sequence_num was preserved (no DELETE+INSERT churn)
            with es5._lock:
                cur = es5._conn.execute("SELECT sequence_num FROM events WHERE event_id = ?", (detected.alert_id,))
                final_seq = cur.fetchone()[0]
            assert final_seq == orig_seq, f"sequence_num changed from {orig_seq} to {final_seq}"

            es5.close()

    def test_10_optional_and_null_fields_roundtrip(self) -> None:
        """Test 10: Alert with all optional/null fields survives conversion and persistence."""
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)

        alert_minimal = Alert(
            alert_id="alt_minimal",
            severity="LOW",
            status=AlertStatus.DETECTED,
            subject_ref=None,
            zone_id=None,
            camera_id="cam-sparse",
            timestamp_ns=100,
            risk_signal_id="sig_sparse",
            hypothesis_ids=(),
            evidence_ids=(),
            explanation="",
            metadata={},
            resolved_ns=None,
            resolution_note=None,
        )
        store.append_alert(alert_minimal)

        reconstructed = store.get_alert("alt_minimal")
        assert reconstructed is not None
        assert reconstructed.alert_id == "alt_minimal"
        assert reconstructed.severity == "LOW"
        assert reconstructed.status == AlertStatus.DETECTED
        assert reconstructed.subject_ref is None
        assert reconstructed.zone_id is None
        assert reconstructed.camera_id == "cam-sparse"
        assert reconstructed.timestamp_ns == 100
        assert reconstructed.risk_signal_id == "sig_sparse"
        assert reconstructed.hypothesis_ids == ()
        assert reconstructed.evidence_ids == ()
        assert reconstructed.explanation == ""
        assert reconstructed.metadata == {}
        assert reconstructed.resolved_ns is None
        assert reconstructed.resolution_note is None

