"""Unit tests for Alert Engine, Alert Store, and Alert Writer (Phase 10.1 — SIH 26187)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import pytest

from gods_eye.alerts.alert_engine import AlertEngine
from gods_eye.alerts.alert_store import SQLiteAlertStore
from gods_eye.alerts.alert_writer import AlertWriter
from gods_eye.schemas.alert import Alert, AlertStatus
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


class TestSQLiteAlertStore:
    def test_store_crud(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "alerts.db")
            store = SQLiteAlertStore(db_path=db_path)
            try:
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
            finally:
                store.close()


class TestAlertWriter:
    def test_async_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "alerts.db")
            store = SQLiteAlertStore(db_path=db_path)
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
                store.close()
