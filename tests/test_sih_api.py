"""Unit tests for FastAPI Command-and-Control API (Phase 10.4 — SIH 26187)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from gods_eye.alerts.alert_store import SQLiteAlertStore
from gods_eye.api.app import create_app
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.evidence.evidence_manager import EvidenceManager
from gods_eye.schemas.alert import Alert, AlertStatus
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import ToolResult


class TestAPIEndpoints:
    def test_root_endpoint(self) -> None:
        app = create_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "God's Eye API"
        assert data["status"] == "operational"

    def test_auth_middleware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GODS_EYE_API_TOKEN", "secret-token-xyz")

        app = create_app()
        client = TestClient(app)

        # Missing token -> 401
        res_no_auth = client.get("/events")
        assert res_no_auth.status_code == 401

        # Invalid token -> 401
        res_bad_auth = client.get("/events", headers={"Authorization": "Bearer wrong-token"})
        assert res_bad_auth.status_code == 401

        # Valid token -> 200
        res_good_auth = client.get("/events", headers={"Authorization": "Bearer secret-token-xyz"})
        assert res_good_auth.status_code == 200

    def test_events_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")
            store = SQLiteEventStore(db_path=db_path)
            try:
                ev = Event(
                    event_id="ev_test_1",
                    event_type=EventType.PERSON_ENTERED_FRAME,
                    global_id="gid_01",
                    camera_id="cam-01",
                    timestamp_ns=1_000_000_000,
                    frame_id=1,
                    confidence=0.95,
                    explanation="Person entered",
                )
                store.append(ev)

                app = create_app(event_store=store)
                client = TestClient(app)

                res = client.get("/events")
                assert res.status_code == 200
                data = res.json()
                assert data["count"] == 1
                assert data["events"][0]["event_id"] == "ev_test_1"
            finally:
                store.close()

    def test_alerts_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "alerts.db")
            store = SQLiteAlertStore(db_path=db_path)
            try:
                alert = Alert(
                    alert_id="alt_api_1",
                    severity="CRITICAL",
                    status=AlertStatus.ACTIVE,
                    subject_ref="person_x",
                    zone_id="zone_vault",
                    camera_id="cam-01",
                    timestamp_ns=1_000_000_000,
                    risk_signal_id="sig_01",
                    explanation="Perimeter intrusion",
                )
                store.append_alert(alert)

                app = create_app(alert_store=store)
                client = TestClient(app)

                # List alerts
                res_list = client.get("/alerts")
                assert res_list.status_code == 200
                assert res_list.json()["count"] == 1

                # Get alert by id
                res_single = client.get("/alerts/alt_api_1")
                assert res_single.status_code == 200
                assert res_single.json()["severity"] == "CRITICAL"

                # 404 for unknown alert
                res_missing = client.get("/alerts/alt_unknown")
                assert res_missing.status_code == 404
            finally:
                store.close()

    def test_evidence_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EvidenceManager(base_path=tmpdir)
            mgr.capture_evidence(
                alert_id="alt_999",
                camera_id="cam-01",
                timestamp_ns=1_000_000_000,
                metadata={"test_key": "test_val"},
            )

            app = create_app(evidence_manager=mgr)
            client = TestClient(app)

            res = client.get("/evidence/alt_999")
            assert res.status_code == 200
            data = res.json()
            assert data["alert_id"] == "alt_999"
            assert len(data["evidence"]) == 1
            assert data["evidence"][0]["test_key"] == "test_val"

    def test_plates_endpoint_delegates_to_dispatcher(self) -> None:
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = ToolResult(
            call_id="call_plate",
            tool_name="get_plate_history",
            success=True,
            data={"readings": [{"plate_text": "DL01AB1234"}], "count": 1},
        )

        app = create_app(tool_dispatcher=mock_dispatcher)
        client = TestClient(app)

        res = client.get("/plates/DL01AB1234")
        assert res.status_code == 200
        data = res.json()
        assert data["plate"] == "DL01AB1234"
        assert mock_dispatcher.dispatch.called
        call_arg = mock_dispatcher.dispatch.call_args[0][0]
        assert call_arg.tool_name == "get_plate_history"
        assert call_arg.arguments["plate_text"] == "DL01AB1234"

    def test_dashboard_endpoint(self) -> None:
        app = create_app()
        client = TestClient(app)
        res = client.get("/dashboard")
        assert res.status_code == 200
        assert "COMMAND & CONTROL" in res.text
