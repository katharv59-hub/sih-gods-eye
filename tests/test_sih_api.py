"""Unit tests for FastAPI Command-and-Control API (Phase 10.4 — SIH 26187)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from gods_eye.alerts.alert_store import AlertStore
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
        event_store = SQLiteEventStore(db_path=":memory:")
        store = AlertStore(event_store=event_store)
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
            event_store.close()

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


class TestAPIErrorHandlingAndFailureSemantics:
    """Rigorous tests for SIH Problem Statement 26187 — Fix #4.

    Proves the distinction between:
    1. Valid empty results (HTTP 200 + empty collection).
    2. Expected client/request errors (HTTP 4xx).
    3. Backend/internal failures (HTTP 500 + generic message, no leakage).
    """

    # ─────────────────────────────────────────────────────────────────
    # 1. VALID EMPTY RESULTS (MUST REMAIN HTTP 200, NOT ERRORS)
    # ─────────────────────────────────────────────────────────────────

    def test_valid_empty_events_returns_200(self) -> None:
        """Legitimately empty event store returns 200 with empty list, count=0."""
        mock_store = MagicMock()
        mock_store.query_events.return_value = []

        app = create_app(event_store=mock_store)
        client = TestClient(app)

        res = client.get("/events")
        assert res.status_code == 200
        data = res.json()
        assert data == {"events": [], "count": 0}

    def test_valid_empty_alerts_returns_200(self) -> None:
        """Legitimately empty alert store returns 200 with empty list, count=0."""
        mock_store = MagicMock()
        mock_store.get_alerts_by_severity.return_value = []

        app = create_app(alert_store=mock_store)
        client = TestClient(app)

        res = client.get("/alerts")
        assert res.status_code == 200
        data = res.json()
        assert data == {"alerts": [], "count": 0}

    def test_valid_empty_cameras_returns_200(self) -> None:
        """Graph store with zero cameras returns 200 with empty cameras list."""
        mock_graph = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_graph._conn.execute.return_value = mock_cursor

        app = create_app(graph_store=mock_graph)
        client = TestClient(app)

        res = client.get("/cameras")
        assert res.status_code == 200
        assert res.json() == {"cameras": []}

    def test_valid_nonexistent_subject_returns_found_false_200(self) -> None:
        """Subject not present in graph store returns 200 with found=False."""
        mock_graph = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_graph._conn.execute.return_value = mock_cursor

        app = create_app(graph_store=mock_graph)
        client = TestClient(app)

        res = client.get("/subjects/subj_nonexistent")
        assert res.status_code == 200
        assert res.json() == {"subject_id": "subj_nonexistent", "found": False}

    def test_valid_vehicle_with_no_events_returns_200(self) -> None:
        """Vehicle with no detections returns 200 with empty events."""
        mock_store = MagicMock()
        mock_store.query_events.return_value = []

        app = create_app(event_store=mock_store)
        client = TestClient(app)

        res = client.get("/vehicles/veh_none")
        assert res.status_code == 200
        assert res.json() == {"vehicle_id": "veh_none", "events": []}

    # ─────────────────────────────────────────────────────────────────
    # 2. EXPECTED CLIENT ERRORS (HTTP 4xx)
    # ─────────────────────────────────────────────────────────────────

    def test_client_error_invalid_event_type_returns_400(self) -> None:
        """Invalid event_type query parameter returns HTTP 400 Bad Request."""
        mock_store = MagicMock()
        app = create_app(event_store=mock_store)
        client = TestClient(app)

        res = client.get("/events?event_type=TOTALLY_INVALID_TYPE_XYZ")
        assert res.status_code == 400
        assert "Invalid event_type" in res.json()["detail"]
        mock_store.query_events.assert_not_called()

    def test_client_error_missing_alert_returns_404(self) -> None:
        """Request for non-existent alert ID returns HTTP 404."""
        mock_store = MagicMock()
        mock_store.get_alert.return_value = None

        app = create_app(alert_store=mock_store)
        client = TestClient(app)

        res = client.get("/alerts/alt_does_not_exist")
        assert res.status_code == 404
        assert res.json() == {"detail": "Alert not found"}

    def test_client_error_empty_plate_returns_400(self) -> None:
        """Whitespace-only plate query returns HTTP 400."""
        app = create_app(tool_dispatcher=MagicMock())
        client = TestClient(app)

        res = client.get("/plates/%20%20")
        assert res.status_code == 400
        assert "Plate text must not be empty" in res.json()["detail"]

    def test_client_error_empty_subject_timeline_returns_400(self) -> None:
        """Whitespace-only subject query returns HTTP 400."""
        app = create_app(tool_dispatcher=MagicMock())
        client = TestClient(app)

        res = client.get("/timeline/%20%20")
        assert res.status_code == 400
        assert "Subject ID must not be empty" in res.json()["detail"]

    # ─────────────────────────────────────────────────────────────────
    # 3. FAILURE INJECTION TESTS (GENUINE BACKEND FAILURES MUST RETURN 500)
    # ─────────────────────────────────────────────────────────────────

    def test_backend_failure_events_store_returns_500(self) -> None:
        """Event store failure must NOT be suppressed into an empty result.

        Deliberately injects an exception containing SQL and internal paths,
        verifying:
        - HTTP status is 500
        - Response is NOT an empty success dict
        - Response does NOT leak SQL, paths, or tracebacks.
        """
        mock_store = MagicMock()
        mock_store.query_events.side_effect = RuntimeError(
            "CRITICAL SQLite DB Error: SELECT * FROM events failed at /var/lib/gods_eye/events.db"
        )

        app = create_app(event_store=mock_store)
        client = TestClient(app)

        res = client.get("/events")
        # Must return 500, NOT 200 with empty events
        assert res.status_code == 500
        assert res.json() != {"events": [], "count": 0}
        assert res.json() == {"detail": "Internal server error"}

        # Verify no information leakage
        raw_text = res.text
        assert "Traceback" not in raw_text
        assert "SELECT * FROM events" not in raw_text
        assert "/var/lib/gods_eye" not in raw_text
        assert "RuntimeError" not in raw_text

    def test_backend_failure_alerts_store_returns_500(self) -> None:
        """Alert store query failure must NOT return an empty alert list."""
        mock_store = MagicMock()
        mock_store.get_alerts_by_severity.side_effect = Exception("Disk I/O failure in alerts storage")

        app = create_app(alert_store=mock_store)
        client = TestClient(app)

        res = client.get("/alerts")
        assert res.status_code == 500
        assert res.json() != {"alerts": [], "count": 0}
        assert res.json() == {"detail": "Internal server error"}
        assert "Disk I/O" not in res.text

    def test_backend_failure_single_alert_returns_500(self) -> None:
        """Single alert lookup database failure must return 500, not 404."""
        mock_store = MagicMock()
        mock_store.get_alert.side_effect = Exception("Alert store query error")

        app = create_app(alert_store=mock_store)
        client = TestClient(app)

        res = client.get("/alerts/alt_corrupted")
        assert res.status_code == 500
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_cameras_returns_500(self) -> None:
        """Graph store failure must NOT return an empty cameras list."""
        mock_graph = MagicMock()
        mock_graph._conn.execute.side_effect = Exception("Database locked: node_cameras")

        app = create_app(graph_store=mock_graph)
        client = TestClient(app)

        res = client.get("/cameras")
        assert res.status_code == 500
        assert res.json() != {"cameras": []}
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_subjects_returns_500(self) -> None:
        """Graph store failure in subject query must NOT return found=False."""
        mock_graph = MagicMock()
        mock_graph._conn.execute.side_effect = Exception("Corrupt table node_identities")

        app = create_app(graph_store=mock_graph)
        client = TestClient(app)

        res = client.get("/subjects/subj_important")
        # Must NOT return 200 with found=False (the old bug)
        assert res.status_code == 500
        assert res.json() != {"subject_id": "subj_important", "found": False}
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_vehicles_returns_500(self) -> None:
        """Event store failure in vehicle query must NOT return empty events."""
        mock_store = MagicMock()
        mock_store.query_events.side_effect = Exception("Event store query failure")

        app = create_app(event_store=mock_store)
        client = TestClient(app)

        res = client.get("/vehicles/veh_123")
        assert res.status_code == 500
        assert res.json() != {"vehicle_id": "veh_123", "events": []}
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_plates_dispatcher_exception_returns_500(self) -> None:
        """Dispatcher exception in plate query must NOT return empty readings."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.side_effect = RuntimeError("Reasoning worker disconnected")

        app = create_app(tool_dispatcher=mock_dispatcher)
        client = TestClient(app)

        res = client.get("/plates/DL01AB1234")
        assert res.status_code == 500
        assert res.json() != {"plate": "DL01AB1234", "readings": []}
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_plates_tool_failure_returns_500(self) -> None:
        """Tool failure result (tr.success=False) must NOT return empty readings."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = ToolResult(
            call_id="call_err",
            tool_name="get_plate_history",
            success=False,
            error_message="Underlying EventStore connection timeout",
        )

        app = create_app(tool_dispatcher=mock_dispatcher)
        client = TestClient(app)

        res = client.get("/plates/DL01AB1234")
        assert res.status_code == 500
        assert res.json() != {"plate": "DL01AB1234", "readings": []}
        assert res.json() == {"detail": "Internal server error"}
        assert "EventStore connection timeout" not in res.text

    def test_backend_failure_timeline_dispatcher_exception_returns_500(self) -> None:
        """Dispatcher exception in timeline query must NOT return empty timeline."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.side_effect = RuntimeError("Timeline engine crash")

        app = create_app(tool_dispatcher=mock_dispatcher)
        client = TestClient(app)

        res = client.get("/timeline/subj_victim")
        assert res.status_code == 500
        assert res.json() != {"subject": "subj_victim", "timeline": []}
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_timeline_tool_failure_returns_500(self) -> None:
        """Tool failure result (tr.success=False) must return 500."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = ToolResult(
            call_id="call_err_tl",
            tool_name="get_identity_timeline",
            success=False,
            error_message="TimelineEngine is not initialized",
        )

        app = create_app(tool_dispatcher=mock_dispatcher)
        client = TestClient(app)

        res = client.get("/timeline/subj_victim")
        assert res.status_code == 500
        assert res.json() != {"subject": "subj_victim", "timeline": []}
        assert res.json() == {"detail": "Internal server error"}

    def test_backend_failure_evidence_manager_returns_500(self) -> None:
        """Evidence manager disk error must NOT return empty evidence list."""
        mock_mgr = MagicMock()
        mock_mgr.get_evidence.side_effect = OSError("Permission denied on /data/evidence/alt_1")

        app = create_app(evidence_manager=mock_mgr)
        client = TestClient(app)

        res = client.get("/evidence/alt_1")
        assert res.status_code == 500
        assert res.json() != {"alert_id": "alt_1", "evidence": []}
        assert res.json() == {"detail": "Internal server error"}
        assert "Permission denied" not in res.text


class TestFaceEvidenceAndWebsiteAPI:
    """Tests for #10C Website Face Enrollment, Evidence Image Serving, and Contracts."""

    def test_evidence_file_serves_snapshot(self) -> None:
        """Evidence image route successfully serves snapshot.jpg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_mgr = EvidenceManager(base_path=tmpdir)
            alert_id = "alt_test_snap"
            ev_id = "ev_snap_01"
            target_dir = Path(tmpdir) / alert_id / ev_id
            target_dir.mkdir(parents=True, exist_ok=True)
            fake_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb"
            (target_dir / "snapshot.jpg").write_bytes(fake_jpg)

            app = create_app(evidence_manager=ev_mgr)
            client = TestClient(app)

            res = client.get(f"/evidence/file/{alert_id}/{ev_id}/snapshot.jpg")
            assert res.status_code == 200
            assert "image/jpeg" in res.headers.get("content-type", "")
            assert res.content == fake_jpg

    def test_evidence_file_serves_face_crop(self) -> None:
        """Evidence image route successfully serves face_crop.jpg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_mgr = EvidenceManager(base_path=tmpdir)
            alert_id = "alt_test_crop"
            ev_id = "ev_crop_01"
            target_dir = Path(tmpdir) / alert_id / ev_id
            target_dir.mkdir(parents=True, exist_ok=True)
            fake_crop = b"\xff\xd8\xff\xe0\x00\x10CROP_DATA_XYZ"
            (target_dir / "face_crop.jpg").write_bytes(fake_crop)

            app = create_app(evidence_manager=ev_mgr)
            client = TestClient(app)

            res = client.get(f"/evidence/file/{alert_id}/{ev_id}/face_crop.jpg")
            assert res.status_code == 200
            assert "image/jpeg" in res.headers.get("content-type", "")
            assert res.content == fake_crop

    def test_evidence_file_invalid_filename_rejected(self) -> None:
        """Filename outside allowlist (e.g. metadata.json or code) must be rejected with 400."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_mgr = EvidenceManager(base_path=tmpdir)
            app = create_app(evidence_manager=ev_mgr)
            client = TestClient(app)

            res = client.get("/evidence/file/alt_1/ev_1/metadata.json")
            assert res.status_code == 400
            assert "Filename not allowed" in res.json()["detail"]

    def test_evidence_file_path_traversal_rejected(self) -> None:
        """Path traversal characters (..) in alert_id, evidence_id or filename must be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_mgr = EvidenceManager(base_path=tmpdir)
            app = create_app(evidence_manager=ev_mgr)
            client = TestClient(app)

            res_parent = client.get("/evidence/file/..%2F..%2Fetc/ev_1/snapshot.jpg")
            assert res_parent.status_code in (400, 403, 404)

            res_direct = client.get("/evidence/file/alert1/../snapshot.jpg")
            assert res_direct.status_code in (400, 403, 404)

    def test_evidence_file_missing_returns_404(self) -> None:
        """Nonexistent evidence file must return 404."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_mgr = EvidenceManager(base_path=tmpdir)
            app = create_app(evidence_manager=ev_mgr)
            client = TestClient(app)

            res = client.get("/evidence/file/nonexistent_alert/nonexistent_ev/snapshot.jpg")
            assert res.status_code == 404
            assert res.json()["detail"] == "Evidence file not found"

    def test_enrolled_face_photo_route_works(self) -> None:
        """GET /faces/photo/{person_id} successfully returns reference photo."""
        enrolled_dir = Path("data/faces/enrolled")
        enrolled_dir.mkdir(parents=True, exist_ok=True)
        test_pid = "person_api_test_photo"
        photo_file = enrolled_dir / f"{test_pid}.jpg"
        fake_photo = b"\xff\xd8\xff\xe0\x00\x10PHOTO_DATA"
        photo_file.write_bytes(fake_photo)

        try:
            app = create_app()
            client = TestClient(app)

            res = client.get(f"/faces/photo/{test_pid}")
            assert res.status_code == 200
            assert "image/jpeg" in res.headers.get("content-type", "")
            assert res.content == fake_photo
        finally:
            if photo_file.exists():
                photo_file.unlink()

    def test_enrolled_face_photo_traversal_rejected(self) -> None:
        """Path traversal in person_id must be strictly rejected with 400 or 403."""
        app = create_app()
        client = TestClient(app)

        res = client.get("/faces/photo/..%2F..%2Fpasswords")
        assert res.status_code in (400, 403, 404)

    def test_dashboard_loads_with_biometric_elements(self) -> None:
        """Dashboard HTML must load and contain face biometrics UI elements."""
        app = create_app()
        client = TestClient(app)

        res = client.get("/dashboard")
        assert res.status_code == 200
        html = res.text
        assert "Face Recognition & Biometrics" in html
        assert 'id="enrollForm"' in html
        assert 'id="faceEventsList"' in html
        assert 'id="evidenceModal"' in html
        assert 'id="modalFaceCropImg"' in html
        assert 'id="modalSnapshotImg"' in html

    def test_face_enrollment_endpoint_functional(self) -> None:
        """POST /faces/enroll must accept name and multipart image and return 201."""
        from gods_eye.memory.graph_store import SQLiteGraphStore
        from gods_eye.schemas.detection import BoundingBox
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "graph.db")
            graph_store = SQLiteGraphStore(db_path=db_path)
            try:
                mock_recognizer = MagicMock()
                mock_recognizer.detect_faces.return_value = [
                    (BoundingBox(10.0, 10.0, 50.0, 50.0), 0.98, np.zeros((1, 15), dtype=np.float32))
                ]
                mock_recognizer.extract_embedding.return_value = np.ones(128, dtype=np.float32)

                app = create_app(graph_store=graph_store, face_recognizer=mock_recognizer)
                client = TestClient(app)

                # 1x1 white JPEG
                import cv2
                img = np.full((100, 100, 3), 255, dtype=np.uint8)
                _, buf = cv2.imencode(".jpg", img)

                res = client.post(
                    "/faces/enroll",
                    data={"name": "Audited Subject"},
                    files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")},
                )
                assert res.status_code == 201
                data = res.json()
                assert data["name"] == "Audited Subject"
                assert data["person_id"].startswith("person_")
                assert data["dimension"] == 128
                assert data["detection_confidence"] == 0.98

                # Verify presence in /faces/enrolled
                res_enrolled = client.get("/faces/enrolled")
                assert res_enrolled.status_code == 200
                enrolled_data = res_enrolled.json()
                assert enrolled_data["count"] == 1
                assert enrolled_data["enrolled"][0]["name"] == "Audited Subject"
                assert "photo_url" in enrolled_data["enrolled"][0]
            finally:
                graph_store.close()

    def test_face_event_frontend_contract_compliance(self) -> None:
        """GET /events with event_type=face_detected must return compliant contract for both KNOWN and UNKNOWN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")
            store = SQLiteEventStore(db_path=db_path)

            # KNOWN event
            ev_known = Event(
                event_id="ev_face_known_01",
                event_type=EventType.FACE_DETECTED,
                global_id="person_auth_01",
                camera_id="cam-front",
                timestamp_ns=1_725_000_000_000_000_000,
                frame_id=101,
                confidence=0.88,
                explanation="Face recognized as Alice",
                metadata={
                    "status": "KNOWN",
                    "person_id": "person_auth_01",
                    "name": "Alice",
                    "similarity": 0.88,
                    "detection_confidence": 0.95,
                    "evidence_id": "ev_face_known_01",
                    "track_id": "trk_01",
                },
            )
            # UNKNOWN event
            ev_unknown = Event(
                event_id="ev_face_unknown_02",
                event_type=EventType.FACE_DETECTED,
                global_id=None,
                camera_id="cam-front",
                timestamp_ns=1_725_000_005_000_000_000,
                frame_id=105,
                confidence=0.91,
                explanation="Unknown face detected",
                metadata={
                    "status": "UNKNOWN",
                    "person_id": None,
                    "name": None,
                    "similarity": 0.12,
                    "detection_confidence": 0.91,
                    "evidence_id": "ev_face_unknown_02",
                    "track_id": "trk_02",
                },
            )
            store.append(ev_known)
            store.append(ev_unknown)

            app = create_app(event_store=store)
            client = TestClient(app)

            res = client.get("/events?event_type=face_detected")
            assert res.status_code == 200
            data = res.json()
            assert data["count"] == 2
            events = data["events"]

            # Validate KNOWN event contract
            known = next(e for e in events if e["event_id"] == "ev_face_known_01")
            assert known["metadata"]["status"] == "KNOWN"
            assert known["metadata"]["name"] == "Alice"
            assert known["metadata"]["similarity"] == 0.88
            assert known["camera_id"] == "cam-front"
            assert known["metadata"]["track_id"] == "trk_01"

            # Validate UNKNOWN event contract
            unknown = next(e for e in events if e["event_id"] == "ev_face_unknown_02")
            assert unknown["metadata"]["status"] == "UNKNOWN"
            assert unknown["metadata"]["name"] is None  # NEVER invented
            assert unknown["metadata"]["person_id"] is None
            assert unknown["metadata"]["similarity"] == 0.12
            assert unknown["metadata"]["detection_confidence"] == 0.91

            store.close()

    def test_delete_enrolled_face_endpoint(self) -> None:
        """DELETE /faces/enrolled/{person_id} deletes person and cleans up photo file."""
        from gods_eye.memory.graph_store import SQLiteGraphStore
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "graph.db")
            graph_store = SQLiteGraphStore(db_path=db_path)
            try:
                pid = "person_delete_test"
                photo_dir = Path("data/faces/enrolled")
                photo_dir.mkdir(parents=True, exist_ok=True)
                test_photo = photo_dir / f"{pid}.jpg"
                test_photo.write_bytes(b"dummy")

                graph_store.enroll_face(
                    person_id=pid,
                    name="Temporary Subject",
                    embedding=np.zeros(128, dtype=np.float32),
                    photo_path=str(test_photo),
                )

                app = create_app(graph_store=graph_store)
                client = TestClient(app)

                # Delete person
                res = client.delete(f"/faces/enrolled/{pid}")
                assert res.status_code == 200
                assert res.json()["status"] == "deleted"
                assert not test_photo.exists()

                # Deleting again returns 404
                res_404 = client.delete(f"/faces/enrolled/{pid}")
                assert res_404.status_code == 404
            finally:
                graph_store.close()

    def test_delete_events_endpoint(self) -> None:
        """DELETE /events?event_type=face_detected clears matching events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "events.db")
            store = SQLiteEventStore(db_path=db_path)
            try:
                ev = Event(
                    event_id="ev_clear_01",
                    event_type=EventType.FACE_DETECTED,
                    global_id=None,
                    camera_id="cam-01",
                    timestamp_ns=1_000_000_000,
                    frame_id=1,
                    confidence=0.9,
                    explanation="Sample face event",
                )
                store.append(ev)

                app = create_app(event_store=store)
                client = TestClient(app)

                # Delete face_detected events
                res = client.delete("/events?event_type=face_detected")
                assert res.status_code == 200
                data = res.json()
                assert data["status"] == "cleared"
                assert data["deleted_count"] == 1

                # Events should now be empty
                res_get = client.get("/events?event_type=face_detected")
                assert res_get.json()["count"] == 0
            finally:
                store.close()



