"""Unit tests for Face Detection Module (Phase 8.3 — SIH 26187).

Verifies detection-only contract, track association, and absence of recognition/embeddings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import time
import numpy as np
import pytest

from gods_eye.face.detector import FaceDetector, FaceDetectionResult
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.track import Track, TrackState


class TestFaceDetector:
    def test_initialization(self) -> None:
        detector = FaceDetector(
            confidence_threshold=0.60,
            scale_factor=1.2,
            min_neighbors=4,
        )
        assert detector._confidence_threshold == 0.60
        assert detector._scale_factor == 1.2
        assert detector._min_neighbors == 4

    def test_detection_only_invariant(self) -> None:
        """Confirms that FaceDetectionResult has NO embedding or biometric identity fields."""
        face = FaceDetectionResult(
            face_id="face-1",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(10, 10, 50, 50),
            confidence=0.85,
        )
        assert not hasattr(face, "embedding")
        assert not hasattr(face, "identity_id")
        assert not hasattr(face, "features")

    def test_detect_blank_frame_no_faces(self) -> None:
        detector = FaceDetector()
        blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        results = detector.detect(
            frame=blank_frame,
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
        )
        assert isinstance(results, list)
        assert len(results) == 0

    def test_track_association(self) -> None:
        detector = FaceDetector()
        # Mock cascade to return 1 face at (100, 100, 50, 50), center is (125, 125)
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(100, 100, 50, 50)]
        detector._cascade = mock_cascade

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Person track covering the face region: (80, 80, 200, 300)
        track = Track(
            track_id="trk_person_1",
            camera_id="cam-01",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(80, 80, 200, 300),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=5,
            lost_frame_count=0,
            detection_history=["det-1"],
        )

        results = detector.detect(
            frame=frame,
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            person_tracks=[track],
        )

        assert len(results) == 1
        assert results[0].associated_person_track_id == "trk_person_1"
        assert results[0].bbox.x1 == 100.0
        assert results[0].bbox.y1 == 100.0


class TestFaceRecognitionAndEnrollment:
    """Tests for Phase 10 Core Face Recognition and Dynamic Enrollment."""

    @pytest.fixture
    def test_images(self) -> tuple[bytes, bytes]:
        """Provides real Market-1501 single-face image bytes."""
        import os

        market_dir = "data/market1501/Market-1501-v15.09.15/bounding_box_test"
        p1 = os.path.join(market_dir, "-1_c1s1_011251_02.jpg")
        p2 = os.path.join(market_dir, "-1_c1s1_011426_05.jpg")
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            return f1.read(), f2.read()

    def test_model_absence_raises_explicit_error(self) -> None:
        """TEST 9 — MODEL ABSENCE: Missing models must fail explicitly, never fallback silently."""
        from gods_eye.face.recognizer import FaceRecognizer

        with pytest.raises(FileNotFoundError) as exc_info:
            FaceRecognizer(detection_model_path="nonexistent_detector.onnx")
        assert "YuNet model file not found" in str(exc_info.value)

        with pytest.raises(FileNotFoundError) as exc_info2:
            FaceRecognizer(recognition_model_path="nonexistent_recognizer.onnx")
        assert "SFace model file not found" in str(exc_info2.value)

    def test_embedding_dimension(self, test_images: tuple[bytes, bytes]) -> None:
        """TEST 8 — EMBEDDING DIMENSION: SFace embedding must strictly be 128-dimensional float32."""
        import cv2
        from gods_eye.face.recognizer import FaceRecognizer

        rec = FaceRecognizer(detection_threshold=0.3)
        assert rec.embedding_dimension == 128

        nparr = np.frombuffer(test_images[0], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = rec.detect_faces(img)
        assert len(faces) == 1
        emb = rec.extract_embedding(img, faces[0][2])
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (128,)
        assert emb.dtype == np.float32

    def test_valid_enrollment(self, test_images: tuple[bytes, bytes], tmp_path: Any) -> None:
        """TEST 1 — VALID ENROLLMENT: Valid 1-face image enrolls successfully with 128-d embedding."""
        from fastapi.testclient import TestClient
        from gods_eye.api.app import create_app
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        db_file = str(tmp_path / "test_graph.db")
        store = SQLiteGraphStore(db_file)
        rec = FaceRecognizer(detection_threshold=0.3)
        app = create_app(graph_store=store, face_recognizer=rec)
        client = TestClient(app)

        resp = client.post(
            "/faces/enroll",
            data={"name": "Alice Cooper"},
            files={"file": ("alice.jpg", test_images[0], "image/jpeg")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "person_id" in data
        assert data["name"] == "Alice Cooper"
        assert data["dimension"] == 128
        assert data["person_id"].startswith("person_")

        # Verify query
        enrolled_resp = client.get("/faces/enrolled")
        assert enrolled_resp.status_code == 200
        enrolled_data = enrolled_resp.json()
        assert enrolled_data["count"] == 1
        assert enrolled_data["enrolled"][0]["person_id"] == data["person_id"]
        assert enrolled_data["enrolled"][0]["name"] == "Alice Cooper"

    def test_persistence_across_store_reopen(self, test_images: tuple[bytes, bytes], tmp_path: Any) -> None:
        """TEST 2 — PERSISTENCE: Enrolled identity persists across SQLite database reopen."""
        import cv2
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        db_file = str(tmp_path / "persistent_faces.db")
        store1 = SQLiteGraphStore(db_file)
        rec = FaceRecognizer(detection_threshold=0.3)

        nparr = np.frombuffer(test_images[0], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = rec.detect_faces(img)
        emb = rec.extract_embedding(img, faces[0][2])

        store1.enroll_face("person_persist_01", "Bob Smith", emb)
        store1.close()

        # Reopen from disk
        store2 = SQLiteGraphStore(db_file)
        records = store2.get_enrolled_faces()
        assert len(records) == 1
        assert records[0].person_id == "person_persist_01"
        assert records[0].name == "Bob Smith"
        assert records[0].dimension == 128
        assert np.allclose(records[0].embedding, emb, atol=1e-6)

        # Also check single lookup
        single = store2.get_enrolled_face("person_persist_01")
        assert single is not None
        assert single.name == "Bob Smith"
        store2.close()

    def test_self_recognition(self, test_images: tuple[bytes, bytes], tmp_path: Any) -> None:
        """TEST 3 — SELF RECOGNITION: Same face image is recognized as KNOWN with matching person_id."""
        from fastapi.testclient import TestClient
        from gods_eye.api.app import create_app
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        store = SQLiteGraphStore(":memory:")
        rec = FaceRecognizer(detection_threshold=0.3)
        app = create_app(graph_store=store, face_recognizer=rec)
        client = TestClient(app)

        # Enroll Alice
        enroll_res = client.post(
            "/faces/enroll",
            data={"name": "Alice"},
            files={"file": ("alice.jpg", test_images[0], "image/jpeg")},
        )
        assert enroll_res.status_code == 201
        alice_id = enroll_res.json()["person_id"]

        # Recognize with same image
        rec_res = client.post(
            "/faces/recognize",
            files={"file": ("alice_query.jpg", test_images[0], "image/jpeg")},
        )
        assert rec_res.status_code == 200
        rec_data = rec_res.json()
        assert rec_data["status"] == "KNOWN"
        assert rec_data["person_id"] == alice_id
        assert rec_data["name"] == "Alice"
        assert rec_data["similarity"] >= rec.recognition_threshold

    def test_unknown_person_not_misassigned(self, test_images: tuple[bytes, bytes], tmp_path: Any) -> None:
        """TEST 4 — UNKNOWN PERSON: Unenrolled face is marked UNKNOWN and never assigned to enrolled identity."""
        from fastapi.testclient import TestClient
        from gods_eye.api.app import create_app
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        store = SQLiteGraphStore(":memory:")
        rec = FaceRecognizer(detection_threshold=0.3)
        app = create_app(graph_store=store, face_recognizer=rec)
        client = TestClient(app)

        # Enroll Alice (image 1)
        enroll_res = client.post(
            "/faces/enroll",
            data={"name": "Alice"},
            files={"file": ("alice.jpg", test_images[0], "image/jpeg")},
        )
        assert enroll_res.status_code == 201

        # Query Bob (image 2) who is NOT enrolled
        rec_res = client.post(
            "/faces/recognize",
            files={"file": ("bob_query.jpg", test_images[1], "image/jpeg")},
        )
        assert rec_res.status_code == 200
        rec_data = rec_res.json()
        assert rec_data["status"] == "UNKNOWN"
        assert rec_data["person_id"] is None
        assert rec_data["name"] is None
        assert rec_data["similarity"] < rec.recognition_threshold

    def test_no_face_enrollment_rejected(self, tmp_path: Any) -> None:
        """TEST 5 — NO FACE: Blank/no-face image is rejected with HTTP 400."""
        import cv2
        from fastapi.testclient import TestClient
        from gods_eye.api.app import create_app
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        store = SQLiteGraphStore(":memory:")
        rec = FaceRecognizer(detection_threshold=0.3)
        app = create_app(graph_store=store, face_recognizer=rec)
        client = TestClient(app)

        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        _, blank_bytes = cv2.imencode(".jpg", blank)

        resp = client.post(
            "/faces/enroll",
            data={"name": "Ghost"},
            files={"file": ("blank.jpg", blank_bytes.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "No face detected" in resp.json()["detail"]

    def test_multiple_faces_enrollment_rejected(self, test_images: tuple[bytes, bytes], tmp_path: Any) -> None:
        """TEST 6 — MULTIPLE FACES: Image with multiple faces is rejected with HTTP 400."""
        import cv2
        from fastapi.testclient import TestClient
        from gods_eye.api.app import create_app
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        store = SQLiteGraphStore(":memory:")
        rec = FaceRecognizer(detection_threshold=0.2)
        app = create_app(graph_store=store, face_recognizer=rec)
        client = TestClient(app)

        nparr1 = np.frombuffer(test_images[0], np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        # Composite frame containing 2 faces
        multi = np.zeros((300, 300, 3), dtype=np.uint8)
        multi[10 : 10 + img1.shape[0], 10 : 10 + img1.shape[1]] = img1
        multi[10 : 10 + img1.shape[0], 150 : 150 + img1.shape[1]] = img1

        _, multi_bytes = cv2.imencode(".jpg", multi)

        resp = client.post(
            "/faces/enroll",
            data={"name": "Crowd"},
            files={"file": ("crowd.jpg", multi_bytes.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "Multiple faces detected" in resp.json()["detail"]

    def test_recognition_determinism(self, test_images: tuple[bytes, bytes]) -> None:
        """TEST 7 — DETERMINISM: Same image produces stable/near-identical embeddings and scores."""
        import cv2
        from gods_eye.face.recognizer import FaceRecognizer

        rec = FaceRecognizer(detection_threshold=0.3)
        nparr = np.frombuffer(test_images[0], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        f1 = rec.detect_faces(img)[0]
        emb1 = rec.extract_embedding(img, f1[2])

        f2 = rec.detect_faces(img)[0]
        emb2 = rec.extract_embedding(img, f2[2])

        assert np.allclose(emb1, emb2, atol=1e-5)
        sim = rec.match(emb1, emb2)
        assert np.isclose(sim, 1.0, atol=1e-4)


class TestLiveFaceRecognition:
    """Automated tests for FIX #10B Live Face Recognition, Evidence, and Canonical Event."""

    @pytest.fixture
    def test_images(self) -> tuple[bytes, bytes]:
        import os

        market_dir = "data/market1501/Market-1501-v15.09.15/bounding_box_test"
        p1 = os.path.join(market_dir, "-1_c1s1_011251_02.jpg")
        p2 = os.path.join(market_dir, "-1_c1s1_011426_05.jpg")
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            return f1.read(), f2.read()

    def test_crop_face_safely_bounds(self) -> None:
        """TEST 4 — SAFE BOUNDS: Face crop never crashes and clamps cleanly even for out-of-frame bboxes."""
        from gods_eye.face.live_engine import LiveFaceEngine

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 1. Bounding box partially outside image left/top
        bbox_out_top_left = BoundingBox(x1=-20.0, y1=-10.0, x2=40.0, y2=40.0)
        crop1 = LiveFaceEngine.crop_face_safely(frame, bbox_out_top_left)
        assert crop1.shape[0] <= 100 and crop1.shape[1] <= 100
        assert crop1.shape[0] > 0 and crop1.shape[1] > 0

        # 2. Bounding box partially outside image bottom/right
        bbox_out_bottom_right = BoundingBox(x1=70.0, y1=80.0, x2=150.0, y2=140.0)
        crop2 = LiveFaceEngine.crop_face_safely(frame, bbox_out_bottom_right)
        assert crop2.shape[0] <= 100 and crop2.shape[1] <= 100
        assert crop2.shape[0] > 0 and crop2.shape[1] > 0

        # 3. Degenerate inverted bbox
        bbox_degenerate = BoundingBox(x1=50.0, y1=50.0, x2=40.0, y2=40.0)
        crop3 = LiveFaceEngine.crop_face_safely(frame, bbox_degenerate)
        assert crop3.shape == (1, 1, 3)

    def test_live_engine_known_and_unknown_and_evidence(
        self, test_images: tuple[bytes, bytes], tmp_path: Any
    ) -> None:
        """TESTS 1, 2, 3, 5, 6, 7, 8:
        1. Person track spatial association
        2. Known live recognition result
        3. Unknown live recognition result
        5. Evidence snapshot.jpg captured
        6. Evidence face_crop.jpg captured
        7. Evidence metadata.json written with recognition fields
        8. Canonical EventType.FACE_DETECTED event persisted in SQLiteEventStore
        """
        import cv2
        import json
        from pathlib import Path
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.live_engine import LiveFaceEngine
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import EnrolledFaceRecord, SQLiteGraphStore
        from gods_eye.schemas.event import EventType

        rec = FaceRecognizer(detection_threshold=0.3)
        graph_store = SQLiteGraphStore(str(tmp_path / "graph.db"))
        event_store = SQLiteEventStore(str(tmp_path / "events.db"))
        evidence_mgr = EvidenceManager(base_path=str(tmp_path / "evidence"))

        # Enroll person Alice using test_images[0]
        nparr0 = np.frombuffer(test_images[0], np.uint8)
        img0 = cv2.imdecode(nparr0, cv2.IMREAD_COLOR)
        f_enroll = rec.detect_faces(img0)[0]
        emb_alice = rec.extract_embedding(img0, f_enroll[2])

        graph_store.enroll_face(
            person_id="person_alice_001",
            name="Alice",
            embedding=emb_alice,
        )

        engine = LiveFaceEngine(
            recognizer=rec,
            graph_store=graph_store,
            event_store=event_store,
            evidence_manager=evidence_mgr,
            throttle_window_s=5.0,
        )

        # Build composite frame: Alice on left (x: 50..114), Unknown Bob on right (x: 300..364)
        nparr1 = np.frombuffer(test_images[1], np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        frame = np.zeros((400, 500, 3), dtype=np.uint8)
        frame[50 : 50 + img0.shape[0], 50 : 50 + img0.shape[1]] = img0
        frame[50 : 50 + img1.shape[0], 300 : 300 + img1.shape[1]] = img1

        # Track 1 for Alice, Track 2 for Bob
        t_alice = Track(
            track_id="trk_1",
            camera_id="CAM-01",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(40, 40, 50 + img0.shape[1] + 10, 50 + img0.shape[0] + 10),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["d1"],
        )
        t_bob = Track(
            track_id="trk_2",
            camera_id="CAM-01",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(290, 40, 300 + img1.shape[1] + 10, 50 + img1.shape[0] + 10),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["d2"],
        )

        results = engine.process_frame(
            frame=frame,
            tracks=[t_alice, t_bob],
            camera_id="CAM-01",
            frame_id=10,
            timestamp_ns=1_000_000_000,
        )

        assert len(results) == 2

        # Sort results by x-coordinate to consistently identify Alice and Bob
        results_sorted = sorted(results, key=lambda r: r.bbox.x1)
        res_alice = results_sorted[0]
        res_bob = results_sorted[1]

        # 1. Association verification
        assert res_alice.track_id == "trk_1"
        assert res_bob.track_id == "trk_2"

        # 2. Known recognition verification
        assert res_alice.status == "KNOWN"
        assert res_alice.person_id == "person_alice_001"
        assert res_alice.name == "Alice"
        assert res_alice.similarity >= rec.recognition_threshold

        # 3. Unknown recognition verification
        assert res_bob.status == "UNKNOWN"
        assert res_bob.person_id is None
        assert res_bob.name is None
        assert res_bob.similarity < rec.recognition_threshold

        # 8. Canonical Event verification
        events = event_store.query_events(event_type=EventType.FACE_DETECTED)
        assert len(events) == 2

        ev_alice = [e for e in events if e.metadata.get("status") == "KNOWN"][0]
        assert ev_alice.global_id == "person_alice_001"
        assert ev_alice.metadata["name"] == "Alice"
        assert ev_alice.metadata["track_id"] == "trk_1"

        ev_bob = [e for e in events if e.metadata.get("status") == "UNKNOWN"][0]
        assert ev_bob.global_id is None
        assert ev_bob.metadata["status"] == "UNKNOWN"
        assert ev_bob.metadata["track_id"] == "trk_2"

        # 5, 6, 7. Evidence verification
        for ev in [ev_alice, ev_bob]:
            ev_id = ev.metadata["evidence_id"]
            ev_records = evidence_mgr.get_evidence(ev_id)
            assert len(ev_records) == 1
            meta = ev_records[0]

            assert meta["has_snapshot"] is True
            assert meta["has_face_crop"] is True
            assert meta["camera_id"] == "CAM-01"
            assert "snapshot_path" in meta
            assert "face_crop_path" in meta
            assert "recognition_status" in meta
            assert Path(meta["snapshot_path"]).exists()
            assert Path(meta["face_crop_path"]).exists()

            # Confirm face_crop is an image with valid dimensions
            face_img = cv2.imread(meta["face_crop_path"])
            assert face_img is not None
            assert face_img.shape[0] > 10 and face_img.shape[1] > 10

    def test_throttling_and_state_change_override(
        self, test_images: tuple[bytes, bytes], tmp_path: Any
    ) -> None:
        """TESTS 9, 10:
        9. Duplicate recognition throttling suppresses same-state events within interval.
        10. Known -> Unknown state change generates an immediate event even within throttle window.
        """
        import cv2
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.live_engine import LiveFaceEngine
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import EnrolledFaceRecord, SQLiteGraphStore
        from gods_eye.schemas.event import EventType

        rec = FaceRecognizer(detection_threshold=0.3)
        graph_store = SQLiteGraphStore(str(tmp_path / "graph.db"))
        event_store = SQLiteEventStore(str(tmp_path / "events.db"))
        evidence_mgr = EvidenceManager(base_path=str(tmp_path / "evidence"))

        # Enroll Alice
        nparr0 = np.frombuffer(test_images[0], np.uint8)
        img_alice = cv2.imdecode(nparr0, cv2.IMREAD_COLOR)
        f0 = rec.detect_faces(img_alice)[0]
        emb0 = rec.extract_embedding(img_alice, f0[2])
        graph_store.enroll_face(
            person_id="p_alice",
            name="Alice",
            embedding=emb0,
        )

        nparr1 = np.frombuffer(test_images[1], np.uint8)
        img_bob = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        engine = LiveFaceEngine(
            recognizer=rec,
            graph_store=graph_store,
            event_store=event_store,
            evidence_manager=evidence_mgr,
            throttle_window_s=5.0,  # 5 seconds
        )

        t1 = Track(
            track_id="trk_live_1",
            camera_id="CAM-01",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(10, 10, 100, 150),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["d1"],
        )

        frame_alice = np.zeros((300, 300, 3), dtype=np.uint8)
        frame_alice[50 : 50 + img_alice.shape[0], 50 : 50 + img_alice.shape[1]] = img_alice

        # Frame 1: t = 0s -> Emits KNOWN Alice event
        engine.process_frame(frame_alice, [t1], "CAM-01", 1, timestamp_ns=0)
        assert event_store.count() == 1

        # Frame 2: t = 1s -> Same state, throttled -> Still 1 event
        engine.process_frame(frame_alice, [t1], "CAM-01", 2, timestamp_ns=1_000_000_000)
        assert event_store.count() == 1

        # Frame 3: t = 2s -> Switch face to Bob (UNKNOWN) on same track -> State change produces immediate event!
        frame_bob = np.zeros((300, 300, 3), dtype=np.uint8)
        frame_bob[50 : 50 + img_bob.shape[0], 50 : 50 + img_bob.shape[1]] = img_bob
        engine.process_frame(frame_bob, [t1], "CAM-01", 3, timestamp_ns=2_000_000_000)
        assert event_store.count() == 2

        # Frame 4: t = 3s -> Still Bob (UNKNOWN), within throttle window -> Suppressed
        engine.process_frame(frame_bob, [t1], "CAM-01", 4, timestamp_ns=3_000_000_000)
        assert event_store.count() == 2

        # Frame 5: t = 8s -> Bob after 6s elapsed (> 5s throttle window) -> Emitted
        engine.process_frame(frame_bob, [t1], "CAM-01", 5, timestamp_ns=8_000_000_000)
        assert event_store.count() == 3

    def test_api_retrieval_of_face_event(
        self, test_images: tuple[bytes, bytes], tmp_path: Any
    ) -> None:
        """TEST 11 — API VERIFICATION: GET /events?event_type=face_detected retrieves the event with metadata."""
        import cv2
        from fastapi.testclient import TestClient
        from gods_eye.api.app import create_app
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.live_engine import LiveFaceEngine
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        rec = FaceRecognizer(detection_threshold=0.3)
        graph_store = SQLiteGraphStore(str(tmp_path / "graph.db"))
        event_store = SQLiteEventStore(str(tmp_path / "events.db"))
        evidence_mgr = EvidenceManager(base_path=str(tmp_path / "evidence"))

        # Enroll Alice
        nparr0 = np.frombuffer(test_images[0], np.uint8)
        img_alice = cv2.imdecode(nparr0, cv2.IMREAD_COLOR)
        f0 = rec.detect_faces(img_alice)[0]
        emb0 = rec.extract_embedding(img_alice, f0[2])
        graph_store.enroll_face(
            person_id="person_alice_api",
            name="Alice Wonderland",
            embedding=emb0,
        )

        engine = LiveFaceEngine(
            recognizer=rec,
            graph_store=graph_store,
            event_store=event_store,
            evidence_manager=evidence_mgr,
        )

        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[50 : 50 + img_alice.shape[0], 50 : 50 + img_alice.shape[1]] = img_alice

        track = Track(
            track_id="trk_api_1",
            camera_id="CAM-API",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(40, 40, 50 + img_alice.shape[1] + 10, 50 + img_alice.shape[0] + 10),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["d1"],
        )

        engine.process_frame(
            frame=frame,
            tracks=[track],
            camera_id="CAM-API",
            frame_id=1,
            timestamp_ns=1_500_000_000,
        )

        app = create_app(event_store=event_store, evidence_manager=evidence_mgr)
        client = TestClient(app)

        resp = client.get("/events?event_type=face_detected")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        ev = data["events"][0]
        assert ev["event_type"] == "face_detected"
        assert ev["camera_id"] == "CAM-API"
        assert ev["global_id"] == "person_alice_api"
        assert ev["metadata"]["status"] == "KNOWN"
        assert ev["metadata"]["name"] == "Alice Wonderland"
        assert ev["metadata"]["track_id"] == "trk_api_1"
        assert "evidence_id" in ev["metadata"]


class TestLiveFaceEngineProductionWiring:
    """Automated tests for #10B-FIX LiveFaceEngine production live pipeline wiring & webcam config."""

    @pytest.fixture
    def test_images(self) -> tuple[bytes, bytes]:
        """Provides real Market-1501 single-face image bytes."""
        import os

        market_dir = "data/market1501/Market-1501-v15.09.15/bounding_box_test"
        p1 = os.path.join(market_dir, "-1_c1s1_011251_02.jpg")
        p2 = os.path.join(market_dir, "-1_c1s1_011426_05.jpg")
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            return f1.read(), f2.read()

    def test_pipeline_startup_creates_live_face_engine(self, tmp_path: Any) -> None:
        """TEST 1: Normal Pipeline startup creates/attaches LiveFaceEngine."""
        from gods_eye.config.settings import Settings
        from gods_eye.detection.detector import Detector
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.ingestion.source import FrameSource
        from gods_eye.pipeline import Pipeline
        from gods_eye.tracking.tracker import Tracker

        class NoopSource(FrameSource):
            def open(self) -> bool:
                return True
            def read(self) -> tuple[bool, np.ndarray | None]:
                return False, None
            def release(self) -> None:
                pass
            def is_opened(self) -> bool:
                return False
            @property
            def resolution(self) -> tuple[int, int]:
                return (640, 480)
            @property
            def fps_nominal(self) -> float:
                return 30.0
            @property
            def source_type(self) -> str:
                return "noop"
            @property
            def is_live(self) -> bool:
                return False

        class ProperDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Any]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class ProperTracker(Tracker):
            def update(self, detections: list[Any], frame_id: int, camera_id: str) -> list[Any]:
                return []
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 0
            @property
            def total_track_count(self) -> int:
                return 0

        settings = Settings(
            event_store_path=str(tmp_path / "events.db"),
            graph_store_path=str(tmp_path / "graph.db"),
            evidence_base_path=str(tmp_path / "evidence"),
            face_detection_model_path="models/face/face_detection_yunet_2023mar.onnx",
            face_recognition_model_path="models/face/face_recognition_sface_2021dec.onnx",
        )

        pipeline = Pipeline(
            camera_id="cam_test_startup",
            source=NoopSource(),
            detector=ProperDetector(),
            tracker=ProperTracker(),
            settings=settings,
            on_result=lambda r: None,
        )

        assert pipeline.live_face_engine is not None
        assert hasattr(pipeline.live_face_engine, "process_frame")

    def test_processed_live_frame_invokes_live_face_engine(self, tmp_path: Any) -> None:
        """TEST 2: A processed live frame actually invokes LiveFaceEngine."""
        from unittest.mock import MagicMock
        from gods_eye.config.settings import Settings
        from gods_eye.detection.detector import Detector
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.ingestion.source import FrameSource
        from gods_eye.pipeline import Pipeline
        from gods_eye.reid.identity_mapper import IdentityResult
        from gods_eye.tracking.tracker import Tracker

        class SingleFrameSource(FrameSource):
            def __init__(self) -> None:
                self._delivered = False
            def open(self) -> bool:
                return True
            def read(self) -> tuple[bool, np.ndarray | None]:
                if not self._delivered:
                    self._delivered = True
                    return True, np.zeros((100, 100, 3), dtype=np.uint8)
                return False, None
            def release(self) -> None:
                pass
            def is_opened(self) -> bool:
                return not self._delivered
            @property
            def resolution(self) -> tuple[int, int]:
                return (100, 100)
            @property
            def fps_nominal(self) -> float:
                return 30.0
            @property
            def source_type(self) -> str:
                return "single"
            @property
            def is_live(self) -> bool:
                return False

        class ProperDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Any]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class ProperTracker(Tracker):
            def update(self, detections: list[Any], frame_id: int, camera_id: str) -> list[Any]:
                return []
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 0
            @property
            def total_track_count(self) -> int:
                return 0

        settings = Settings(
            frame_queue_size=5,
            detection_queue_size=5,
            track_queue_size=5,
        )

        mock_face_engine = MagicMock()
        delivered_results: list[IdentityResult] = []

        pipeline = Pipeline(
            camera_id="cam_invoke_test",
            source=SingleFrameSource(),
            detector=ProperDetector(),
            tracker=ProperTracker(),
            settings=settings,
            on_result=delivered_results.append,
            live_face_engine=mock_face_engine,
        )

        pipeline.start()
        # Wait for frame to flow through the pipeline
        timeout = time.time() + 3.0
        while time.time() < timeout and not delivered_results:
            time.sleep(0.05)
        pipeline.stop(timeout=2.0)

        assert len(delivered_results) == 1
        assert mock_face_engine.process_frame.called
        call_kwargs = mock_face_engine.process_frame.call_args[1]
        assert call_kwargs["camera_id"] == "cam_invoke_test"
        assert call_kwargs["frame"] is not None
        assert call_kwargs["frame"].shape == (100, 100, 3)

    def test_existing_track_objects_reach_live_face_engine(self, tmp_path: Any) -> None:
        """TEST 3: The existing Track objects reach LiveFaceEngine."""
        from unittest.mock import MagicMock
        from gods_eye.config.settings import Settings
        from gods_eye.detection.detector import Detector
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.ingestion.source import FrameSource
        from gods_eye.pipeline import Pipeline
        from gods_eye.reid.identity_mapper import IdentityResult
        from gods_eye.schemas.detection import BoundingBox
        from gods_eye.schemas.track import Track, TrackState
        from gods_eye.tracking.tracker import Tracker

        class SingleFrameSource(FrameSource):
            def __init__(self) -> None:
                self._delivered = False
            def open(self) -> bool:
                return True
            def read(self) -> tuple[bool, np.ndarray | None]:
                if not self._delivered:
                    self._delivered = True
                    return True, np.zeros((100, 100, 3), dtype=np.uint8)
                return False, None
            def release(self) -> None:
                pass
            def is_opened(self) -> bool:
                return not self._delivered
            @property
            def resolution(self) -> tuple[int, int]:
                return (100, 100)
            @property
            def fps_nominal(self) -> float:
                return 30.0
            @property
            def source_type(self) -> str:
                return "single"
            @property
            def is_live(self) -> bool:
                return False

        sample_track = Track(
            track_id="trk_production_reach_42",
            camera_id="cam_track_reach",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(10, 10, 80, 80),
            velocity=(1.0, 2.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["d1"],
        )

        class ProperDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Any]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class ProperTracker(Tracker):
            def update(self, detections: list[Any], frame_id: int, camera_id: str) -> list[Track]:
                return [sample_track]
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 1
            @property
            def total_track_count(self) -> int:
                return 1

        settings = Settings(
            frame_queue_size=5,
            detection_queue_size=5,
            track_queue_size=5,
        )

        mock_face_engine = MagicMock()
        delivered_results: list[IdentityResult] = []

        pipeline = Pipeline(
            camera_id="cam_track_reach",
            source=SingleFrameSource(),
            detector=ProperDetector(),
            tracker=ProperTracker(),
            settings=settings,
            on_result=delivered_results.append,
            live_face_engine=mock_face_engine,
        )

        pipeline.start()
        timeout = time.time() + 3.0
        while time.time() < timeout and not delivered_results:
            time.sleep(0.05)
        pipeline.stop(timeout=2.0)

        assert len(delivered_results) == 1
        assert mock_face_engine.process_frame.called
        call_kwargs = mock_face_engine.process_frame.call_args[1]
        passed_tracks = call_kwargs["tracks"]
        assert len(passed_tracks) == 1
        assert passed_tracks[0].track_id == "trk_production_reach_42"
        assert passed_tracks[0].bbox.x1 == 10.0

    def test_known_face_through_production_pipeline(
        self, test_images: tuple[bytes, bytes], tmp_path: Any
    ) -> None:
        """TEST 4: A known face processed through the production pipeline generates FACE_DETECTED + known identity + evidence."""
        import cv2
        from gods_eye.config.settings import Settings
        from gods_eye.detection.detector import Detector
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.ingestion.source import FrameSource
        from gods_eye.memory.graph_store import SQLiteGraphStore
        from gods_eye.pipeline import Pipeline
        from gods_eye.reid.identity_mapper import IdentityResult
        from gods_eye.schemas.detection import BoundingBox
        from gods_eye.schemas.event import EventType
        from gods_eye.schemas.track import Track, TrackState
        from gods_eye.tracking.tracker import Tracker

        rec = FaceRecognizer(detection_threshold=0.3)
        graph_store = SQLiteGraphStore(str(tmp_path / "graph.db"))
        event_store = SQLiteEventStore(str(tmp_path / "events.db"))
        evidence_mgr = EvidenceManager(base_path=str(tmp_path / "evidence"))

        # Real enrollment of Alice with actual YuNet + SFace embeddings (no fake embeddings)
        nparr0 = np.frombuffer(test_images[0], np.uint8)
        img_alice = cv2.imdecode(nparr0, cv2.IMREAD_COLOR)
        faces = rec.detect_faces(img_alice)
        assert len(faces) == 1
        emb_alice = rec.extract_embedding(img_alice, faces[0][2])
        graph_store.enroll_face(
            person_id="person_alice_prod",
            name="Alice Production",
            embedding=emb_alice,
        )

        # Prepare frame with Alice embedded
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        canvas[50 : 50 + img_alice.shape[0], 50 : 50 + img_alice.shape[1]] = img_alice

        person_track = Track(
            track_id="trk_prod_alice_1",
            camera_id="cam_prod_face",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(40, 40, 50 + img_alice.shape[1] + 10, 50 + img_alice.shape[0] + 10),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["det_1"],
        )

        class SingleFrameSource(FrameSource):
            def __init__(self, frame: np.ndarray) -> None:
                self._frame = frame
                self._delivered = False
            def open(self) -> bool:
                return True
            def read(self) -> tuple[bool, np.ndarray | None]:
                if not self._delivered:
                    self._delivered = True
                    return True, self._frame.copy()
                return False, None
            def release(self) -> None:
                pass
            def is_opened(self) -> bool:
                return not self._delivered
            @property
            def resolution(self) -> tuple[int, int]:
                return (400, 400)
            @property
            def fps_nominal(self) -> float:
                return 30.0
            @property
            def source_type(self) -> str:
                return "test"
            @property
            def is_live(self) -> bool:
                return False

        class ProperDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Any]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class ProperTracker(Tracker):
            def update(self, detections: list[Any], frame_id: int, camera_id: str) -> list[Track]:
                return [person_track]
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 1
            @property
            def total_track_count(self) -> int:
                return 1

        settings = Settings(
            frame_queue_size=5,
            detection_queue_size=5,
            track_queue_size=5,
            event_store_path=str(tmp_path / "events.db"),
            graph_store_path=str(tmp_path / "graph.db"),
            evidence_base_path=str(tmp_path / "evidence"),
        )

        delivered: list[IdentityResult] = []

        pipeline = Pipeline(
            camera_id="cam_prod_face",
            source=SingleFrameSource(canvas),
            detector=ProperDetector(),
            tracker=ProperTracker(),
            settings=settings,
            on_result=delivered.append,
            face_recognizer=rec,
            event_store=event_store,
            graph_store=graph_store,
            evidence_manager=evidence_mgr,
        )

        pipeline.start()
        timeout = time.time() + 5.0
        while time.time() < timeout and not delivered:
            time.sleep(0.05)
        pipeline.stop(timeout=2.0)

        assert len(delivered) == 1
        events = event_store.query_events(event_type=EventType.FACE_DETECTED)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == EventType.FACE_DETECTED
        assert ev.global_id == "person_alice_prod"
        assert ev.metadata["status"] == "KNOWN"
        assert ev.metadata["name"] == "Alice Production"
        assert ev.metadata["track_id"] == "trk_prod_alice_1"
        assert "evidence_id" in ev.metadata

        # Verify evidence files exist on disk
        ev_records = evidence_mgr.get_evidence(ev.event_id)
        assert len(ev_records) == 1
        meta = ev_records[0]
        assert Path(meta["snapshot_path"]).exists()
        assert Path(meta["face_crop_path"]).exists()

    def test_unknown_face_through_production_pipeline(
        self, test_images: tuple[bytes, bytes], tmp_path: Any
    ) -> None:
        """TEST 5: Unknown face through the production integration remains UNKNOWN."""
        import cv2
        from gods_eye.config.settings import Settings
        from gods_eye.detection.detector import Detector
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.ingestion.source import FrameSource
        from gods_eye.memory.graph_store import SQLiteGraphStore
        from gods_eye.pipeline import Pipeline
        from gods_eye.reid.identity_mapper import IdentityResult
        from gods_eye.schemas.detection import BoundingBox
        from gods_eye.schemas.event import EventType
        from gods_eye.schemas.track import Track, TrackState
        from gods_eye.tracking.tracker import Tracker

        rec = FaceRecognizer(detection_threshold=0.3)
        graph_store = SQLiteGraphStore(str(tmp_path / "graph.db"))
        event_store = SQLiteEventStore(str(tmp_path / "events.db"))
        evidence_mgr = EvidenceManager(base_path=str(tmp_path / "evidence"))

        # Enroll Alice ONLY
        nparr0 = np.frombuffer(test_images[0], np.uint8)
        img_alice = cv2.imdecode(nparr0, cv2.IMREAD_COLOR)
        faces_alice = rec.detect_faces(img_alice)
        emb_alice = rec.extract_embedding(img_alice, faces_alice[0][2])
        graph_store.enroll_face(person_id="p_alice", name="Alice", embedding=emb_alice)

        # Feed Bob (test_images[1]) who is NOT enrolled
        nparr1 = np.frombuffer(test_images[1], np.uint8)
        img_bob = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        canvas[50 : 50 + img_bob.shape[0], 50 : 50 + img_bob.shape[1]] = img_bob

        person_track = Track(
            track_id="trk_bob_unknown",
            camera_id="cam_bob",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(40, 40, 50 + img_bob.shape[1] + 10, 50 + img_bob.shape[0] + 10),
            velocity=(0.0, 0.0),
            first_frame_id=1,
            last_frame_id=1,
            lost_frame_count=0,
            detection_history=["det_1"],
        )

        class SingleFrameSource(FrameSource):
            def __init__(self, frame: np.ndarray) -> None:
                self._frame = frame
                self._delivered = False
            def open(self) -> bool:
                return True
            def read(self) -> tuple[bool, np.ndarray | None]:
                if not self._delivered:
                    self._delivered = True
                    return True, self._frame.copy()
                return False, None
            def release(self) -> None:
                pass
            def is_opened(self) -> bool:
                return not self._delivered
            @property
            def resolution(self) -> tuple[int, int]:
                return (400, 400)
            @property
            def fps_nominal(self) -> float:
                return 30.0
            @property
            def source_type(self) -> str:
                return "test"
            @property
            def is_live(self) -> bool:
                return False

        class ProperDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Any]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class ProperTracker(Tracker):
            def update(self, detections: list[Any], frame_id: int, camera_id: str) -> list[Track]:
                return [person_track]
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 1
            @property
            def total_track_count(self) -> int:
                return 1

        settings = Settings(
            frame_queue_size=5,
            detection_queue_size=5,
            track_queue_size=5,
            event_store_path=str(tmp_path / "events.db"),
            graph_store_path=str(tmp_path / "graph.db"),
            evidence_base_path=str(tmp_path / "evidence"),
        )

        delivered: list[IdentityResult] = []

        pipeline = Pipeline(
            camera_id="cam_bob",
            source=SingleFrameSource(canvas),
            detector=ProperDetector(),
            tracker=ProperTracker(),
            settings=settings,
            on_result=delivered.append,
            face_recognizer=rec,
            event_store=event_store,
            graph_store=graph_store,
            evidence_manager=evidence_mgr,
        )

        pipeline.start()
        timeout = time.time() + 5.0
        while time.time() < timeout and not delivered:
            time.sleep(0.05)
        pipeline.stop(timeout=2.0)

        assert len(delivered) == 1
        events = event_store.query_events(event_type=EventType.FACE_DETECTED)
        assert len(events) == 1
        ev = events[0]
        assert ev.metadata["status"] == "UNKNOWN"
        assert ev.metadata["person_id"] is None
        assert ev.metadata["track_id"] == "trk_bob_unknown"

    def test_webcam_device_index_env_propagation(self, monkeypatch: Any) -> None:
        """TEST 6: The configured GODS_EYE_WEBCAM_DEVICE_INDEX is propagated to WebcamSource."""
        from gods_eye.config.settings import Settings
        from gods_eye.ingestion.source import WebcamSource

        monkeypatch.setenv("GODS_EYE_WEBCAM_DEVICE_INDEX", "2")
        settings = Settings.from_env()
        assert settings.webcam_device_index == 2

        source = WebcamSource(device_index=settings.webcam_device_index)
        assert source.device_index == 2

    def test_webcam_default_behavior_compatible(self, monkeypatch: Any) -> None:
        """TEST 7: Default behavior remains compatible (0) when no environment variable is provided."""
        from gods_eye.config.settings import Settings
        from gods_eye.ingestion.source import WebcamSource

        monkeypatch.delenv("GODS_EYE_WEBCAM_DEVICE_INDEX", raising=False)
        settings = Settings.from_env()
        assert settings.webcam_device_index == 0

        source = WebcamSource(device_index=settings.webcam_device_index)
        assert source.device_index == 0

    def test_no_second_videocapture_created_by_live_face_engine(
        self, test_images: tuple[bytes, bytes], tmp_path: Any, monkeypatch: Any
    ) -> None:
        """TEST 8: No second VideoCapture is created by LiveFaceEngine."""
        import cv2
        from unittest.mock import MagicMock
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.live_engine import LiveFaceEngine
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.memory.graph_store import SQLiteGraphStore

        rec = FaceRecognizer(detection_threshold=0.3)
        graph_store = SQLiteGraphStore(str(tmp_path / "graph.db"))
        event_store = SQLiteEventStore(str(tmp_path / "events.db"))
        evidence_mgr = EvidenceManager(base_path=str(tmp_path / "evidence"))

        engine = LiveFaceEngine(
            recognizer=rec,
            graph_store=graph_store,
            event_store=event_store,
            evidence_manager=evidence_mgr,
        )

        # Wrap / spy on cv2.VideoCapture
        original_vc = cv2.VideoCapture
        vc_call_count = [0]

        def spy_videocapture(*args: Any, **kwargs: Any) -> Any:
            vc_call_count[0] += 1
            return original_vc(*args, **kwargs)

        monkeypatch.setattr(cv2, "VideoCapture", spy_videocapture)

        nparr0 = np.frombuffer(test_images[0], np.uint8)
        img_alice = cv2.imdecode(nparr0, cv2.IMREAD_COLOR)

        engine.process_frame(
            frame=img_alice,
            tracks=[],
            camera_id="cam_check_no_vc",
            frame_id=1,
            timestamp_ns=1_000_000_000,
        )

        # Confirm LiveFaceEngine created 0 VideoCapture instances
        assert vc_call_count[0] == 0

    def test_multi_camera_pipeline_wires_live_face_engine(self, tmp_path: Any) -> None:
        """Confirms MultiCameraPipeline automatically attaches and invokes LiveFaceEngine."""
        from unittest.mock import MagicMock
        from gods_eye.camera_graph.camera_graph import CameraGraph
        from gods_eye.config.settings import Settings
        from gods_eye.detection.detector import Detector
        from gods_eye.events.event_store import SQLiteEventStore
        from gods_eye.evidence.evidence_manager import EvidenceManager
        from gods_eye.face.recognizer import FaceRecognizer
        from gods_eye.ingestion.frame_packet import FramePacket
        from gods_eye.ingestion.source import FrameSource
        from gods_eye.memory.graph_store import SQLiteGraphStore
        from gods_eye.pipeline_multi import MultiCameraPipeline
        from gods_eye.schemas.camera import CameraNode
        from gods_eye.tracking.tracker import Tracker

        class MockSource(FrameSource):
            def open(self) -> bool:
                return True
            def read(self) -> tuple[bool, np.ndarray | None]:
                return False, None
            def release(self) -> None:
                pass
            def is_opened(self) -> bool:
                return False
            @property
            def resolution(self) -> tuple[int, int]:
                return (100, 100)
            @property
            def fps_nominal(self) -> float:
                return 30.0
            @property
            def source_type(self) -> str:
                return "mock"
            @property
            def is_live(self) -> bool:
                return False

        class ProperDetector(Detector):
            def detect(self, packet: FramePacket) -> list[Any]:
                return []
            def warmup(self) -> None:
                pass
            @property
            def device(self) -> str:
                return "cpu"
            @property
            def model_name(self) -> str:
                return "stub"

        class ProperTracker(Tracker):
            def update(self, detections: list[Any], frame_id: int, camera_id: str) -> list[Any]:
                return []
            def reset(self) -> None:
                pass
            @property
            def active_track_count(self) -> int:
                return 0
            @property
            def total_track_count(self) -> int:
                return 0

        settings = Settings(
            event_store_path=str(tmp_path / "events.db"),
            graph_store_path=str(tmp_path / "graph.db"),
            evidence_base_path=str(tmp_path / "evidence"),
            face_detection_model_path="models/face/face_detection_yunet_2023mar.onnx",
            face_recognition_model_path="models/face/face_recognition_sface_2021dec.onnx",
        )

        mock_id_mapper = MagicMock()
        cam_node = CameraNode(
            camera_id="cam_multi_01",
            display_name="Multi Cam 1",
            source_uri="mock://0",
            location_label="Hallway",
            resolution=(100, 100),
            fps_nominal=30.0,
        )

        multi_pipeline = MultiCameraPipeline(
            camera_configs=[cam_node],
            source_factory=lambda uri: MockSource(),
            detector=ProperDetector(),
            tracker_factory=lambda cid: ProperTracker(),
            identity_mapper=mock_id_mapper,
            camera_graph=CameraGraph(nodes=[cam_node]),
            settings=settings,
            on_result=lambda r: None,
        )

        assert multi_pipeline.live_face_engine is not None
        assert hasattr(multi_pipeline.live_face_engine, "process_frame")



