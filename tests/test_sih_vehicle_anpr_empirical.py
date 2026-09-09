"""Empirical Vehicle + ANPR Validation Tests — Phase 8 (SIH 26187 Fix #5).

Tests proving that the existing vehicle + ANPR pipeline executes on real visual data:
- Real CCTV visual input (MOT17-04 video)
- Real vehicle detections via YOLOv8
- Canonical ByteTrack vehicle tracking with track continuity
- Plate region candidate detection
- EasyOCR inference with uncertainty preservation on distant/night CCTV
- Real plate OCR text recognition above confidence floor
- Vehicle track to plate association via VehicleTracker
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from benchmarks.benchmark_vehicle_anpr import (
    run_cctv_video_benchmark,
    run_ground_truth_ocr_benchmark,
    run_readable_plate_benchmark,
)
from gods_eye.anpr.ocr import PlateOCR, PlateReadingResult
from gods_eye.anpr.plate_detector import PlateDetector
from gods_eye.api.app import create_app
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.event import Event, EventType
from gods_eye.schemas.reasoning import ToolCall
from gods_eye.schemas.vehicle import (
    VEHICLE_CLASSES,
    VehicleDetection,
    VehiclePlateAssociation,
    VehicleTrack,
    VehicleTrackState,
)
from gods_eye.vehicle.detector import VehicleDetector
from gods_eye.vehicle.tracker import VehicleTracker
from fastapi.testclient import TestClient

_CCTV_VIDEO_PATH = "tests/data/demo_videos/mot17_04_medium_density.mp4"
_PLATE_IMAGE_PATH = "tests/data/vehicle_samples/readable_plate_sample.png"
_GT_SAMPLE_PATH = "tests/data/vehicle_samples/maine_sample.png"


@pytest.fixture(scope="module")
def sample_cctv_frame() -> np.ndarray:
    """Extract a representative frame with vehicles from MOT17-04."""
    cap = cv2.VideoCapture(_CCTV_VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open CCTV video: {_CCTV_VIDEO_PATH}"
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None
    return frame


class TestEmpiricalVehicleDetection:
    """Validates real visual inference via VehicleDetector on CCTV frames."""

    def test_detector_detects_real_vehicles_in_cctv_frame(self, sample_cctv_frame: np.ndarray) -> None:
        detector = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
        h, w = sample_cctv_frame.shape[:2]

        dets = detector.detect(
            frame=sample_cctv_frame,
            camera_id="cam-cctv-01",
            frame_id=5,
            timestamp_ns=1_000_000_000,
        )

        assert len(dets) >= 1, "Expected at least 1 vehicle in MOT17-04 frame 5"

        for d in dets:
            assert d.vehicle_class in VEHICLE_CLASSES
            assert 0.0 <= d.confidence <= 1.0
            assert d.source_resolution == (w, h)
            assert 0 <= d.bbox.x1 < d.bbox.x2 <= w
            assert 0 <= d.bbox.y1 < d.bbox.y2 <= h
            assert d.detection_id.startswith("vdet_cam-cctv-01_5_")

        classes_found = {d.vehicle_class for d in dets}
        assert "car" in classes_found or "motorcycle" in classes_found


class TestEmpiricalByteTrackTracking:
    """Validates canonical ByteTrack vehicle tracking across sequential CCTV frames."""

    def test_tracking_continuity_across_cctv_frames(self) -> None:
        cap = cv2.VideoCapture(_CCTV_VIDEO_PATH)
        detector = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
        tracker = VehicleTracker(stream_prefix="vtrk")

        seen_track_ids: set[str] = set()

        for f in range(1, 15):
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ns = 1_000_000_000 + f * 33_333_333
            dets = detector.detect(frame, "cam-cctv-01", f, timestamp_ns)
            tracks = tracker.update(dets, f, "cam-cctv-01")

            for t in tracks:
                seen_track_ids.add(t.track_id)
                assert t.track_id.startswith("vtrk_cam-cctv-01_")
                assert t.camera_id == "cam-cctv-01"
                assert t.state in (VehicleTrackState.ACTIVE, VehicleTrackState.LOST)
                assert t.vehicle_class in VEHICLE_CLASSES

        cap.release()

        assert len(seen_track_ids) >= 1
        assert tracker.total_track_count >= 1


class TestEmpiricalPlateDetectionAndUncertainty:
    """Validates plate detection and OCR uncertainty preservation on real CCTV frames."""

    def test_plate_detection_candidate_regions(self, sample_cctv_frame: np.ndarray) -> None:
        detector = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
        plate_detector = PlateDetector(confidence_threshold=0.40)

        dets = detector.detect(sample_cctv_frame, "cam-cctv-01", 5, 1_000_000_000)
        cars = [d for d in dets if d.vehicle_class == "car"]
        assert len(cars) >= 1

        car_bbox = cars[0].bbox
        plates = plate_detector.detect_plates(
            frame=sample_cctv_frame,
            vehicle_bbox=car_bbox,
            camera_id="cam-cctv-01",
            frame_id=5,
            timestamp_ns=1_000_000_000,
            vehicle_track_id="vtrk_test_1",
        )

        for p in plates:
            assert p.camera_id == "cam-cctv-01"
            assert p.frame_id == 5
            assert p.vehicle_track_id == "vtrk_test_1"
            assert p.plate_image is not None
            assert p.plate_image.size > 0
            assert p.confidence >= 0.40

    def test_ocr_uncertainty_preservation_on_distant_cctv(self, sample_cctv_frame: np.ndarray) -> None:
        """Distant night CCTV plates must NOT produce fabricated text; must return 'uncertain'."""
        detector = VehicleDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
        plate_detector = PlateDetector(confidence_threshold=0.40)
        ocr = PlateOCR(min_confidence=0.60, use_easyocr=True)

        dets = detector.detect(sample_cctv_frame, "cam-cctv-01", 5, 1_000_000_000)
        cars = [d for d in dets if d.vehicle_class == "car"]
        assert len(cars) >= 1

        plates = plate_detector.detect_plates(
            frame=sample_cctv_frame,
            vehicle_bbox=cars[0].bbox,
            camera_id="cam-cctv-01",
            frame_id=5,
            timestamp_ns=1_000_000_000,
            vehicle_track_id="vtrk_test_1",
        )

        for p in plates:
            res = ocr.read_plate(p.plate_image)
            assert res.is_uncertain is True
            assert res.plate_text == "uncertain"


class TestVehiclePlateAssociationConfidence:
    """Validates deterministic vehicle<->plate association confidence semantics (Fix #5A)."""

    def test_valid_association_contains_confidence(self) -> None:
        tracker = VehicleTracker(stream_prefix="vtrk")
        track_id = "vtrk_cam-01_100"

        assoc = tracker.associate_plate(
            track_id=track_id,
            plate_text="DL01AB1234",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
        )

        assert isinstance(assoc, VehiclePlateAssociation)
        assert assoc.track_id == track_id
        assert assoc.plate_text == "DL01AB1234"
        assert assoc.confidence is not None
        assert 0.0 <= assoc.confidence <= 1.0

    def test_strong_evidence_yields_higher_confidence_than_weak(self) -> None:
        tracker = VehicleTracker(stream_prefix="vtrk")

        # Create an active track in tracker by updating with a detection
        det = VehicleDetection(
            detection_id="vdet_1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(100, 100, 300, 300),
            confidence=0.90,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks = tracker.update([det], frame_id=1, camera_id="cam-01")
        track_id = tracks[0].track_id

        # 1. Strong evidence: plate fully inside vehicle lower half + high OCR confidence
        strong_plate_bbox = BoundingBox(150, 220, 250, 260)
        strong_vehicle_bbox = BoundingBox(100, 100, 300, 300)
        strong_assoc = tracker.associate_plate(
            track_id=track_id,
            plate_text="DL01AB1234",
            ocr_confidence=0.95,
            is_uncertain=False,
            plate_bbox=strong_plate_bbox,
            vehicle_bbox=strong_vehicle_bbox,
        )

        # 2. Weak evidence: plate poorly overlapping + low OCR confidence
        weak_plate_bbox = BoundingBox(50, 50, 120, 120)  # largely outside vehicle
        weak_assoc = tracker.associate_plate(
            track_id="vtrk_unknown_track",
            plate_text="MH12CD5678",
            ocr_confidence=0.35,
            is_uncertain=False,
            plate_bbox=weak_plate_bbox,
            vehicle_bbox=strong_vehicle_bbox,
        )

        assert strong_assoc.confidence > weak_assoc.confidence
        assert strong_assoc.confidence >= 0.70
        assert weak_assoc.confidence < 0.60

    def test_uncertain_plate_observation_confidence_bounded(self) -> None:
        """Uncertain or unreadable plate observations must NEVER receive high association confidence."""
        tracker = VehicleTracker(stream_prefix="vtrk")

        assoc = tracker.associate_plate(
            track_id="vtrk_test_1",
            plate_text="uncertain",
            is_uncertain=True,
            ocr_confidence=0.0,
        )

        assert assoc.is_uncertain is True
        assert assoc.confidence <= 0.30

    def test_vehicle_track_preserves_association_and_confidence(self) -> None:
        tracker = VehicleTracker(stream_prefix="vtrk")
        det = VehicleDetection(
            detection_id="vdet_1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(100, 100, 300, 300),
            confidence=0.90,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks = tracker.update([det], frame_id=1, camera_id="cam-01")
        track_id = tracks[0].track_id

        tracker.associate_plate(track_id, "KA04MH9999", confidence=0.88)

        # Next frame update
        det2 = VehicleDetection(
            detection_id="vdet_2",
            camera_id="cam-01",
            frame_id=2,
            timestamp_ns=1_033_333_333,
            bbox=BoundingBox(102, 102, 302, 302),
            confidence=0.92,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks2 = tracker.update([det2], frame_id=2, camera_id="cam-01")

        assert len(tracks2) == 1
        t = tracks2[0]
        assert t.track_id == track_id
        assert t.plate_text == "KA04MH9999"
        assert t.association_confidence == 0.88
        assert t.plate_association is not None
        assert t.plate_association.confidence == 0.88
        assert t.plate_association.plate_text == "KA04MH9999"


class TestGroundTruthANPROCR:
    """Validates real OCR recognition against known independent ground truth (Fix #5A)."""

    def test_ground_truth_ocr_crop_exact_match(self) -> None:
        """Ground-truth registration crop from official specimen plate must match 'SAMPLE' exactly."""
        assert Path(_GT_SAMPLE_PATH).exists(), f"Ground-truth asset missing: {_GT_SAMPLE_PATH}"
        img = cv2.imread(_GT_SAMPLE_PATH)
        assert img is not None

        # Registration number crop: [68:273, 91:610]
        crop = img[68:273, 91:610]
        ocr = PlateOCR(min_confidence=0.60, use_easyocr=True)
        res = ocr.read_plate(crop)

        expected_ground_truth = "SAMPLE"

        # 1. OCR executed and returned text
        assert len(res.plate_text) > 0
        # 2. Text matches known ground truth exactly
        assert res.plate_text.strip() == expected_ground_truth
        # 3. OCR confidence is high (well above 0.60 floor)
        assert res.confidence >= 0.90
        # 4. Uncertainty state is False
        assert res.is_uncertain is False

    def test_ground_truth_ocr_full_plate_distinguishes_concatenation(self) -> None:
        """Full plate OCR contains state name and slogans; must distinguish token match from exact match."""
        img = cv2.imread(_GT_SAMPLE_PATH)
        assert img is not None

        ocr = PlateOCR(min_confidence=0.60, use_easyocr=True)
        res = ocr.read_plate(img)

        expected_ground_truth = "SAMPLE"

        # The ground truth 'SAMPLE' is correctly recognized as a token within the full plate
        assert expected_ground_truth in res.plate_text.split()
        # Honest empirical distinction: full plate OCR is NOT an exact match due to slogans/headers
        assert res.plate_text.strip() != expected_ground_truth
        assert res.confidence >= 0.80
        assert res.is_uncertain is False


class TestEmpiricalBenchmarkHarnessExecution:
    """Validates that the reproducible benchmark functions execute and return valid metrics."""

    def test_cctv_benchmark_harness(self) -> None:
        results = run_cctv_video_benchmark(_CCTV_VIDEO_PATH, max_frames=5)
        assert results["dataset"]["frames_processed"] == 5
        assert results["vehicle_detection"]["total_detections"] >= 5
        assert results["bytetrack_tracking"]["total_unique_tracks"] >= 1
        assert "car" in results["vehicle_detection"]["class_distribution"]
        assert results["performance"]["throughput_fps"] > 0
        assert "association" in results
        assert results["association"]["total_associations"] >= 0
        assert "canonical_event_provenance" in results
        assert results["canonical_event_provenance"]["persisted_vehicle_detected_events"] >= 1

    def test_ground_truth_benchmark_harness(self) -> None:
        results = run_ground_truth_ocr_benchmark(_GT_SAMPLE_PATH)
        assert results["ground_truth"]["expected_registration_text"] == "SAMPLE"
        assert results["registration_crop_ocr"]["exact_match"] is True
        assert results["registration_crop_ocr"]["confidence"] >= 0.90
        assert results["association"]["association_success"] is True
        assert results["association"]["association_confidence"] >= 0.70
        assert "canonical_event_provenance" in results
        assert results["canonical_event_provenance"]["tool13_success"] is True
        assert results["canonical_event_provenance"]["tool13_count"] >= 1
        assert results["canonical_event_provenance"]["api_plate_status"] == 200
        assert results["canonical_event_provenance"]["api_vehicle_status"] == 200
        assert results["canonical_event_provenance"]["api_unobserved_plate_status"] == 200
        assert results["canonical_event_provenance"]["api_unobserved_vehicle_status"] == 200


class TestCanonicalVehicleANPRProvenanceAndAdversarial:
    """Validates direct provenance of vehicle/ANPR objects into canonical events and C2 API (Fix #7)."""

    def test_vehicle_track_to_canonical_event(self) -> None:
        """VehicleTrack.to_event() must produce valid EventType.VEHICLE_DETECTED preserving all fields."""
        track = VehicleTrack(
            track_id="vtrk_cam01_42",
            camera_id="cam-01",
            state=VehicleTrackState.ACTIVE,
            bbox=BoundingBox(x1=100.0, y1=150.0, x2=350.0, y2=400.0),
            velocity=(1.5, -0.5),
            vehicle_class="truck",
            first_frame_id=10,
            last_frame_id=25,
            lost_frame_count=0,
            plate_text="KA01MJ5000",
            association_confidence=0.88,
        )

        ev = track.to_event(timestamp_ns=1_500_000_000, frame_id=25, confidence=0.91)

        assert isinstance(ev, Event)
        assert ev.event_type == EventType.VEHICLE_DETECTED
        assert ev.global_id == "vtrk_cam01_42"
        assert ev.camera_id == "cam-01"
        assert ev.frame_id == 25
        assert ev.timestamp_ns == 1_500_000_000
        assert ev.confidence == 0.91
        assert ev.metadata["vehicle_track_id"] == "vtrk_cam01_42"
        assert ev.metadata["vehicle_class"] == "truck"
        assert ev.metadata["plate_text"] == "KA01MJ5000"
        assert ev.metadata["association_confidence"] == 0.88
        assert ev.metadata["bbox"] == [100.0, 150.0, 350.0, 400.0]
        assert ev.metadata["velocity"] == [1.5, -0.5]

    def test_vehicle_plate_association_to_canonical_event(self) -> None:
        """VehiclePlateAssociation.to_event() must produce EventType.ANPR_READING."""
        assoc = VehiclePlateAssociation(
            track_id="vtrk_cam01_42",
            plate_text="KA01MJ5000",
            confidence=0.88,
            camera_id="cam-01",
            frame_id=25,
            timestamp_ns=1_500_000_000,
            is_uncertain=False,
            ocr_confidence=0.92,
            spatial_confidence=0.85,
        )

        ev = assoc.to_event(event_id="ev_anpr_test_01")

        assert isinstance(ev, Event)
        assert ev.event_id == "ev_anpr_test_01"
        assert ev.event_type == EventType.ANPR_READING
        assert ev.global_id == "vtrk_cam01_42"
        assert ev.camera_id == "cam-01"
        assert ev.timestamp_ns == 1_500_000_000
        assert ev.metadata["plate_text"] == "KA01MJ5000"
        assert ev.metadata["ocr_confidence"] == 0.92
        assert ev.metadata["vehicle_track_id"] == "vtrk_cam01_42"
        assert ev.metadata["association_confidence"] == 0.88
        assert ev.metadata["is_uncertain"] is False
        assert ev.confidence == 0.92

    def test_uncertain_ocr_event_preserves_uncertainty_without_fabrication(self) -> None:
        """Uncertain OCR must remain is_uncertain=True with empty/raw text and no fabricated plate."""
        assoc = VehiclePlateAssociation(
            track_id="vtrk_cam01_99",
            plate_text="",
            confidence=0.0,
            camera_id="cam-01",
            frame_id=30,
            timestamp_ns=1_600_000_000,
            is_uncertain=True,
            ocr_confidence=0.15,
        )

        ev = assoc.to_event()

        assert ev.metadata["is_uncertain"] is True
        assert ev.metadata["plate_text"] == ""
        assert ev.confidence == 0.15
        assert "Uncertain" in ev.explanation

    def test_standalone_plate_reading_result_to_event(self) -> None:
        """PlateReadingResult.to_event() adapts an unassociated reading into canonical ANPR_READING."""
        reading = PlateReadingResult(
            plate_text="DL01AB1234",
            confidence=0.89,
            raw_text="DL01AB1234",
            is_uncertain=False,
        )

        ev = reading.to_event(
            camera_id="cam-gate",
            timestamp_ns=2_000_000_000,
            frame_id=100,
            vehicle_track_id="vtrk_gate_01",
            association_confidence=0.85,
        )

        assert ev.event_type == EventType.ANPR_READING
        assert ev.metadata["plate_text"] == "DL01AB1234"
        assert ev.metadata["ocr_confidence"] == 0.89
        assert ev.metadata["vehicle_track_id"] == "vtrk_gate_01"
        assert ev.metadata["association_confidence"] == 0.85

    def test_tracker_adapter_methods(self) -> None:
        """VehicleTracker adapter methods to_event and to_anpr_event delegate cleanly."""
        tracker = VehicleTracker(stream_prefix="vtrk")
        track = VehicleTrack(
            track_id="vtrk_stream_1",
            camera_id="cam-01",
            state=VehicleTrackState.ACTIVE,
            bbox=BoundingBox(x1=10.0, y1=20.0, x2=80.0, y2=90.0),
            velocity=(0.0, 0.0),
            vehicle_class="car",
            first_frame_id=1,
            last_frame_id=5,
            lost_frame_count=0,
        )
        assoc = tracker.associate_plate(
            track_id="vtrk_stream_1",
            plate_text="MH14AA1111",
            is_uncertain=False,
            ocr_confidence=0.90,
            camera_id="cam-01",
            frame_id=5,
            timestamp_ns=1_000_000_000,
        )

        ev_veh = tracker.to_event(track, timestamp_ns=1_000_000_000)
        assert ev_veh.event_type == EventType.VEHICLE_DETECTED
        assert ev_veh.metadata["vehicle_track_id"] == "vtrk_stream_1"

        ev_anpr = tracker.to_anpr_event(assoc)
        assert ev_anpr.event_type == EventType.ANPR_READING
        assert ev_anpr.metadata["plate_text"] == "MH14AA1111"

    def test_association_confidence_bounds(self) -> None:
        """Association confidence must strictly enforce [0.0, 1.0] bounds."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            VehiclePlateAssociation(track_id="vtrk_1", plate_text="ABC", confidence=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            VehiclePlateAssociation(track_id="vtrk_1", plate_text="ABC", confidence=-0.1)

    def test_canonical_persistence_and_c2_api_retrieval(self, tmp_path: Path) -> None:
        """Test full provenance: track/association -> EventStore -> Tool 13 -> C2 API with negative cases."""
        db_path = tmp_path / "provenance_test.db"
        event_store = SQLiteEventStore(db_path=db_path)

        # 1. Create and persist vehicle event
        track = VehicleTrack(
            track_id="vtrk_c2_test",
            camera_id="cam-east",
            state=VehicleTrackState.ACTIVE,
            bbox=BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
            velocity=(1.0, 0.0),
            vehicle_class="car",
            first_frame_id=1,
            last_frame_id=10,
            lost_frame_count=0,
            plate_text="DL08CC5555",
            association_confidence=0.92,
        )
        ev_veh = track.to_event(timestamp_ns=1_000_000_000, frame_id=10, confidence=0.90)
        event_store.append(ev_veh)

        # 2. Create and persist ANPR event
        assoc = VehiclePlateAssociation(
            track_id="vtrk_c2_test",
            plate_text="DL08CC5555",
            confidence=0.92,
            camera_id="cam-east",
            frame_id=10,
            timestamp_ns=1_000_000_000,
            is_uncertain=False,
            ocr_confidence=0.94,
        )
        ev_anpr = assoc.to_event()
        event_store.append(ev_anpr)

        # 3. Tool 13 retrieval
        dispatcher = ToolDispatcher(event_store=event_store)
        tc = ToolCall(
            call_id="call_t13_1",
            tool_name="get_plate_history",
            arguments={"plate_text": "DL08CC5555"},
        )
        tr = dispatcher.dispatch(tc)
        assert tr.success is True
        assert tr.data["count"] == 1
        assert tr.data["readings"][0]["plate_text"] == "DL08CC5555"

        # 4. C2 API retrieval
        app = create_app(event_store=event_store, tool_dispatcher=dispatcher)
        client = TestClient(app)

        # Positive plate query
        r_plate = client.get("/plates/DL08CC5555")
        assert r_plate.status_code == 200
        assert "DL08CC5555" in str(r_plate.json())

        # Positive vehicle query
        r_veh = client.get("/vehicles/vtrk_c2_test")
        assert r_veh.status_code == 200
        veh_events = r_veh.json().get("events", [])
        assert len(veh_events) == 1
        assert veh_events[0]["event_id"] == ev_veh.event_id

        # Negative unobserved plate (valid empty 200, not error)
        r_plate_neg = client.get("/plates/UNOBSERVED_PLATE_000")
        assert r_plate_neg.status_code == 200
        neg_data = r_plate_neg.json()
        assert neg_data["plate"] == "UNOBSERVED_PLATE_000"
        readings_sub = neg_data["readings"]
        if isinstance(readings_sub, dict):
            assert readings_sub.get("count", 0) == 0
        else:
            assert len(readings_sub) == 0

        # Negative unobserved vehicle (valid empty 200, not error)
        r_veh_neg = client.get("/vehicles/vtrk_unobserved_999")
        assert r_veh_neg.status_code == 200
        assert r_veh_neg.json().get("events", []) == []

        # Negative empty plate query (400 client error)
        r_plate_empty = client.get("/plates/%20%20")
        assert r_plate_empty.status_code == 400

        # Close store to release Windows file lock
        event_store.close()


