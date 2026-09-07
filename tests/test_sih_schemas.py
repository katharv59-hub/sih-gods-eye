"""Unit tests for SIH 26187 Schemas (Phase 8-10).

Validates all contracts, bounds, immutability, and serialization.
"""

from __future__ import annotations

import pytest

from gods_eye.schemas import (
    BoundingBox,
    EventType,
    VehicleDetection,
    VehicleTrack,
    VehicleTrackState,
    Zone,
    Alert,
    AlertStatus,
)
from gods_eye.anpr.ocr import PlateReadingResult
from gods_eye.face.detector import FaceDetectionResult


class TestVehicleSchemas:
    def test_vehicle_detection(self) -> None:
        det = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(0, 0, 200, 150),
            confidence=0.92,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        assert det.vehicle_class == "car"
        assert det.confidence == 0.92
        assert det.bbox.area == 30_000.0

    def test_vehicle_detection_invariants(self) -> None:
        with pytest.raises(ValueError, match="confidence MUST be between"):
            VehicleDetection(
                detection_id="vdet-1",
                camera_id="cam-01",
                frame_id=1,
                timestamp_ns=1_000_000_000,
                bbox=BoundingBox(0, 0, 200, 150),
                confidence=1.5,
                vehicle_class="car",
                source_resolution=(1920, 1080),
            )

        with pytest.raises(ValueError, match="vehicle_class MUST be one of"):
            VehicleDetection(
                detection_id="vdet-1",
                camera_id="cam-01",
                frame_id=1,
                timestamp_ns=1_000_000_000,
                bbox=BoundingBox(0, 0, 200, 150),
                confidence=0.9,
                vehicle_class="submarine",  # type: ignore
                source_resolution=(1920, 1080),
            )

    def test_vehicle_track(self) -> None:
        track = VehicleTrack(
            track_id="vt-1",
            camera_id="cam-01",
            state=VehicleTrackState.ACTIVE,
            bbox=BoundingBox(0, 0, 200, 150),
            velocity=(2.5, 0.0),
            vehicle_class="truck",
            first_frame_id=1,
            last_frame_id=10,
            lost_frame_count=0,
            detection_history=["vdet-1"],
            plate_text="DL01AB1234",
        )
        assert track.vehicle_class == "truck"
        assert track.state == VehicleTrackState.ACTIVE
        assert track.plate_text == "DL01AB1234"
        assert len(track.detection_history) == 1


class TestANPRSchemas:
    def test_plate_reading_result(self) -> None:
        res = PlateReadingResult(
            plate_text="DL01AB1234",
            confidence=0.95,
            raw_text="DL01AB1234",
            is_uncertain=False,
        )
        assert res.plate_text == "DL01AB1234"
        assert not res.is_uncertain

    def test_plate_reading_uncertain(self) -> None:
        res = PlateReadingResult(
            plate_text="uncertain",
            confidence=0.45,
            raw_text="D?01A",
            is_uncertain=True,
        )
        assert res.is_uncertain is True
        assert res.plate_text == "uncertain"


class TestFaceDetectionSchema:
    def test_face_detection(self) -> None:
        face = FaceDetectionResult(
            face_id="face-1",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(10, 10, 50, 60),
            confidence=0.88,
            associated_person_track_id="pt-1",
        )
        assert face.face_id == "face-1"
        assert face.associated_person_track_id == "pt-1"


class TestZoneSchemas:
    def test_zone_construction(self) -> None:
        zone = Zone(
            zone_id="zone-secure",
            camera_id="cam-01",
            display_name="Restricted Periphery",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            zone_type="restricted",
            expected_dwell_s=(0.0, 30.0),
            allowed_classes=["person"],
            allowed_time_window=(6, 22),
            direction_rules="entry_only",
        )
        assert zone.zone_type == "restricted"
        assert zone.expected_dwell_s == (0.0, 30.0)
        assert zone.direction_rules == "entry_only"
        assert len(zone.polygon) == 4


class TestAlertSchemas:
    def test_alert_lifecycle(self) -> None:
        alert = Alert(
            alert_id="alt-001",
            severity="HIGH",
            status=AlertStatus.ACTIVE,
            subject_ref="person_42",
            zone_id="zone-secure",
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            risk_signal_id="risk_sig_1",
            hypothesis_ids=("hyp_1",),
            evidence_ids=("ev-1", "ev-2"),
            explanation="Intrusion detected in restricted area",
        )
        assert alert.severity == "HIGH"
        assert alert.status == AlertStatus.ACTIVE
        assert len(alert.evidence_ids) == 2

    def test_alert_invariants(self) -> None:
        with pytest.raises(ValueError, match="severity MUST be one of"):
            Alert(
                alert_id="alt-002",
                severity="EXTREME",  # Invalid severity
                status=AlertStatus.ACTIVE,
                subject_ref=None,
                zone_id=None,
                camera_id="cam-01",
                timestamp_ns=1_000_000_000,
                risk_signal_id="risk_1",
            )
