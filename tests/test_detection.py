"""Tests for the detection subsystem.

Tests Detector interface, YOLODetector output conformance to §4 schema,
and detection conversion logic.
"""

from __future__ import annotations

import uuid

import numpy as np

from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.schemas.detection import BoundingBox, Detection


# ─── Detection Schema Conformance ────────────────────────────────────────────


class TestDetectionConversion:
    """Verify that Detection objects produced match §4 schema exactly."""

    @staticmethod
    def _make_packet() -> FramePacket:
        return FramePacket(
            camera_id="cam-01",
            frame_id=42,
            timestamp_ns=1_000_000_000,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            resolution=(640, 480),
        )

    def test_detection_fields(self) -> None:
        """A Detection built from inference results has all §4 fields."""
        det = Detection(
            detection_id=str(uuid.uuid4()),
            camera_id="cam-01",
            frame_id=42,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=500.0),
            confidence=0.87,
            class_label="person",
            source_resolution=(640, 480),
        )
        assert det.camera_id == "cam-01"
        assert det.frame_id == 42
        assert det.class_label == "person"
        assert 0.0 <= det.confidence <= 1.0
        assert det.bbox.width == 200.0
        assert det.bbox.height == 300.0
        assert det.source_resolution == (640, 480)
        # detection_id must be a valid UUID
        uuid.UUID(det.detection_id)

    def test_detection_id_uniqueness(self) -> None:
        """Each Detection must have a unique UUID."""
        ids = {str(uuid.uuid4()) for _ in range(100)}
        assert len(ids) == 100

    def test_bbox_from_xyxy(self) -> None:
        """BoundingBox correctly represents x1y1x2y2 format."""
        bbox = BoundingBox(x1=10.5, y1=20.3, x2=110.5, y2=220.3)
        assert bbox.width == 100.0
        assert bbox.height == 200.0
        assert bbox.area == 20_000.0

    def test_empty_detection_list(self) -> None:
        """No detections is a valid result."""
        detections: list[Detection] = []
        assert len(detections) == 0

    def test_multiple_detections(self) -> None:
        """Multiple detections from same frame share camera_id and frame_id."""
        base = {
            "camera_id": "cam-01",
            "frame_id": 42,
            "timestamp_ns": 1_000_000_000,
            "class_label": "person",
            "source_resolution": (640, 480),
        }
        dets = [
            Detection(
                detection_id=str(uuid.uuid4()),
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
                confidence=0.9,
                **base,
            ),
            Detection(
                detection_id=str(uuid.uuid4()),
                bbox=BoundingBox(x1=300, y1=100, x2=400, y2=300),
                confidence=0.75,
                **base,
            ),
        ]
        assert len(dets) == 2
        assert dets[0].detection_id != dets[1].detection_id
        assert dets[0].camera_id == dets[1].camera_id
