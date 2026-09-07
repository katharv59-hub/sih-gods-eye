"""Unit tests for Face Detection Module (Phase 8.3 — SIH 26187).

Verifies detection-only contract, track association, and absence of recognition/embeddings.
"""

from __future__ import annotations

from unittest.mock import MagicMock
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
