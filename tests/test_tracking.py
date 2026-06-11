"""Tests for the tracking subsystem.

Tests Tracker interface, ByteTrackTracker with synthetic detections,
track state transitions, and detection history.
"""

from __future__ import annotations

import uuid

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import TrackState
from gods_eye.tracking.bytetrack_tracker import ByteTrackTracker


def _make_detection(
    x1: float, y1: float, x2: float, y2: float,
    confidence: float = 0.9,
    camera_id: str = "cam-01",
    frame_id: int = 1,
) -> Detection:
    return Detection(
        detection_id=str(uuid.uuid4()),
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp_ns=frame_id * 33_000_000,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=confidence,
        class_label="person",
        source_resolution=(640, 480),
    )


class TestByteTrackTracker:
    @staticmethod
    def _make_tracker() -> ByteTrackTracker:
        settings = Settings(
            detection_confidence_threshold=0.25,
            track_lost_timeout=5,
            track_dead_timeout=10,
        )
        return ByteTrackTracker(settings)

    def test_empty_detections(self) -> None:
        tracker = self._make_tracker()
        tracks = tracker.update([], frame_id=1, camera_id="cam-01")
        assert isinstance(tracks, list)

    def test_single_detection_creates_track(self) -> None:
        tracker = self._make_tracker()
        dets = [_make_detection(100, 100, 200, 300, confidence=0.9)]
        tracks = tracker.update(dets, frame_id=1, camera_id="cam-01")
        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        assert len(active) >= 1
        assert active[0].camera_id == "cam-01"
        assert active[0].first_frame_id == 1

    def test_track_continuity(self) -> None:
        """Same detection position across frames should maintain the same track_id."""
        tracker = self._make_tracker()
        track_ids: list[str] = []

        for fid in range(1, 6):
            dets = [_make_detection(100, 100, 200, 300, frame_id=fid)]
            tracks = tracker.update(dets, frame_id=fid, camera_id="cam-01")
            active = [t for t in tracks if t.state == TrackState.ACTIVE]
            if active:
                track_ids.append(active[0].track_id)

        # All frames should have the same track_id
        assert len(set(track_ids)) == 1

    def test_track_becomes_lost(self) -> None:
        """Track transitions to LOST when detections stop."""
        tracker = self._make_tracker()

        # 3 frames with detection
        for fid in range(1, 4):
            dets = [_make_detection(100, 100, 200, 300, frame_id=fid)]
            tracker.update(dets, frame_id=fid, camera_id="cam-01")

        # Next frame: no detections → track should become LOST
        tracks = tracker.update([], frame_id=4, camera_id="cam-01")
        lost = [t for t in tracks if t.state == TrackState.LOST]
        assert len(lost) >= 1
        assert lost[0].lost_frame_count >= 1

    def test_multiple_tracks(self) -> None:
        """Two separated detections should create two distinct tracks."""
        tracker = self._make_tracker()
        dets = [
            _make_detection(10, 10, 50, 100, confidence=0.9),
            _make_detection(400, 10, 500, 100, confidence=0.85),
        ]
        tracks = tracker.update(dets, frame_id=1, camera_id="cam-01")
        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        assert len(active) == 2
        assert active[0].track_id != active[1].track_id

    def test_track_schema_fields(self) -> None:
        """Verify Track objects have all §4 required fields."""
        tracker = self._make_tracker()
        dets = [_make_detection(100, 100, 200, 300)]
        tracks = tracker.update(dets, frame_id=1, camera_id="cam-01")
        active = [t for t in tracks if t.state == TrackState.ACTIVE]
        assert len(active) >= 1
        t = active[0]
        assert isinstance(t.track_id, str)
        assert t.camera_id == "cam-01"
        assert t.state == TrackState.ACTIVE
        assert isinstance(t.bbox, BoundingBox)
        assert t.first_frame_id == 1
        assert t.last_frame_id == 1
        assert t.lost_frame_count == 0
        assert isinstance(t.detection_history, list)

    def test_reset(self) -> None:
        tracker = self._make_tracker()
        dets = [_make_detection(100, 100, 200, 300)]
        tracker.update(dets, frame_id=1, camera_id="cam-01")
        assert tracker.total_track_count >= 1
        tracker.reset()
        assert tracker.total_track_count == 0
        assert tracker.active_track_count == 0
