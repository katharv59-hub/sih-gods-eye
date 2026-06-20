"""Unit tests for canonical schemas (§4).

Coverage target: 100% — these are contracts.
"""

from __future__ import annotations

import numpy as np

from gods_eye.schemas import (
    BoundingBox,
    CameraNode,
    Detection,
    Event,
    EventType,
    Identity,
    IdentityStatus,
    Track,
    TrackState,
)


class TestBoundingBox:
    def test_properties(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        assert bbox.width == 100.0
        assert bbox.height == 100.0
        assert bbox.area == 10_000.0
        assert bbox.center == (60.0, 70.0)

    def test_zero_area(self) -> None:
        bbox = BoundingBox(x1=5.0, y1=5.0, x2=5.0, y2=5.0)
        assert bbox.area == 0.0


class TestDetection:
    def test_construction(self) -> None:
        det = Detection(
            detection_id="det-001",
            camera_id="cam-01",
            frame_id=42,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(0, 0, 100, 200),
            confidence=0.95,
            class_label="person",
            source_resolution=(1920, 1080),
        )
        assert det.confidence == 0.95
        assert det.class_label == "person"


class TestTrack:
    def test_states(self) -> None:
        assert TrackState.ACTIVE.value == "active"
        assert TrackState.LOST.value == "lost"
        assert TrackState.DEAD.value == "dead"

    def test_construction(self) -> None:
        track = Track(
            track_id="t-1",
            camera_id="cam-01",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(0, 0, 50, 50),
            velocity=(2.5, -1.3),
            first_frame_id=0,
            last_frame_id=10,
            lost_frame_count=0,
            detection_history=["det-001", "det-002"],
        )
        assert track.state == TrackState.ACTIVE
        assert len(track.detection_history) == 2
        assert track.velocity == (2.5, -1.3)

    def test_velocity_zero_default(self) -> None:
        """First frame of a track should have zero velocity."""
        track = Track(
            track_id="t-2",
            camera_id="cam-01",
            state=TrackState.ACTIVE,
            bbox=BoundingBox(0, 0, 50, 50),
            velocity=(0.0, 0.0),
            first_frame_id=0,
            last_frame_id=0,
            lost_frame_count=0,
        )
        assert track.velocity == (0.0, 0.0)


class TestIdentity:
    def test_status_values(self) -> None:
        assert IdentityStatus.ACTIVE.value == "active"
        assert IdentityStatus.LOST.value == "lost"
        assert IdentityStatus.PURGED.value == "purged"

    def test_construction(self) -> None:
        embedding = np.random.randn(512).astype(np.float32)
        identity = Identity(
            global_id="id-uuid-001",
            status=IdentityStatus.ACTIVE,
            embedding=embedding,
            first_seen_ns=1_000_000_000,
            last_seen_ns=2_000_000_000,
            last_camera_id="cam-01",
        )
        assert identity.embedding.shape == (512,)
        assert identity.purge_at_ns is None


class TestEvent:
    def test_event_types(self) -> None:
        assert EventType.PERSON_ENTERED_FRAME.value == "person_entered_frame"
        assert EventType.CROSS_CAMERA_TRANSITION.value == "cross_camera_transition"

    def test_construction(self) -> None:
        event = Event(
            event_id="evt-001",
            event_type=EventType.PERSON_ENTERED_FRAME,
            global_id=None,
            camera_id="cam-01",
            timestamp_ns=1_000_000_000,
            frame_id=42,
            metadata={"direction": "left"},
        )
        assert event.global_id is None


class TestCameraNode:
    def test_construction(self) -> None:
        cam = CameraNode(
            camera_id="cam-01",
            display_name="Main Entrance",
            source_uri="rtsp://192.168.1.100:554/stream",
            location_label="Building A - Front Door",
            resolution=(1920, 1080),
            fps_nominal=30.0,
            adjacent_cameras=["cam-02", "cam-03"],
        )
        assert len(cam.adjacent_cameras) == 2
        assert cam.transition_priors == {}
        assert cam.overlap_regions == []
