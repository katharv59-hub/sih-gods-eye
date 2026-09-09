"""Unit tests for Vehicle Detection and Tracking (Phase 8.1 — SIH 26187).

Validates that VehicleTracker strictly delegates to the canonical ByteTrack
implementation, isolates track IDs from person tracking, preserves vehicle
classes and velocity, and avoids any custom IOU tracker implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.config.settings import Settings
from gods_eye.schemas.detection import BoundingBox, Detection
from gods_eye.schemas.track import TrackState
from gods_eye.schemas.vehicle import VehicleDetection, VehicleTrack, VehicleTrackState
from gods_eye.tracking.bytetrack import ByteTrackTracker
from gods_eye.vehicle.detector import VehicleDetector
from gods_eye.vehicle.tracker import VehicleTracker


class TestVehicleTrackerArchitecture:
    """Architectural tests proving ByteTrack reuse and elimination of custom IOU tracker."""

    def test_no_custom_iou_tracker_regression(self) -> None:
        """Regression test: fail if someone later reintroduces an independent IOU tracker.

        Ensures VehicleTracker:
        1. Does not implement custom _iou calculation.
        2. Does not maintain an independent _tracks dictionary.
        3. Does not maintain an independent _next_id counter.
        4. Reuses ByteTrackTracker internally.
        """
        # 1. No custom IOU method
        assert not hasattr(
            VehicleTracker, "_iou"
        ), "Architecture violation: VehicleTracker must not implement a custom _iou method."

        tracker = VehicleTracker()

        # 2. Reuses ByteTrackTracker
        assert hasattr(tracker, "_tracker"), "VehicleTracker must wrap an internal tracker."
        assert isinstance(
            tracker._tracker, ByteTrackTracker
        ), "Architecture violation: VehicleTracker must wrap ByteTrackTracker."

        # 3. No duplicate track or ID management
        assert not hasattr(
            tracker, "_tracks"
        ), "Architecture violation: VehicleTracker must not maintain its own _tracks dict."
        assert not hasattr(
            tracker, "_next_id"
        ), "Architecture violation: VehicleTracker must not maintain its own _next_id counter."

    def test_vehicle_tracking_uses_bytetrack(self) -> None:
        """Verify VehicleTracker delegates tracking to ByteTrackTracker."""
        custom_settings = Settings(
            detection_confidence_threshold=0.25,
            track_lost_timeout=15,
            track_dead_timeout=45,
        )
        bytetrack = ByteTrackTracker(settings=custom_settings, stream_prefix="vtrk")
        tracker = VehicleTracker(tracker=bytetrack)

        assert tracker.underlying_tracker is bytetrack
        assert tracker.active_track_count == 0
        assert tracker.total_track_count == 0

    def test_person_and_vehicle_track_ids_cannot_collide(self) -> None:
        """Verify that person track IDs and vehicle track IDs are strictly isolated."""
        person_tracker = ByteTrackTracker(stream_prefix="")
        vehicle_tracker = VehicleTracker(stream_prefix="vtrk")

        # Create 1 person detection and 1 vehicle detection on camera 'cam-01'
        person_det = Detection(
            detection_id="pdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(100, 100, 150, 250),
            confidence=0.9,
            class_label="person",
            source_resolution=(1920, 1080),
        )
        vehicle_det = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(100, 100, 250, 200),
            confidence=0.9,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )

        p_tracks = person_tracker.update([person_det], frame_id=1, camera_id="cam-01")
        v_tracks = vehicle_tracker.update([vehicle_det], frame_id=1, camera_id="cam-01")

        assert len(p_tracks) == 1
        assert len(v_tracks) == 1

        person_track_id = p_tracks[0].track_id
        vehicle_track_id = v_tracks[0].track_id

        # Person tracker assigns raw string ID ("1"), vehicle tracker assigns prefixed ID ("vtrk_cam-01_1")
        assert person_track_id == "1"
        assert vehicle_track_id.startswith("vtrk_")
        assert person_track_id != vehicle_track_id
        assert not person_track_id.startswith("vtrk_")


class TestVehicleTrackerFunctionality:
    """Functional tests for vehicle tracking behavior via ByteTrack."""

    def test_empty_detections(self) -> None:
        tracker = VehicleTracker()
        tracks = tracker.update(detections=[], frame_id=1, camera_id="cam-01")
        assert len(tracks) == 0
        assert tracker.active_track_count == 0

    def test_single_detection_creates_track(self) -> None:
        tracker = VehicleTracker()
        det = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(100, 100, 200, 200),
            confidence=0.9,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks = tracker.update(detections=[det], frame_id=1, camera_id="cam-01")
        assert len(tracks) == 1
        assert tracks[0].state == VehicleTrackState.ACTIVE
        assert tracks[0].vehicle_class == "car"
        assert tracks[0].first_frame_id == 1
        assert tracks[0].velocity == (0.0, 0.0)
        assert tracks[0].track_id.startswith("vtrk_")

    def test_multiple_vehicle_tracks(self) -> None:
        """Multiple spatially separated vehicles are tracked simultaneously."""
        tracker = VehicleTracker()
        det_car = VehicleDetection(
            detection_id="vdet-car-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(10, 10, 80, 80),
            confidence=0.9,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        det_truck = VehicleDetection(
            detection_id="vdet-truck-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(300, 300, 450, 450),
            confidence=0.88,
            vehicle_class="truck",
            source_resolution=(1920, 1080),
        )

        tracks = tracker.update(
            detections=[det_car, det_truck], frame_id=1, camera_id="cam-01"
        )
        assert len(tracks) == 2
        active_ids = {t.track_id for t in tracks}
        assert len(active_ids) == 2
        classes = {t.vehicle_class for t in tracks}
        assert classes == {"car", "truck"}

    def test_vehicle_classes_preserved_across_frames(self) -> None:
        """Vehicle class label remains attached across successive frames."""
        tracker = VehicleTracker()
        det1 = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(50, 50, 150, 150),
            confidence=0.9,
            vehicle_class="motorcycle",
            source_resolution=(1920, 1080),
        )
        tracker.update([det1], frame_id=1, camera_id="cam-01")

        det2 = VehicleDetection(
            detection_id="vdet-2",
            camera_id="cam-01",
            frame_id=2,
            timestamp_ns=1_033_000_000,
            bbox=BoundingBox(55, 52, 155, 152),
            confidence=0.91,
            vehicle_class="motorcycle",
            source_resolution=(1920, 1080),
        )
        tracks = tracker.update([det2], frame_id=2, camera_id="cam-01")
        assert len(tracks) == 1
        assert tracks[0].vehicle_class == "motorcycle"

    def test_track_id_stability_across_frames(self) -> None:
        """Vehicle track ID remains stable across consecutive frames."""
        tracker = VehicleTracker()
        observed_ids: list[str] = []

        for fid in range(1, 6):
            offset = (fid - 1) * 3.0
            det = VehicleDetection(
                detection_id=f"vdet-{fid}",
                camera_id="cam-01",
                frame_id=fid,
                timestamp_ns=1_000_000_000 + fid * 33_000_000,
                bbox=BoundingBox(100 + offset, 100, 200 + offset, 200),
                confidence=0.9,
                vehicle_class="bus",
                source_resolution=(1920, 1080),
            )
            tracks = tracker.update([det], frame_id=fid, camera_id="cam-01")
            active = [t for t in tracks if t.state == VehicleTrackState.ACTIVE]
            assert len(active) == 1
            observed_ids.append(active[0].track_id)

        # Track ID must remain completely identical across all 5 frames
        assert len(set(observed_ids)) == 1

    def test_track_continuity_and_velocity(self) -> None:
        """Verify velocity calculation from consecutive bounding box centers."""
        tracker = VehicleTracker()
        det1 = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(10, 10, 50, 50),
            confidence=0.9,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks1 = tracker.update(detections=[det1], frame_id=1, camera_id="cam-01")
        assert len(tracks1) == 1
        assert tracks1[0].velocity == (0.0, 0.0)

        # Frame 2: moved by (4, 2)
        det2 = VehicleDetection(
            detection_id="vdet-2",
            camera_id="cam-01",
            frame_id=2,
            timestamp_ns=1_040_000_000,
            bbox=BoundingBox(14, 12, 54, 52),
            confidence=0.92,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks2 = tracker.update(detections=[det2], frame_id=2, camera_id="cam-01")
        assert len(tracks2) == 1
        assert tracks2[0].state == VehicleTrackState.ACTIVE
        assert tracks2[0].velocity == (4.0, 2.0)
        assert len(tracks2[0].detection_history) == 2

    def test_track_lost_and_dead_lifecycle(self) -> None:
        """Verify ACTIVE -> LOST -> DEAD lifecycle conforming to ByteTrack."""
        tracker = VehicleTracker(max_lost_frames=2, max_dead_frames=4)
        det = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(10, 10, 50, 50),
            confidence=0.9,
            vehicle_class="truck",
            source_resolution=(1920, 1080),
        )
        tracks1 = tracker.update(detections=[det], frame_id=1, camera_id="cam-01")
        assert len(tracks1) == 1
        assert tracks1[0].state == VehicleTrackState.ACTIVE

        # Frame 2: missed -> becomes LOST
        tracks2 = tracker.update(detections=[], frame_id=2, camera_id="cam-01")
        lost_tracks = [t for t in tracks2 if t.state == VehicleTrackState.LOST]
        assert len(lost_tracks) == 1
        assert lost_tracks[0].lost_frame_count == 1

        # Miss frames up to dead timeout (frame 3, 4, 5)
        tracker.update(detections=[], frame_id=3, camera_id="cam-01")
        tracker.update(detections=[], frame_id=4, camera_id="cam-01")

        # Frame 5: lost_frame_count >= max_dead_frames(4) -> track evicted
        tracks5 = tracker.update(detections=[], frame_id=5, camera_id="cam-01")
        assert len(tracks5) == 0
        assert tracker.total_track_count == 0

    def test_plate_association_and_reset(self) -> None:
        """Verify license plate association and reset functionality."""
        tracker = VehicleTracker()
        det = VehicleDetection(
            detection_id="vdet-1",
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            bbox=BoundingBox(10, 10, 50, 50),
            confidence=0.9,
            vehicle_class="car",
            source_resolution=(1920, 1080),
        )
        tracks = tracker.update([det], frame_id=1, camera_id="cam-01")
        tid = tracks[0].track_id

        tracker.associate_plate(tid, "MH12AB1234")

        # Next frame update preserves plate
        tracks2 = tracker.update([det], frame_id=2, camera_id="cam-01")
        assert tracks2[0].plate_text == "MH12AB1234"

        # Reset clears tracks and plates
        tracker.reset()
        assert tracker.active_track_count == 0
        assert tracker.total_track_count == 0


class TestVehicleDetector:
    def test_detector_initialization(self) -> None:
        detector = VehicleDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.35,
        )
        assert detector._confidence_threshold == 0.35
        assert detector._model_path == "yolov8n.pt"

    def test_detect_blank_frame_handles_gracefully(self) -> None:
        detector = VehicleDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(
            frame=frame,
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
        )
        assert isinstance(dets, list)
