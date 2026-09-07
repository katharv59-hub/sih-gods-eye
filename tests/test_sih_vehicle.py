"""Unit tests for Vehicle Detection and Tracking (Phase 8.1 — SIH 26187)."""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.vehicle import VehicleDetection, VehicleTrack, VehicleTrackState
from gods_eye.vehicle.detector import VehicleDetector
from gods_eye.vehicle.tracker import VehicleTracker


class TestVehicleTracker:
    def test_iou_calculation(self) -> None:
        box_a = BoundingBox(0, 0, 10, 10)
        box_b = BoundingBox(0, 0, 10, 10)
        assert pytest.approx(VehicleTracker._iou(box_a, box_b)) == 1.0

        box_c = BoundingBox(20, 20, 30, 30)
        assert VehicleTracker._iou(box_a, box_c) == 0.0

        box_d = BoundingBox(5, 0, 15, 10)
        # overlap is 5x10 = 50, area_a = 100, area_d = 100, union = 150 -> 50/150 = 1/3
        assert pytest.approx(VehicleTracker._iou(box_a, box_d), 0.01) == 0.333

    def test_empty_detections(self) -> None:
        tracker = VehicleTracker()
        tracks = tracker.update(detections=[], frame_id=1, camera_id="cam-01")
        assert len(tracks) == 0

    def test_single_detection_creates_track(self) -> None:
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
        tracks = tracker.update(detections=[det], frame_id=1, camera_id="cam-01")
        assert len(tracks) == 1
        assert tracks[0].state == VehicleTrackState.ACTIVE
        assert tracks[0].vehicle_class == "car"
        assert tracks[0].first_frame_id == 1
        assert tracks[0].velocity == (0.0, 0.0)

    def test_track_continuity_and_velocity(self) -> None:
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
        tracker.update(detections=[det1], frame_id=1, camera_id="cam-01")

        # Frame 2: moved slightly by (4, 2)
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
        tracker.update(detections=[det], frame_id=1, camera_id="cam-01")

        # Frame 2: missed -> lost_count=1 < max_lost_frames(2) -> still ACTIVE
        tracks_f2 = tracker.update(detections=[], frame_id=2, camera_id="cam-01")
        assert len(tracks_f2) == 1

        # Frame 3: missed -> lost_count=2 >= max_lost_frames(2) -> state transitions to LOST
        tracks_f3 = tracker.update(detections=[], frame_id=3, camera_id="cam-01")
        assert len(tracks_f3) == 0

        # Frame 4: missed -> lost_count=3 -> still LOST
        tracker.update(detections=[], frame_id=4, camera_id="cam-01")

        # Frame 5: missed -> lost_count=4 >= max_dead_frames(4) -> pruned from tracks
        tracker.update(detections=[], frame_id=5, camera_id="cam-01")
        assert "vtrk_cam-01_1" not in tracker._tracks


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
        # Even if model is not loaded or blank image, detect should return list without crashing
        dets = detector.detect(
            frame=frame,
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
        )
        assert isinstance(dets, list)
