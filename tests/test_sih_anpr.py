"""Unit tests for ANPR License Plate Detection and OCR (Phase 8.2 — SIH 26187)."""

from __future__ import annotations

import numpy as np
import pytest

from gods_eye.anpr.ocr import PlateOCR, PlateReadingResult
from gods_eye.anpr.plate_detector import PlateDetector, PlateRegion
from gods_eye.schemas.detection import BoundingBox


class TestPlateDetector:
    def test_initialization(self) -> None:
        detector = PlateDetector(
            min_plate_width_ratio=0.2,
            confidence_threshold=0.5,
        )
        assert detector._min_plate_width_ratio == 0.2
        assert detector._confidence_threshold == 0.5

    def test_detect_plates_empty_crop(self) -> None:
        detector = PlateDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Invalid small bounding box
        bbox = BoundingBox(0, 0, 5, 5)
        regions = detector.detect_plates(
            frame=frame,
            vehicle_bbox=bbox,
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
        )
        assert regions == []

    def test_detect_plates_valid_crop(self) -> None:
        detector = PlateDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic white plate rectangle on the frame
        frame[350:390, 200:350] = 255
        bbox = BoundingBox(150, 250, 400, 450)
        regions = detector.detect_plates(
            frame=frame,
            vehicle_bbox=bbox,
            camera_id="cam-01",
            frame_id=1,
            timestamp_ns=1_000_000_000,
            vehicle_track_id="vtrk_1",
        )
        assert isinstance(regions, list)


class TestPlateOCR:
    def test_ocr_empty_image(self) -> None:
        ocr = PlateOCR(min_confidence=0.60, use_easyocr=False)
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        res = ocr.read_plate(empty_img)
        assert res.is_uncertain is True
        assert res.plate_text == "uncertain"
        assert res.confidence == 0.0

    def test_ocr_fallback_blank_image(self) -> None:
        ocr = PlateOCR(min_confidence=0.60, use_easyocr=False)
        blank_img = np.zeros((50, 150, 3), dtype=np.uint8)
        res = ocr.read_plate(blank_img)
        assert res.is_uncertain is True
        assert res.plate_text == "uncertain"

    def test_ocr_confidence_floor_enforcement(self) -> None:
        # High confidence floor (0.95) should mark fallback readings as uncertain
        ocr = PlateOCR(min_confidence=0.95, use_easyocr=False)
        test_img = np.ones((50, 150, 3), dtype=np.uint8) * 200
        res = ocr.read_plate(test_img)
        assert res.is_uncertain is True
        assert res.plate_text == "uncertain"
