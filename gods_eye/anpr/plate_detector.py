"""License Plate Detector — Phase 8.2 (SIH 26187).

Detects license plate regions within vehicle bounding boxes.
Uses a lightweight approach: crop vehicle bbox, run plate region detection.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox

_log = get_logger("anpr.plate_detector")


@dataclass
class PlateRegion:
    """Detected license plate region within a vehicle bounding box.

    Attributes:
        plate_id: Unique plate region identifier.
        camera_id: Source camera.
        frame_id: Frame counter.
        timestamp_ns: Unix nanoseconds.
        bbox: Plate bounding box in absolute pixel coordinates.
        confidence: Detection confidence [0.0, 1.0].
        vehicle_track_id: Associated vehicle track ID.
        plate_image: Cropped plate region as numpy array (BGR).
    """

    plate_id: str
    camera_id: str
    frame_id: int
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    vehicle_track_id: Optional[str] = None
    plate_image: Optional[np.ndarray] = None


class PlateDetector:
    """License plate region detector.

    Uses a simple approach for the SIH demo:
    1. Crop vehicle bounding box from frame
    2. Use Canny edge detection + contour analysis to find plate-like rectangles
    3. Return PlateRegion with cropped plate image for OCR

    Can be upgraded to a dedicated YOLO plate model for production.
    """

    def __init__(
        self,
        min_plate_width_ratio: float = 0.15,
        max_plate_width_ratio: float = 0.80,
        min_aspect_ratio: float = 1.5,
        max_aspect_ratio: float = 6.0,
        confidence_threshold: float = 0.40,
    ) -> None:
        self._min_plate_width_ratio = min_plate_width_ratio
        self._max_plate_width_ratio = max_plate_width_ratio
        self._min_aspect_ratio = min_aspect_ratio
        self._max_aspect_ratio = max_aspect_ratio
        self._confidence_threshold = confidence_threshold

    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_bbox: BoundingBox,
        camera_id: str,
        frame_id: int,
        timestamp_ns: int,
        vehicle_track_id: Optional[str] = None,
    ) -> list[PlateRegion]:
        """Detect license plate regions within a vehicle bounding box.

        Args:
            frame: Full BGR frame as numpy array.
            vehicle_bbox: Vehicle bounding box in absolute coords.
            camera_id: Source camera identifier.
            frame_id: Frame counter.
            timestamp_ns: Unix nanoseconds.
            vehicle_track_id: Associated vehicle track ID.

        Returns:
            List of PlateRegion objects.
        """
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            _log.warning("opencv_not_available", msg="Cannot detect plates without OpenCV")
            return []

        h, w = frame.shape[:2]
        # Crop vehicle region
        x1 = max(0, int(vehicle_bbox.x1))
        y1 = max(0, int(vehicle_bbox.y1))
        x2 = min(w, int(vehicle_bbox.x2))
        y2 = min(h, int(vehicle_bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return []

        vehicle_crop = frame[y1:y2, x1:x2]
        vh, vw = vehicle_crop.shape[:2]

        if vh < 20 or vw < 40:
            return []

        # Focus on bottom 60% of vehicle (plates are usually in lower portion)
        plate_region_y_start = int(vh * 0.4)
        bottom_crop = vehicle_crop[plate_region_y_start:, :]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plates: list[PlateRegion] = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) < 4:
                continue

            rx, ry, rw, rh = cv2.boundingRect(approx)

            # Check aspect ratio (plates are wider than tall)
            if rh == 0:
                continue
            aspect = rw / rh

            if not (self._min_aspect_ratio <= aspect <= self._max_aspect_ratio):
                continue

            # Check relative size
            width_ratio = rw / vw
            if not (self._min_plate_width_ratio <= width_ratio <= self._max_plate_width_ratio):
                continue

            # Compute confidence based on shape regularity
            hull_area = cv2.contourArea(cv2.convexHull(approx))
            contour_area = cv2.contourArea(approx)
            solidity = contour_area / hull_area if hull_area > 0 else 0
            conf = min(1.0, solidity * 0.8 + 0.2)

            if conf < self._confidence_threshold:
                continue

            # Convert coordinates back to full frame
            abs_x1 = x1 + rx
            abs_y1 = y1 + plate_region_y_start + ry
            abs_x2 = abs_x1 + rw
            abs_y2 = abs_y1 + rh

            plate_crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
            plate_id = f"plate_{camera_id}_{frame_id}_{uuid.uuid4().hex[:8]}"

            plates.append(
                PlateRegion(
                    plate_id=plate_id,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    bbox=BoundingBox(
                        x1=float(abs_x1),
                        y1=float(abs_y1),
                        x2=float(abs_x2),
                        y2=float(abs_y2),
                    ),
                    confidence=conf,
                    vehicle_track_id=vehicle_track_id,
                    plate_image=plate_crop if plate_crop.size > 0 else None,
                )
            )

            # Only return best plate per vehicle
            break

        return plates
