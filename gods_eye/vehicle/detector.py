"""Vehicle Detector — Phase 8.1 (SIH 26187).

Wraps YOLOv8 to detect vehicles (car, motorcycle, truck, bus, van).
Reuses the existing YOLO infrastructure from gods_eye/detection/ but
filters for COCO vehicle classes instead of person class.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.vehicle import VehicleDetection

_log = get_logger("vehicle.detector")

# COCO class IDs → vehicle class mapping
COCO_VEHICLE_MAP: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """YOLOv8-based vehicle detector.

    Uses the same YOLO model as person detection but filters for
    vehicle classes. Can optionally use a dedicated vehicle model.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.30,
        device: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazy-load YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]

                self._model = YOLO(self._model_path)
                if self._device:
                    self._model.to(self._device)
                _log.info(
                    "vehicle_detector_loaded",
                    model=self._model_path,
                    device=self._device or "auto",
                )
            except ImportError:
                _log.warning("ultralytics_not_available", msg="Using mock vehicle detector")
                self._model = None
        return self._model

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_id: int,
        timestamp_ns: int,
    ) -> list[VehicleDetection]:
        """Run vehicle detection on a frame.

        Args:
            frame: BGR image as numpy array.
            camera_id: Source camera identifier.
            frame_id: Frame counter.
            timestamp_ns: Unix nanoseconds timestamp.

        Returns:
            List of VehicleDetection objects.
        """
        model = self._ensure_model()
        if model is None:
            return []

        h, w = frame.shape[:2]
        source_resolution = (w, h)

        try:
            results = model(
                frame,
                conf=self._confidence_threshold,
                verbose=False,
                classes=list(COCO_VEHICLE_MAP.keys()),
            )
        except Exception as exc:
            _log.error("vehicle_detection_error", error=str(exc))
            return []

        detections: list[VehicleDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                if cls_id not in COCO_VEHICLE_MAP:
                    continue

                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                xyxy = box.xyxy[0]
                x1 = float(xyxy[0].item()) if hasattr(xyxy[0], "item") else float(xyxy[0])
                y1 = float(xyxy[1].item()) if hasattr(xyxy[1], "item") else float(xyxy[1])
                x2 = float(xyxy[2].item()) if hasattr(xyxy[2], "item") else float(xyxy[2])
                y2 = float(xyxy[3].item()) if hasattr(xyxy[3], "item") else float(xyxy[3])

                vehicle_class = COCO_VEHICLE_MAP[cls_id]
                det_id = f"vdet_{camera_id}_{frame_id}_{uuid.uuid4().hex[:8]}"

                detections.append(
                    VehicleDetection(
                        detection_id=det_id,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        timestamp_ns=timestamp_ns,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=conf,
                        vehicle_class=vehicle_class,
                        source_resolution=source_resolution,
                    )
                )

        return detections
