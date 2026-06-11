"""YOLOv8 detector — concrete implementation using ultralytics.

Model: YOLOv8n (nano)
CUDA: FP16 (half precision)
CPU: FP32
Person class only (COCO class 0)
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

from gods_eye.config.settings import Settings
from gods_eye.detection.detector import Detector
from gods_eye.ingestion.frame_packet import FramePacket
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox, Detection

# COCO class index for "person"
_PERSON_CLASS_ID: int = 0


class YOLODetector(Detector):
    """YOLOv8n person detector.

    Loads ultralytics YOLOv8n, runs inference on FramePackets,
    and converts results to canonical Detection schema (§4).

    Not thread-safe — must be called from a single thread.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = get_logger("detection")

        # Resolve device (§6 CPU-only degraded mode)
        use_cuda = torch.cuda.is_available() and not settings.cpu_only
        self._device: str = "cuda" if use_cuda else "cpu"
        self._half: bool = use_cuda  # FP16 only on CUDA

        self._log.info(
            "detector_init",
            model=settings.detection_model,
            device=self._device,
            half=self._half,
        )

        # Load model
        self._model: Any = YOLO(settings.detection_model)
        self._model.to(self._device)

        self._confidence_threshold = settings.detection_confidence_threshold
        self._target_classes: tuple[str, ...] = settings.detection_target_classes

        self._log.info(
            "detector_ready",
            confidence_threshold=self._confidence_threshold,
            target_classes=list(self._target_classes),
        )

    def warmup(self) -> None:
        """Run dummy inference to warm up CUDA kernels and JIT."""
        self._log.info("detector_warmup_start", device=self._device)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            verbose=False,
            device=self._device,
            half=self._half,
        )
        self._log.info("detector_warmup_complete")

    def detect(self, packet: FramePacket) -> list[Detection]:
        """Run YOLOv8 inference and convert to canonical Detections.

        Args:
            packet: FramePacket from ingestion layer.

        Returns:
            List of Detection objects for person-class detections
            above the confidence threshold.
        """
        results = self._model.predict(
            packet.frame,
            verbose=False,
            device=self._device,
            half=self._half,
            conf=self._confidence_threshold,
            classes=[_PERSON_CLASS_ID],
        )

        detections: list[Detection] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_val: float = float(boxes.conf[i].cpu().numpy())
            cls_id: int = int(boxes.cls[i].cpu().numpy())

            # Filter: person class only
            if cls_id != _PERSON_CLASS_ID:
                continue

            detection = Detection(
                detection_id=str(uuid.uuid4()),
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                timestamp_ns=packet.timestamp_ns,
                bbox=BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                ),
                confidence=conf_val,
                class_label="person",
                source_resolution=packet.resolution,
            )
            detections.append(detection)

        return detections

    @property
    def device(self) -> str:
        """Current execution device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Loaded model name."""
        return self._settings.detection_model
