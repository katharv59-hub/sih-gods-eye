"""Detection layer — YOLOv8 wrapper, detection pipeline.

Phase 1 deliverable: YOLOv8 detection outputting List[Detection].
"""

from gods_eye.detection.detector import Detector
from gods_eye.detection.yolo_detector import YOLODetector

__all__ = [
    "Detector",
    "YOLODetector",
]
