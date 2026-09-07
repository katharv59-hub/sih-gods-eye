"""Face Detector — Phase 8.3 (SIH 26187).

Detects faces in video frames. Outputs face_id, camera_id, timestamp_ns,
bbox, confidence, associated_person_track_id.

EXPLICITLY OUT OF SCOPE: face recognition/embeddings.
This requires the same legal-basis-ADR prerequisite pattern the master spec
applied to Phase 6 identity profiling. Flagged here, not built.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox

_log = get_logger("face.detector")


@dataclass
class FaceDetectionResult:
    """A single face detection result.

    Attributes:
        face_id: Unique face detection identifier.
        camera_id: Source camera.
        timestamp_ns: Unix nanoseconds.
        bbox: Face bounding box in absolute pixel coordinates.
        confidence: Detection confidence [0.0, 1.0].
        associated_person_track_id: Person track this face belongs to, if known.
    """

    face_id: str
    camera_id: str
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    associated_person_track_id: Optional[str] = None


class FaceDetector:
    """Face detector using YOLO or Haar cascade.

    For the SIH demo, uses OpenCV's built-in Haar cascade face detector
    as a lightweight option. Can be upgraded to YOLOv8-face or RetinaFace.

    WARNING: This module performs DETECTION ONLY.
    Face recognition/embeddings are NOT implemented.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: tuple[int, int] = (30, 30),
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_size = min_face_size
        self._cascade: Any = None

    def _ensure_cascade(self) -> Any:
        """Lazy-load Haar cascade classifier."""
        if self._cascade is None:
            try:
                import cv2  # type: ignore[import-untyped]

                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
                if self._cascade.empty():
                    _log.warning("face_cascade_empty", path=cascade_path)
                    self._cascade = None
                else:
                    _log.info("face_detector_loaded", cascade="haarcascade_frontalface_default")
            except (ImportError, AttributeError):
                _log.warning("opencv_not_available", msg="Face detector unavailable")
                self._cascade = None
        return self._cascade

    def detect(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp_ns: int,
        person_tracks: Optional[list[Any]] = None,
    ) -> list[FaceDetectionResult]:
        """Detect faces in a frame.

        Args:
            frame: BGR image as numpy array.
            camera_id: Source camera identifier.
            timestamp_ns: Unix nanoseconds.
            person_tracks: Optional list of person Track objects for association.

        Returns:
            List of FaceDetectionResult objects.
        """
        cascade = self._ensure_cascade()
        if cascade is None:
            return []

        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_face_size,
        )

        results: list[FaceDetectionResult] = []

        for x, y, w, h in faces:
            face_center = (x + w / 2, y + h / 2)

            # Estimate confidence from face size relative to frame
            frame_h, frame_w = frame.shape[:2]
            size_ratio = (w * h) / (frame_w * frame_h)
            # Larger relative faces → higher confidence (heuristic)
            conf = min(1.0, max(self._confidence_threshold, 0.5 + size_ratio * 50))

            if conf < self._confidence_threshold:
                continue

            # Try to associate with a person track
            associated_track_id: Optional[str] = None
            if person_tracks:
                for trk in person_tracks:
                    if hasattr(trk, "bbox"):
                        trk_bbox = trk.bbox
                        if (
                            trk_bbox.x1 <= face_center[0] <= trk_bbox.x2
                            and trk_bbox.y1 <= face_center[1] <= trk_bbox.y2
                        ):
                            associated_track_id = str(getattr(trk, "track_id", None))
                            break

            face_id = f"face_{camera_id}_{uuid.uuid4().hex[:8]}"
            results.append(
                FaceDetectionResult(
                    face_id=face_id,
                    camera_id=camera_id,
                    timestamp_ns=timestamp_ns,
                    bbox=BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                    ),
                    confidence=conf,
                    associated_person_track_id=associated_track_id,
                )
            )

        return results
