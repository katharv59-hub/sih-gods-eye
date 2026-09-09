"""Face Recognizer — Phase 10 (SIH 26187).

Uses OpenCV's official YuNet face detector (cv2.FaceDetectorYN) and
SFace face recognizer (cv2.FaceRecognizerSF) to perform face detection,
face alignment, 128-d L2-normalized feature extraction, and cosine
similarity comparison against an enrolled identity gallery.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.detection import BoundingBox

_log = get_logger("face.recognizer")

# Default OpenCV Zoo SFace cosine similarity threshold.
# Values >= 0.363 indicate the same identity per official OpenCV benchmarks.
DEFAULT_FACE_RECOGNITION_THRESHOLD: float = 0.363


@dataclass
class FaceRecognitionResult:
    """Result of recognizing a face against an enrolled gallery.

    Attributes:
        status: "KNOWN" if best similarity >= threshold, else "UNKNOWN".
        person_id: Enrolled person identifier if KNOWN, else None.
        name: Enrolled person name if KNOWN, else None.
        similarity: Cosine similarity score bounded to [0.0, 1.0].
        bbox: Face bounding box in pixel coordinates.
        detection_confidence: Face detection confidence [0.0, 1.0].
        timestamp_ns: Capture timestamp in Unix nanoseconds.
    """

    status: str
    person_id: Optional[str]
    name: Optional[str]
    similarity: float
    bbox: BoundingBox
    detection_confidence: float
    timestamp_ns: Optional[int] = None
    camera_id: Optional[str] = None
    track_id: Optional[str] = None
    global_id: Optional[str] = None


class FaceRecognizer:
    """Production face recognition engine wrapping OpenCV YuNet and SFace models.

    Models:
        - YuNet: face_detection_yunet_2023mar.onnx (FaceDetectorYN)
        - SFace: face_recognition_sface_2021dec.onnx (FaceRecognizerSF)

    Characteristics:
        - 128-dimensional L2-normalized float32 embedding
        - Sub-10ms CPU inference
        - Strict cosine similarity matching with threshold floor
    """

    def __init__(
        self,
        detection_model_path: str = "models/face/face_detection_yunet_2023mar.onnx",
        recognition_model_path: str = "models/face/face_recognition_sface_2021dec.onnx",
        recognition_threshold: float = DEFAULT_FACE_RECOGNITION_THRESHOLD,
        detection_threshold: float = 0.60,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self._detection_model_path = Path(detection_model_path)
        self._recognition_model_path = Path(recognition_model_path)
        self._recognition_threshold = recognition_threshold
        self._detection_threshold = detection_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k

        if not self._detection_model_path.exists():
            raise FileNotFoundError(
                f"YuNet model file not found at: {self._detection_model_path}. "
                "Download official OpenCV Zoo model face_detection_yunet_2023mar.onnx"
            )

        if not self._recognition_model_path.exists():
            raise FileNotFoundError(
                f"SFace model file not found at: {self._recognition_model_path}. "
                "Download official OpenCV Zoo model face_recognition_sface_2021dec.onnx"
            )

        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self._detection_model_path),
                config="",
                input_size=(320, 320),
                score_threshold=self._detection_threshold,
                nms_threshold=self._nms_threshold,
                top_k=self._top_k,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(self._recognition_model_path),
                config="",
            )
            _log.info(
                "face_recognizer_initialized",
                detector=str(self._detection_model_path),
                recognizer=str(self._recognition_model_path),
                threshold=self._recognition_threshold,
            )
        except Exception as exc:
            _log.error("face_recognizer_init_failed", error=str(exc))
            raise RuntimeError(f"Failed to initialize FaceDetectorYN/FaceRecognizerSF: {exc}") from exc

    @property
    def recognition_threshold(self) -> float:
        return self._recognition_threshold

    @property
    def embedding_dimension(self) -> int:
        return 128

    def detect_faces(
        self, frame: np.ndarray
    ) -> list[tuple[BoundingBox, float, np.ndarray]]:
        """Detect all faces in a BGR frame using YuNet.

        Args:
            frame: BGR image numpy array.

        Returns:
            List of tuples: (BoundingBox, confidence, 15-dim face array).
        """
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)

        results: list[tuple[BoundingBox, float, np.ndarray]] = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                conf = float(f[14])
                bbox = BoundingBox(x1=x, y1=y, x2=x + bw, y2=y + bh)
                results.append((bbox, conf, f))
        return results

    def extract_embedding(
        self, frame: np.ndarray, face_data: np.ndarray
    ) -> np.ndarray:
        """Extract a 128-dimensional L2-normalized embedding from a detected face.

        Args:
            frame: Original BGR image numpy array.
            face_data: 15-dim face vector from detect_faces().

        Returns:
            1D numpy array of shape (128,) dtype float32.
        """
        aligned_face = self._recognizer.alignCrop(frame, face_data)
        feature = self._recognizer.feature(aligned_face)
        embedding = feature.flatten().astype(np.float32)

        if embedding.shape[0] != 128:
            raise ValueError(
                f"Unexpected SFace embedding dimension: {embedding.shape[0]} (expected 128)"
            )
        return embedding

    def match(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two face embeddings.

        Args:
            embedding1: First 128-dim embedding.
            embedding2: Second 128-dim embedding.

        Returns:
            Cosine similarity score bounded to [0.0, 1.0].
        """
        e1 = embedding1.reshape(1, -1).astype(np.float32)
        e2 = embedding2.reshape(1, -1).astype(np.float32)
        score = self._recognizer.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE)
        return max(0.0, min(1.0, float(score)))

    def recognize(
        self,
        frame: np.ndarray,
        gallery: list[Any],
        threshold: Optional[float] = None,
        timestamp_ns: Optional[int] = None,
    ) -> FaceRecognitionResult:
        """Recognize a single face in a frame against an enrolled gallery.

        Requires exactly one face in the frame.

        Args:
            frame: BGR image numpy array.
            gallery: List of enrolled face objects or dicts with 'embedding', 'person_id', 'name'.
            threshold: Cosine threshold override. Defaults to self.recognition_threshold.
            timestamp_ns: Optional timestamp for result.

        Returns:
            FaceRecognitionResult with status KNOWN or UNKNOWN.
        """
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        if len(faces) > 1:
            raise ValueError("Multiple faces detected; exactly one face required")

        bbox, conf, face_data = faces[0]
        query_emb = self.extract_embedding(frame, face_data)

        thresh = threshold if threshold is not None else self._recognition_threshold

        best_sim = -1.0
        best_record: Optional[Any] = None

        for record in gallery:
            ref_emb = (
                record.embedding
                if hasattr(record, "embedding")
                else record["embedding"]
            )
            if isinstance(ref_emb, (bytes, bytearray)):
                ref_emb = np.frombuffer(ref_emb, dtype=np.float32)
            elif isinstance(ref_emb, list):
                ref_emb = np.array(ref_emb, dtype=np.float32)

            sim = self.match(ref_emb, query_emb)
            if sim > best_sim:
                best_sim = sim
                best_record = record

        best_sim_bounded = max(0.0, best_sim)

        if best_record is not None and best_sim >= thresh:
            pid = (
                best_record.person_id
                if hasattr(best_record, "person_id")
                else best_record["person_id"]
            )
            pname = (
                best_record.name
                if hasattr(best_record, "name")
                else best_record["name"]
            )
            return FaceRecognitionResult(
                status="KNOWN",
                person_id=str(pid),
                name=str(pname),
                similarity=best_sim_bounded,
                bbox=bbox,
                detection_confidence=conf,
                timestamp_ns=timestamp_ns,
            )
        else:
            return FaceRecognitionResult(
                status="UNKNOWN",
                person_id=None,
                name=None,
                similarity=best_sim_bounded if best_sim >= 0 else 0.0,
                bbox=bbox,
                detection_confidence=conf,
                timestamp_ns=timestamp_ns,
            )
