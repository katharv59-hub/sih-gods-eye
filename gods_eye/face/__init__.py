"""Face Detection Module — Phase 8.3 (SIH 26187).

Detection only — face recognition/embeddings are explicitly OUT OF SCOPE.
Requires legal-basis ADR prerequisite (same as Phase 6 identity profiling).
"""

from gods_eye.face.detector import FaceDetector, FaceDetectionResult
from gods_eye.face.live_engine import LiveFaceEngine
from gods_eye.face.recognizer import FaceRecognizer, FaceRecognitionResult

__all__ = [
    "FaceDetector",
    "FaceDetectionResult",
    "FaceRecognizer",
    "FaceRecognitionResult",
    "LiveFaceEngine",
]
