"""Background Modeling — Phase 3.5 (§16, ADR-005).

Provides GMM (OpenCV BackgroundSubtractorMOG2) background subtractor
with running average fallback for CPU mode. Evaluates scene motion ratio
and calculates background model stability confidence score [0.0, 1.0].
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from gods_eye.config.settings import Settings
from gods_eye.observability.logger import get_logger
from gods_eye.schemas.environment import BackgroundModel

_log = get_logger("environmental.background_model")

try:
    import cv2  # type: ignore[import-untyped]

    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False


class BackgroundModeler:
    """Background model generator and stability evaluator per camera.

    Not thread-safe — owned exclusively by ``EnvironmentalWorker`` for
    its assigned camera stream (spec §9 single-writer).
    """

    def __init__(self, camera_id: str, settings: Settings) -> None:
        self._camera_id = camera_id
        self._cpu_only = settings.cpu_only
        self._learning_rate = settings.bg_learning_rate
        self._warmup_bg_frames = settings.warmup_bg_frames
        self._warmup_bg_confidence = settings.warmup_bg_confidence

        self._version = 1
        self._created_ns = time.time_ns()
        self._last_updated_ns = self._created_ns
        self._frame_count = 0

        self._mog2: Optional[cv2.BackgroundSubtractorMOG2] = None
        self._running_avg: Optional[np.ndarray] = None
        self._model_type = "running_average"

        if _OPENCV_AVAILABLE and not self._cpu_only:
            try:
                self._mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=True
                )
                self._model_type = "gmm"
            except Exception as exc:
                _log.warning(
                    "mog2_init_failed_using_running_avg",
                    camera_id=camera_id,
                    error=str(exc),
                )

        self._fg_ratio_history: list[float] = []
        self._history_capacity = 100

    def apply(
        self, frame: np.ndarray, timestamp_ns: int
    ) -> tuple[np.ndarray, BackgroundModel]:
        """Process a raw video frame, update background model, return fg mask & metadata.

        Args:
            frame: BGR frame array (HxWxC).
            timestamp_ns: Unix nanoseconds capture timestamp.

        Returns:
            Tuple of (binary foreground mask array, BackgroundModel record).
        """
        self._frame_count += 1
        self._last_updated_ns = timestamp_ns

        # Convert to grayscale for background modeling
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if _OPENCV_AVAILABLE:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = (
                    0.299 * frame[:, :, 2]
                    + 0.587 * frame[:, :, 1]
                    + 0.114 * frame[:, :, 0]
                ).astype(np.uint8)
        else:
            gray = frame

        if self._mog2 is not None:
            fg_mask = self._mog2.apply(gray, learningRate=self._learning_rate)
            # Remove shadows (value 127 in OpenCV MOG2)
            fg_mask = np.where(fg_mask > 200, 255, 0).astype(np.uint8)
        else:
            fg_mask, gray_float = self._apply_running_average(gray)

        # Calculate foreground motion ratio
        total_pixels = fg_mask.size
        fg_pixels = int(np.count_nonzero(fg_mask))
        fg_ratio = (
            float(fg_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        )

        self._fg_ratio_history.append(fg_ratio)
        if len(self._fg_ratio_history) > self._history_capacity:
            self._fg_ratio_history.pop(0)

        confidence = self._compute_confidence()
        warm_up_complete = (
            self._frame_count >= self._warmup_bg_frames
            and confidence >= self._warmup_bg_confidence
        )

        model_record = BackgroundModel(
            camera_id=self._camera_id,
            model_type=self._model_type,
            version=self._version,
            created_ns=self._created_ns,
            last_updated_ns=self._last_updated_ns,
            warm_up_complete=warm_up_complete,
            frame_count=self._frame_count,
            confidence=confidence,
        )

        return fg_mask, model_record

    def _apply_running_average(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU fallback running average background subtractor."""
        gray_float = gray.astype(np.float32)
        if self._running_avg is None:
            self._running_avg = gray_float.copy()

        # running_avg = (1 - alpha) * running_avg + alpha * gray
        self._running_avg = (
            1.0 - self._learning_rate
        ) * self._running_avg + self._learning_rate * gray_float
        diff = np.abs(gray_float - self._running_avg)
        fg_mask = np.where(diff > 30.0, 255, 0).astype(np.uint8)
        return fg_mask, gray_float

    def _compute_confidence(self) -> float:
        """Calculate background model stability confidence [0.0, 1.0].

        Based on foreground motion ratio variance over recent history.
        Low variance = stable background model = high confidence.
        """
        if len(self._fg_ratio_history) < 10:
            return min(1.0, 0.5 + (len(self._fg_ratio_history) * 0.03))

        std_dev = float(np.std(self._fg_ratio_history))
        # High std_dev (> 0.2) reduces confidence; low std_dev (< 0.05) keeps confidence high
        stability = max(0.0, 1.0 - (std_dev * 3.0))
        return round(float(np.clip(stability, 0.1, 1.0)), 3)

    def reset(self) -> None:
        """Reset background model state (e.g. after scene reset)."""
        self._version += 1
        self._frame_count = 0
        self._running_avg = None
        self._fg_ratio_history.clear()
        if self._mog2 is not None and _OPENCV_AVAILABLE:
            try:
                self._mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=True
                )
            except Exception:
                pass
        _log.info(
            "background_model_reset",
            camera_id=self._camera_id,
            new_version=self._version,
        )
