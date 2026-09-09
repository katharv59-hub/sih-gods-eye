"""License Plate OCR — Phase 8.2 (SIH 26187).

Reads text from detected plate regions with configurable confidence floor.
Below ANPR_MIN_CONFIDENCE → plate_text = "uncertain", still persisted, never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gods_eye.observability.logger import get_logger

_log = get_logger("anpr.ocr")

ANPR_MIN_CONFIDENCE_DEFAULT: float = 0.60


@dataclass
class PlateReadingResult:
    """OCR result for a license plate region.

    Attributes:
        plate_text: Recognized plate text, or "uncertain" if below confidence.
        confidence: OCR confidence [0.0, 1.0].
        raw_text: Raw OCR output before filtering.
        is_uncertain: True if confidence < ANPR_MIN_CONFIDENCE.
    """

    plate_text: str
    confidence: float
    raw_text: str
    is_uncertain: bool

    def to_event(
        self,
        camera_id: str = "cam-01",
        timestamp_ns: int = 0,
        frame_id: int = 1,
        vehicle_track_id: Optional[str] = None,
        association_confidence: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> Any:
        """Adapt this OCR reading result into a canonical Event(ANPR_READING)."""
        import uuid
        from gods_eye.schemas.event import Event, EventType

        ev_id = event_id or f"ev_anpr_{uuid.uuid4().hex[:12]}"
        explanation = (
            f"Plate '{self.plate_text}' read with OCR confidence {self.confidence:.2f}"
            if not self.is_uncertain
            else f"Uncertain plate reading '{self.plate_text}' (confidence {self.confidence:.2f} below floor)"
        )
        metadata: dict[str, Any] = {
            "plate_text": self.plate_text,
            "confidence": self.confidence,
            "ocr_confidence": self.confidence,
            "raw_text": self.raw_text,
            "is_uncertain": self.is_uncertain,
            "vehicle_track_id": vehicle_track_id,
            "association_confidence": association_confidence,
        }
        return Event(
            event_id=ev_id,
            event_type=EventType.ANPR_READING,
            global_id=vehicle_track_id,
            camera_id=camera_id,
            zone_id=None,
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            confidence=self.confidence,
            explanation=explanation,
            metadata=metadata,
        )



class PlateOCR:
    """License plate text recognizer.

    Uses a simple approach for the SIH demo:
    1. Preprocess plate image (grayscale, threshold, denoise)
    2. Attempt OCR using pytesseract or EasyOCR if available
    3. Fallback to contour-based character counting if no OCR library

    Enforces ANPR_MIN_CONFIDENCE: below floor → plate_text = "uncertain".
    """

    def __init__(
        self,
        min_confidence: float = ANPR_MIN_CONFIDENCE_DEFAULT,
        use_easyocr: bool = True,
    ) -> None:
        self._min_confidence = min_confidence
        self._use_easyocr = use_easyocr
        self._reader: Optional[object] = None

    def read_plate(self, plate_image: np.ndarray) -> PlateReadingResult:
        """Read text from a cropped plate image.

        Args:
            plate_image: Cropped BGR plate image as numpy array.

        Returns:
            PlateReadingResult with text and confidence.
        """
        if plate_image is None or plate_image.size == 0:
            return PlateReadingResult(
                plate_text="uncertain",
                confidence=0.0,
                raw_text="",
                is_uncertain=True,
            )

        # Try EasyOCR first
        if self._use_easyocr:
            result = self._try_easyocr(plate_image)
            if result is not None:
                return result

        # Fallback: morphological analysis to estimate if plate-like text exists
        return self._fallback_analysis(plate_image)

    def _try_easyocr(self, plate_image: np.ndarray) -> Optional[PlateReadingResult]:
        """Attempt OCR using EasyOCR library."""
        try:
            import easyocr  # type: ignore[import-untyped]

            if self._reader is None:
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

            results = self._reader.readtext(plate_image)  # type: ignore[union-attr]

            if not results:
                return PlateReadingResult(
                    plate_text="uncertain",
                    confidence=0.0,
                    raw_text="",
                    is_uncertain=True,
                )

            # Combine text from all detected regions
            texts = []
            total_conf = 0.0
            for _, text, conf in results:
                texts.append(text.strip())
                total_conf += conf

            raw_text = " ".join(texts).upper()
            avg_conf = total_conf / len(results)

            # Clean plate text: keep alphanumeric
            clean_text = "".join(c for c in raw_text if c.isalnum() or c == " ")

            if avg_conf < self._min_confidence:
                return PlateReadingResult(
                    plate_text="uncertain",
                    confidence=avg_conf,
                    raw_text=raw_text,
                    is_uncertain=True,
                )

            return PlateReadingResult(
                plate_text=clean_text if clean_text.strip() else "uncertain",
                confidence=avg_conf,
                raw_text=raw_text,
                is_uncertain=False,
            )

        except ImportError:
            _log.debug("easyocr_not_available", msg="Falling back to morphological analysis")
            return None
        except Exception as exc:
            _log.warning("easyocr_error", error=str(exc))
            return None

    def _fallback_analysis(self, plate_image: np.ndarray) -> PlateReadingResult:
        """Fallback plate analysis when no OCR library is available.

        Uses morphological analysis to detect if plate-like content exists.
        Returns 'uncertain' with a confidence estimate based on edge density.
        """
        try:
            import cv2  # type: ignore[import-untyped]

            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Count non-zero pixels as a rough "has text" estimate
            text_ratio = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])

            # Plates typically have 30-70% text coverage
            if 0.2 <= text_ratio <= 0.8:
                conf = 0.40  # Low confidence without actual OCR
            else:
                conf = 0.10

            return PlateReadingResult(
                plate_text="uncertain",
                confidence=conf,
                raw_text="[ocr_unavailable]",
                is_uncertain=True,
            )
        except ImportError:
            return PlateReadingResult(
                plate_text="uncertain",
                confidence=0.0,
                raw_text="[opencv_unavailable]",
                is_uncertain=True,
            )

    def to_event(
        self,
        reading: PlateReadingResult,
        camera_id: str = "cam-01",
        timestamp_ns: int = 0,
        frame_id: int = 1,
        vehicle_track_id: Optional[str] = None,
        association_confidence: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> Any:
        """Adapt a PlateReadingResult into a canonical Event(ANPR_READING)."""
        return reading.to_event(
            camera_id=camera_id,
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            vehicle_track_id=vehicle_track_id,
            association_confidence=association_confidence,
            event_id=event_id,
        )

