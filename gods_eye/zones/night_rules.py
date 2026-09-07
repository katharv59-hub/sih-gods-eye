"""Night-Time Movement Engine — Phase 9.3 (SIH 26187).

Combines existing LightingCondition (Phase 3.5) with zone intrusion
events to emit NIGHT_MOVEMENT events with a higher confidence bar
(NIGHT_EVENT_MIN_CONFIDENCE = 0.80).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.environment import LightingCondition

_log = get_logger("zones.night_rules")

NIGHT_EVENT_MIN_CONFIDENCE_DEFAULT: float = 0.80

# Lighting conditions considered "night"
NIGHT_CONDITIONS: set[LightingCondition] = {
    LightingCondition.NIGHT_LIT,
    LightingCondition.NIGHT_DARK,
    LightingCondition.DUSK_DAWN,
}


@dataclass
class NightMovementEvent:
    """Night-time movement event.

    Attributes:
        zone_id: Zone where night movement was detected.
        camera_id: Source camera.
        subject_ref: Pseudonymous subject reference.
        timestamp_ns: When the event was detected.
        lighting_condition: Current lighting condition.
        associated_event_type: The zone event that triggered this (e.g. RESTRICTED_ZONE_INTRUSION).
        confidence: Combined confidence score.
        explanation: Human-readable explanation.
    """

    zone_id: str
    camera_id: str
    subject_ref: str
    timestamp_ns: int
    lighting_condition: LightingCondition
    associated_event_type: str
    confidence: float
    explanation: str


class NightMovementEngine:
    """Night-time movement correlation engine.

    Combines LightingCondition with zone intrusion or off-hours movement
    events. Only emits NIGHT_MOVEMENT when:
    1. Lighting is night-time (NIGHT_LIT, NIGHT_DARK, DUSK_DAWN)
    2. A zone intrusion or off-hours movement event co-occurs
    3. Confidence exceeds NIGHT_EVENT_MIN_CONFIDENCE (0.80)
    """

    def __init__(
        self,
        min_confidence: float = NIGHT_EVENT_MIN_CONFIDENCE_DEFAULT,
    ) -> None:
        self._min_confidence = min_confidence

    def evaluate(
        self,
        subject_ref: str,
        zone_id: str,
        camera_id: str,
        timestamp_ns: int,
        lighting_condition: LightingCondition,
        associated_event_type: str,
        event_confidence: float,
    ) -> Optional[NightMovementEvent]:
        """Evaluate whether a zone event constitutes a night-time movement.

        Args:
            subject_ref: Subject pseudonym.
            zone_id: Zone where the event occurred.
            camera_id: Source camera.
            timestamp_ns: Event timestamp.
            lighting_condition: Current lighting condition.
            associated_event_type: The triggering zone event type.
            event_confidence: Confidence of the triggering event.

        Returns:
            NightMovementEvent if conditions met, None otherwise.
        """
        # Only trigger for night-time conditions
        if lighting_condition not in NIGHT_CONDITIONS:
            return None

        # Apply higher confidence bar for night events
        if event_confidence < self._min_confidence:
            _log.debug(
                "night_movement_suppressed_low_confidence",
                subject_ref=subject_ref,
                confidence=event_confidence,
                threshold=self._min_confidence,
            )
            return None

        # Valid night movement event
        explanation = (
            f"Night-time movement detected: subject {subject_ref} triggered "
            f"{associated_event_type} in zone {zone_id} during "
            f"{lighting_condition.value} conditions. "
            f"Night confidence threshold ({self._min_confidence}) met."
        )

        return NightMovementEvent(
            zone_id=zone_id,
            camera_id=camera_id,
            subject_ref=subject_ref,
            timestamp_ns=timestamp_ns,
            lighting_condition=lighting_condition,
            associated_event_type=associated_event_type,
            confidence=event_confidence,
            explanation=explanation,
        )

    def build_evidence_payload(self, event: NightMovementEvent) -> dict:
        """Build evidence payload dict for hypothesis/risk evaluation."""
        return {
            "event_type": "NIGHT_MOVEMENT",
            "zone_id": event.zone_id,
            "subject_ref": event.subject_ref,
            "lighting_condition": event.lighting_condition.value,
            "associated_event_type": event.associated_event_type,
            "confidence": event.confidence,
        }
