"""Dwell / Loitering Engine — Phase 9.2 (SIH 26187).

Per-zone configurable dwell-time monitoring. Feeds zone_dwell sigma value
directly into SituationalRiskEvaluator's existing zone_dwell > 5.0σ
CRITICAL-rule term. Does NOT invent a separate loitering severity scale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.camera import Zone

_log = get_logger("zones.dwell_engine")


@dataclass
class DwellEvent:
    """Loitering/dwell anomaly event.

    Attributes:
        zone_id: Zone where loitering was detected.
        camera_id: Source camera.
        subject_ref: Pseudonymous subject reference.
        timestamp_ns: When the event was generated.
        dwell_seconds: How long the subject has been in the zone.
        expected_max_dwell_s: Zone's configured maximum expected dwell time.
        zone_dwell_sigma: Deviation in sigma units from zone baseline.
        confidence: Detection confidence.
        explanation: Human-readable explanation.
    """

    zone_id: str
    camera_id: str
    subject_ref: str
    timestamp_ns: int
    dwell_seconds: float
    expected_max_dwell_s: float
    zone_dwell_sigma: float
    confidence: float
    explanation: str


class DwellEngine:
    """Zone dwell-time monitoring engine.

    Tracks how long subjects remain in each zone and computes sigma
    deviation from zone baseline. Output feeds directly into the
    evaluator's existing zone_dwell > 5.0σ CRITICAL-rule term.
    """

    def __init__(
        self,
        default_max_dwell_s: float = 300.0,
        default_std_dwell_s: float = 60.0,
    ) -> None:
        self._default_max_dwell_s = default_max_dwell_s
        self._default_std_dwell_s = default_std_dwell_s

        # {(subject_ref, zone_id): entry_timestamp_ns}
        self._zone_entries: dict[tuple[str, str], int] = {}

    def record_zone_presence(
        self,
        subject_ref: str,
        zone_id: str,
        camera_id: str,
        timestamp_ns: int,
        confidence: float,
        zone: Optional[Zone] = None,
    ) -> Optional[DwellEvent]:
        """Record that a subject is present in a zone and check for loitering.

        Args:
            subject_ref: Subject pseudonym.
            zone_id: Zone where subject is present.
            camera_id: Source camera.
            timestamp_ns: Current timestamp.
            confidence: Detection confidence.
            zone: Optional Zone config for dwell thresholds.

        Returns:
            DwellEvent if loitering threshold exceeded, None otherwise.
        """
        key = (subject_ref, zone_id)

        if key not in self._zone_entries:
            self._zone_entries[key] = timestamp_ns
            return None

        entry_ns = self._zone_entries[key]
        dwell_s = (timestamp_ns - entry_ns) / 1_000_000_000.0

        # Get expected dwell parameters from zone config or defaults
        if zone is not None and zone.expected_dwell_s[1] > 0:
            max_dwell_s = zone.expected_dwell_s[1]
            # Estimate std as 1/3 of the expected range
            min_dwell_s = zone.expected_dwell_s[0]
            std_s = max((max_dwell_s - min_dwell_s) / 3.0, 10.0)
        else:
            max_dwell_s = self._default_max_dwell_s
            std_s = self._default_std_dwell_s

        # Compute sigma deviation
        if std_s > 0:
            zone_dwell_sigma = (dwell_s - max_dwell_s) / std_s
        else:
            zone_dwell_sigma = 0.0

        # Only generate event if dwell exceeds expected maximum
        if dwell_s <= max_dwell_s:
            return None

        zone_name = zone.display_name if zone else zone_id
        explanation = (
            f"Subject {subject_ref} has been in zone '{zone_name}' ({zone_id}) "
            f"for {dwell_s:.1f}s. Expected maximum dwell: {max_dwell_s:.1f}s. "
            f"Deviation: {zone_dwell_sigma:.1f}σ."
        )

        return DwellEvent(
            zone_id=zone_id,
            camera_id=camera_id,
            subject_ref=subject_ref,
            timestamp_ns=timestamp_ns,
            dwell_seconds=dwell_s,
            expected_max_dwell_s=max_dwell_s,
            zone_dwell_sigma=zone_dwell_sigma,
            confidence=confidence,
            explanation=explanation,
        )

    def record_zone_exit(self, subject_ref: str, zone_id: str) -> None:
        """Record that a subject has left a zone."""
        key = (subject_ref, zone_id)
        self._zone_entries.pop(key, None)

    def build_evidence_payload(self, dwell_event: DwellEvent) -> dict:
        """Build evidence payload dict for SituationalRiskEvaluator.

        Sets zone_dwell_sigma so the evaluator's CRITICAL rule
        (zone_dwell > 5.0σ) can fire.
        """
        return {
            "zone_id": dwell_event.zone_id,
            "subject_ref": dwell_event.subject_ref,
            "zone_dwell_sigma": dwell_event.zone_dwell_sigma,
            "dwell_seconds": dwell_event.dwell_seconds,
            "confidence": dwell_event.confidence,
        }

    def reset(self) -> None:
        """Clear all tracked entries."""
        self._zone_entries.clear()
