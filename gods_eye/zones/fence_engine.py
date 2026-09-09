"""Virtual Fence Engine — Phase 9.1 (SIH 26187).

Point-in-polygon + edge-crossing detection against zone boundaries.
Supports both person Track and VehicleTrack.

CRITICAL: This engine sets is_restricted_zone = True on evidence objects,
which is the literal missing wire for SituationalRiskEvaluator's CRITICAL rule.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from gods_eye.observability.logger import get_logger
from gods_eye.schemas.camera import Zone
from gods_eye.schemas.detection import BoundingBox
from gods_eye.schemas.event import Event, EventType

_log = get_logger("zones.fence_engine")


@dataclass
class FenceEvent:
    """Result of a fence crossing or restricted zone intrusion check.

    Attributes:
        event_type: "VIRTUAL_FENCE_CROSSED" or "RESTRICTED_ZONE_INTRUSION"
        zone_id: Zone that was crossed/intruded.
        camera_id: Source camera.
        subject_ref: Pseudonymous subject reference (person or vehicle).
        timestamp_ns: When the event occurred.
        is_restricted_zone: True if the zone is restricted.
        confidence: Confidence of the detection [0.0, 1.0].
        bbox_center: (x, y) normalized center of the subject at time of event.
        explanation: Human-readable explanation.
    """

    event_type: str
    zone_id: str
    camera_id: str
    subject_ref: str
    timestamp_ns: int
    is_restricted_zone: bool
    confidence: float
    bbox_center: tuple[float, float]
    explanation: str


class FenceEngine:
    """Virtual fence / geofence engine.

    Checks if tracked subjects (persons or vehicles) are inside zone polygons.
    For restricted zones, sets is_restricted_zone=True on the output,
    which feeds directly into SituationalRiskEvaluator's CRITICAL rule.
    """

    def __init__(self) -> None:
        self._previous_positions: dict[str, tuple[str, float, float]] = {}
        # {subject_ref: (zone_id_or_empty, norm_x, norm_y)}

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        """Ray-casting algorithm for point-in-polygon test.

        Args:
            point: (x, y) normalized coordinates.
            polygon: List of (x, y) normalized polygon vertices.

        Returns:
            True if point is inside the polygon.
        """
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def check_subject(
        self,
        subject_ref: str,
        bbox: BoundingBox,
        camera_id: str,
        timestamp_ns: int,
        source_resolution: tuple[int, int],
        zones: list[Zone],
        confidence: float = 0.90,
        subject_class: str = "person",
    ) -> list[FenceEvent]:
        """Check if a subject is inside any zone and generate fence events.

        Args:
            subject_ref: Pseudonymous subject reference.
            bbox: Subject bounding box in absolute pixel coordinates.
            camera_id: Source camera.
            timestamp_ns: Current timestamp.
            source_resolution: (width, height) of source frame.
            zones: List of Zone objects to check against.
            confidence: Detection confidence of the subject.
            subject_class: "person" or vehicle class name.

        Returns:
            List of FenceEvent objects for any triggered zones.
        """
        w, h = source_resolution
        if w <= 0 or h <= 0:
            return []

        # Normalize bbox center to [0,1]
        cx, cy = bbox.center
        norm_x = cx / w
        norm_y = cy / h

        events: list[FenceEvent] = []

        prev = self._previous_positions.get(subject_ref)
        prev_zone_id = prev[0] if prev else ""

        for zone in zones:
            if not zone.polygon or len(zone.polygon) < 3:
                continue

            # Check if subject class is allowed in zone
            if hasattr(zone, "allowed_classes") and zone.allowed_classes:
                if subject_class not in zone.allowed_classes:
                    # Subject class not normally allowed → zone intrusion
                    pass  # Continue checking, this is actually more suspicious

            is_inside = self._point_in_polygon((norm_x, norm_y), zone.polygon)
            was_inside = prev_zone_id == zone.zone_id if prev else False
            is_restricted = zone.zone_type == "restricted"

            # Check time window restrictions
            time_violation = False
            if hasattr(zone, "allowed_time_window") and zone.allowed_time_window is not None:
                import time as _time

                current_hour = _time.gmtime(timestamp_ns // 1_000_000_000).tm_hour
                start_h, end_h = zone.allowed_time_window
                if start_h <= end_h:
                    time_violation = not (start_h <= current_hour < end_h)
                else:
                    time_violation = end_h <= current_hour < start_h

            if is_inside and not was_inside:
                # Subject just entered the zone
                event_type = (
                    "RESTRICTED_ZONE_INTRUSION" if is_restricted else "VIRTUAL_FENCE_CROSSED"
                )

                explanation = (
                    f"Subject {subject_ref} ({subject_class}) entered zone "
                    f"'{zone.display_name}' ({zone.zone_id})"
                )
                if is_restricted:
                    explanation += " — RESTRICTED ZONE"
                if time_violation:
                    explanation += " — outside allowed time window"

                events.append(
                    FenceEvent(
                        event_type=event_type,
                        zone_id=zone.zone_id,
                        camera_id=camera_id,
                        subject_ref=subject_ref,
                        timestamp_ns=timestamp_ns,
                        is_restricted_zone=is_restricted or time_violation,
                        confidence=confidence,
                        bbox_center=(norm_x, norm_y),
                        explanation=explanation,
                    )
                )

            if is_inside:
                self._previous_positions[subject_ref] = (zone.zone_id, norm_x, norm_y)
            elif was_inside and not is_inside:
                # Subject left the zone
                self._previous_positions[subject_ref] = ("", norm_x, norm_y)

        if not any(
            self._point_in_polygon((norm_x, norm_y), z.polygon)
            for z in zones
            if z.polygon and len(z.polygon) >= 3
        ):
            self._previous_positions[subject_ref] = ("", norm_x, norm_y)

        return events

    def build_evidence_payload(self, fence_event: FenceEvent) -> dict:
        """Build evidence payload dict suitable for SituationalRiskEvaluator.

        This is the critical wire: sets is_restricted_zone in the payload
        so the evaluator's CRITICAL rule can fire.
        """
        return {
            "event_type": fence_event.event_type,
            "zone_id": fence_event.zone_id,
            "subject_ref": fence_event.subject_ref,
            "is_restricted_zone": fence_event.is_restricted_zone,
            "confidence": fence_event.confidence,
            "bbox_center_x": fence_event.bbox_center[0],
            "bbox_center_y": fence_event.bbox_center[1],
        }

    def to_event(self, fence_event: FenceEvent, frame_id: int = 1) -> Event:
        """Adapt a FenceEvent into a canonical Event record."""
        ev_type = (
            EventType.RESTRICTED_ZONE_INTRUSION
            if fence_event.event_type.upper() == "RESTRICTED_ZONE_INTRUSION"
            else EventType.VIRTUAL_FENCE_CROSSED
        )
        return Event(
            event_id=f"ev_fence_{fence_event.subject_ref}_{fence_event.timestamp_ns}",
            event_type=ev_type,
            global_id=fence_event.subject_ref,
            camera_id=fence_event.camera_id,
            timestamp_ns=fence_event.timestamp_ns,
            frame_id=frame_id,
            confidence=fence_event.confidence,
            explanation=fence_event.explanation,
            zone_id=fence_event.zone_id,
            metadata=self.build_evidence_payload(fence_event),
        )

    def reset(self) -> None:
        """Clear all tracked positions."""
        self._previous_positions.clear()

