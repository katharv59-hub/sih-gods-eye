"""Event schemas — §4 Canonical Data Schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Types of events emitted by the system."""

    PERSON_ENTERED_FRAME = "person_entered_frame"
    PERSON_LEFT_FRAME = "person_left_frame"
    IDENTITY_CONFIRMED = "identity_confirmed"
    IDENTITY_LOST = "identity_lost"
    CROSS_CAMERA_TRANSITION = "cross_camera_transition"
    IDENTITY_PURGED = "identity_purged"
    # Environmental events (Phase 3.5)
    SCENE_LIGHTING_CHANGED = "scene_lighting_changed"
    OCCUPANCY_THRESHOLD = "occupancy_threshold"
    BACKGROUND_MODEL_UPDATED = "background_model_updated"
    ENVIRONMENTAL_ANOMALY = "environmental_anomaly"
    # System events (Phase 3.5)
    SYSTEM_MODE_CHANGED = "system_mode_changed"
    MODEL_DRIFT_DETECTED = "model_drift_detected"
    WARM_UP_COMPLETE = "warm_up_complete"
    # Vehicle perception events (Phase 8 — SIH 26187)
    VEHICLE_DETECTED = "vehicle_detected"
    PLATE_DETECTED = "plate_detected"
    ANPR_READING = "anpr_reading"
    FACE_DETECTED = "face_detected"
    # Spatial & event intelligence events (Phase 9 — SIH 26187)
    VIRTUAL_FENCE_CROSSED = "virtual_fence_crossed"
    RESTRICTED_ZONE_INTRUSION = "restricted_zone_intrusion"
    LOITERING_DETECTED = "loitering_detected"
    NIGHT_MOVEMENT = "night_movement"
    TRAJECTORY_ANOMALY = "trajectory_anomaly"
    # Alert lifecycle events (Phase 10 — SIH 26187)
    ALERT_CREATED = "alert_created"
    ALERT_RESOLVED = "alert_resolved"


@dataclass
class Event:
    """An immutable event record in the append-only event log.

    Attributes:
        event_id: UUID, unique per event.
        event_type: Category of this event.
        global_id: Identity UUID, None if identity not yet resolved.
        camera_id: Source camera identifier.
        zone_id: Logical zone within camera FOV, None if not applicable.
        timestamp_ns: Unix nanoseconds.
        frame_id: Frame counter at time of event.
        confidence: Confidence score in [0.0, 1.0] — REQUIRED (Rule 7).
        explanation: Human-readable reason — REQUIRED (Rule 8).
        metadata: Event-type-specific payload (flat, JSON-serializable).
    """

    event_id: str
    event_type: EventType
    global_id: Optional[str]
    camera_id: str
    timestamp_ns: int
    frame_id: int
    confidence: float = 0.0
    explanation: str = ""
    zone_id: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)
