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


@dataclass
class Event:
    """An immutable event record in the append-only event log.

    Attributes:
        event_id: UUID, unique per event.
        event_type: Category of this event.
        global_id: Identity UUID, None if identity not yet resolved.
        camera_id: Source camera identifier.
        timestamp_ns: Unix nanoseconds.
        frame_id: Frame counter at time of event.
        metadata: Event-type-specific payload (flat, JSON-serializable).
    """

    event_id: str
    event_type: EventType
    global_id: Optional[str]
    camera_id: str
    timestamp_ns: int
    frame_id: int
    metadata: dict[str, object] = field(default_factory=dict)
