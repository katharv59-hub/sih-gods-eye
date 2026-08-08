"""Timeline & Replay Schemas — Phase 4.4 (§14).

Defines read-only data models for reconstructed spatiotemporal timelines
and deterministic replay frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from gods_eye.schemas.event import Event


@dataclass(frozen=True)
class VisitSegment:
    """A contiguous visit by an identity at a camera location."""

    global_id: str
    camera_id: str
    zone_id: Optional[str]
    start_ns: int
    end_ns: Optional[int]
    dwell_s: float
    observation_count: int
    is_complete: bool


@dataclass(frozen=True)
class IdentityTimeline:
    """Reconstructed temporal history for a single global identity."""

    global_id: str
    start_ns: int
    end_ns: int
    visits: list[VisitSegment] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class CameraTimeline:
    """Reconstructed activity history for a specific camera."""

    camera_id: str
    start_ns: int
    end_ns: int
    active_identities: list[str] = field(default_factory=list)
    visit_segments: list[VisitSegment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


@dataclass(frozen=True)
class ZoneTimeline:
    """Reconstructed activity history for a specific spatial zone."""

    zone_id: str
    start_ns: int
    end_ns: int
    occupancy_segments: list[dict[str, Any]] = field(default_factory=list)
    unique_identities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayFrame:
    """A single deterministically ordered record in a replay stream."""

    timestamp_ns: int
    sequence_num: int
    record_id: str
    record_type: str  # "event", "observation", "transition", "occupancy"
    data: dict[str, Any]
